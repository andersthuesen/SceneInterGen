import torch

from models.smpl import SMPL_SKELETON
from models.utils import *
from models.cfg_sampler import ClassifierFreeSampleModel
from models.blocks import *
from utils.utils import *
from smplx import SMPLLayer

from scipy.spatial.transform import Rotation as R

from models.gaussian_diffusion import (
    MotionDiffusion,
    space_timesteps,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType,
)

from einops import rearrange


class MotionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.input_feats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION

        self.query_token = nn.Parameter(torch.randn(1, self.latent_dim))

        self.embed_motion = nn.Linear(self.input_feats * 2, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(
            self.latent_dim, self.dropout, max_len=2000
        )

        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            seqTransEncoderLayer, num_layers=self.num_layers
        )
        self.out_ln = nn.LayerNorm(self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 512)

    def forward(self, batch):
        x, mask = batch["motions"], batch["mask"]
        B, T, D = x.shape

        x = x.reshape(B, T, 2, -1)[..., :-4].reshape(B, T, -1)

        x_emb = self.embed_motion(x)

        emb = torch.cat(
            [
                self.query_token[torch.zeros(B, dtype=torch.long, device=x.device)][
                    :, None
                ],
                x_emb,
            ],
            dim=1,
        )

        seq_mask = mask > 0.5
        token_mask = torch.ones((B, 1), dtype=bool, device=x.device)
        valid_mask = torch.cat([token_mask, seq_mask], dim=1)

        h = self.sequence_pos_encoder(emb)
        h = self.transformer(h, src_key_padding_mask=~valid_mask)
        h = self.out_ln(h)
        motion_emb = self.out(h[:, 0])

        batch["motion_emb"] = motion_emb

        return batch


class InterDenoiser(nn.Module):
    def __init__(
        self,
        motion_dim: int,
        cam_ext_dim: int,
        desc_dim: int,
        kpts_dim: int,
        num_classes: int,
        num_actions: int,
        max_num_people: int = 16,
        bias=False,
        latent_dim=1024,
        num_frames=240,
        ff_size=1024,
        num_layers=16,
        num_heads=8,
        dropout=0.1,
        activation="gelu",
        cfg_weight=0.0,
        points_group_size=64,
    ):
        super().__init__()

        self.cfg_weight = cfg_weight
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.bias = bias
        self.max_num_people = max_num_people
        self.points_group_size = points_group_size

        self.motion_dim = motion_dim
        self.cam_ext_dim = cam_ext_dim

        # 2000 accommodates both the 1000 diffusion timesteps and 200 seconds of motion generation.
        # (training motion lenghts are 10s of 10fps = 100 frames in total)
        self.pe = PositionalEncoding(self.latent_dim, dropout=0, max_len=2000)

        self.emb_pos = TimestepEmbedder(self.latent_dim, self.pe)

        self.emb_t = TimestepEmbedder(self.latent_dim, self.pe)

        self.emb_cam_ext = nn.Linear(cam_ext_dim, self.latent_dim)

        self.emb_kpts = nn.Linear(kpts_dim, self.latent_dim)
        self.emb_id = nn.Embedding(max_num_people + 1, self.latent_dim)


        # Add an extra identity for the no conditioning case
        self.emb_class = nn.Embedding(num_classes + 1, self.latent_dim)
        # Zero out the no conditioning identity
        self.emb_class.weight.data[0].fill_(0.0)

        self.emb_action = nn.Embedding(num_actions + 1, self.latent_dim)
        # Zero out the no conditioning action
        self.emb_action.weight.data[0].fill_(0.0)

        # Input Embedding
        self.emb_motion = nn.Linear(motion_dim, self.latent_dim)

        # Embed the description
        self.emb_desc = nn.Linear(desc_dim, self.latent_dim)

        self.emb_points_group = nn.Linear(points_group_size * 3, self.latent_dim)

        self.unemb_motion = nn.Linear(self.latent_dim, motion_dim)

        self.unemb_cam_ext = nn.Linear(self.latent_dim, cam_ext_dim)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                batch_first=True,
                d_model=latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                bias=bias,
                activation=activation,
            ),
            num_layers=num_layers,
        )

    def forward(
        self,
        # Required
        x: torch.Tensor,
        timesteps: torch.Tensor,
        # Conditioning
        kpts: Optional[torch.Tensor] = None,
        classes: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        description_emb: Optional[torch.Tensor] = None,
        # Masks
        motion_mask: Optional[torch.Tensor] = None,
        description_mask: Optional[torch.Tensor] = None,
        kpts_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, P, T = x.shape[:3]

        motion, cam_ext = x.split((
            self.motion_dim, self.cam_ext_dim
        ), dim=-1)

        pos_emb = self.emb_pos(torch.arange(T, device=motion.device, dtype=torch.long))

        motion_emb = (
            self.emb_motion(motion)
            # Embed position
            + pos_emb[None, None, :, :]
            # Embed identity (Generate a random "id" for each person)
            + self.emb_id(
                torch.randperm(self.max_num_people, device=motion.device)[:P]
            )[None, :, None, :]
        )

        if classes is not None:
            motion_emb += self.emb_class(classes)[:, :, None, :]

        if actions is not None:
            motion_emb[..., : actions.shape[-1], :] += self.emb_action(actions)

        if kpts is not None and kpts_mask is not None:
            # All kpts are within the frame.
            all_kpts_mask = kpts_mask.all(dim=-1)
            motion_emb[all_kpts_mask] += self.emb_kpts(kpts.flatten(start_dim=-2))[all_kpts_mask]

        # Prepare for transformer
        motion_emb_flat = rearrange(motion_emb, "B P T D -> B (P T) D")

        # Embed diffusion timestep
        t_emb = self.emb_t(timesteps)[:, None, :]

        t_mask_flat = torch.ones(
            t_emb.shape[:-1], dtype=torch.bool, device=t_emb.device
        )

        motion_mask_flat = (
            torch.ones(
                motion_emb_flat.shape[:-1],
                dtype=torch.bool,
                device=motion_emb_flat.device,
            )
            if motion_mask is None
            else motion_mask.flatten(start_dim=1)
        )

        mask_flat = torch.cat(
            (t_mask_flat, motion_mask_flat),
            dim=1,
        )

        # Apply transformer
        in_flat = torch.cat((t_emb, motion_emb_flat), dim=1)

        # Assume same static camera extrinsics for all people and frames
        cam_ext = cam_ext.mean(dim=(1, 2))
        cam_ext_emb = self.emb_cam_ext(cam_ext)

        cam_ext_emb_flat = cam_ext_emb.unsqueeze(1)

        mask_flat = torch.cat((
            torch.ones_like(cam_ext_emb_flat[..., 0], dtype=torch.bool),
            mask_flat
        ), dim=1)

        in_flat = torch.cat((
            cam_ext_emb_flat,
            in_flat
        ), dim=1)

        if description_emb is not None:
            desc_emb_emb_flat = self.emb_desc(description_emb).unsqueeze(1)
            desc_mask_flat = description_mask.unsqueeze(1)

            mask_flat = torch.cat((desc_mask_flat, mask_flat), dim=1)
            in_flat = torch.cat((desc_emb_emb_flat, in_flat), dim=1)

        # Reverse the mask due to : https://pytorch.org/docs/master/generated/torch.nn.Transformer.html#torch.nn.Transformer.forward
        # [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by the attention.
        # # If a BoolTensor is provided, the positions with the value of True will be ignored
        # while the position with the value of False will be unchanged.
        out_flat = self.transformer(in_flat, src_key_padding_mask=~mask_flat)

        # if torch.isnan(out_flat).any():
        #     print("NAN in output")
        #     raise ValueError("NAN in output")

        if description_emb is not None:
            # Pop the description embedding
            out_flat = out_flat[:, 1:]

        # Pop the camera extrinsics
        cam_ext_emb_out, out_flat = out_flat[:, 0], out_flat[:, 1:]

        # Pop the timestep embedding
        _, motion_emb_flat_out = out_flat[:, 0], out_flat[:, 1:]

        # Rearrange motion back to its original shape
        motion_emb_out = rearrange(motion_emb_flat_out, "B (P T) D -> B P T D", P=P)

        motion_out = self.unemb_motion(motion_emb_out)
        cam_ext_out = self.unemb_cam_ext(cam_ext_emb_out)

        # Combine motion and camera extrinsics again
        x_out = torch.cat((
            motion_out,
            cam_ext_out[:, None, None, :].repeat(1, P, T, 1)
        ), dim=-1)

        return x_out


class InterDiffusion(nn.Module):
    def __init__(self, cfg, sampling_strategy="ddim50"):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION
        self.motion_rep = cfg.MOTION_REP

        self.cfg_weight = cfg.CFG_WEIGHT
        self.diffusion_steps = cfg.DIFFUSION_STEPS
        self.beta_scheduler = cfg.BETA_SCHEDULER
        self.sampler = cfg.SAMPLER
        self.sampling_strategy = sampling_strategy

        self.net = InterDenoiser(
            motion_dim=cfg.MOTION_DIM,
            cam_ext_dim=cfg.CAM_EXT_DIM,
            desc_dim=cfg.DESC_DIM,
            kpts_dim=cfg.KPTS_DIM,
            num_classes=cfg.NUM_CLASSES,
            num_actions=cfg.NUM_ACTIONS,
            latent_dim=self.latent_dim,
            ff_size=self.ff_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            activation=self.activation,
            cfg_weight=self.cfg_weight,
        )

        self.betas = get_named_beta_schedule(self.beta_scheduler, self.diffusion_steps)

        timestep_respacing = [self.diffusion_steps]
        self.diffusion = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            motion_rep=self.motion_rep,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )
        self.sampler = create_named_schedule_sampler(self.sampler, self.diffusion)

        self.smpl = SMPLLayer(model_path=cfg.SMPL_MODEL_PATH)

    def batch_cond_mask(
        self, cond: Optional[torch.Tensor], mask_prob=0.1
    ) -> Optional[torch.Tensor]:
        if cond is None:
            return None

        mask = (
            torch.rand(cond.shape[:1] + (1,) * (cond.dim() - 1), device=cond.device)
            > mask_prob
        )
        return cond * mask

    def compute_loss(self, batch, mean: torch.Tensor, std: torch.Tensor):
        x_start: torch.Tensor = batch["x"]
        mask = batch["mask"]

        batch_size, *_ = x_start.shape

        cam_f = batch["cam_f"]
        kpts = batch["kpts"]

        kpts_mask = mask[..., None] & torch.all((-1 < kpts) & (kpts < 1), dim=-1)

        t, weights = self.sampler.sample(batch_size, x_start.device)
        output = self.diffusion.training_losses(
            model=self.net,
            x_start=x_start,
            mask=mask,
            weights=weights,
            t=t,
            t_bar=self.cfg.T_BAR,
            mean=mean,
            std=std,
            cam_f=cam_f,
            kpts=kpts,
            kpts_mask=kpts_mask,
            model_kwargs={
                "motion_mask": batch["motion_mask"],
                "kpts": kpts,
                "kpts_mask": kpts_mask
            },
        )
        return output

    def forward(self, batch):
        num_batches = batch["num_batches"]
        num_frames = batch["num_frames"]
        num_people = batch["num_people"]

        timestep_respacing = self.sampling_strategy
        self.diffusion_test = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            motion_rep=self.motion_rep,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )

        self.cfg_model = ClassifierFreeSampleModel(self.net, self.cfg_weight)
        output = self.diffusion_test.ddim_sample_loop(
            self.cfg_model,
            (num_batches, num_people, num_frames, self.nfeats),
            clip_denoised=False,
            progress=True,
            model_kwargs={"motion_mask": batch["motion_mask"]},
        )

        return {"output": output}
