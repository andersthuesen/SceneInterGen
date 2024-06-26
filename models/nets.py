import torch

from models.utils import *
from models.cfg_sampler import ClassifierFreeSampleModel
from models.blocks import *
from utils.utils import *
from smplx import SMPLLayer

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
        num_features,
        num_classes,
        num_actions,
        max_num_people=16,
        bias=False,
        latent_dim=512,
        num_frames=240,
        ff_size=1024,
        num_layers=8,
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
        self.input_feats = num_features
        self.bias = bias
        self.max_num_people = max_num_people
        self.points_group_size = points_group_size

        self.emb_pos = PositionalEncoding(self.latent_dim, dropout=0)
        self.emb_t = TimestepEmbedder(self.latent_dim, self.emb_pos)

        self.emb_id = nn.Embedding(max_num_people + 1, self.latent_dim)

        # Add an extra identity for the no conditioning case
        self.emb_class = nn.Embedding(num_classes + 1, self.latent_dim)
        # Zero out the no conditioning identity
        self.emb_class.weight.data[0].fill_(0.0)

        self.emb_action = nn.Embedding(num_actions + 1, self.latent_dim)
        # Zero out the no conditioning action
        self.emb_action.weight.data[0].fill_(0.0)

        # Input Embedding
        self.emb_motion = nn.Linear(self.input_feats, self.latent_dim)

        # Embed the description
        self.emb_desc = nn.Linear(768, self.latent_dim)

        self.emb_points_group = nn.Linear(points_group_size * 3, self.latent_dim)

        self.unemb_motion = nn.Linear(self.latent_dim, self.input_feats)

        # self.blocks = nn.ModuleList()
        # for i in range(num_layers):
        #     self.blocks.append(
        #         TransformerBlock(
        #             num_heads=num_heads,
        #             latent_dim=latent_dim,
        #             dropout=dropout,
        #             ff_size=ff_size,
        #         )
        #     )
        # # Output Module
        # self.out = zero_module(FinalLayer(self.latent_dim, self.input_feats))

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
        motion: torch.Tensor,
        timesteps: torch.Tensor,
        # Conditioning
        classes: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        description_emb: Optional[torch.Tensor] = None,
        object_points: Optional[torch.Tensor] = None,
        # Masks
        motion_mask: Optional[torch.Tensor] = None,
        description_mask: Optional[torch.Tensor] = None,
        object_points_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, P, T = motion.shape[:3]

        motion_emb = (
            self.emb_motion(motion)
            # Embed position
            + self.emb_pos.pe[:T, :][None, None, :, :]
            # Embed identity (Generate a random "id" for each person)
            + self.emb_id(
                torch.randperm(self.max_num_people, device=motion.device)[:P]
            )[None, :, None, :]
        )

        if classes is not None:
            motion_emb += self.emb_class(classes)[:, :, None, :]

        if actions is not None:
            motion_emb[..., : actions.shape[-1], :] += self.emb_action(actions)

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

        if description_emb is not None:
            desc_emb_emb_flat = self.emb_desc(description_emb).unsqueeze(1)
            desc_mask_flat = description_mask.unsqueeze(1)

            mask_flat = torch.cat((desc_mask_flat, mask_flat), dim=1)
            in_flat = torch.cat((desc_emb_emb_flat, in_flat), dim=1)

        if object_points is not None:
            # Group points into 32
            object_point_groups = object_points.split(64, dim=1)
            object_point_group_masks = object_points_mask.split(64, dim=1)

            point_group_embs = torch.stack(
                [
                    self.emb_points_group(
                        torch.nn.functional.pad(
                            object_points_group.flatten(start_dim=1),
                            (
                                0,
                                (self.points_group_size - object_points_group.shape[1])
                                * object_points_group.shape[2],
                            ),
                            "constant",
                        )
                    )
                    for object_points_group in object_point_groups
                ],
                dim=1,
            )

            point_group_masks = torch.stack(
                [
                    torch.nn.functional.pad(
                        object_points_mask.flatten(start_dim=1),
                        (0, (self.points_group_size - object_points_mask.shape[1])),
                        "constant",
                    )
                    for object_points_mask in object_point_group_masks
                ],
                dim=1,
            ).any(dim=-1)

            mask_flat = torch.cat((point_group_masks, mask_flat), dim=1)
            in_flat = torch.cat((point_group_embs, in_flat), dim=1)

        # Reverse the mask due to : https://pytorch.org/docs/master/generated/torch.nn.Transformer.html#torch.nn.Transformer.forward
        # [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by the attention.
        # # If a BoolTensor is provided, the positions with the value of True will be ignored
        # while the position with the value of False will be unchanged.
        out_flat = self.transformer(in_flat, src_key_padding_mask=~mask_flat)

        if object_points is not None:
            # Pop the object points embedding
            out_flat = out_flat[:, point_group_masks.shape[1] :]

        if description_emb is not None:
            # Pop the description embedding
            out_flat = out_flat[:, 1:]

        # Pop the timestep embedding
        out_flat = out_flat[:, 1:]

        # Rearrange back to the original shape
        out = rearrange(out_flat, "B (P T) D -> B P T D", P=P)

        return self.unemb_motion(out)


class InterDiffusion(nn.Module):
    def __init__(self, cfg, sampling_strategy="ddim50"):
        super().__init__()
        self.cfg = cfg
        self.nfeats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION
        self.motion_rep = cfg.MOTION_REP
        self.num_actions = cfg.NUM_ACTIONS
        self.num_classes = cfg.NUM_CLASSES

        self.cfg_weight = cfg.CFG_WEIGHT
        self.diffusion_steps = cfg.DIFFUSION_STEPS
        self.beta_scheduler = cfg.BETA_SCHEDULER
        self.sampler = cfg.SAMPLER
        self.sampling_strategy = sampling_strategy

        self.net = InterDenoiser(
            num_features=self.nfeats,
            num_classes=self.num_classes,
            num_actions=self.num_actions,
            latent_dim=self.latent_dim,
            ff_size=self.ff_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            activation=self.activation,
            cfg_weight=self.cfg_weight,
        )

        self.diffusion_steps = self.diffusion_steps
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

    def mask_cond(
        self, cond: Optional[torch.Tensor], mask_prob=0.1
    ) -> Optional[torch.Tensor]:
        # Mask out whole batch with 10% probability (classifier free guidance)
        if cond is None:
            return None

        mask = (
            torch.randn(cond.shape[:1] + (1,) * (cond.dim() - 1), device=cond.device)
            < mask_prob
        )
        return cond * mask

    def compute_loss(self, batch, mean: torch.Tensor, std: torch.Tensor):
        x_start = batch["motion"]

        motion_mask = batch["motion_mask"]

        classes = self.mask_cond(batch["classes"])
        actions = self.mask_cond(batch["actions"])
        object_points_mask = self.mask_cond(batch["object_points_mask"])
        description_mask = self.mask_cond(batch["description_mask"])

        t, _ = self.sampler.sample(x_start.shape[0], x_start.device)
        output = self.diffusion.training_losses(
            model=self.net,
            smpl=self.smpl,
            x_start=x_start,
            mask=motion_mask,
            t=t,
            t_bar=self.cfg.T_BAR,
            mean=mean,
            std=std,
            model_kwargs={
                "motion_mask": motion_mask,
                "classes": classes,
                "actions": actions,
                "object_points": batch["object_points"],
                "object_points_mask": object_points_mask,
                "description_emb": batch["description_emb"],
                "description_mask": description_mask,
            },
        )
        return output

    def forward(self, batch):
        classes = batch["classes"]
        actions = batch["actions"]

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
            actions.shape + (self.nfeats,),
            clip_denoised=False,
            progress=True,
            model_kwargs={
                "classes": classes,
                "actions": actions,
            },
        )
        return {"output": output}
