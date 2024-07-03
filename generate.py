import os.path
import clip
from datasets.teton import (
    SMPL_JOINTS_DIMS,
    SMPL_JOINTS_SIZE,
)

import torch
import lightning as L

from os.path import join as pjoin
from models import *
from configs import get_config
from models.smpl import SMPL_SKELETON
from utils.plot_script import *
from utils.preprocess import *
from utils import paramUtil


from models.intergen import InterGen


class LitGenModel(L.LightningModule):
    def __init__(self, model: InterGen, cfg):
        super().__init__()
        # cfg init
        self.cfg = cfg

        self.automatic_optimization = False

        self.save_root = pjoin(self.cfg.GENERAL.CHECKPOINT, self.cfg.GENERAL.EXP_NAME)
        self.model_dir = pjoin(self.save_root, "model")
        self.meta_dir = pjoin(self.save_root, "meta")
        self.log_dir = pjoin(self.save_root, "log")

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # train model init
        self.model = model

    def generate_loop(self, batch):
        return self.model.forward_test(batch)


if __name__ == "__main__":
    model_cfg = get_config("configs/model.yaml")

    mean = torch.load("mean.pt")
    std = torch.load("std.pt")

    model = InterGen(model_cfg, mean, std)

    if model_cfg.CHECKPOINT:
        ckpt = torch.load(model_cfg.CHECKPOINT, map_location="cpu")
        for k in list(ckpt["state_dict"].keys()):
            if "model" in k:
                ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print("checkpoint state loaded!")

    device = torch.device("cpu")  # Force CPU for now
    clip_model, _ = clip.load("ViT-L/14@336px", device=device, jit=False)
    model = model.to(device)

    num_frames = 50
    num_people = 1

    # Setup conditioning
    # classes = torch.zeros(1, num_people, dtype=torch.long, device=device)
    # classes[:, 0] = PERSON_CLASSES.index("patient") + 1
    # classes[:, 1] = PERSON_CLASSES.index("medical_staff") + 1

    # actions = torch.zeros(1, num_people, num_frames, dtype=torch.long, device=device)
    # actions[:, 0] = ACTIONS.index("laying_in_bed") + 1
    # actions[:, 1, :] = ACTIONS.index("standing_on_floor") + 1

    prompt = "The person is walking forwards"

    tokens = clip.tokenize(prompt).to(device)
    x = clip_model.token_embedding(tokens)
    pe_tokens = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = pe_tokens.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)
    embs = clip_model.ln_final(x).type(clip_model.dtype)

    # Generate motion
    out = model.forward_test(
        {
            "motion_mask": None,
            "num_batches": 1,
            "num_people": num_people,
            "num_frames": num_frames,
            "classes": None,
            "actions": None,
            "description_tokens": tokens,
            "description_embs": embs,
            "object_points": None,
            "object_points_mask": None,
        }
    )

    output = out["output"]
    output = output * std.to(device) + mean.to(device)

    joints, joint_vels, smpl_6d = output.split(
        [SMPL_JOINTS_SIZE, SMPL_JOINTS_SIZE, 23 * 3 * 2], dim=-1
    )

    joints = joints.view(output.shape[:-1] + SMPL_JOINTS_DIMS[0])

    import matplotlib.pyplot as plt

    os.system("rm -rf results/*.png")
    for t in range(num_frames):
        plt.cla()
        fig = plt.figure()
        ax = plt.axes(projection="3d")

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(0, 4)

        # Set labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Change the view angle so we look into the X-Y plane with X as the horizontal axis and Y as the vertical
        ax.view_init(elev=10, azim=-45)

        COLORS = (
            "red",
            "green",
        )

        for person_idx, person_joints in enumerate(joints[0, :, t]):
            ax.scatter3D(
                person_joints[:24, 0].cpu().numpy(),
                person_joints[:24, 1].cpu().numpy(),
                person_joints[:24, 2].cpu().numpy(),
                label=f"Person {person_idx + 1}",
                color=COLORS[person_idx],
                s=1,
            )
            for i, j in SMPL_SKELETON:
                ax.plot(
                    [person_joints[i, 0].item(), person_joints[j, 0].item()],
                    [person_joints[i, 1].item(), person_joints[j, 1].item()],
                    [person_joints[i, 2].item(), person_joints[j, 2].item()],
                    color=COLORS[person_idx],
                )

        plt.legend()
        plt.savefig(f"results/frame_{t}.png")

    # Run ffmpeg
    os.system(
        f"ffmpeg -y -r 10 -i results/frame_%d.png -vcodec libx264 -pix_fmt yuv420p results/plot.mp4"
    )
