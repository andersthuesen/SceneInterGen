import os
import os.path
from os.path import join as pjoin
import clip
from datasets.teton import (
    SMPL_JOINTS_DIMS,
    SMPL_JOINTS_SIZE,
)

import torch
import click

from multiprocessing import Pool

# from models import *
from configs import get_config
from models.smpl import SMPL_SKELETON
from models.intergen import InterGen
from geometry import rot6d_to_rotmat

from models.intergen import InterGen

from tqdm import tqdm

import matplotlib.pyplot as plt

from functools import partial


def render_frame(prompt: str, joints: torch.Tensor, foot_contact: torch.Tensor, out: str, t: int):
    plt.cla()
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    fig.suptitle(f"\"{prompt}\"")

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 4)

    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Change the view angle so we look into the X-Y plane with X as the horizontal axis and Y as the vertical
    ax.view_init(elev=10, azim=-45)

    COLORS = ("red", "green", "blue", "yellow")

    for person_idx, person_joints in enumerate(joints[:, t]):
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
    plt.savefig(pjoin(out, f"frame_{t}.png"))


@click.command()
@click.argument("prompt", type=str)
@click.option("-p", type=int, default=2, help="Number of people in the scene")
@click.option("-f", type=int, default=100, help="Number of frames to generate")
@click.option("--out", type=str, help="Output path")
@click.option("--device", type=str, default="cpu", help="Device to run on")
def cli(prompt: str, p: int, f: int, out: str, device: str):
    device = torch.device(device)

    num_people = p
    num_frames = f

    model_cfg = get_config("configs/model.yaml")
    mean = torch.load("mean.pt", map_location=device)
    std = torch.load("std.pt", map_location=device)

    model = InterGen(model_cfg, mean, std)

    if model_cfg.CHECKPOINT:
        ckpt = torch.load(model_cfg.CHECKPOINT, map_location="cpu")
        for k in list(ckpt["state_dict"].keys()):
            if "model" in k:
                ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print("checkpoint state loaded!")

    
    clip_model, _ = clip.load("ViT-L/14@336px", device=device, jit=False)
    model = model.to(device)

    clip_model.eval()
    model.eval()

    tokens = clip.tokenize(prompt, truncate=True).to(device)
    x = clip_model.token_embedding(tokens).type(
        clip_model.dtype
    )
    pe_tokens = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = pe_tokens.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    embs = clip_model.ln_final(x)  # Normalize the final layer


    prompt = "person walks in full circle"
    
    motion = torch.load("/work3/s183926/data/humanml3d/001337/motion.pt")
    num_people, num_frames = motion.shape[:2]
    tokens = torch.load("/work3/s183926/data/humanml3d/001337/description_tokens.pt")[0][None]
    embs = torch.load("/work3/s183926/data/humanml3d/001337/description_embs.pt")[0][None]

    # Generate motion
    result = model.forward_test(
        {
            "motion_mask": None,
            "num_batches": 1,
            "num_people": num_people,
            "num_frames": num_frames,
            "description_token": tokens.float(),
            "description_emb": embs.float(),
        }
    )

    output = result["output"] * std + mean

    joints, joint_vels, pose_6d, foot_contact = output.split(
        [SMPL_JOINTS_SIZE, SMPL_JOINTS_SIZE, 23 * 3 * 2, 4], dim=-1
    )

    joints = joints.view(output.shape[:-1] + SMPL_JOINTS_DIMS[0])
    pose_6d = pose_6d.view(output.shape[:-1] + (23, 3, 2))
    pose = rot6d_to_rotmat(pose_6d)

    os.makedirs(out, exist_ok=True)

    torch.save(joints.cpu().clone(), pjoin(out, "joints.pt"))
    torch.save(pose.cpu().clone(), pjoin(out, "poses.pt"))

    print(prompt, joints.shape, foot_contact.shape, out)

    with Pool() as p:
        for _ in tqdm(
            p.imap_unordered(partial(render_frame, prompt, joints.cpu()[0], foot_contact.cpu()[0], out), range(num_frames)),
            desc="Rendering frames",
            total=num_frames,
        ):
            pass

    # Run ffmpeg
    os.system(
        f"ffmpeg -y -r 10 -i {out}/frame_%d.png -vcodec libx264 -pix_fmt yuv420p {out}/motion.mp4"
    )


if __name__ == "__main__":
    cli()
