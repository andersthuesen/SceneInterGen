import torch
from configs import get_config
from datasets import DataModule
from models.smpl import SMPL_SKELETON
import matplotlib.pyplot as plt
import time
from datasets.teton import (
    SMPL_JOINTS_DIMS,
    SMPL_JOINTS_SIZE,
)

import smplx


if __name__ == "__main__":
    model_cfg = get_config("configs/model.yaml")
    train_cfg = get_config("configs/train.yaml")
    data_cfg = get_config("configs/datasets.yaml")

    smpl = smplx.SMPLLayer(
        model_path=model_cfg.SMPL_MODEL_PATH,
    )
    mean = torch.load("mean.pt")
    std = torch.load("std.pt")

    datamodule = DataModule(
        data_cfg,
        1,
        1,
        smpl,
        mean,
        std,
    )
    datamodule.setup()

    dataloader = datamodule.train_dataloader()

    for batch in dataloader:
        (
            motion,
            motion_mask,
            classes,
            actions,
            object_points,
            object_points_mask,
            description_tokens,
            description_embs,
        ) = batch

        motion = motion * std + mean

        joints, joint_vels, smpl_6d = motion.split(
            [SMPL_JOINTS_SIZE, SMPL_JOINTS_SIZE, 23 * 3 * 2], dim=-1
        )
        joints = joints.view(motion.shape[:-1] + SMPL_JOINTS_DIMS[0])

        # server = viser.ViserServer()

        for t in range(joints.shape[2]):
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

            if object_points is not None:
                obj_points = object_points * std[:3] + mean[:3]
                ax.scatter3D(
                    obj_points[0, :, 0].cpu().numpy(),
                    obj_points[0, :, 1].cpu().numpy(),
                    obj_points[0, :, 2].cpu().numpy(),
                    label="Object",
                    color="blue",
                    s=1,
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
            plt.savefig("results/plot.png")
            time.sleep(0.05)
