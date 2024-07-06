import os
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

from scipy.spatial.transform import Rotation as R

import smplx

import matplotlib.pyplot as plt


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

    smpl_out = smpl()

    # print("yolo", smpl_out.joints.shape)

    feet_ids = [7, 10, 8, 11]

    for batch in dataloader:
        joints = batch["joints"]
        joint_vels = batch["joint_vels"]

        feet_h = joints[..., feet_ids, 2]
        feet_vels = joint_vels[..., feet_ids, :]
        feet_vel = feet_vels.pow(2).sum(dim=-1)

        contact = (feet_vel < 0.001) & (
            feet_h[:, :, 1:] < torch.Tensor([0.12, 0.05, 0.12, 0.05])
        )

        plt.figure()
        for c in contact[0]:
            plt.plot(c)
            # feet_vels = p[..., feet_ids, :]
            # feet_pos = p[..., feet_ids, :]

            # foot_vel = feet_vels.pow(2).sum(dim=-1)
            # foot_h = feet_vels

            # plt.plot(foot_vel)
            # plt.plot(foot_h)
        # plt.plot(foot_vel > 0.001)

        # Horizontal line
        plt.axhline(y=0.001, color="r", linestyle="--", label="Velocity threshold")
        plt.savefig("test.png")
        input("Press for next")

        # kpts = batch["keypoints"]
        # origin_kpt = batch["origin_kpt"]

        # print(
        #     batch["x_len"], batch["num_frames"], batch["num_people"], batch["x"].shape
        # )

        # plt.figure()

        # # Star symbol
        # plt.scatter(origin_kpt[0, 0], origin_kpt[0, 1], c="black", s=10, marker="*")

        # for p, c in zip(range(kpts.shape[1]), ["r", "g", "b"]):
        #     for kpt in kpts[0, p, 0]:
        #         plt.scatter(kpt[0], kpt[1], c=c, s=1)

        #     for i, j in SMPL_SKELETON:
        #         plt.plot(
        #             [kpts[0, p, 0, i, 0], kpts[0, p, 0, j, 0]],
        #             [kpts[0, p, 0, i, 1], kpts[0, p, 0, j, 1]],
        #             c=c,
        #         )

        # plt.xlim(-1, 1)
        # plt.ylim(-1, 1)
        # plt.savefig("test.png")
        # input("Press for next")

        # translation = batch["translation"]
        # global_orient = batch["global_orient"]
        # body_pose = batch["body_pose"]

        # random_rot = torch.from_numpy(R.random().as_matrix()).float()

        # smpl_out = smpl(
        #     global_orient=global_orient.view(-1, 3, 3),
        #     body_pose=body_pose.view(-1, 23, 3, 3),
        #     transl=translation.view(-1, 3),
        # )

        # joints = smpl_out.joints.view(*translation.shape[:1], -1, 3)
        # joints_rot = joints @ random_rot.T

        # smpl_out_rot = smpl(
        #     global_orient=torch.inverse(
        #         torch.inverse(global_orient.view(-1, 3, 3)) @ random_rot.T
        #     ),
        #     body_pose=body_pose.view(-1, 23, 3, 3),
        #     transl=translation.view(-1, 3) @ random_rot.T,
        # )

        # rot_joints = smpl_out_rot.joints.view(*translation.shape[:1], -1, 3)

        # print(joints.shape, rot_joints.shape)

        # # joints = smpl_out.joints
        # # joints_rot = (joints - translation.view(-1, 1, 3)) @ random_rot.T
        # # rot_joints = smpl_out_rot.joints

        # print("diff", joints_rot - rot_joints)
        # print()
        # assert torch.allclose(joints_rot, rot_joints), "Rotation failed"

        # print("Rotation successful")
        # (
        #     motion,
        #     motion_mask,
        #     classes,
        #     actions,
        #     object_points,
        #     object_points_mask,
        #     description_tokens,
        #     description_embs,
        # ) = batch

        # motion = motion * std + mean

        # joints, joint_vels, smpl_6d = motion.split(
        #     [SMPL_JOINTS_SIZE, SMPL_JOINTS_SIZE, 23 * 3 * 2], dim=-1
        # )
        # joints = joints.view(motion.shape[:-1] + SMPL_JOINTS_DIMS[0])

        # os.system("rm -rf test/*")

        # for t in range(joints.shape[2]):
        #     plt.cla()

        #     fig = plt.figure()
        #     ax = plt.axes(projection="3d")
        #     ax.set_xlim(-2, 2)
        #     ax.set_ylim(-2, 2)
        #     ax.set_zlim(0, 4)

        #     # Set labels
        #     ax.set_xlabel("X")
        #     ax.set_ylabel("Y")
        #     ax.set_zlabel("Z")

        #     # Change the view angle so we look into the X-Y plane with X as the horizontal axis and Y as the vertical
        #     ax.view_init(elev=10, azim=-45)

        #     COLORS = (
        #         "red",
        #         "green",
        #     )

        #     if object_points is not None:
        #         obj_points = object_points * std[:3] + mean[:3]
        #         ax.scatter3D(
        #             obj_points[0, :, 0].cpu().numpy(),
        #             obj_points[0, :, 1].cpu().numpy(),
        #             obj_points[0, :, 2].cpu().numpy(),
        #             label="Object",
        #             color="blue",
        #             s=1,
        #         )

        #     for person_idx, person_joints in enumerate(joints[0, :, t]):
        #         ax.scatter3D(
        #             person_joints[:24, 0].cpu().numpy(),
        #             person_joints[:24, 1].cpu().numpy(),
        #             person_joints[:24, 2].cpu().numpy(),
        #             label=f"Person {person_idx + 1}",
        #             color=COLORS[person_idx],
        #             s=1,
        #         )
        #         for i, j in SMPL_SKELETON:
        #             ax.plot(
        #                 [person_joints[i, 0].item(), person_joints[j, 0].item()],
        #                 [person_joints[i, 1].item(), person_joints[j, 1].item()],
        #                 [person_joints[i, 2].item(), person_joints[j, 2].item()],
        #                 color=COLORS[person_idx],
        #             )

        #     plt.legend()
        #     plt.savefig(f"test/frame_{t}.png")

        # os.system(
        #     f"ffmpeg -y -i test/frame_%d.png -framerate 10 -r 10 -c:v libx264 -pix_fmt yuv420p test/plot.mp4"
        # )

        # input("Press for next")
