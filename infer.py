import os.path
from datasets.teton import (
    ACTIONS,
    PERSON_CLASSES,
    SMPL_6D_SIZE,
    SMPL_6D_SIZES,
    SMPL_JOINTS_DIMS,
    SMPL_JOINTS_SIZE,
)

import torch
import lightning as L
import scipy.ndimage.filters as filters

from os.path import join as pjoin
from models import *
from configs import get_config
from models.smpl import SMPL_SKELETON
from utils.plot_script import *
from utils.preprocess import *
from utils import paramUtil

from geometry import rot6d_to_rotmat

from models.intergen import InterGen


class LitGenModel(L.LightningModule):
    def __init__(self, model, cfg):
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

    def plot_t2m(self, mp_data, result_path, caption):
        mp_joint = []
        for joint in mp_data:
            mp_joint.append(joint)

        plot_3d_motion(
            result_path, paramUtil.t2m_kinematic_chain, mp_joint, title=caption, fps=10
        )

    def generate_one_sample(self, name, batch):
        self.model.eval()

        motion_output = self.generate_loop(batch)
        result_path = f"results/{name}.mp4"
        if not os.path.exists("results"):
            os.makedirs("results")

        # self.plot_t2m(
        #     [motion_output[0], motion_output[1]], result_path, batch["prompt"]
        # )

    def generate_loop(self, batch):
        return self.model.forward_test(batch)


def build_models(cfg, mean, std):
    if cfg.NAME == "InterGen":
        model = InterGen(cfg, mean, std)
    return model


if __name__ == "__main__":
    # torch.manual_seed(37)
    model_cfg = get_config("configs/model.yaml")
    infer_cfg = get_config("configs/infer.yaml")

    mean = torch.load("mean.pt")
    std = torch.load("std.pt")

    model = build_models(model_cfg, mean, std)

    if model_cfg.CHECKPOINT:
        ckpt = torch.load(model_cfg.CHECKPOINT, map_location="cpu")
        for k in list(ckpt["state_dict"].keys()):
            if "model" in k:
                ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print("checkpoint state loaded!")

    device = torch.device("cuda:0")
    litmodel = LitGenModel(model, infer_cfg).to(device)

    classes = torch.zeros(1, 2, dtype=torch.long, device=device)

    classes[:, 0] = PERSON_CLASSES.index("patient") + 1
    classes[:, 1] = PERSON_CLASSES.index("medical_staff") + 1
    # classes[:, 2] = PERSON_CLASSES.index("medical_staff") + 1

    actions = torch.zeros(1, 2, 200, dtype=torch.long, device=device)
    actions[:, 0, :40] = ACTIONS.index("laying_in_bed") + 1
    actions[:, 0, 40:] = ACTIONS.index("sitting_in_walker") + 1
    # actions[:, 0, 30:50] = ACTIONS.index("sitting_on_bed_edge") + 1
    # actions[:, 0, 50:] = ACTIONS.index("standing_on_floor") + 1

    actions[:, 1, :] = ACTIONS.index("standing_on_floor") + 1
    # actions[:, 2, :] = ACTIONS.index("standing_on_floor") + 1

    out = litmodel.generate_loop(
        {
            "classes": classes,
            "actions": actions,
        },
    )

    output = out["output"]

    output = output * std.to(device) + mean.to(device)

    smpl_6d, joints, joint_vels = output.split(
        [SMPL_6D_SIZE, SMPL_JOINTS_SIZE, SMPL_JOINTS_SIZE], dim=-1
    )

    # smpl_trans, smpl_6d_rot, smpl_6d_pose = smpl_6d.split(SMPL_6D_SIZES, dim=-1)

    # smpl_rot = rot6d_to_rotmat(smpl_6d_rot.view(output.shape[:-1] + (3, 2)))

    # smpl_pose = rot6d_to_rotmat(smpl_6d_pose.view(output.shape[:-1] + (23, 3, 2)))

    # # print(smpl_trans.shape, smpl_rot.shape, smpl_pose.shape)

    # motion = torch.cat(
    #     [
    #         smpl_trans,
    #         smpl_rot.view(output.shape[:-1] + (-1,)),
    #         smpl_pose.view(output.shape[:-1] + (-1,)),
    #     ],
    #     dim=-1,
    # )

    # torch.save(motion, "results/motion.pt")

    # exit(0)

    joints = joints.view(output.shape[:-1] + SMPL_JOINTS_DIMS[0])
    joint_vels = joint_vels.view(output.shape[:-1] + SMPL_JOINTS_DIMS[0])

    #
    # joints = joints[..., :1, :, :] + joint_vels.cumsum(dim=2)

    import matplotlib.pyplot as plt
    import time

    min_x = joints[..., :24, 0].min().item()
    max_x = joints[..., :24, 0].max().item()

    min_y = joints[..., :24, 1].min().item()
    max_y = joints[..., :24, 1].max().item()

    min_z = joints[..., :24, 2].min().item()
    max_z = joints[..., :24, 2].max().item()

    # server = viser.ViserServer()

    for t in range(100):
        plt.cla()
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(max_y, min_y)
        ax.set_zlim(min_z, max_z)

        # Set labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Change the view angle so we look into the X-Y plane with X as the horizontal axis and Y as the vertical
        ax.view_init(elev=100, azim=-90)

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
        plt.savefig("results/plot.png")
        time.sleep(0.25)

    # while True:
    #     pass

    # for i in range(100):
    #     plt.cla()
    #     plt.xlim(min_x, max_x)
    #     plt.ylim(min_y, max_y)
    #     plt.scatter(
    #         joints[0, 0, i, :22, 0].cpu().numpy(),
    #         joints[0, 0, i, :22, 1].cpu().numpy(),
    #         label="Person 1",
    #     )
    #     plt.scatter(
    #         joints[0, 1, i, :22, 0].cpu().numpy(),
    #         joints[0, 1, i, :22, 1].cpu().numpy(),
    #         label="Person 2",
    #     )
    #     plt.legend()
    #     plt.savefig("results/plot.png")
    #     time.sleep(0.25)

    # litmodel.plot_t2m(
    #     joints[0].cpu().numpy(),
    #     "results/test.mp4",
    #     "Hello world",
    # )
