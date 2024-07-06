import os
import math
import torch
import pickle
import random
from torch.utils.data import Dataset
from typing import Callable, TypeVar, Tuple, List, Optional
import common.data_types as data_types
from smplx import SMPLLayer

from scipy.spatial.transform import Rotation as R

from collections import defaultdict

from geometry import rotmat_to_rot6d

from tqdm import tqdm

Meta = Tuple[str, str, str, str]
T = TypeVar("T")

PERSON_CLASSES = ["person", "patient", "medical_staff"]
ACTIONS = [
    "laying_in_bed",
    "laying_on_floor",
    "sitting_in_bed",
    "sitting_in_chair",
    "sitting_in_lift",
    "standing_on_floor",
    "walking_with_walking_aid",
    "sitting_in_walker",
    "sitting_on_bed_edge",
    "sitting_on_floor",
    "kneeling",
    "crawling",
]

SMPL_DIMS = (
    (3,),  # Global translation
    (3, 3),  # Global orientation
    (23, 3, 3),  # Pose
)
SMPL_SIZES = tuple(math.prod(feat) for feat in SMPL_DIMS)
SMPL_SIZE = sum(SMPL_SIZES)


SMPL_JOINTS_DIMS = ((24, 3),)
SMPL_JOINTS_SIZES = tuple(math.prod(feat) for feat in SMPL_JOINTS_DIMS)
SMPL_JOINTS_SIZE = sum(SMPL_JOINTS_SIZES)


SMPL_6D_DIMS = ((3,), (3, 2), (23, 3, 2))
SMPL_6D_SIZES = tuple(math.prod(feat) for feat in SMPL_6D_DIMS)
SMPL_6D_SIZE = sum(SMPL_6D_SIZES)

# Mapping from left to right joints
SMPL_LEFT_RIGHT_MAP = [
    (0, 0),
    (1, 2),
    (3, 3),
    (4, 5),
    (6, 6),
    (7, 8),
    (9, 9),
    (10, 11),
    (12, 12),
    (13, 14),
    (15, 15),
    (16, 17),
    (18, 19),
    (20, 21),
    (22, 23),
]

SMPL_POSE_FLIP = [
    1,
    0,
    2,
    4,
    3,
    5,
    7,
    6,
    8,
    10,
    9,
    11,
    13,
    12,
    14,
    16,
    15,
    18,
    17,
    20,
    19,
    22,
    21,
]


class TetonDataset(Dataset):
    def __init__(
        self,
        root_path: str,
        transform: Callable = None,
        augment: Callable = None,
        cache=False,
        motion_filename="optimized_motion_transformed.pt",
    ):
        super(TetonDataset, self).__init__()
        self.transform = transform
        self.augment = augment
        self.cache = cache
        self.motion_filename = motion_filename
        path_cache_path = os.path.join(root_path, "path_cache.pkl")
        self.paths = (
            pickle.load(open(path_cache_path, "rb"))
            if cache and os.path.exists(path_cache_path)
            else [
                path
                for path, _, files in tqdm(os.walk(root_path), desc="Loading dataset")
                if motion_filename in files
            ]
        )

        if cache and not os.path.exists(path_cache_path):
            pickle.dump(self.paths, open(path_cache_path, "wb"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index) -> dict:
        path = self.paths[index]

        if self.cache and os.path.exists(os.path.join(path, "cache.pkl")):
            try:
                data = pickle.load(open(os.path.join(path, "cache.pkl"), "rb"))
                if self.augment:
                    data = self.augment(data)
                return data
            except EOFError as e:
                print(f"Error loading cache: {e}")
                print(f"Path: {path}")
                print("Regenerating data...")
                pass

        motion_path = os.path.join(path, self.motion_filename)

        motion = torch.load(motion_path).float()
        motion_mask, translation, global_orient, body_pose = motion.split(
            [1, *SMPL_SIZES], dim=-1
        )

        motion_mask = motion_mask.squeeze(-1).bool()
        global_orient = global_orient.view(*motion_mask.shape, 3, 3)
        body_pose = body_pose.view(*motion_mask.shape, 23, 3, 3)

        classes_path = os.path.join(path, "class.pt")
        actions_path = os.path.join(path, "action.pt")
        object_points_path = os.path.join(path, "object_points_transformed.pt")
        description_tokens_path = os.path.join(path, "description_tokens.pt")
        description_embs_path = os.path.join(path, "description_embs.pt")

        classes = torch.load(classes_path) if os.path.exists(classes_path) else None
        actions = torch.load(actions_path) if os.path.exists(actions_path) else None
        description_tokens = (
            torch.load(description_tokens_path)
            if os.path.exists(description_tokens_path)
            else None
        )
        description_embs = (
            torch.load(description_embs_path)
            if os.path.exists(description_embs_path)
            else None
        )

        object_points = (
            torch.load(object_points_path)
            if os.path.exists(object_points_path)
            else None
        )

        object_points_mask = (
            torch.ones(
                object_points.shape[0], dtype=torch.bool, device=object_points.device
            )
            if object_points is not None
            else None
        )

        data = {
            # "path": path,
            "translation": translation,
            "global_orient": global_orient,
            "body_pose": body_pose,
            "motion_mask": motion_mask,
            "classes": classes,
            "actions": actions,
            "object_points": object_points,
            "object_points_mask": object_points_mask,
            "description_tokens": description_tokens,
            "description_embs": description_embs,
        }

        if self.transform:
            data = self.transform(data)

        if self.cache:
            pickle.dump(data, open(os.path.join(path, "cache.pkl"), "wb"))

        if self.augment:
            data = self.augment(data)

        return data


class AppendSMPLJoints:
    def __init__(self, smpl: SMPLLayer):
        self.smpl = smpl

    def __call__(self, data: dict) -> dict:
        translation = data["translation"]
        global_orient = data["global_orient"]
        body_pose = data["body_pose"]

        smpl_out = self.smpl.forward(
            global_orient=global_orient.view(-1, 3, 3),
            body_pose=body_pose.view(-1, 23, 3, 3),
            transl=translation.view(-1, 3),
            return_verts=False,
            return_full_pose=False,
        )

        joints = smpl_out.joints.view(*translation.shape[:-1], -1, 3)[..., :24, :]

        return {**data, "joints": joints}


class AppendJointVelocities:
    def __call__(self, data: dict) -> dict:
        joints = data["joints"]
        joint_vels = joints[:, 1:] - joints[:, :-1]

        return {**data, "joint_vels": joint_vels}


class AppendRandomCamera:
    def __call__(self, data: dict) -> dict:
        # Pick random camera elevation and radius
        cam_r = torch.rand(1) * 9 + 1  # From 1 to 10 meters
        cam_h = torch.rand(1) * 4.5 - 0.5  # From -0.5 to 4 meters

        cam_rot = torch.rand(1) * 2 * torch.pi

        # Camera is looking in the y- direction. left right is x and up and down is z.
        cam_pos = torch.tensor(
            [cam_r * torch.sin(cam_rot), cam_r * torch.cos(cam_rot), cam_h]
        )

        # Compute angle from camera to person with some noise
        cam_tilt = (
            -torch.pi / 2
            - torch.atan(cam_h / (cam_r + 1e-6))
            + torch.randn(1) * 0.05 * torch.pi
        )

        cam_roll = torch.randn(1) * 0.01 * torch.pi
        cam_azim = torch.randn(1) * 0.01 * torch.pi + cam_rot

        cam_R = torch.from_numpy(
            R.from_euler(
                "xyz",
                torch.cat((cam_tilt, cam_azim, cam_roll)),
            ).as_matrix()
        ).type_as(cam_pos)

        cam_t = -cam_pos @ cam_R.T

        dist = torch.norm(cam_pos)

        # Random focal length
        cam_f = torch.tensor([dist / 2]) + torch.randn(1) * 0.25

        return {
            **data,
            "cam_R": cam_R,
            "cam_t": cam_t,
            "cam_f": cam_f,
        }


class AppendFootContacts:
    def __call__(seof, data: dict) -> dict:
        feet_ids = [7, 10, 8, 11]

        joints = data["joints"]
        joint_vels = data["joint_vels"]

        feet_h = joints[..., feet_ids, 2]
        feet_vels = joint_vels[..., feet_ids, :]
        feet_vel = feet_vels.pow(2).sum(dim=-1)

        foot_contact = (feet_vel < 0.001) & (
            feet_h[:, 1:] < torch.Tensor([0.12, 0.05, 0.12, 0.05])
        )

        return {**data, "foot_contact": foot_contact}


class AppendRenderedKeypoints:
    def __call__(self, data: dict) -> dict:
        joints = data["joints"]
        cam_R = data["cam_R"]
        cam_t = data["cam_t"]
        cam_f = data["cam_f"]

        # Translate joints to camera space
        joints_cam = joints @ cam_R.T + cam_t[None, None, :]

        # Project joints
        kpts = joints_cam[..., :2] / (joints_cam[..., 2:3] + 1e-6) * cam_f

        origin_cam = torch.zeros(3) @ cam_R.T + cam_t
        origin_kpt = origin_cam[:2] / (origin_cam[2] + 1e-6) * cam_f

        return {
            **data,
            "kpts": kpts,
            "origin_kpt": origin_kpt,
        }


class ToNonCannonical:
    def __init__(self, smpl: SMPLLayer):
        self.smpl = smpl

    def __call__(self, data: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        motions, *rest = data
        translation, global_orient, body_pose = motions.view(-1, SMPL_SIZE).split(
            SMPL_SIZES,
            dim=-1,
        )

        out = self.smpl.forward(
            global_orient=global_orient,
            body_pose=body_pose,
            transl=translation,
            return_verts=False,
            return_full_pose=False,
        )

        body_pose_6d = body_pose.reshape(*body_pose.shape[:-1], 23, 3, 3)[..., :2]
        joints = out.joints.view(*motions.shape[:-1], -1, 3)[..., :24, :]

        joint_vels = torch.cat(
            (
                torch.zeros_like(joints[..., :1, :]),
                joints[..., 1:, :] - joints[..., :-1, :],
            ),
            dim=-2,
        )

        x = torch.cat(
            (
                joints.view(*motions.shape[:-1], -1),
                joint_vels.view(*motions.shape[:-1], -1),
                body_pose_6d.reshape(*motions.shape[:-1], -1),
            ),
            dim=-1,
        )

        # Append SMPL joints to the end of the motion tensor
        return x, *rest


class ChooseRandomDescription:
    def __call__(self, data: dict) -> dict:
        if "description_tokens" not in data or "description_embs" not in data:
            return data

        description_tokens = data["description_tokens"]
        description_embs = data["description_embs"]

        # Choose random index
        idx = random.randrange(0, len(description_tokens))

        return {
            **data,
            "description_token": description_tokens[idx],
            "description_emb": description_embs[idx],
        }


class ToRepresentation:
    def __call__(self, data: dict) -> dict:
        joints = data["joints"]
        joint_vels = data["joint_vels"]
        body_pose = data["body_pose"]
        foot_contact = data["foot_contact"]

        body_pose_6d = rotmat_to_rot6d(body_pose)

        P, _, J, D = joints.shape
        x = torch.cat(
            (
                joints.flatten(start_dim=-2),
                torch.cat(
                    (torch.zeros(P, 1, J, D), joint_vels),
                    dim=1,
                ).flatten(start_dim=-2),
                body_pose_6d.flatten(start_dim=-3),
                torch.cat((torch.zeros(P, 1, 4), foot_contact), dim=1).flatten(
                    start_dim=-1
                ),
            ),
            dim=-1,
        )

        mask = data["motion_mask"]

        return {**data, "x": x, "mask": mask}


def collate_pose_annotations(
    data: List[dict[str, Optional[torch.Tensor]]]
) -> dict[str, Optional[torch.Tensor]]:
    batch_size = len(data)
    batch = {}

    # Collate all tensors in the batch
    for k in set(k for d in data for k in d.keys()):
        xs = [(i, d[k]) for i, d in enumerate(data) if k in d and d[k] is not None]
        if not xs:
            continue

        max_dims = [max(dim) for dim in zip(*[x.shape for (_, x) in xs])]
        batch[k] = torch.zeros(
            (batch_size, *max_dims),
            device=xs[0][1].device,
            dtype=xs[0][1].dtype,
        )
        for i, d in xs:
            batch[k][i, *[slice(0, dim) for dim in d.shape]] = d

    return batch
