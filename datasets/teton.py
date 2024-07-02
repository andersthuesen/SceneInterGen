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

import sys

from tqdm import tqdm

sys.modules["data_types"] = data_types

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

    def __getitem__(
        self, index
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base_path = self.paths[index]

        if self.cache and os.path.exists(os.path.join(base_path, "cache.pkl")):
            out = pickle.load(open(os.path.join(base_path, "cache.pkl"), "rb"))
            if self.augment:
                out = self.augment(out)
            return out

        motion_path = os.path.join(base_path, self.motion_filename)

        motion = torch.load(motion_path).float()

        motion_mask, motion = motion.split([1, SMPL_SIZE], dim=-1)
        motion_mask = motion_mask.squeeze(-1).bool()

        classes_path = os.path.join(base_path, "class.pt")
        actions_path = os.path.join(base_path, "action.pt")
        object_points_path = os.path.join(base_path, "object_points_transformed.pt")
        description_tokens_path = os.path.join(base_path, "description_tokens.pt")
        description_embs_path = os.path.join(base_path, "description_embs.pt")

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

        out = (
            motion,
            motion_mask,
            classes,
            actions,
            object_points,
            object_points_mask,
            description_tokens,
            description_embs,
        )

        if self.transform:
            out = self.transform(out)

        if self.cache:
            pickle.dump(out, open(os.path.join(base_path, "cache.pkl"), "wb"))

        if self.augment:
            out = self.augment(out)

        return out


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
    def __call__(self, data: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        *rest, description_tokens, description_embs = data

        if description_tokens is None or description_embs is None:
            return *rest, None, None

        # Choose random index
        idx = random.randrange(0, len(description_tokens))
        return *rest, description_tokens[idx], description_embs[idx]


class RandomTranslate:
    def __init__(self, max_x=0.5, max_y=0.5, max_z=0.0):
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z

    def __call__(self, data: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        motion, motion_mask, classes, actions, object_points, *rest = data
        joints, joint_vels, pose = motion.split(
            (
                SMPL_JOINTS_SIZE,
                SMPL_JOINTS_SIZE,
                SMPL_6D_SIZES[-1],  # 6D pose
            ),
            dim=-1,
        )

        joints = joints.view(joints.shape[:-1] + SMPL_JOINTS_DIMS[0])

        # Translate the joints
        translation = (torch.rand(3) * 2 - 1) * torch.tensor(
            [self.max_x, self.max_y, self.max_z]
        )

        joints_translated = joints + translation
        object_points_translated = (
            (object_points + translation) if object_points is not None else None
        )

        motion_translated = torch.cat(
            (
                joints_translated.view(*motion.shape[:-1], -1),
                joint_vels,
                pose,
            ),
            dim=-1,
        )

        return (
            motion_translated,
            motion_mask,
            classes,
            actions,
            object_points_translated,
            *rest,
        )


class RandomRotate:
    def __init__(self, max_angle=360):
        self.max_angle = max_angle

    def __call__(self, data: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        motion, motion_mask, classes, actions, object_points, *rest = data
        joints, joint_vels, pose = motion.split(
            (
                SMPL_JOINTS_SIZE,
                SMPL_JOINTS_SIZE,
                SMPL_6D_SIZES[-1],  # 6D pose
            ),
            dim=-1,
        )

        joints = joints.view(joints.shape[:-1] + SMPL_JOINTS_DIMS[0])
        joint_vels = joint_vels.view(joint_vels.shape[:-1] + SMPL_JOINTS_DIMS[0])

        # Rotate the joints
        r = R.from_euler("z", torch.rand(1) * self.max_angle, degrees=True)
        rotmat = r.as_matrix()[0]

        joints_rotated = (joints @ rotmat.T).type(joints.dtype)
        joint_vels_rotated = (joint_vels @ rotmat.T).type(joint_vels.dtype)

        object_points_rotated = (
            (object_points @ rotmat.T).type(object_points.dtype)
            if object_points is not None
            else None
        )

        motion_rotated = torch.cat(
            (
                joints_rotated.view(*motion.shape[:-1], -1),
                joint_vels_rotated.view(*motion.shape[:-1], -1),
                pose,
            ),
            dim=-1,
        )

        return (
            motion_rotated,
            motion_mask,
            classes,
            actions,
            object_points_rotated,
            *rest,
        )


def collate_pose_annotations(
    batch: List[
        Tuple[
            torch.Tensor,  # motion
            torch.Tensor,  # mask
            Optional[torch.Tensor],  # classes
            Optional[torch.Tensor],  # actions
            Optional[torch.Tensor],  # object_points
        ]
    ]
):
    batch_size = len(batch)

    outs = []
    for inpts in zip(*batch):
        non_none_inpts = [inp for inp in inpts if inp is not None]
        if not non_none_inpts:
            outs.append(None)
            continue

        max_dims = [max(dim) for dim in zip(*[inp.shape for inp in non_none_inpts])]

        out = torch.zeros(
            (batch_size, *max_dims),
            device=non_none_inpts[0].device,
            dtype=non_none_inpts[0].dtype,
        )
        for i, inp in enumerate(inpts):
            if inp is None:
                continue

            out[i, *[slice(0, dim) for dim in inp.shape]] = inp

        outs.append(out)

    return outs
