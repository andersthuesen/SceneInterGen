import os
import math
import torch
import pickle
from torch.utils.data import Dataset
from typing import Callable, TypeVar, Tuple, List
import common.data_types as data_types
from matching import MatchedTracks
from smplx import SMPLLayer

import sys

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


class TetonPoseAnnotationsDataset(Dataset):
    def __init__(
        self,
        root_path: str,
        transform: Callable = None,
        augment: Callable = None,
        cache=False,
    ):
        super(TetonPoseAnnotationsDataset, self).__init__()
        self.transform = transform
        self.augment = augment
        self.cache = cache
        self.paths = [
            os.path.join(root_path, department_dir, device_dir, date_dir, datetime_dir)
            for department_dir in os.listdir(root_path)
            for device_dir in os.listdir(os.path.join(root_path, department_dir))
            for date_dir in os.listdir(
                os.path.join(root_path, department_dir, device_dir)
            )
            for datetime_dir in os.listdir(
                os.path.join(root_path, department_dir, device_dir, date_dir)
            )
            if os.path.exists(
                os.path.join(
                    root_path,
                    department_dir,
                    device_dir,
                    date_dir,
                    datetime_dir,
                    "matched.pkl",
                )
            )
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(
        self, index
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base_path = self.paths[index]

        if self.cache and os.path.exists(os.path.join(base_path, "cache.pkl")):
            return pickle.load(open(os.path.join(base_path, "cache.pkl"), "rb"))

        matched_path = os.path.join(base_path, "matched.pkl")

        matched: MatchedTracks = (
            pickle.load(open(matched_path, "rb"))
            if os.path.exists(matched_path)
            else None
        )

        num_people = len(matched)

        frame_ids = [
            int(frame_id)
            for (_, annotation) in matched.values()
            for frame_id in annotation["frames"]
        ]

        min_frame_id = min(frame_ids)
        max_frame_id = max(frame_ids)

        num_frames = max_frame_id - min_frame_id + 1

        motion = torch.zeros(num_people, num_frames, SMPL_SIZE)
        mask = torch.zeros(num_people, num_frames, dtype=torch.bool)
        classes = torch.zeros(num_people, dtype=torch.long)
        actions = torch.zeros(num_people, num_frames, dtype=torch.long)
        pose_mask = torch.zeros(num_people, num_frames, dtype=torch.bool)

        # Find the minimum frame ID to use as the starting index

        for person_idx, (track, annotation) in enumerate(matched.values()):
            person_class_idx = PERSON_CLASSES.index(annotation.get("class", "person"))
            classes[person_idx] = person_class_idx + 1  # 0 is reserved for "unknown"

            for frame_id, frame in annotation["frames"].items():
                frame_id = int(frame_id)
                frame_idx = frame_id - min_frame_id - 1

                mask[person_idx, frame_idx] = True

                action_idx = frame["action"]
                actions[person_idx, frame_idx] = (
                    action_idx + 1
                )  # 0 is reserved for "unknown"

                if frame_id in track:
                    person = track[frame_id]

                    pose_mask[person_idx, frame_idx] = True
                    motion[person_idx, frame_idx] = torch.cat(
                        (
                            person.pred_cam_t_full,
                            person.smpl_global_orient,
                            person.smpl_pose,
                        )
                    )

        out = motion, mask, classes, actions, pose_mask

        if self.transform:
            out = self.transform(out)

        if self.cache:
            pickle.dump(out, open(os.path.join(base_path, "cache.pkl"), "wb"))

        if self.augment:
            # This runs after
            out = self.augment(out)

        return out


class AppendSMPLJoints:
    def __init__(self, smpl: SMPLLayer):
        self.smpl = smpl

    def __call__(self, data: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        motions, *rest = data
        translation, global_orient, body_pose = torch.split(
            motions.view(-1, SMPL_SIZE),
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

        # Append SMPL joints to the end of the motion tensor
        return (
            torch.cat(
                (motions, out.joints[..., :24, :].view(*motions.shape[:-1], -1)),
                dim=-1,
            ),
            *rest,
        )


class AppendJointVelocities:
    def __call__(
        self,
        data: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        motions, masks, classes, actions, pose_masks = data
        joint_pos = motions[..., -SMPL_JOINTS_SIZE:].reshape(
            *motions.shape[:-1], *SMPL_JOINTS_SIZES
        )

        # Create a tensor to hold the joint velocities and masks
        joint_vel = torch.zeros_like(joint_pos)
        joint_vel_masks = torch.zeros_like(pose_masks, dtype=torch.bool)

        joint_vel[:, 1:] = joint_pos[:, 1:] - joint_pos[:, :-1]
        joint_vel_masks[:, 1:] = pose_masks[:, 1:] & pose_masks[:, :-1]

        # Zero out joint velocities where the pose is not present
        joint_vel[~joint_vel_masks] = 0

        return (
            torch.cat((motions, joint_vel.view(*motions.shape[:-1], -1)), dim=-1),
            masks,
            classes,
            actions,
            pose_masks,
            joint_vel_masks,
        )


class SMPL6D:
    def __call__(self, data: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        motions, *rest_data = data
        translation, global_orient, body_pose, rest = torch.split(
            motions,
            (*SMPL_SIZES, motions.shape[-1] - SMPL_SIZE),
            dim=-1,
        )

        global_orient_6d = global_orient.reshape(*global_orient.shape[:-1], 3, 3)[
            ..., :2
        ]
        body_pose_6d = body_pose.reshape(*body_pose.shape[:-1], 23, 3, 3)[..., :2]

        return (
            torch.cat(
                (
                    translation,
                    global_orient_6d.reshape(*motions.shape[:-1], -1),
                    body_pose_6d.reshape(*motions.shape[:-1], -1),
                    rest,
                ),
                dim=-1,
            ),
            *rest_data,
        )


def collate_pose_annotations(
    batch: List[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]
):
    batch_size = len(batch)
    max_dims = [max(dims) for dims in zip(*[m.shape for m, *_ in batch])]

    outs = []
    for inpts in zip(*batch):
        max_dims = [max(dim) for dim in zip(*[inp.shape for inp in inpts])]
        out = torch.zeros(
            batch_size,
            *max_dims,
            dtype=inpts[0].dtype,
            device=inpts[0].device,
        )
        for i, inp in enumerate(inpts):
            out[i, *[slice(0, dim) for dim in inp.shape]] = inp

        outs.append(out)

    return outs

    # motions = torch.zeros(batch_size, *max_dims)
    # masks = torch.zeros(batch_size, *max_dims[:2], dtype=torch.bool)
    # classes = torch.zeros(batch_size, *max_dims[:1], dtype=torch.long)
    # actions = torch.zeros(batch_size, *max_dims[:2], dtype=torch.long)
    # pose_masks = torch.zeros(batch_size, *max_dims[:2], dtype=torch.bool)

    # for i, (motion, mask, cls, act, pose_mask) in enumerate(batch):
    #     motions[i, : motion.shape[0], : motion.shape[1]] = motion
    #     masks[i, : mask.shape[0], : mask.shape[1]] = mask
    #     classes[i, : cls.shape[0]] = cls
    #     actions[i, : act.shape[0], : act.shape[1]] = act
    #     pose_masks[i, : pose_mask.shape[0], : pose_mask.shape[1]] = pose_mask

    # return motions, masks, classes, actions, pose_masks
