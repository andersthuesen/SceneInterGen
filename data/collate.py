from typing import Tuple, List
import math
import torch

from datasets.teton import Meta
from datasets.teton import PERSON_CLASSES, ACTIONS
from matching import MatchedTracks


TETON_TO_SMPL = {
    0: 24,  # "nose",
    1: 25,  # "right_eye",
    2: 26,  # "left_eye",
    3: 27,  # "right_ear",
    4: 28,  # "left_ear",
    # 5: 17, #"right_shoulder",
    # 6: 16, #"left_shoulder",
    7: 19,  # "right_elbow",
    8: 18,  # "left_elbow",
    9: 21,  # "right_wrist",
    10: 20,  # "left_wrist",
    # 11: "right_pinky_knuckle",
    # 12: "left_pinky_knuckle",
    # 13: "right_index_knuckle",
    # 14: "left_index_knuckle",
    # 15: "right_thumb_knuckle",
    # 16: "left_thumb_knuckle",
    # 17: 2, #"right_hip",
    # 18: 1, #"left_hip",
    19: 5,  # "right_knee",
    20: 4,  # "left_knee",
    21: 34,  # "right_ankle",
    22: 31,  # "left_ankle",
    23: 34,  # "right_heel",
    24: 31,  # "left_heel",
    25: 32,  # "right_foot_index",
    26: 29,  # "left_foot_index"
}


KEYPOINTS_SHAPE = (27, 2)
SMPL_POSE_DIMENSIONS = (3, 3 * 3, 23 * 3 * 3)


def collate_pose_annotations(batch: List[Tuple[dict, dict, str]]):
    """
    Assembles the mini-batches with keypoint annotations and SMPL pseudo ground truth where
    the avg. reprojection error is below 100 pixels (top 10% best predictions).
    """

    batch_size = len(batch)
    max_objs = max(
        [
            len([obj_id for obj_id in poses_annotations if obj_id != "meta"])
            for poses_annotations, _, _ in batch
        ]
    )

    frame_ids = [
        int(frame_id)
        for poses_annotations, _, _ in batch
        for obj_id, obj_annotation in poses_annotations.items()
        if obj_id != "meta"
        for frame_id in obj_annotation["frames"]
    ]
    max_frame_id = max(frame_ids) if frame_ids else 0
    min_frame_id = min(frame_ids) if frame_ids else 0
    max_frames = max_frame_id - min_frame_id + 1

    in_frame = torch.zeros((batch_size, max_objs, max_frames), dtype=torch.bool)
    kpts = torch.zeros((batch_size, max_objs, max_frames, math.prod(KEYPOINTS_SHAPE)))
    identities = torch.zeros((batch_size, max_objs), dtype=torch.long)
    actions = torch.zeros((batch_size, max_objs, max_frames), dtype=torch.long)

    has_pose = torch.zeros((batch_size, max_objs, max_frames), dtype=torch.bool)
    poses = torch.zeros((batch_size, max_objs, max_frames, sum(SMPL_POSE_DIMENSIONS)))
    paths = []

    for batch_idx, (poses_annotations, smpl_pseudo_gt, path) in enumerate(batch):
        paths.append(path)

        for obj_idx, (obj_id, obj_annotation) in enumerate(poses_annotations.items()):
            if obj_id == "meta":
                continue

            # Sometimes we get a "label" other times a "class"
            person_class = (
                obj_annotation["label"]
                if "label" in obj_annotation
                else obj_annotation["class"]
            )
            person_class_idx = PERSON_CLASSES.index(person_class)
            identities[batch_idx, obj_idx] = person_class_idx + 1

            for frame_id, frame_annotation in obj_annotation["frames"].items():
                frame_idx = int(frame_id) - min_frame_id

                teton_kpts = frame_annotation["keypoints"]
                kpts[batch_idx, obj_idx, frame_idx] = teton_kpts[:27, :2].reshape(-1)
                action_idx = frame_annotation["action"]
                actions[batch_idx, obj_idx, frame_idx] = action_idx + 1
                in_frame[batch_idx, obj_idx, frame_idx] = True

                if (
                    smpl_pseudo_gt is None
                    or obj_id not in smpl_pseudo_gt
                    or frame_id not in smpl_pseudo_gt[obj_id]
                ):
                    # print(f"SMPL pseudo GT not found for {obj_id}")
                    continue

                smpl_pose = smpl_pseudo_gt[obj_id][frame_id]
                reproj_error = torch.tensor(smpl_pose["reprojection_error_per_joint"])

                # Only populate z-coordinate if reproj error is bottom 10% of all reproj errors, <0.01
                if reproj_error.mean() < 100:
                    poses[batch_idx, obj_idx, frame_idx] = torch.cat(
                        (
                            smpl_pose["pred_cam_t_full"].view(-1),
                            torch.tensor(smpl_pose["global_orient"]).view(-1),
                            torch.tensor(smpl_pose["body_pose"]).view(-1),
                        )
                    )
                    has_pose[batch_idx, obj_idx, frame_idx] = True

    # For debugging purposes
    # all_kpts = kpts[has_pose].view(-1, *KEYPOINTS_SHAPE)
    # print(
    #     "kpts",
    #     all_kpts.mean(dim=(0, 1)),
    #     all_kpts.std(dim=(0, 1)),
    # )

    # all_trans = poses[has_pose][..., :3]
    # print(
    #     "trans",
    #     all_trans.mean(dim=list(range(all_trans.dim() - 1))),
    #     all_trans.std(dim=list(range(all_trans.dim() - 1))),
    # )

    return in_frame, kpts, identities, actions, has_pose, poses, paths


FEATURE_SIZE = 3 + (24 * 3 * 3)


def collate_matched_tracks(args: Tuple[Tuple[MatchedTracks, Meta]]):
    matches, metas = zip(*args)
    matches: List[MatchedTracks]
    metas: List[Meta]

    num_matches = len(matches)
    max_objs = max(len(match) for match in matches)

    # Find the maximum number of frames
    frame_ids = [
        frame_idx
        for match in matches
        for track, _ in match.values()
        for frame_idx in track.keys()
    ]
    max_frame_id = max(frame_ids)
    min_frame_id = min(frame_ids)
    max_frames = max_frame_id - min_frame_id + 1

    # Allocate tensor
    pose_seqs = torch.zeros(num_matches, max_objs, max_frames, FEATURE_SIZE)

    # The mask indicates whether the pose is valid
    valids = torch.zeros(num_matches, max_objs, max_frames, dtype=torch.bool)

    # One hot encode the person class. This remains the same over time.
    identities = torch.zeros(num_matches, max_objs, dtype=torch.long)

    # One hot encode the actions, which can change over time.
    actions = torch.zeros(num_matches, max_objs, max_frames, dtype=torch.long)

    for match_idx, match in enumerate(matches):
        for obj_idx, (track, annotation) in enumerate(match.values()):
            # One hot encode the person class
            if "class" in annotation:
                class_idx = PERSON_CLASSES.index(annotation["class"])
                identities[match_idx, obj_idx] = class_idx + 1

            for frame_id, frame in track.items():
                frame_idx = frame_id - min_frame_id
                # Mark as valid, as we have a frame prediction
                valids[match_idx, obj_idx, frame_idx] = True

                # One hot encode the action
                if str(frame_idx) in annotation["frames"]:
                    action_idx = annotation["frames"][str(frame_idx)]["action"]
                    actions[match_idx, obj_idx, frame_idx] = action_idx + 1

                # Concatenate the global translation and the SMPL pose
                pose_seqs[match_idx, obj_idx, frame_idx] = torch.cat(
                    (
                        frame.pred_cam_t_full,
                        frame.smpl_global_orient,
                        frame.smpl_pose,
                    )
                )

    # Normalize the translations to be roughly between -1 and 1 just like the rotation matrices
    # pose_seqs[..., 2] = (pose_seqs[:, :, :, 2] - 120) / 60
    # pose_seqs[..., :2] = pose_seqs[..., :2] / 2

    return pose_seqs, identities, actions, valids, metas
