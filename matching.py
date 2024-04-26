import sys
import torch
from typing import Dict, Tuple

from common.data_types import ObjectID, PoseAnnotation, PoseAnnotations, Track
from tracking import Tracker, Tracks


MatchedTracks = Dict[ObjectID, Tuple[Track, PoseAnnotation]]


def match_tracks_to_annotations(
    tracked: Tracks, annotations: PoseAnnotations, miss_cost=1
) -> MatchedTracks:
    transposed_tracks = Tracker.transpose_tracks(tracked)

    if len(transposed_tracks) == 0:
        return {}

    filtered_annotations = {
        obj_id: obj for obj_id, obj in annotations.items() if obj_id != "meta"
    }

    annotation_id_to_idx = {
        obj_id: idx for idx, obj_id in enumerate(filtered_annotations.keys())
    }
    annotation_idx_to_id = {
        idx: obj_id for idx, obj_id in enumerate(filtered_annotations.keys())
    }

    track_id_to_idx = {
        obj_id: idx for idx, obj_id in enumerate(transposed_tracks.keys())
    }
    track_idx_to_id = {
        idx: obj_id for idx, obj_id in enumerate(transposed_tracks.keys())
    }

    costs = torch.zeros(
        len(filtered_annotations),
        len(transposed_tracks),
    )

    # Compute the pairwise costs between every annotation and track
    for annotation_obj_id, annotation_obj in filtered_annotations.items():
        for track_obj_id, track_obj in transposed_tracks.items():
            if annotation_obj_id == "meta":
                continue

            annotation_obj_idx = annotation_id_to_idx[annotation_obj_id]
            track_obj_idx = track_id_to_idx[track_obj_id]
            for frame_id, annotation_obj_frame in annotation_obj["frames"].items():
                frame_idx = int(frame_id)

                if frame_idx not in track_obj:
                    costs[annotation_obj_idx][track_obj_idx] += miss_cost
                    continue

                track_obj_frame = track_obj[frame_idx]

                pred_kpts_2d = track_obj_frame.kpts_2d[:, :2]
                # Make sure anno kpts is only as long as pred kpts. (poses_annotations_vit.json has 33 joints)
                anno_kpts_2d = torch.tensor(annotation_obj_frame["keypoints"])[
                    : len(pred_kpts_2d), :2
                ]
                # 2D keypoints cost
                cost = torch.norm(pred_kpts_2d - anno_kpts_2d)
                costs[annotation_obj_idx][track_obj_idx] += cost

            # Normalize cost
            costs[annotation_obj_idx][track_obj_idx] /= len(annotation_obj["frames"])

    # Compute greedy matching
    # TODO:  Implement a better matching algorithm (e.g. Hungarian algorithm)
    matches = {}
    if costs.nelement() == 0:
        print("WARNING: No costs to match", file=sys.stderr)

    while costs.nelement() > 0 and torch.min(costs) < torch.inf:
        best_match = costs.argmin()
        annotation_idx, track_idx = (
            best_match // costs.shape[1],
            best_match % costs.shape[1],
        )
        annotation_id = annotation_idx_to_id[annotation_idx.item()]
        track_id = track_idx_to_id[track_idx.item()]

        if track_id in matches:
            # Remove duplicate matches
            print(
                f"WARNING: Duplicate match for track {track_id} and annotation {annotation_id}",
                file=sys.stderr,
            )
            costs[annotation_idx, track_idx] = torch.inf
        else:
            matches[track_id] = annotation_id
            costs[annotation_idx, :] = torch.inf

    # Return mapped tracks
    # TODO: Do some kind of interpolation of missing frames?
    matched_tracks = {
        annotation_obj_id: (
            transposed_tracks[track_obj_id],
            filtered_annotations[annotation_obj_id],
        )
        for track_obj_id, annotation_obj_id in matches.items()
    }

    return matched_tracks
