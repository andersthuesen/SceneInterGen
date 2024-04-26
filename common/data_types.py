import torch
from typing import TypedDict, OrderedDict, Dict, Union, List, Tuple, NamedTuple
from enum import Enum
from dataclasses import dataclass
from collections import OrderedDict as OrderedDictClass


FrameID = int
ObjectID = str


Coordinate3D = Tuple[float, float, float]
Box = Tuple[float, float, float, float]
Keypoint2D = Tuple[float, float, float]  # x, y, score

PoseAnnotationFrame = TypedDict(
    "PoseAnnotationFrame",
    {
        "box": Box,
        "keypoints": List[Keypoint2D],
        "action": int,
        "occluded": bool,
    },
)


class ObjectClass(Enum):
    PERSON = "person"
    PATIENT = "patient"


PoseAnnotation = TypedDict(
    "PoseAnnotation",
    {"frames": OrderedDict[str, PoseAnnotationFrame], "class": ObjectClass},
)

PoseAnnotationMeta = TypedDict(
    "PoseAnnotationMeta",
    {
        # TOOD: Add the fields
    },
)

PoseAnnotationMetaDict = TypedDict("Meta", {"meta": PoseAnnotationMeta})

PoseAnnotations = Union[Dict[ObjectID, PoseAnnotation], PoseAnnotationMetaDict]


@dataclass
class ObjectInfo:
    class_: ObjectClass


@dataclass
class TransposedPoseAnnotation:
    frames: OrderedDict[FrameID, Dict[ObjectID, PoseAnnotationFrame]]
    objects: Dict[ObjectID, ObjectInfo]
    meta: PoseAnnotationMeta

    def __init__(self, frames=None, objects=None, meta=None):
        self.frames = frames if frames is not None else OrderedDictClass()
        self.objects = objects if objects is not None else {}
        self.meta = meta if meta is not None else PoseAnnotationMeta()


FramePrediction = TypedDict(
    "FramePrediction",
    {
        "bbox": torch.Tensor,
        "person_probs": torch.Tensor,
        "action_probs": torch.Tensor,
        "2d_kpts": torch.Tensor,
        "smpl_global_orient": torch.Tensor,
        "smpl_betas": torch.Tensor,
        "pred_cam_t_full": torch.Tensor,
        "person_cls": torch.Tensor,
        "action_cls": torch.Tensor,
        "is_valid_dynamic": torch.Tensor,
        "static_bbox": torch.Tensor,
        "static_batch_idx": torch.Tensor,
        "static_cls": torch.Tensor,
        "is_valid_static": torch.Tensor,
        "3d_kpts": torch.Tensor,
    },
)


VideoPredictions = OrderedDict[FrameID, FramePrediction]


@dataclass
class DynamicPrediction:
    bbox: torch.Tensor
    person_cls: torch.Tensor
    person_probs: torch.Tensor
    action_cls: torch.Tensor
    action_probs: torch.Tensor
    kpts_2d: torch.Tensor
    kpts_3d: torch.Tensor
    smpl_global_orient: torch.Tensor
    smpl_betas: torch.Tensor
    smpl_pose: torch.Tensor
    pred_cam_t_full: torch.Tensor

    @staticmethod
    def deserialize(data: dict) -> "DynamicPrediction":
        return DynamicPrediction(
            bbox=torch.tensor(data["bbox"]),
            person_cls=torch.tensor(data["person_cls"]),
            person_probs=torch.tensor(data["person_probs"]),
            action_cls=torch.tensor(data["action_cls"]),
            action_probs=torch.tensor(data["action_probs"]),
            kpts_2d=torch.tensor(data["kpts_2d"]),
            kpts_3d=torch.tensor(data["kpts_3d"]),
            smpl_global_orient=torch.tensor(data["smpl_global_orient"]),
            smpl_betas=torch.tensor(data["smpl_betas"]),
            smpl_pose=torch.tensor(data["smpl_pose"]),
            pred_cam_t_full=torch.tensor(data["pred_cam_t_full"]),
        )

    @staticmethod
    def deserialize_torch(data: dict) -> "DynamicPrediction":
        return DynamicPrediction(
            bbox=data["bbox"],
            person_cls=data["person_cls"],
            person_probs=data["person_probs"],
            action_cls=data["action_cls"],
            action_probs=data["action_probs"],
            kpts_2d=data["kpts_2d"],
            kpts_3d=data["kpts_3d"],
            smpl_global_orient=data["smpl_global_orient"],
            smpl_betas=data["smpl_betas"],
            smpl_pose=data["smpl_pose"],
            pred_cam_t_full=data["pred_cam_t_full"],
        )

    def serialize(self) -> dict:
        return {
            "bbox": self.bbox.tolist(),
            "person_cls": self.person_cls.tolist(),
            "person_probs": self.person_probs.tolist(),
            "action_cls": self.action_cls.tolist(),
            "action_probs": self.action_probs.tolist(),
            "kpts_2d": self.kpts_2d.tolist(),
            "kpts_3d": self.kpts_3d.tolist(),
            "smpl_global_orient": self.smpl_global_orient.tolist(),
            "smpl_betas": self.smpl_betas.tolist(),
            "smpl_pose": self.smpl_pose.tolist(),
            "pred_cam_t_full": self.pred_cam_t_full.tolist(),
        }

    def serialize_torch(self) -> dict:
        return {
            "bbox": self.bbox,
            "person_cls": self.person_cls,
            "person_probs": self.person_probs,
            "action_cls": self.action_cls,
            "action_probs": self.action_probs,
            "kpts_2d": self.kpts_2d,
            "kpts_3d": self.kpts_3d,
            "smpl_global_orient": self.smpl_global_orient,
            "smpl_betas": self.smpl_betas,
            "smpl_pose": self.smpl_pose,
            "pred_cam_t_full": self.pred_cam_t_full,
        }


DynamicPredictions = OrderedDict[FrameID, List[DynamicPrediction]]

Track = OrderedDict[FrameID, DynamicPrediction]
TransposedTracks = Dict[ObjectID, Track]
