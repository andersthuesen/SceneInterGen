import lightning.pytorch as pl
import torch
from .interhuman import InterHumanDataset

from .teton import (
    AppendFootContacts,
    AppendJointVelocities,
    AppendRandomCamera,
    AppendRenderedKeypoints,
    AppendSMPLJoints,
    TetonDataset,
    ChooseRandomDescription,
    ToRepresentation,
    collate_pose_annotations,
)

from transforms.normalization import Normalize

from torchvision.transforms import Compose

from datasets.evaluator import (
    EvaluationDataset,
    get_dataset_motion_loader,
    get_motion_loader,
)

from typing import Optional

import smplx


__all__ = [
    "InterHumanDataset",
    "EvaluationDataset",
    "get_dataset_motion_loader",
    "get_motion_loader",
]


def build_loader(cfg, data_cfg):
    # setup data
    if data_cfg.NAME == "interhuman":
        train_dataset = InterHumanDataset(data_cfg)
    else:
        raise NotImplementedError

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=1,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )

    return loader


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
        batch_size,
        num_workers,
        smpl: smplx.SMPLLayer,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ):
        """
        Initialize LightningDataModule for ProHMR training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
            dataset_cfg (CfgNode): Dataset configuration file
        """
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.smpl = smpl
        self.mean = mean
        self.std = std

    def setup(self, stage=None):
        """
        Create train and validation datasets
        """
        datasets = []
        for dataset_name, dataset_cfg in self.cfg.items():
            print(f"Using {dataset_name}")
            dataset = TetonDataset(
                root_path=dataset_cfg.DATA_ROOT,
                files_list=dataset_cfg.FILES_LIST,
                transform=Compose(
                    [
                        AppendSMPLJoints(self.smpl),
                        AppendJointVelocities(),
                        AppendFootContacts(),
                    ]
                ),
                augment=(
                    Compose(
                        [
                            ChooseRandomDescription(),
                            ToRepresentation(),
                        ]
                        + (
                            [Normalize(self.mean, self.std)]
                            if self.mean is not None and self.std is not None
                            else []
                        )
                    )
                ),
                motion_filename=dataset_cfg.MOTION_FILENAME,
                cache=dataset_cfg.CACHE,
            )
            datasets.append(dataset)

        self.train_dataset = torch.utils.data.ConcatDataset(datasets)

    def train_dataloader(self):
        """
        Return train dataloader
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_pose_annotations,
        )
