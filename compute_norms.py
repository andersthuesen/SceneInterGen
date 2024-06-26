import torch
from datasets import DataModule
from configs import get_config
from tqdm import tqdm

import smplx

if __name__ == "__main__":
    model_cfg = get_config("configs/model.yaml")
    train_cfg = get_config("configs/train.yaml")
    data_cfg = get_config("configs/datasets.yaml")

    smpl = smplx.SMPLLayer(
        model_path=model_cfg.SMPL_MODEL_PATH,
    )

    datamodule = DataModule(data_cfg, 1, train_cfg.TRAIN.NUM_WORKERS, smpl)

    # Use this code to calculate mean and std
    datamodule.setup()

    means = []
    stds = []

    for motion, mask, *_ in tqdm(
        datamodule.train_dataloader(), desc="Computing mean and std"
    ):
        masked_motion = motion[mask]
        mean = masked_motion.mean(dim=0)
        std = masked_motion.std(dim=0)

        if mean.isnan().any():
            print("mean is nan")
        elif std.isnan().any():
            print("std is nan")
        else:
            means.append(mean)
            stds.append(std)

    mean = torch.stack(means).mean(dim=0)
    std = torch.stack(stds).mean(dim=0)

    # Save mean and std
    torch.save(mean, "mean.pt")
    torch.save(std, "std.pt")
