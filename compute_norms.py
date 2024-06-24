import torch
from datasets import DataModule
from configs import get_config
from tqdm import tqdm


if __name__ == "__main__":
    train_cfg = get_config("configs/train.yaml")
    data_cfg = get_config("configs/datasets.yaml").teton

    datamodule = DataModule(data_cfg, 1, train_cfg.TRAIN.NUM_WORKERS)

    # Use this code to calculate mean and std
    datamodule.setup()

    means = []
    stds = []

    for motion, mask, *_ in tqdm(
        datamodule.train_dataloader(), desc="Computing mean and std"
    ):
        pass

    exit(0)
    # masked_motion = motion[mask]
    # mean = masked_motion.mean(dim=0)
    # std = masked_motion.std(dim=0)

    # if mean.isnan().any():
    #     print("mean is nan")
    # elif std.isnan().any():
    #     print("std is nan")
    # else:
    #     means.append(mean)
    #     stds.append(std)

    mean = torch.stack(means).mean(dim=0)
    std = torch.stack(stds).mean(dim=0)

    # Save mean and std
    torch.save(mean, "means.pt")
    torch.save(std, "stds.pt")
