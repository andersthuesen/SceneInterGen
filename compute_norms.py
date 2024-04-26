import torch
from datasets import DataModule
from configs import get_config
from tqdm import tqdm


if __name__ == "__main__":
    train_cfg = get_config("configs/train.yaml")
    data_cfg = get_config("configs/datasets.yaml").teton

    datamodule = DataModule(
        data_cfg, train_cfg.TRAIN.BATCH_SIZE, train_cfg.TRAIN.NUM_WORKERS
    )

    # Use this code to calculate mean and std
    datamodule.setup()

    means = []
    stds = []

    total = 100

    for i, (motion, mask, *_) in tqdm(
        enumerate(datamodule.train_dataloader()),
        total=total,
        desc="Computing mean and std",
    ):

        masked_motion = motion[mask]
        means.append(masked_motion.mean(dim=0))
        stds.append(masked_motion.std(dim=0))

        if i > total:
            break

    mean = torch.stack(means).mean(dim=0)
    std = torch.stack(stds).mean(dim=0)

    # Save mean and std
    torch.save(mean, "means.pt")
    torch.save(std, "stds.pt")
