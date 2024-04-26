import torch


class Normalize:
    def __init__(self, means_path: str, stds_path: str):
        self.means = torch.load(means_path)
        self.stds = torch.load(stds_path)

    def __call__(self, data):
        x, *rest = data
        return (x - self.means) / self.stds, *rest


class UnNormalize:
    def __init__(self, means_path: str, stds_path: str):
        self.means = torch.load(means_path)
        self.stds = torch.load(stds_path)

    def __call__(self, data):
        x, *rest = data
        return x * self.stds + self.means, *rest
