import torch


class Normalize:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        x, *rest = data
        return (x - self.mean) / self.std, *rest


class Denormalize:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        x, *rest = data
        return x * self.std + self.mean, *rest
