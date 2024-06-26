import torch

EPSILON = 1e-6


class Normalize:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps=EPSILON):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, data):
        x, *rest = data
        if x.isnan().any():
            raise ValueError("NaNs in input")

        return (x - self.mean) / (self.std + self.eps), *rest


class Denormalize:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps=EPSILON):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, data):
        x, *rest = data
        if x.isnan().any():
            raise ValueError("NaNs in input")

        return x * (self.std + self.eps) + self.mean, *rest
