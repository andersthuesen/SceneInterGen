import torch

EPSILON = 1e-6


class Normalize:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps=EPSILON):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, data: dict):
        x = data["x"]

        x_norm = (x - self.mean) / (self.std + self.eps)

        return {**data, "x": x_norm}
