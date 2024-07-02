import torch

EPSILON = 1e-7


class Normalize:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps=EPSILON):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, data):
        (
            motion,
            motion_mask,
            classes,
            actions,
            object_points,
            object_points_mask,
            description_tokens,
            description_embs,
        ) = data

        motion_normalized = (motion - self.mean) / (self.std + self.eps)
        object_points_normalized = (
            (object_points - self.mean[:3]) / (self.std[:3] + self.eps)
            if object_points is not None
            else None
        )

        return (
            motion_normalized,
            motion_mask,
            classes,
            actions,
            object_points_normalized,
            object_points_mask,
            description_tokens,
            description_embs,
        )
