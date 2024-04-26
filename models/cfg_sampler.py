import torch
import torch.nn as nn


class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model, cfg_scale):
        super().__init__()
        self.model = model  # model is the actual model to run
        self.s = cfg_scale

    def forward(self, x, timesteps, **kwargs):
        out_cond = self.model(x, timesteps, **kwargs)
        out_uncond = self.model(x, timesteps)

        cfg_out = self.s * out_cond + (1 - self.s) * out_uncond
        return cfg_out
