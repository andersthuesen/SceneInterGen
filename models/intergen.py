from torch import nn
from models.nets import InterDiffusion


class InterGen(nn.Module):
    def __init__(self, cfg, mean, std):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.LATENT_DIM
        self.decoder = InterDiffusion(cfg, sampling_strategy=cfg.STRATEGY)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def compute_loss(self, batch):
        losses = self.decoder.compute_loss(batch, self.mean, self.std)
        return losses["total"], losses

    def decode_motion(self, batch):
        batch.update(self.decoder(batch))
        return batch

    def forward(self, batch):
        return self.compute_loss(batch)

    def forward_test(self, batch):
        batch.update(self.decode_motion(batch))
        return batch
