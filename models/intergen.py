from torch import nn
from models import *


class InterGen(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.LATENT_DIM
        self.decoder = InterDiffusion(cfg, sampling_strategy=cfg.STRATEGY)

        self.clipTransEncoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=768,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=2,
        )
        self.clip_ln = nn.LayerNorm(768)

    def compute_loss(self, batch):
        losses = self.decoder.compute_loss(batch)
        return losses["total"], losses

    def decode_motion(self, batch):
        batch.update(self.decoder(batch))
        return batch

    def forward(self, batch):
        return self.compute_loss(batch)

    def forward_test(self, batch):
        batch.update(self.decode_motion(batch))
        return batch
