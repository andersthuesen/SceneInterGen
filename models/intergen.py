import torch
from torch import nn
from models.nets import InterDiffusion


class InterGen(nn.Module):
    def __init__(self, cfg, mean, std):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.LATENT_DIM
        self.decoder = InterDiffusion(cfg, sampling_strategy=cfg.STRATEGY)

        # Same as intergen model
        clipTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.clipTransEncoder = nn.TransformerEncoder(
            clipTransEncoderLayer, num_layers=2
        )
        self.clip_ln = nn.LayerNorm(768)

        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def compute_loss(self, batch):
        # batch = self.text_process(batch)
        losses = self.decoder.compute_loss(batch, self.mean, self.std)
        return losses["total"], losses

    def decode_motion(self, batch):
        batch.update(self.decoder(batch))
        return batch

    def forward(self, batch):
        return self.compute_loss(batch)

    def forward_test(self, batch):
        # batch = self.text_process(batch)
        batch.update(self.decode_motion(batch))
        return batch

    def text_process(self, batch):
        tokens: torch.Tensor = batch["description_token"]
        embs: torch.Tensor = batch["description_emb"]

        if tokens is None or embs is None:
            batch["description_mask"] = None
            batch["description_emb"] = None
            return batch

        mask = tokens.max(dim=-1).values != 0

        emb = self.clipTransEncoder(embs)
        emb = self.clip_ln(emb)
        emb = emb[torch.arange(emb.size(0)), tokens.argmax(dim=-1)]

        # Update batch with new embeddings and masks
        batch["description_mask"] = mask
        batch["description_emb"] = emb

        return batch
