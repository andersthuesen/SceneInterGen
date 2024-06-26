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

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def compute_loss(self, batch):
        batch = self.text_process(batch)
        losses = self.decoder.compute_loss(batch, self.mean, self.std)
        return losses["total"], losses

    def decode_motion(self, batch):
        batch.update(self.decoder(batch))
        return batch

    def forward(self, batch):
        return self.compute_loss(batch)

    def forward_test(self, batch):
        batch = self.text_process(batch)
        batch.update(self.decode_motion(batch))
        return batch

    def text_process(self, batch):
        tokens = batch["description_tokens"]
        embs = batch["description_embs"]

        # Merge batch and number of descriptions into one dimension
        tokens_flat = tokens.reshape(-1, *tokens.shape[2:])
        embs_flat = embs.reshape(-1, *embs.shape[2:])

        # Apply our transformer encoder
        out_flat = self.clipTransEncoder(embs_flat)
        out_flat = self.clip_ln(out_flat)

        # Extract embedding for EOS tokens
        cond_flat = out_flat[
            torch.arange(out_flat.shape[0]),
            # Find EOS token which should contain all information
            tokens_flat.argmax(dim=-1),
        ]

        # # Find the average of the embeddings (mask out pure padding tokens)
        cond_mask = tokens.max(dim=-1, keepdim=True).values > 0
        num_descs = cond_mask.sum(dim=1)

        cond = self.clip_ln(
            (cond_mask * cond_flat.reshape(*embs.shape[:2], cond_flat.shape[-1])).sum(
                dim=1
            )
            / num_descs
        )

        desc_mask = num_descs.squeeze(-1) > 0

        # Update batch with new embeddings and masks
        batch["description_mask"] = desc_mask
        batch["description_emb"] = cond

        return batch
