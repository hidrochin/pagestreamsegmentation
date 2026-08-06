"""
PSS model variants (B0 / B1 / I1) assembled on the shared LayoutXLM page encoder.

All variants consume the same batch dict (see pss/data/collate.py) and output
per-page boundary + type logits. They differ only in how they use context:

    B0  per-page     : boundary/type from e_i alone (no context) — the floor.
    B1  page-pair    : boundary from [e_{i-1}, e_i, |Δ|, e_{i-1}·e_i]; type from e_i.
    I1  sequence     : context over the whole window (TemporalCNN or Transformer),
                       then boundary/type from the contextualized page rep.

Batch dict (B streams x P pages per window):
    input_ids [B,P,T]  bbox [B,P,T,4]  attention_mask [B,P,T]  image [B,P,3,224,224]
    page_mask [B,P]    boundary_labels [B,P]   type_labels [B,P]   (labels: -100 = ignore)
"""

import torch
import torch.nn.functional as F
from torch import nn

from pss.model.page_encoder import build_page_encoder
from pss.model.sequence_heads import (
    BoundaryHead,
    TemporalCNN,
    TransformerOverPages,
    TypeHead,
)


class PSSModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.variant = cfg.model.variant
        self.n_types = cfg.model.n_types
        self.type_loss_weight = float(cfg.model.type_loss_weight)

        self.encoder = build_page_encoder(cfg)
        d = self.encoder.output_dim

        self.context = None
        if self.variant == "I1":
            sh = cfg.model.seq_head
            if sh.type == "cnn":
                self.context = TemporalCNN(
                    d, sh.n_layers, sh.kernel_size, sh.dropout, sh.hidden_size
                )
            else:
                self.context = TransformerOverPages(
                    d, sh.n_layers, sh.n_heads, sh.dropout, max_pages=cfg.data.max_pages
                )
            boundary_in = type_in = self.context.out_dim
        elif self.variant == "B1":
            boundary_in, type_in = 4 * d, d
        else:  # B0
            boundary_in = type_in = d

        self.boundary_head = BoundaryHead(boundary_in)
        self.type_head = TypeHead(type_in, self.n_types)

        # up-weight the minority breaking-point class (TABME: ~35% positives)
        self.register_buffer(
            "bd_weight", torch.tensor([1.0, float(cfg.model.boundary_pos_weight)])
        )

    # -- page encoding: fold [B,P,...] into [B*P,...], encode, unfold ------------
    def encode(self, batch):
        b, p = batch["input_ids"].shape[:2]

        def flat(x):
            return x.reshape(b * p, *x.shape[2:])

        emb = self.encoder(
            flat(batch["input_ids"]),
            flat(batch["bbox"]),
            flat(batch["attention_mask"]),
            flat(batch["image"]),
        )
        return emb.reshape(b, p, -1)  # [B, P, D]

    def forward(self, batch):
        page_emb = self.encode(batch)  # [B, P, D]

        if self.variant == "I1":
            ctx = self.context(page_emb, batch.get("page_mask"))
            boundary_logits = self.boundary_head(ctx)
            type_logits = self.type_head(ctx)
        elif self.variant == "B1":
            prev = torch.cat(
                [torch.zeros_like(page_emb[:, :1]), page_emb[:, :-1]], dim=1
            )
            pair = torch.cat(
                [prev, page_emb, (prev - page_emb).abs(), prev * page_emb], dim=-1
            )
            boundary_logits = self.boundary_head(pair)
            type_logits = self.type_head(page_emb)
        else:  # B0
            boundary_logits = self.boundary_head(page_emb)
            type_logits = self.type_head(page_emb)

        out = {"boundary_logits": boundary_logits, "type_logits": type_logits}
        if "boundary_labels" in batch:
            out.update(self._loss(boundary_logits, type_logits, batch))
        return out

    def _loss(self, boundary_logits, type_logits, batch):
        bl = boundary_logits.reshape(-1, 2)
        by = batch["boundary_labels"].reshape(-1)
        boundary_loss = F.cross_entropy(
            bl, by, weight=self.bd_weight.to(bl.dtype), ignore_index=-100
        )
        tl = type_logits.reshape(-1, self.n_types)
        ty = batch["type_labels"].reshape(-1)
        type_loss = F.cross_entropy(tl, ty, ignore_index=-100)
        loss = boundary_loss + self.type_loss_weight * type_loss
        return {"loss": loss, "boundary_loss": boundary_loss, "type_loss": type_loss}


def build_model(cfg):
    return PSSModel(cfg)
