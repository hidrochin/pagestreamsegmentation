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
        # Where the type head reads from in I1 (see forward). "context" is the
        # original behavior (type off the temporal-CNN output); "page"/"
        # page_plus_context" decouple typing from the deep context so a run's
        # dominant type can't smear across its pages (per-page type is what B1
        # already does). No effect on B0/B1, which type from e_i regardless.
        self.type_from = cfg.model.get("type_from", "context")
        # Where the boundary head reads in I1 (see forward). "context" is the
        # original behavior (boundary off the temporal-CNN output). "pairwise"
        # feeds the B1-style adjacent-page contrast [e_{i-1}, e_i, |Δ|, e_{i-1}·e_i]
        # on the RAW page embeddings — the context CNN smooths adjacent pages, so a
        # boundary's |Δ| is smallest exactly where we want it largest; raw keeps the
        # title-word jump sharp. "pairwise_context" concatenates the contextual h_i
        # so the CNN still informs (and receives gradient for) the boundary decision.
        # No effect on B0 (per-page) or B1 (already pairwise).
        self.boundary_from = cfg.model.get("boundary_from", "context")
        # >0 turns the type CE into focal loss (down-weights easy majority pages).
        self.type_focal_gamma = float(cfg.model.get("type_focal_gamma", 0.0) or 0.0)

        self.encoder = build_page_encoder(cfg)
        d = self.encoder.output_dim

        self.context = None
        if self.variant == "I1":
            sh = cfg.model.seq_head
            if sh.type == "cnn":
                self.context = TemporalCNN(
                    d,
                    sh.n_layers,
                    sh.kernel_size,
                    sh.dropout,
                    sh.hidden_size,
                    residual=sh.get("residual", False),
                )
            else:
                self.context = TransformerOverPages(
                    d, sh.n_layers, sh.n_heads, sh.dropout, max_pages=cfg.data.max_pages
                )
            if self.boundary_from == "context":
                boundary_in = self.context.out_dim
            elif self.boundary_from == "pairwise":
                boundary_in = 4 * d
            elif self.boundary_from == "pairwise_context":
                boundary_in = 4 * d + self.context.out_dim
            else:
                raise ValueError(f"unknown model.boundary_from={self.boundary_from!r}")
            if self.type_from == "context":
                type_in = self.context.out_dim
            elif self.type_from == "page":
                type_in = d
            elif self.type_from == "page_plus_context":
                if self.context.out_dim != d:
                    raise ValueError(
                        "model.type_from=page_plus_context needs the context output "
                        "dim to equal the encoder dim — set model.seq_head.hidden_size=0."
                    )
                type_in = d
            else:
                raise ValueError(f"unknown model.type_from={self.type_from!r}")
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
        # optional per-type CE weights (e.g. inverse frequency) to counter type
        # imbalance. Non-persistent: it's a training-only knob, kept out of the
        # state dict so checkpoints load regardless of how (or whether) it was set.
        tw = cfg.model.get("type_class_weights", None)
        if tw in (None, "auto"):  # "auto" is resolved to a list in train.py pre-build
            self.type_weight = None
        else:
            self.register_buffer(
                "type_weight",
                torch.tensor([float(x) for x in tw], dtype=torch.float32),
                persistent=False,
            )

    @staticmethod
    def _pairwise(emb):
        """Adjacent-page contrast features [e_{i-1}, e_i, |e_i-e_{i-1}|, e_{i-1}·e_i]
        over a [B, P, D] sequence -> [B, P, 4D]. Page 0's "previous" is zeros. Shared
        by B1 and I1's pairwise boundary head."""
        prev = torch.cat([torch.zeros_like(emb[:, :1]), emb[:, :-1]], dim=1)
        return torch.cat([prev, emb, (prev - emb).abs(), prev * emb], dim=-1)

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
            if self.boundary_from == "context":
                bd_feat = ctx
            elif self.boundary_from == "pairwise":
                bd_feat = self._pairwise(page_emb)
            else:  # pairwise_context: sharp raw contrast ++ smoothed context
                bd_feat = torch.cat([self._pairwise(page_emb), ctx], dim=-1)
            boundary_logits = self.boundary_head(bd_feat)
            if self.type_from == "page":
                type_feat = page_emb
            elif self.type_from == "page_plus_context":
                type_feat = ctx + page_emb
            else:  # context
                type_feat = ctx
            type_logits = self.type_head(type_feat)
        elif self.variant == "B1":
            boundary_logits = self.boundary_head(self._pairwise(page_emb))
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
        type_loss = self._type_loss(tl, ty)
        loss = boundary_loss + self.type_loss_weight * type_loss
        return {"loss": loss, "boundary_loss": boundary_loss, "type_loss": type_loss}

    def _type_loss(self, tl, ty):
        w = self.type_weight.to(tl.dtype) if self.type_weight is not None else None
        if self.type_focal_gamma > 0:
            return _focal_ce(tl, ty, self.type_focal_gamma, weight=w, ignore_index=-100)
        return F.cross_entropy(tl, ty, weight=w, ignore_index=-100)


def _focal_ce(logits, target, gamma, weight=None, ignore_index=-100):
    """Multi-class focal loss with optional per-class weight, ignoring ignore_index.
    Down-weights well-classified (usually majority) pages by (1 - p_t)^gamma so the
    type head stops collapsing to the dominant class. Reduces to (weighted) CE at
    gamma=0; normalized by the summed class weight to match F.cross_entropy."""
    valid = target != ignore_index
    if not bool(valid.any()):
        return logits.sum() * 0.0  # keep the graph, contribute nothing
    logits = logits[valid]
    target = target[valid]
    logp = F.log_softmax(logits, dim=-1)
    logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    pt = logpt.exp()
    loss = -((1.0 - pt) ** gamma) * logpt
    if weight is not None:
        wt = weight[target]
        return (loss * wt).sum() / wt.sum().clamp_min(1e-8)
    return loss.mean()


def build_model(cfg):
    return PSSModel(cfg)
