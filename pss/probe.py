"""
Model-behavior probes for a trained PSS checkpoint — *how* it decides, not *how
well* (that's pss.evaluate). Built on the same stitched forward pass, so what you
see here is the representation the scored model actually computes.

Two questions this answers, tied to the primary recipe (boundary reads the raw
adjacent-page contrast; type reads the raw per-page embedding):

  1. "Does the encoder separate documents at all?" — the boundary head's sharpest
     signal is the contrast between the RAW LayoutXLM embeddings of adjacent pages
     (pss/model/variants.py::_pairwise). We trace cosine(e_{i-1}, e_i) along each
     stream and overlay the true boundaries. If the dips in similarity land on the
     boundaries, the embedding is doing the job; if not, the encoder — not the
     head — is the bottleneck (e.g. the S-vs-I title cue isn't being picked up).
       -> per-stream trace PNGs (adjacent_cos + per-layer drift)
       -> boundary_separation.png : the distribution of adjacent cosine-similarity
          at true new-document pages vs. continuation pages. Well-separated
          distributions == the encoder linearly separates boundaries. This is the
          headline diagnostic.

  2. "Does the deep TemporalCNN wash out per-page identity?" (the type-collapse
     mechanism) — for the I1 CNN we tap the page representation after each conv
     layer and measure how far it drifts from the page's own raw embedding
     (cosine(e_i, h^l_i)). If drift climbs toward 0 by the last layer, every page
     has become a blend of its neighbors — exactly what smears a run's dominant
     type across its pages. Lets you *see* whether seq_head.residual=true keeps
     per-page identity alive.
       -> the lower panel of each per-stream trace PNG (pages x layers heatmap)

Run:
    python -m pss.probe --config=configs/pss.yaml \
        pretrained_model_file=pss_runs/i1_layoutxlm/checkpoints/<run_id>/last.ckpt \
        eval_mode=test probe.out_dir=pss_runs/i1_layoutxlm/probe probe.max_streams=8
"""

import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pss.config import get_config
from pss.data import PSSDataset, collate_streams
from pss.evaluate import load_eval_net


class _EmbAccumulator:
    """Per-(stream, absolute-page) averaging of raw page embeddings and per-layer
    context reps over overlapping windows — the embedding analogue of
    pss/stitch.py::StreamAccumulator (which averages logits)."""

    def __init__(self):
        self.emb_sum = defaultdict(dict)  # fid -> {page: [D] tensor}
        self.layer_sum = defaultdict(dict)  # fid -> {page: [L, C] tensor}
        self.cnt = defaultdict(dict)  # fid -> {page: int}
        self.bd_true = defaultdict(dict)  # fid -> {page: 0/1}

    def add(self, meta, page_emb, layer_reps, bd_true):
        # page_emb [B,P,D]; layer_reps [B,P,L,C] or None; bd_true [B,P]
        for i, m in enumerate(meta):
            fid, start, length = m["folder_id"], m["start"], m["length"]
            for j in range(length):
                p = start + j
                self.emb_sum[fid][p] = self.emb_sum[fid].get(p, 0) + page_emb[i, j]
                if layer_reps is not None:
                    self.layer_sum[fid][p] = (
                        self.layer_sum[fid].get(p, 0) + layer_reps[i, j]
                    )
                self.cnt[fid][p] = self.cnt[fid].get(p, 0) + 1
                self.bd_true[fid][p] = int(bd_true[i, j])

    def stream_ids(self):
        return list(self.emb_sum.keys())

    def stream(self, fid):
        """(idxs, emb [N,D], layers [N,L,C] or None, bd [N])."""
        idxs = sorted(self.emb_sum[fid])
        c = torch.tensor([self.cnt[fid][p] for p in idxs]).float().unsqueeze(-1)
        emb = torch.stack([self.emb_sum[fid][p] for p in idxs]).float() / c
        bd = np.array([self.bd_true[fid][p] for p in idxs])
        layers = None
        if fid in self.layer_sum and self.layer_sum[fid]:
            layers = torch.stack([self.layer_sum[fid][p] for p in idxs]).float()
            layers = layers / c.unsqueeze(-1)  # [N,L,C]
        return idxs, emb, layers, bd


def _adjacent_cos(emb):
    """cos(e_{i-1}, e_i) along a [N, D] stream; index 0 is nan (no predecessor)."""
    if emb.shape[0] < 2:
        return np.array([np.nan] * emb.shape[0])
    sims = F.cosine_similarity(emb[1:], emb[:-1], dim=-1).numpy()
    return np.concatenate([[np.nan], sims])


def _layer_drift(emb, layers):
    """cos(e_i, h^l_i) per page per layer -> [N, L]. High == the page kept its own
    identity through layer l; low == it dissolved into its neighbors' context."""
    n, l, _ = layers.shape
    out = np.zeros((n, l))
    for li in range(l):
        out[:, li] = F.cosine_similarity(emb, layers[:, li], dim=-1).numpy()
    return out


def _plot_stream(fid, cos_adj, drift, bd, out_path):
    has_drift = drift is not None
    fig, axes = plt.subplots(
        2 if has_drift else 1,
        1,
        figsize=(max(6, 0.35 * len(cos_adj)), 6 if has_drift else 3.2),
        squeeze=False,
    )
    x = np.arange(len(cos_adj))

    ax = axes[0][0]
    ax.plot(x, cos_adj, "-o", ms=3, color="#1f77b4", label="cos(e_{i-1}, e_i)")
    for p in x[bd == 1]:
        ax.axvline(p, color="#d62728", ls="--", lw=1, alpha=0.7)
    ax.axvline(np.nan, color="#d62728", ls="--", lw=1, label="true new-doc")  # legend
    ax.set_ylabel("adjacent cosine")
    ax.set_title(f"stream {fid} — adjacent-page embedding similarity")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(min(0, np.nanmin(cos_adj)) - 0.02, 1.01)

    if has_drift:
        ax2 = axes[1][0]
        im = ax2.imshow(
            drift.T, aspect="auto", cmap="viridis", vmin=0, vmax=1, origin="lower"
        )
        ax2.set_yticks(range(drift.shape[1]))
        ax2.set_yticklabels([f"conv{li+1}" for li in range(drift.shape[1])])
        ax2.set_xlabel("page index")
        ax2.set_ylabel("layer")
        ax2.set_title(
            "per-page identity retention  cos(e_i, h^l_i)  (low = washed out)"
        )
        for p in x[bd == 1]:
            ax2.axvline(p, color="#d62728", ls="--", lw=1, alpha=0.6)
        fig.colorbar(im, ax=ax2)
    else:
        axes[0][0].set_xlabel("page index")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_separation(bd_sims, cont_sims, out_path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bins = (
        np.linspace(min(bd_sims.min(), cont_sims.min()), 1.0, 30)
        if len(bd_sims) and len(cont_sims)
        else 30
    )
    ax.hist(
        cont_sims,
        bins=bins,
        alpha=0.6,
        label=f"continuation (n={len(cont_sims)})",
        color="#2ca02c",
        density=True,
    )
    ax.hist(
        bd_sims,
        bins=bins,
        alpha=0.6,
        label=f"new-document (n={len(bd_sims)})",
        color="#d62728",
        density=True,
    )
    ax.set_xlabel("cos(e_{i-1}, e_i)")
    ax.set_ylabel("density")
    ax.set_title("Adjacent-page similarity at boundaries vs continuations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    cfg = get_config()
    mode = cfg.get("eval_mode", "test")
    probe_cfg = cfg.get("probe", {})
    out_dir = probe_cfg.get("out_dir") or os.path.join(cfg.workspace, "probe")
    max_streams = int(probe_cfg.get("max_streams", 8) or 8)
    os.makedirs(out_dir, exist_ok=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    net = load_eval_net(cfg, device)

    # per-layer drift only makes sense for the I1 CNN context stack
    cnn_ctx = (
        cfg.model.variant == "I1"
        and cfg.model.seq_head.type == "cnn"
        and net.context is not None
    )

    ds = PSSDataset(cfg, mode)
    loader = DataLoader(
        ds,
        batch_size=cfg.val.batch_size,
        shuffle=False,
        num_workers=cfg.val.num_workers,
        collate_fn=collate_streams,
    )

    acc = _EmbAccumulator()
    with torch.no_grad():
        for batch in loader:
            gpu_batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            page_emb = net.encode(gpu_batch)  # [B,P,D]
            layer_reps = None
            if cnn_ctx:
                _, layers = net.context(
                    page_emb, gpu_batch.get("page_mask"), return_layers=True
                )
                layer_reps = torch.stack(layers, dim=2).cpu()  # [B,P,L,C]
            acc.add(batch["meta"], page_emb.cpu(), layer_reps, batch["boundary_labels"])

    # per-stream trace figures (first `max_streams` streams that contain a boundary)
    bd_all, cont_all = [], []
    n_plotted = 0
    for fid in acc.stream_ids():
        idxs, emb, layers, bd = acc.stream(fid)
        cos_adj = _adjacent_cos(emb)
        # aggregate separability over adjacent pairs (skip page 0's nan)
        for k in range(1, len(cos_adj)):
            (bd_all if bd[k] == 1 else cont_all).append(cos_adj[k])
        if n_plotted < max_streams and (bd == 1).sum() >= 1 and len(idxs) >= 2:
            drift = _layer_drift(emb, layers) if layers is not None else None
            safe = str(fid).replace("/", "_")
            _plot_stream(
                fid, cos_adj, drift, bd, os.path.join(out_dir, f"stream_{safe}.png")
            )
            n_plotted += 1

    bd_all = np.array(bd_all)
    cont_all = np.array(cont_all)
    if len(bd_all) and len(cont_all):
        _plot_separation(
            bd_all, cont_all, os.path.join(out_dir, "boundary_separation.png")
        )

    print(
        f"[probe] wrote {n_plotted} per-stream traces + boundary_separation.png -> {out_dir}"
    )
    if len(bd_all) and len(cont_all):
        # lower adjacent similarity at a boundary is the signal; a clean gap means
        # the raw embeddings linearly separate new-doc pages from continuations.
        print(
            f"[probe] adjacent cos(e_i-1,e_i):  new-doc mean={bd_all.mean():.4f}"
            f"  continuation mean={cont_all.mean():.4f}"
            f"  gap={cont_all.mean() - bd_all.mean():+.4f}"
        )
        thr = np.linspace(-1, 1, 401)
        # accuracy of the best single-threshold split (boundary if cos < thr)
        best = max(
            ((bd_all < t).mean() * len(bd_all) + (cont_all >= t).mean() * len(cont_all))
            / (len(bd_all) + len(cont_all))
            for t in thr
        )
        print(
            f"[probe] best single-threshold boundary accuracy from raw cos = {best:.4f}"
        )


if __name__ == "__main__":
    main()
