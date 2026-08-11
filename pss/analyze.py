"""
Visual model-behavior diagnostics for a trained PSS checkpoint.

Reuses ``pss.evaluate``'s stream stitching (``collect_predictions``) so the
underlying predictions are identical to what ``pss.evaluate`` scores — this is
a *visualization* layer on top of that, not a separate scoring path. Produces,
under ``analyze.out_dir``:

  - boundary_confusion.png   2x2 confusion matrix (continue vs new-document),
                              page-level, after stream stitching.
  - type_confusion.png       NxN document-type confusion matrix (majority-voted
                              per true document segment, same as the doc macro-F1
                              in pss.evaluate).
  - type_prf.png             per-type precision/recall/F1 bars, to see which
                              document types the model handles well vs. poorly
                              (macro-F1 alone hides this).

Run:
    python -m pss.analyze --config=configs/pss.yaml \
        pretrained_model_file=pss_runs/i1_layoutxlm/checkpoints/<run_id>/last.ckpt \
        eval_mode=test analyze.out_dir=pss_runs/i1_layoutxlm/analysis
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pss.config import get_config
from pss.data.dataset import read_class_names
from pss.evaluate import collect_predictions, load_eval_net


def _confusion_matrix(pairs, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in pairs:
        cm[true, pred] += 1
    return cm


def _plot_confusion(cm, labels, title, out_path):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(4, 0.6 * n) + 1, max(4, 0.6 * n)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _per_class_prf(cm):
    """cm[true, pred]. Returns (precision, recall, f1) arrays, one per class."""
    n = cm.shape[0]
    prec, rec, f1 = np.zeros(n), np.zeros(n), np.zeros(n)
    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        prec[c], rec[c] = p, r
        f1[c] = 2 * p * r / (p + r) if p + r else 0.0
    return prec, rec, f1


def _plot_prf_bars(labels, prec, rec, f1, title, out_path):
    n = len(labels)
    x = np.arange(n)
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * n), 4.5))
    ax.bar(x - w, prec, w, label="precision")
    ax.bar(x, rec, w, label="recall")
    ax.bar(x + w, f1, w, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    cfg = get_config()
    mode = cfg.get("eval_mode", "test")
    out_dir = cfg.analyze.out_dir or os.path.join(cfg.workspace, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    net = load_eval_net(cfg, device)
    result = collect_predictions(cfg, net, device, mode)

    # -- boundary confusion (page-level, after stitching) ------------------------
    bd_pairs = list(zip(result["all_true"], result["all_pred"]))
    cm_bd = _confusion_matrix(bd_pairs, 2)
    bd_labels = ["continue", "new-doc"]
    _plot_confusion(
        cm_bd,
        bd_labels,
        f"Boundary confusion ({mode}, variant={cfg.model.variant})",
        os.path.join(out_dir, "boundary_confusion.png"),
    )

    # -- document-type confusion (majority-voted per true segment) ---------------
    class_names = read_class_names(cfg.data.root)
    cm_ty = _confusion_matrix(result["type_pairs"], cfg.model.n_types)
    _plot_confusion(
        cm_ty,
        class_names,
        f"Document-type confusion ({mode}, variant={cfg.model.variant})",
        os.path.join(out_dir, "type_confusion.png"),
    )

    prec, rec, f1 = _per_class_prf(cm_ty)
    _plot_prf_bars(
        class_names,
        prec,
        rec,
        f1,
        f"Per-type P/R/F1 ({mode}, variant={cfg.model.variant})",
        os.path.join(out_dir, "type_prf.png"),
    )

    print(f"[analyze] wrote boundary_confusion.png, type_confusion.png, type_prf.png -> {out_dir}")
    print("[analyze] per-type F1:")
    for name, f1_c, support in zip(class_names, f1, cm_ty.sum(axis=1)):
        print(f"  {name:<20s} F1={f1_c:.4f}  (n={int(support)})")


if __name__ == "__main__":
    main()
