"""
PSS metrics.

Page-level (boundary): Precision / Recall / F1 on the breaking-point class, and
Cohen's kappa (both robust to the ~35/65 class imbalance).
Stream-level: MNDD (mean |#pred_docs - #true_docs|, lower better) and STP
(fraction of streams segmented perfectly).
Typing: per-document macro-F1 (majority vote within each true document).

Logit-based helpers (used for window-level validation) ignore label == -100.
Pred-based helpers (used by evaluate.py after stream stitching) take hard labels.
"""

import torch


# -- core (hard predictions) -----------------------------------------------------
def prf_from_preds(pred, true, positive=1):
    pred = torch.as_tensor(pred)
    true = torch.as_tensor(true)
    tp = int(((pred == positive) & (true == positive)).sum())
    fp = int(((pred == positive) & (true != positive)).sum())
    fn = int(((pred != positive) & (true == positive)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def kappa_from_preds(pred, true):
    pred = torch.as_tensor(pred).float()
    true = torch.as_tensor(true).float()
    if pred.numel() == 0:
        return 0.0
    po = float((pred == true).float().mean())
    p1p, p1l = float((pred == 1).float().mean()), float((true == 1).float().mean())
    pe = p1p * p1l + (1 - p1p) * (1 - p1l)
    return (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0


# -- logit wrappers (window-level validation) ------------------------------------
def _flat_valid(logits, labels):
    c = logits.shape[-1]
    logits = logits.reshape(-1, c)
    labels = labels.reshape(-1)
    keep = labels != -100
    return logits[keep].argmax(-1), labels[keep]


def boundary_prf(logits, labels):
    pred, true = _flat_valid(logits, labels)
    if true.numel() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return prf_from_preds(pred, true)


def cohen_kappa(logits, labels):
    pred, true = _flat_valid(logits, labels)
    return kappa_from_preds(pred, true)


def type_accuracy(logits, labels):
    pred, true = _flat_valid(logits, labels)
    if true.numel() == 0:
        return 0.0
    return float((pred == true).float().mean())


# -- stream / document level (evaluate.py) ---------------------------------------
def stream_metrics(streams):
    """streams: list of (pred_boundary[list[int]], true_boundary[list[int]]).
    #documents in a stream == #breaking-points (page 0 is always one)."""
    if not streams:
        return {"mndd": 0.0, "stp": 0.0}
    ndd, perfect = [], 0
    for pred, true in streams:
        ndd.append(abs(sum(pred) - sum(true)))
        perfect += int(pred == true)
    n = len(streams)
    return {"mndd": sum(ndd) / n, "stp": perfect / n}


def type_macro_f1(pairs, n_types):
    """pairs: list of (true_type, pred_type) per document. Macro over present classes."""
    tp = [0] * n_types
    fp = [0] * n_types
    fn = [0] * n_types
    for t, p in pairs:
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    f1s = []
    for c in range(n_types):
        if tp[c] + fp[c] + fn[c] == 0:
            continue  # class absent from this eval set
        prec = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0
