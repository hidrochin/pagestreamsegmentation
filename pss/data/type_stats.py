"""Per-type page-frequency stats over a split's synthesized folders, and the
inverse-frequency class weights derived from them for the type loss.

Pure stdlib (reuses ``resolve.py``) so it runs before the torch model is built —
``pss.train`` resolves ``model.type_class_weights="auto"`` through here, and it's
handy standalone to eyeball how imbalanced a corpus is:

    python -m pss.data.type_stats --root datasets/pss --mode train
"""

import argparse
from collections import Counter

from pss.data.resolve import load_json, read_class_names, read_index


def type_page_counts(root, mode="train"):
    """Return (class_names, [page_count_per_class]) over ``mode``'s folders,
    counting every page reference by its inherited document type."""
    names = read_class_names(root)
    idx = {n: i for i, n in enumerate(names)}
    counts = Counter()
    for rel, _ in read_index(root, mode):
        folder = load_json(root, rel)
        for pr in folder["pages"]:
            t = pr.get("type")
            if t in idx:
                counts[idx[t]] += 1
    return names, [counts.get(i, 0) for i in range(len(names))]


def inverse_freq_weights(root, mode="train", beta=1.0, normalize=True, cap=None):
    """Per-class type-loss weights ~ (total_pages / class_pages) ** beta.

    beta=1 is full inverse frequency; 0.5 is a milder correction. Classes with no
    pages in the split get a neutral weight of 1.0. With ``normalize`` the present
    classes' weights are rescaled to mean 1.0 so the overall loss magnitude (and
    thus the effective LR) doesn't shift when weighting is turned on. ``cap`` (if
    set) clips present weights into ``[1/cap, cap]`` to avoid a tiny class blowing
    up the loss. Returns (names, counts, weights)."""
    names, counts = type_page_counts(root, mode)
    total = sum(counts)
    weights = [((total / c) ** beta) if c > 0 else 1.0 for c in counts]
    if normalize:
        present = [w for w, c in zip(weights, counts) if c > 0]
        mean = (sum(present) / len(present)) if present else 1.0
        weights = [
            (w / mean if c > 0 else 1.0) for w, c in zip(weights, counts)
        ]
    if cap:
        weights = [
            (min(cap, max(1.0 / cap, w)) if c > 0 else 1.0)
            for w, c in zip(weights, counts)
        ]
    return names, counts, weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/pss")
    ap.add_argument("--mode", default="train")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--cap", type=float, default=None)
    a = ap.parse_args()
    names, counts, weights = inverse_freq_weights(
        a.root, a.mode, beta=a.beta, cap=a.cap
    )
    total = sum(counts) or 1
    print(f"[type_stats] {a.mode}: {sum(counts)} pages over {len(names)} types")
    for n, c, w in zip(names, counts, weights):
        print(f"    {n:<28} pages={c:<8} ({100*c/total:5.1f}%)  weight={w:.3f}")


if __name__ == "__main__":
    main()
