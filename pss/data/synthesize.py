"""
TABME Algorithm 1 — synthesize page streams ("folders") from single documents.

Concatenate random sequences of whole, type-labeled documents into folders:
- number of documents per folder ~ Poisson(lambda), at least 1;
- documents sampled without repetition within a split;
- the SPLIT IS AT THE DOCUMENT LEVEL (a document never appears in two splits),
  which prevents leakage — the critical detail from TABME.

The first page of each source document is a breaking-point (boundary=1); every
page inherits its document's type. Folders store page *references*, so 100k
folders don't duplicate page content.

Run:
    python -m pss.data.synthesize --root datasets/pss \
        --lambda 5 --n_train 100000 --n_val 5000 --n_test 5000 --seed 42
"""

import argparse
import glob
import json
import math
import os
import random


def _poisson(lam, rng):
    """Knuth's Poisson sampler (stdlib only — no numpy dependency here)."""
    el = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= el:
            return k - 1


def _load_doc_meta(root):
    """Return list of {doc_id, type, n_pages} for every docs/*.json, and the
    sorted set of types."""
    metas, types = [], set()
    for path in sorted(glob.glob(os.path.join(root, "docs", "*.json"))):
        with open(path, encoding="utf-8") as fp:
            d = json.load(fp)
        metas.append(
            {"doc_id": d["doc_id"], "type": d["type"], "n_pages": len(d["pages"])}
        )
        types.add(d["type"])
    return metas, sorted(types)


def _split_docs(metas, splits, rng):
    docs = list(metas)
    rng.shuffle(docs)
    n = len(docs)
    n_tr = int(splits[0] * n)
    n_va = int(splits[1] * n)
    return {
        "train": docs[:n_tr],
        "val": docs[n_tr : n_tr + n_va],
        "test": docs[n_tr + n_va :],
    }


def _make_folder(split_docs, lam, rng):
    k = max(1, _poisson(lam, rng))
    k = min(k, len(split_docs))
    chosen = rng.sample(split_docs, k)
    pages = []
    for d in chosen:
        for pidx in range(d["n_pages"]):
            pages.append(
                {
                    "doc_id": d["doc_id"],
                    "page_idx": pidx,
                    "boundary": 1 if pidx == 0 else 0,
                    "type": d["type"],
                }
            )
    return pages


def synthesize(root, lam=5.0, n_folders=None, splits=(0.9, 0.05, 0.05), seed=42):
    n_folders = n_folders or {"train": 100000, "val": 5000, "test": 5000}
    rng = random.Random(seed)

    metas, types = _load_doc_meta(root)
    if not metas:
        raise FileNotFoundError(f"No docs/*.json found under {root}")

    with open(os.path.join(root, "class_names.txt"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(types) + "\n")

    by_split = _split_docs(metas, splits, rng)

    for split, docs in by_split.items():
        if not docs:
            continue
        out_dir = os.path.join(root, "folders", split)
        os.makedirs(out_dir, exist_ok=True)
        index = []
        for i in range(n_folders[split]):
            fid = f"{split}_{i:07d}"
            pages = _make_folder(docs, lam, rng)
            rel = os.path.join("folders", split, f"{fid}.json")
            with open(os.path.join(root, rel), "w", encoding="utf-8") as fp:
                json.dump({"folder_id": fid, "pages": pages}, fp)
            index.append(f"{rel}\t{len(pages)}")
        with open(
            os.path.join(root, f"preprocessed_files_{split}.txt"), "w", encoding="utf-8"
        ) as fp:
            fp.write("\n".join(index) + "\n")
        print(f"[{split}] {len(index)} folders, {len(docs)} source docs")

    print(f"types ({len(types)}): {types}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/pss")
    ap.add_argument("--lambda", dest="lam", type=float, default=5.0)
    ap.add_argument("--n_train", type=int, default=100000)
    ap.add_argument("--n_val", type=int, default=5000)
    ap.add_argument("--n_test", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    synthesize(
        a.root,
        lam=a.lam,
        n_folders={"train": a.n_train, "val": a.n_val, "test": a.n_test},
        seed=a.seed,
    )


if __name__ == "__main__":
    main()
