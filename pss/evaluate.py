"""
Evaluate a trained PSS checkpoint with full stream-level stitching.

    python -m pss.evaluate --config=configs/pss.yaml \
        pretrained_model_file=pss_runs/i1_layoutxlm/checkpoints/last.ckpt \
        eval_mode=test

Overlapping sliding windows are stitched back per stream (logits averaged over
overlaps), then we report breaking-point Precision/Recall/F1 + kappa (page level),
MNDD + STP (stream level), and document-type macro-F1 (typing scored on the true
segments, so it is decoupled from segmentation errors).
"""

from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader

from pss import metrics
from pss.config import get_config
from pss.data import PSSDataset, collate_streams
from pss.lightning_module import PSSLightningModule


def load_weights(net, ckpt_path):
    # weights_only=False: Lightning checkpoints hold non-tensor objects (hyper-params,
    # callbacks); torch>=2.6 flipped this default to True, which would refuse to load.
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    new = {}
    for k, v in sd.items():
        new[k[len("net.") :] if k.startswith("net.") else k] = v
    missing, unexpected = net.load_state_dict(new, strict=False)
    if missing:
        print(f"[load] missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"[load] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")


def main():
    cfg = get_config()
    mode = cfg.get("eval_mode", "test")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    module = PSSLightningModule(cfg)
    net = module.net
    load_weights(net, cfg.pretrained_model_file)
    net.to(device).eval()

    ds = PSSDataset(cfg, mode)
    loader = DataLoader(
        ds,
        batch_size=cfg.val.batch_size,
        shuffle=False,
        num_workers=cfg.val.num_workers,
        collate_fn=collate_streams,
    )

    # per-folder, per-page accumulators (averaged over overlapping windows)
    bd_sum = defaultdict(dict)
    ty_sum = defaultdict(dict)
    cnt = defaultdict(dict)
    bd_true = defaultdict(dict)
    ty_true = defaultdict(dict)

    with torch.no_grad():
        for batch in loader:
            gpu_batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = net(gpu_batch)
            bl = out["boundary_logits"].float().cpu()
            tl = out["type_logits"].float().cpu()
            by = batch["boundary_labels"]
            ty = batch["type_labels"]

            for i, meta in enumerate(batch["meta"]):
                fid, start, length = meta["folder_id"], meta["start"], meta["length"]
                for j in range(length):
                    p = start + j
                    bd_sum[fid][p] = bd_sum[fid].get(p, 0) + bl[i, j]
                    ty_sum[fid][p] = ty_sum[fid].get(p, 0) + tl[i, j]
                    cnt[fid][p] = cnt[fid].get(p, 0) + 1
                    bd_true[fid][p] = int(by[i, j])
                    ty_true[fid][p] = int(ty[i, j])

    # stitch + score
    streams = []  # (pred_boundary, true_boundary) per stream
    type_pairs = []  # (true_type, pred_type) per document
    all_pred, all_true = [], []

    for fid in bd_sum:
        idxs = sorted(bd_sum[fid])
        pred_bd = [int((bd_sum[fid][p] / cnt[fid][p]).argmax()) for p in idxs]
        true_bd = [bd_true[fid][p] for p in idxs]
        pred_ty = [int((ty_sum[fid][p] / cnt[fid][p]).argmax()) for p in idxs]
        streams.append((pred_bd, true_bd))
        all_pred.extend(pred_bd)
        all_true.extend(true_bd)

        # segment by TRUE boundaries; majority-vote the type within each document
        docs, cur = [], []
        for k in range(len(idxs)):
            if true_bd[k] == 1 and cur:
                docs.append(cur)
                cur = []
            cur.append(k)
        if cur:
            docs.append(cur)
        for d in docs:
            pv = Counter(pred_ty[k] for k in d).most_common(1)[0][0]
            tv = ty_true[fid][idxs[d[0]]]
            type_pairs.append((tv, pv))

    prf = metrics.prf_from_preds(all_pred, all_true)
    kappa = metrics.kappa_from_preds(all_pred, all_true)
    sm = metrics.stream_metrics(streams)
    type_f1 = metrics.type_macro_f1(type_pairs, cfg.model.n_types)

    print(f"\n=== PSS eval ({mode}, variant={cfg.model.variant}) ===")
    print(f"boundary  P={prf['precision']:.4f}  R={prf['recall']:.4f}  F1={prf['f1']:.4f}  kappa={kappa:.4f}")
    print(f"stream    MNDD={sm['mndd']:.4f}  STP={sm['stp']:.4f}")
    print(f"typing    doc macro-F1={type_f1:.4f}")


if __name__ == "__main__":
    main()
