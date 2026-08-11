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

from collections import Counter

import torch
from torch.utils.data import DataLoader

from pss import metrics
from pss.config import get_config
from pss.data import PSSDataset, collate_streams
from pss.lightning_module import PSSLightningModule
from pss.stitch import StreamAccumulator


def load_weights(net, ckpt_path, strict=True):
    # weights_only=False: Lightning checkpoints hold non-tensor objects (hyper-params,
    # callbacks); torch>=2.6 flipped this default to True, which would refuse to load.
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    new = {}
    for k, v in sd.items():
        new[k[len("net.") :] if k.startswith("net.") else k] = v
    missing, unexpected = net.load_state_dict(new, strict=False)
    if missing or unexpected:
        msg = (
            f"[load] missing keys: {len(missing)} {missing}\n"
            f"[load] unexpected keys: {len(unexpected)} {unexpected}"
        )
        if strict:
            raise RuntimeError(
                f"{msg}\ncheckpoint {ckpt_path} does not match the current model "
                "(model.variant/page_embed/seq_head.type mismatch?). Pass "
                "allow_partial_load=true to load anyway."
            )
        print(msg)


def load_eval_net(cfg, device):
    """Build the model, load a checkpoint (strict by default), move to device+eval.
    Shared by evaluate.py and analyze.py so both score the exact same weights."""
    module = PSSLightningModule(cfg)
    net = module.net
    load_weights(
        net, cfg.pretrained_model_file, strict=not cfg.get("allow_partial_load", False)
    )
    net.to(device).eval()
    return net


def collect_predictions(cfg, net, device, mode):
    """Run `net` over `mode`'s stitched streams. Returns a dict with everything
    evaluate.py/analyze.py need: per-stream (pred, true) boundary sequences,
    per-document (true, pred) type pairs (typed on the TRUE segmentation, so
    typing is decoupled from segmentation errors), and flat page-level
    boundary predictions/labels."""
    ds = PSSDataset(cfg, mode)
    loader = DataLoader(
        ds,
        batch_size=cfg.val.batch_size,
        shuffle=False,
        num_workers=cfg.val.num_workers,
        collate_fn=collate_streams,
    )

    acc = StreamAccumulator()

    with torch.no_grad():
        for batch in loader:
            gpu_batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = net(gpu_batch)
            bl = out["boundary_logits"].float().cpu()
            tl = out["type_logits"].float().cpu()
            acc.add_batch(
                batch["meta"], bl, tl, batch["boundary_labels"], batch["type_labels"]
            )

    streams = []  # (pred_boundary, true_boundary) per stream
    type_pairs = []  # (true_type, pred_type) per document
    all_pred, all_true = [], []

    for fid in acc.stream_ids():
        idxs, pred_bd, pred_ty = acc.stitch(fid)
        true_bd = [acc.bd_true[fid][p] for p in idxs]
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
            tv = acc.ty_true[fid][idxs[d[0]]]
            type_pairs.append((tv, pv))

    return {
        "streams": streams,
        "type_pairs": type_pairs,
        "all_pred": all_pred,
        "all_true": all_true,
    }


def main():
    cfg = get_config()
    mode = cfg.get("eval_mode", "test")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    net = load_eval_net(cfg, device)
    result = collect_predictions(cfg, net, device, mode)
    all_pred, all_true = result["all_pred"], result["all_true"]
    streams, type_pairs = result["streams"], result["type_pairs"]

    prf = metrics.prf_from_preds(all_pred, all_true)
    kappa = metrics.kappa_from_preds(all_pred, all_true)
    sm = metrics.stream_metrics(streams)
    type_f1 = metrics.type_macro_f1(type_pairs, cfg.model.n_types)

    print(f"\n=== PSS eval ({mode}, variant={cfg.model.variant}) ===")
    print(
        f"boundary  P={prf['precision']:.4f}  R={prf['recall']:.4f}  F1={prf['f1']:.4f}  kappa={kappa:.4f}"
    )
    print(f"stream    MNDD={sm['mndd']:.4f}  STP={sm['stp']:.4f}")
    print(f"typing    doc macro-F1={type_f1:.4f}")


if __name__ == "__main__":
    main()
