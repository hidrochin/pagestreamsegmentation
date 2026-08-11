"""
Production inference: fixed-width sliding-window PSS over a live page stream, no
ground truth required (unlike ``pss.evaluate``).

Reuses the same window boundaries (``build_stream_windows``) and page encoding
(``encode_page``) as training/eval, and the same overlap-averaging stitch
(``pss.stitch.StreamAccumulator``) as ``pss.evaluate`` — only the forward pass and
padding are different.

Training runs on a full GPU with `data.max_pages` chosen for accuracy; the deployed
hardware instead forces a FIXED window width per forward pass. Any window shorter
than that width (a stream's tail, or a whole stream under one window) is padded up
to it with all-zero tensors + page_mask=0 — the same convention ``collate_streams``
already uses to pad the page axis across a training batch — instead of repeating
the last real page. A repeated real page would inject fabricated content into the
model's context; a masked pad slot is a path the model is already trained to
ignore (page_mask=0 -> excluded from loss/metrics, and per collate.py, safe
through LayoutLMv2/XLM since visual tokens stay attendable under an all-zero text
mask). For best accuracy, set ``infer.window_pages`` to the same value used for
``data.max_pages`` at training time — the model has only ever seen that context.

Run:
    python -m pss.infer --config=configs/pss.yaml \
        pretrained_model_file=pss_runs/i1_layoutxlm/checkpoints/last.ckpt \
        infer.stream=stream.json infer.window_pages=10 infer.window_stride=6
"""

import json

import torch

from pss.config import get_config
from pss.data.page_codec import encode_page
from pss.data.windowing import build_stream_windows
from pss.evaluate import load_weights
from pss.lightning_module import PSSLightningModule
from pss.model.page_encoder import build_image_processor, build_tokenizer
from pss.stitch import StreamAccumulator


def pad_window(pages_enc, width):
    """pages_enc: list of encode_page() dicts for the real pages in one window
    (len <= width). Pads up to `width` with all-zero tensors, page_mask=0 for the
    padded slots — mirrors collate_streams' batch-padding convention."""
    t = pages_enc[0]["input_ids"].shape[0]
    img_shape = pages_enc[0]["image"].shape

    input_ids = torch.zeros(width, t, dtype=torch.long)
    attention_mask = torch.zeros(width, t, dtype=torch.long)
    bbox = torch.zeros(width, t, 4, dtype=torch.long)
    image = torch.zeros(width, *img_shape, dtype=torch.float32)
    page_mask = torch.zeros(width, dtype=torch.long)

    for i, p in enumerate(pages_enc):
        input_ids[i] = p["input_ids"]
        attention_mask[i] = p["attention_mask"]
        bbox[i] = p["bbox"]
        image[i] = p["image"]
        page_mask[i] = 1

    return input_ids, attention_mask, bbox, image, page_mask


def run_stream(
    net,
    device,
    tokenizer,
    image_processor,
    stream_doc,
    image_root,
    max_seq_length,
    window_pages,
    window_stride,
):
    """stream_doc: {"doc_id", "pages": [...]} in the docs/*.json page schema
    (``type`` is ignored — that's what we're predicting). Returns a list of
    {"page_idx", "pred_boundary", "pred_type"} in stream order."""
    pages = stream_doc["pages"]
    fid = stream_doc.get("doc_id", "stream")
    acc = StreamAccumulator()

    net.eval()
    with torch.no_grad():
        for start, length in build_stream_windows(
            len(pages), window_pages, window_stride
        ):
            encs = [
                encode_page(
                    tokenizer,
                    image_processor,
                    pages[start + j],
                    image_root,
                    max_seq_length,
                )
                for j in range(length)
            ]
            input_ids, attention_mask, bbox, image, page_mask = pad_window(
                encs, window_pages
            )
            batch = {
                "input_ids": input_ids.unsqueeze(0).to(device),
                "attention_mask": attention_mask.unsqueeze(0).to(device),
                "bbox": bbox.unsqueeze(0).to(device),
                "image": image.unsqueeze(0).to(device),
                "page_mask": page_mask.unsqueeze(0).to(device),
            }
            out = net(batch)
            bl = out["boundary_logits"].float().cpu()
            tl = out["type_logits"].float().cpu()
            # length = REAL page count for this window; the padded tail (if any)
            # is never accumulated, so it can't pollute the stitched average.
            acc.add_batch(
                [{"folder_id": fid, "start": start, "length": length}], bl, tl
            )

    idxs, pred_bd, pred_ty = acc.stitch(fid)
    return [
        {"page_idx": p, "pred_boundary": b, "pred_type": t}
        for p, b, t in zip(idxs, pred_bd, pred_ty)
    ]


def main():
    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    module = PSSLightningModule(cfg)
    net = module.net
    load_weights(net, cfg.pretrained_model_file)
    net.to(device).eval()

    tokenizer = build_tokenizer(cfg.model.backbone)
    image_processor = build_image_processor(cfg.model.backbone)

    with open(cfg.infer.stream, encoding="utf-8") as fp:
        stream_doc = json.load(fp)

    image_root = cfg.infer.image_root or cfg.data.root

    preds = run_stream(
        net,
        device,
        tokenizer,
        image_processor,
        stream_doc,
        image_root,
        cfg.model.max_seq_length,
        cfg.infer.window_pages,
        cfg.infer.window_stride,
    )

    if cfg.infer.out:
        with open(cfg.infer.out, "w", encoding="utf-8") as fp:
            json.dump(preds, fp, indent=2)
        print(f"[infer] wrote {len(preds)} page predictions to {cfg.infer.out}")
    else:
        print(json.dumps(preds, indent=2))


if __name__ == "__main__":
    main()
