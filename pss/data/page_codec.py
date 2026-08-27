"""Page -> model-input tensor encoding, shared by ``PSSDataset`` (training/eval,
pages looked up from ``docs/<doc_id>.json``) and ``pss.infer`` (production, pages
come from a live stream but are in the same schema).
"""

import os

from PIL import Image


def norm_boxes(words, width, height):
    """Pixel boxes -> int 0..1000, clamped. Returns (texts, boxes).

    Each axis is sorted (min then max) *before* clamping so the result always
    satisfies ``0 <= x0 <= x1 <= 1000`` and ``0 <= y0 <= y1 <= 1000``. LayoutLMv2/XLM
    looks up its 2-D position embeddings on the box *spans* (``x1-x0``, ``y1-y0``)
    with ``nn.Embedding(1024)``; a negative span (an inverted box, which real OCR
    routinely emits) is an out-of-range index -> CUDA device-side assert. Clamping
    each coordinate independently would preserve such an inversion, so we normalize
    ordering here."""
    texts, boxes = [], []
    sx = 1000.0 / max(1, width)
    sy = 1000.0 / max(1, height)
    for w in words:
        x0, y0, x1, y1 = w["box"]
        xa, xb = sorted((x0 * sx, x1 * sx))
        ya, yb = sorted((y0 * sy, y1 * sy))
        bx = [
            min(1000, max(0, int(xa))),
            min(1000, max(0, int(ya))),
            min(1000, max(0, int(xb))),
            min(1000, max(0, int(yb))),
        ]
        texts.append(w["text"])
        boxes.append(bx)
    if not texts:  # empty page — give one dummy token so the tokenizer is happy
        texts, boxes = ["[UNK]"], [[0, 0, 0, 0]]
    return texts, boxes


def encode_page(tokenizer, image_processor, page, image_root, max_seq_length):
    """``page``: {"width", "height", "image", "words"} (the docs/*.json page schema).
    ``image`` is resolved relative to ``image_root`` (``data.root``).

    Returns {"input_ids": [T], "attention_mask": [T], "bbox": [T,4], "image": [3,H,W]}.
    """
    texts, boxes = norm_boxes(page["words"], page["width"], page["height"])

    enc = tokenizer(
        text=texts,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
        return_tensors="pt",
    )
    img = Image.open(os.path.join(image_root, page["image"])).convert("RGB")
    pv = image_processor(img, return_tensors="pt")
    pixel = pv["pixel_values"] if "pixel_values" in pv else pv["image"]

    return {
        "input_ids": enc["input_ids"][0],
        "attention_mask": enc["attention_mask"][0],
        "bbox": enc["bbox"][0],
        "image": pixel[0],
    }
