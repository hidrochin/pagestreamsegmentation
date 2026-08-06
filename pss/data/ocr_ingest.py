"""
Adapter from your OCR model's output to the PSS single-document JSON schema.

You already have an OCR detection model. Wire it in ``ingest_ocr`` below: given a
page image, return a list of (text, [x0, y0, x1, y1]) word boxes in PIXEL
coordinates. The rest (box normalization, tokenization, image processing) is
handled downstream by PSSDataset, so this adapter only has to produce the schema.

Builders here write ``docs/<doc_id>.json`` (+ expect images under images/<doc_id>/),
which pss.data.synthesize then concatenates into training streams.
"""

import json
import os


def make_word(text, box):
    x0, y0, x1, y1 = box
    return {"text": str(text), "box": [int(x0), int(y0), int(x1), int(y1)]}


def make_page(width, height, image_relpath, words):
    """words: iterable of (text, box) or of dicts from make_word."""
    norm = [w if isinstance(w, dict) else make_word(*w) for w in words]
    return {
        "width": int(width),
        "height": int(height),
        "image": image_relpath,  # relative to data.root
        "words": norm,
    }


def make_doc(doc_id, doc_type, pages):
    return {"doc_id": str(doc_id), "type": str(doc_type), "pages": list(pages)}


def save_doc(root, doc):
    out_dir = os.path.join(root, "docs")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{doc['doc_id']}.json")
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(doc, fp)
    return path


def ingest_ocr(page_image_path):
    """PLUG YOUR OCR MODEL IN HERE.

    Input:  path to a single page image.
    Return: list of (text, [x0, y0, x1, y1]) in pixel coords.

    Example wiring:
        dets = your_ocr_model.run(page_image_path)   # -> boxes + transcriptions
        return [(d.text, d.xyxy) for d in dets]
    """
    raise NotImplementedError(
        "Connect your OCR detection model here — return (text, [x0,y0,x1,y1]) per word."
    )
