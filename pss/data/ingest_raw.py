"""
Convert a raw folder of ``{images/, ocr/}`` pairs into the PSS ``docs/`` schema.

Input layout (declared by a `raw_sources.yaml`, see ``configs/raw_sources.yaml``):

    <raw_root>/<source path>/images/<prefix>_<page>.jpg
    <raw_root>/<source path>/ocr/<prefix>_<page>.txt      one word per line:
                                                           "x0,y0,x1,y1<TAB>text"

Images sharing a filename prefix (everything before the trailing "_<page-number>")
are grouped into one multi-page document, ordered by that page number. Each source
path is labeled with a ``type`` from the yaml — the raw folder name is never assumed
to be the type (see ``configs/raw_sources.yaml`` for why: it's more durable than
either renaming folders on disk or a flat folder-name->type dict).

Re-running is incremental: a manifest (``<out_root>/.ingest_manifest.json``) records
each document's source file mtimes, so unchanged documents are skipped and only
new/modified ones are (re)converted. This is what lets you add a new ``sources:``
entry, or drop more images into an existing folder, and rerun the exact same command
to extend ``datasets/pss/docs/`` without reprocessing everything.

After this, (re-)run ``python -m pss.data.synthesize`` to rebuild ``class_names.txt``
/ ``folders/`` / ``preprocessed_files_*.txt`` from the full, updated set of docs.

Run:
    python -m pss.data.ingest_raw --config configs/raw_sources.yaml --out_root datasets/pss
"""

import argparse
import glob
import json
import os
import re
import shutil

from omegaconf import OmegaConf
from PIL import Image

from pss.data.ocr_ingest import make_doc, make_page, parse_sidecar_txt, save_doc

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
DEFAULT_PAGE_PATTERN = r"^(?P<doc>.+)_(?P<page>\d+)$"


def _sanitize(text):
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")


def _manifest_path(out_root):
    return os.path.join(out_root, ".ingest_manifest.json")


def _load_manifest(out_root):
    path = _manifest_path(out_root)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fp:
            return json.load(fp)
    return {}


def _save_manifest(out_root, manifest):
    os.makedirs(out_root, exist_ok=True)
    with open(_manifest_path(out_root), "w", encoding="utf-8") as fp:
        json.dump(manifest, fp)


def expand_sources(raw_root, sources):
    """Each ``sources[].path`` is a glob relative to raw_root; expand to concrete
    folders (must contain images/ and ocr/ subfolders), tagged with their type."""
    expanded = []
    for src in sources:
        pattern = os.path.join(raw_root, src["path"])
        is_glob = any(c in src["path"] for c in "*?[")
        matches = sorted(glob.glob(pattern)) if is_glob else [pattern]
        for m in matches:
            if os.path.isdir(os.path.join(m, "images")) and os.path.isdir(
                os.path.join(m, "ocr")
            ):
                expanded.append((m, src["type"]))
            else:
                print(f"[ingest_raw] skip (no images/+ocr/ subfolders): {m}")
    return expanded


def group_pages(images_dir, pattern):
    """Group image files under images_dir by filename prefix; sort each group by
    its numeric page suffix. Returns dict[doc_prefix] -> list[(page_no, stem, path, ext)].
    """
    rx = re.compile(pattern)
    groups = {}
    for path in sorted(glob.glob(os.path.join(images_dir, "*"))):
        stem, ext = os.path.splitext(os.path.basename(path))
        if ext.lower() not in IMAGE_EXTS:
            continue
        m = rx.match(stem)
        if not m:
            print(f"[ingest_raw] skip (doesn't match page pattern {pattern!r}): {path}")
            continue
        groups.setdefault(m.group("doc"), []).append(
            (int(m.group("page")), stem, path, ext)
        )
    for prefix in groups:
        groups[prefix].sort(key=lambda t: t[0])
    return groups


def _source_signature(pages, ocr_dir):
    """One value per page-image + its OCR sidecar mtime; used to detect changes."""
    sig = []
    for _, stem, img_path, _ in pages:
        ocr_path = os.path.join(ocr_dir, f"{stem}.txt")
        sig.append(
            [
                os.path.getmtime(img_path),
                os.path.getmtime(ocr_path) if os.path.exists(ocr_path) else None,
            ]
        )
    return sig


def convert_document(doc_id, doc_type, pages, ocr_dir, out_root):
    """pages: sorted list of (page_no, stem, img_path, ext). Writes
    docs/<doc_id>.json + images/<doc_id>/page_NNN.<ext>, returns the doc dict."""
    img_out_dir = os.path.join(out_root, "images", doc_id)
    if os.path.isdir(img_out_dir):
        shutil.rmtree(img_out_dir)
    os.makedirs(img_out_dir, exist_ok=True)

    doc_pages = []
    for i, (_, stem, img_path, ext) in enumerate(pages):
        ocr_path = os.path.join(ocr_dir, f"{stem}.txt")
        words = parse_sidecar_txt(ocr_path) if os.path.exists(ocr_path) else []

        dst_name = f"page_{i:03d}{ext.lower()}"
        shutil.copy2(img_path, os.path.join(img_out_dir, dst_name))
        with Image.open(img_path) as im:
            width, height = im.size

        image_relpath = os.path.join("images", doc_id, dst_name)
        doc_pages.append(make_page(width, height, image_relpath, words))

    doc = make_doc(doc_id, doc_type, doc_pages)
    save_doc(out_root, doc)
    return doc


def ingest(raw_root, sources, out_root, page_pattern=DEFAULT_PAGE_PATTERN):
    manifest = _load_manifest(out_root)
    n_converted, n_skipped = 0, 0

    for folder, doc_type in expand_sources(raw_root, sources):
        images_dir = os.path.join(folder, "images")
        ocr_dir = os.path.join(folder, "ocr")
        rel = _sanitize(os.path.relpath(folder, raw_root))

        for doc_prefix, pages in group_pages(images_dir, page_pattern).items():
            doc_id = f"{rel}__{_sanitize(doc_prefix)}"
            sig = _source_signature(pages, ocr_dir)

            if manifest.get(doc_id, {}).get("sig") == sig:
                n_skipped += 1
                continue

            convert_document(doc_id, doc_type, pages, ocr_dir, out_root)
            manifest[doc_id] = {"sig": sig, "type": doc_type, "n_pages": len(pages)}
            n_converted += 1

    _save_manifest(out_root, manifest)
    print(f"[ingest_raw] converted {n_converted} doc(s), skipped {n_skipped} unchanged")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to a raw_sources.yaml")
    ap.add_argument("--out_root", default="datasets/pss")
    ap.add_argument("--page_pattern", default=DEFAULT_PAGE_PATTERN)
    a = ap.parse_args()

    cfg = OmegaConf.load(a.config)
    ingest(cfg.raw_root, cfg.sources, a.out_root, a.page_pattern)


if __name__ == "__main__":
    main()
