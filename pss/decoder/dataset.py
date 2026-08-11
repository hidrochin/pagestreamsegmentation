"""
Builds decoder training/eval examples ({"prompt","completion","images"} JSONL)
from the same ``docs/*.json`` + ``folders/*/*.json`` the encoder pipeline reads
(see ``pss/data/resolve.py``) — no changes to ingestion/synthesis needed.

Caches the built JSONL under ``<data.root>/decoder_cache/<mode>_<context_mode>_
<modality>.jsonl`` so repeated training runs against the same config don't
re-serialize every window. Delete the cache file (or pass ``force=True``) after
changing ``decoder.context_mode``/``decoder.modality``/``data.max_pages`` etc.

    python -m pss.decoder.dataset --config=configs/pss_decoder.yaml mode=train
"""

import json
import os

from pss.data.resolve import DocCache, folder_window_pages, read_class_names, read_index
from pss.data.windowing import build_stream_windows
from pss.decoder.prompt import (
    build_pair_completion,
    build_pair_prompt,
    build_window_completion,
    build_window_prompt,
    serialize_page_text,
)


def _image_path(root, page):
    return os.path.join(root, page["image"])


def _pair_windows(n):
    """Every page, one at a time — the paper's page-pair formulation."""
    return [(i, 1) for i in range(n)]


def build_examples(cfg, split):
    """Yields {"prompt", "completion", "images", "meta"} dicts for one split.
    ``meta`` (folder_id/start/length/true_boundary/true_type — the latter as
    int ids, same encoding as the encoder pipeline's ``type2id``) is unused by
    training but lets pss.decoder.evaluate reconstruct per-stream predictions
    without re-resolving pages. Windows are built with
    pss.data.windowing.build_stream_windows in "window" mode — the exact
    function the encoder pipeline uses — so the two tracks see identical
    windows when compared head-to-head."""
    root = cfg.data.root
    modality = cfg.decoder.modality
    type_names = read_class_names(root)
    type2id = {c: i for i, c in enumerate(type_names)}
    folders = read_index(root, split)
    cache = DocCache(root)

    for rel, n in folders:
        if cfg.decoder.context_mode == "pair":
            windows = _pair_windows(n)
        else:
            windows = build_stream_windows(n, cfg.data.max_pages, cfg.data.window_stride)

        for start, length in windows:
            folder, pairs = folder_window_pages(root, rel, start, length, cache)
            fid = folder.get("folder_id", rel)
            true_boundary = [pr["boundary"] for pr, _ in pairs]
            true_type = [type2id.get(pr["type"], -100) for pr, _ in pairs]
            meta = {
                "folder_id": fid,
                "start": start,
                "length": length,
                "true_boundary": true_boundary,
                "true_type": true_type,
            }

            if cfg.decoder.context_mode == "pair":
                pr, page = pairs[0]
                prior_text = ""
                prior_image = None
                if start > 0:
                    _, prior_pairs = folder_window_pages(root, rel, start - 1, 1, cache)
                    prior_pr, prior_page = prior_pairs[0]
                    prior_text = serialize_page_text(prior_page)
                    prior_image = _image_path(root, prior_page)
                page_text = serialize_page_text(page)
                prompt = build_pair_prompt(prior_text, page_text, type_names)
                type_name = pr["type"] if pr["boundary"] else None
                completion = build_pair_completion(pr["boundary"], type_name)
                images = [prior_image, _image_path(root, page)] if modality == "vision" else []
                yield {"prompt": prompt, "completion": completion, "images": images, "meta": meta}
            else:
                page_texts = [serialize_page_text(p) for _, p in pairs]
                prompt = build_window_prompt(page_texts, type_names)
                boundaries = [pr["boundary"] for pr, _ in pairs]
                types = [pr["type"] if pr["boundary"] else None for pr, _ in pairs]
                completion = build_window_completion(boundaries, types)
                images = [_image_path(root, p) for _, p in pairs] if modality == "vision" else []
                yield {"prompt": prompt, "completion": completion, "images": images, "meta": meta}


def _cache_path(cfg, split):
    name = f"{split}_{cfg.decoder.context_mode}_{cfg.decoder.modality}.jsonl"
    return os.path.join(cfg.data.root, "decoder_cache", name)


def build_cache(cfg, split, force=False):
    """Materialize a split's examples to JSONL under decoder_cache/. Skips
    rebuilding if the cache already exists and force=False. Returns the path."""
    path = _cache_path(cfg, split)
    if os.path.exists(path) and not force:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    n = 0
    with open(tmp, "w", encoding="utf-8") as fp:
        for ex in build_examples(cfg, split):
            fp.write(json.dumps(ex) + "\n")
            n += 1
    os.replace(tmp, path)
    print(f"[decoder.dataset] wrote {n} examples ({split}) -> {path}")
    return path


def load_cache(path):
    with open(path, encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def main():
    from pss.decoder.config import get_config

    cfg = get_config()
    mode = cfg.get("mode", "train")
    build_cache(cfg, mode, force=cfg.get("force", False))


if __name__ == "__main__":
    main()
