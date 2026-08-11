"""Shared, pure-stdlib folder/page resolution: turns a synthesized stream's page
references (``folders/<split>/<fid>.json``) into their underlying document
content (``docs/<doc_id>.json``). Used by both the encoder pipeline
(``pss.data.dataset.PSSDataset``) and ``pss.decoder`` so this JSON-traversal
logic lives in exactly one place.

No torch/transformers import here on purpose: ``pss.decoder`` runs in its own
venv (see ``requirements-decoder.txt``) and must be able to import this module
without pulling in the encoder stack (``pss.model.page_encoder``'s
detectron2/transformers dependency).
"""

import json
import os


def load_json(root, relpath):
    with open(os.path.join(root, relpath), encoding="utf-8") as fp:
        return json.load(fp)


def read_class_names(root):
    with open(os.path.join(root, "class_names.txt"), encoding="utf-8") as fp:
        return [ln.strip() for ln in fp if ln.strip()]


def read_index(root, mode):
    """preprocessed_files_<mode>.txt -> list[(folder_relpath, n_pages)]."""
    path = os.path.join(root, f"preprocessed_files_{mode}.txt")
    folders = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rel, n = line.split("\t")
            folders.append((rel, int(n)))
    return folders


class DocCache:
    """Caches parsed docs/<doc_id>.json across many page lookups within a split."""

    def __init__(self, root):
        self.root = root
        self._docs = {}

    def doc(self, doc_id):
        if doc_id not in self._docs:
            self._docs[doc_id] = load_json(self.root, f"docs/{doc_id}.json")
        return self._docs[doc_id]

    def page(self, page_ref):
        """page_ref: {"doc_id", "page_idx", "boundary", "type", ...} from a
        folder json. Returns the raw docs/*.json page dict
        ({"width", "height", "image", "words"})."""
        doc = self.doc(page_ref["doc_id"])
        return doc["pages"][page_ref["page_idx"]]


def folder_window_pages(root, folder_relpath, start, length, doc_cache=None):
    """Load a folder json and resolve one window's page references to their
    underlying page content. Returns (folder_dict, list[(page_ref, page)])."""
    cache = doc_cache or DocCache(root)
    folder = load_json(root, folder_relpath)
    refs = folder["pages"][start : start + length]
    return folder, [(pr, cache.page(pr)) for pr in refs]
