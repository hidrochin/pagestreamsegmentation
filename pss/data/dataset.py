"""
PSSDataset — yields one sliding *window* of a page stream per item.

A folder (stream) longer than ``max_pages`` is cut into overlapping windows
(stride = ``window_stride``); shorter folders are a single window. Each window's
pages are tokenized with the LayoutXLM tokenizer (word boxes normalized to
0..1000) and their images normalized with the matching image processor.

The tokenizer aligns subword boxes automatically and inserts the special-token
boxes ([CLS]->[0,0,0,0], [SEP]->[1000,1000,1000,1000]).
"""

import torch
from torch.utils.data.dataset import Dataset

from pss.data.page_codec import encode_page
from pss.data.resolve import DocCache, load_json, read_class_names, read_index
from pss.data.windowing import build_stream_windows
from pss.model.page_encoder import build_image_processor, build_tokenizer

__all__ = ["PSSDataset", "read_class_names"]


class PSSDataset(Dataset):
    def __init__(self, cfg, mode, tokenizer=None, image_processor=None):
        self.cfg = cfg
        self.root = cfg.data.root
        self.mode = mode
        self.max_pages = cfg.data.max_pages
        self.stride = cfg.data.window_stride
        self.max_seq_length = cfg.model.max_seq_length

        self.tokenizer = tokenizer or build_tokenizer(cfg.model.backbone)
        self.image_processor = image_processor or build_image_processor(
            cfg.model.backbone
        )

        self.class_names = read_class_names(self.root)
        self.type2id = {c: i for i, c in enumerate(self.class_names)}

        self.folders = read_index(self.root, self.mode)  # list[(folder_relpath, n_pages)]
        self.windows = self._build_windows()  # list[(folder_idx, start, length)]
        self._doc_cache = DocCache(self.root)

    # -- indexing ----------------------------------------------------------------
    def _build_windows(self):
        windows = []
        for fi, (_, n) in enumerate(self.folders):
            for start, length in build_stream_windows(n, self.max_pages, self.stride):
                windows.append((fi, start, length))
        return windows

    def __len__(self):
        return len(self.windows)

    # -- loading -----------------------------------------------------------------
    def _encode_page(self, page_ref):
        page = self._doc_cache.page(page_ref)
        enc = encode_page(
            self.tokenizer, self.image_processor, page, self.root, self.max_seq_length
        )
        enc["boundary"] = int(page_ref["boundary"])
        enc["type"] = self.type2id.get(page_ref["type"], -100)
        return enc

    def __getitem__(self, idx):
        fi, start, length = self.windows[idx]
        rel, _ = self.folders[fi]
        folder = load_json(self.root, rel)
        page_refs = folder["pages"][start : start + length]

        pages = [self._encode_page(pr) for pr in page_refs]
        return {
            "input_ids": torch.stack([p["input_ids"] for p in pages]),  # [P, T]
            "attention_mask": torch.stack([p["attention_mask"] for p in pages]),
            "bbox": torch.stack([p["bbox"] for p in pages]),  # [P, T, 4]
            "image": torch.stack([p["image"] for p in pages]),  # [P, 3, 224, 224]
            "boundary_labels": torch.tensor(
                [p["boundary"] for p in pages], dtype=torch.long
            ),
            "type_labels": torch.tensor([p["type"] for p in pages], dtype=torch.long),
            "meta": {
                "folder_id": folder.get("folder_id", rel),
                "start": start,
                "length": length,
            },
        }
