"""
Collate a batch of variable-length page windows into padded tensors.

Each dataset item is one window: P real pages, each already tokenized to T text
tokens + a 224x224 image. We pad the page axis to the batch's max P.

Padded pages are safe to run through LayoutLMv2/XLM even with an all-zero text
attention mask: the 49 visual tokens are always attendable, so no attention row
is all-masked. We still mark them with page_mask=0 and set their labels to -100
so they contribute nothing to the loss or metrics.

Output batch dict:
    input_ids      [B, P, T]   long
    bbox           [B, P, T, 4] long   (0..1000)
    attention_mask [B, P, T]   long
    image          [B, P, 3, H, W] float
    page_mask      [B, P]      long    (1 = real page)
    boundary_labels[B, P]      long    (0/1, -100 = pad)
    type_labels    [B, P]      long    (type id, -100 = pad)
    meta           list[dict]  (per-stream {folder_id, start, length}) for eval stitching
"""

import torch


def collate_streams(samples):
    b = len(samples)
    p = max(s["input_ids"].shape[0] for s in samples)
    t = samples[0]["input_ids"].shape[1]
    img_shape = samples[0]["image"].shape[1:]  # (3, H, W)

    input_ids = torch.zeros(b, p, t, dtype=torch.long)
    attention_mask = torch.zeros(b, p, t, dtype=torch.long)
    bbox = torch.zeros(b, p, t, 4, dtype=torch.long)
    image = torch.zeros(b, p, *img_shape, dtype=torch.float32)
    page_mask = torch.zeros(b, p, dtype=torch.long)
    boundary_labels = torch.full((b, p), -100, dtype=torch.long)
    type_labels = torch.full((b, p), -100, dtype=torch.long)
    meta = []

    for i, s in enumerate(samples):
        n = s["input_ids"].shape[0]
        input_ids[i, :n] = s["input_ids"]
        attention_mask[i, :n] = s["attention_mask"]
        bbox[i, :n] = s["bbox"]
        image[i, :n] = s["image"]
        page_mask[i, :n] = 1
        boundary_labels[i, :n] = s["boundary_labels"]
        type_labels[i, :n] = s["type_labels"]
        meta.append(s["meta"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "bbox": bbox,
        "image": image,
        "page_mask": page_mask,
        "boundary_labels": boundary_labels,
        "type_labels": type_labels,
        "meta": meta,
    }
