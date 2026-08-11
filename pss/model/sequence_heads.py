"""
Context-over-pages modules and prediction heads.

The sequence head turns per-page embeddings [B, P, D] into context-aware page
representations [B, P, C] (P = pages in a window). Two options:
- TemporalCNN     : TABME's proven head — 1D convs over the page axis with
                    circular padding (primary; cheap, strong).
- TransformerOverPages : CoSMo-style encoder over pages (ablation).

Heads map a page representation to boundary (2-class) and type (n_types) logits.
"""

import torch
from torch import nn


class TemporalCNN(nn.Module):
    """Stack of Conv1d over the page axis (TABME). Zero padding at the sequence
    ends (not circular — pages don't wrap); ReLU + Dropout after each conv.
    Channels are preserved (or set to ``hidden_size``) so the heads see D-dim
    contextual features — the final logit projection lives in the heads, not the
    conv stack. Padded pages (``page_mask==0``) are re-zeroed after every layer so
    their (bias-only) activations can never leak into real pages' receptive
    fields, matching TransformerOverPages' masking guarantee."""

    def __init__(self, dim, n_layers=5, kernel_size=3, dropout=0.2, hidden_size=0):
        super().__init__()
        c = hidden_size or dim
        pad = kernel_size // 2
        blocks = []
        in_c = dim
        for _ in range(n_layers):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_c, c, kernel_size, padding=pad),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                )
            )
            in_c = c
        self.blocks = nn.ModuleList(blocks)
        self.out_dim = in_c

    def forward(self, x, page_mask=None):  # x: [B, P, D]
        m = page_mask.unsqueeze(-1).to(x.dtype) if page_mask is not None else None
        h = x * m if m is not None else x
        h = h.transpose(1, 2)  # [B, D, P]
        m_t = m.transpose(1, 2) if m is not None else None  # [B, 1, P]
        for block in self.blocks:
            h = block(h)
            if m_t is not None:
                h = h * m_t
        return h.transpose(1, 2)  # [B, P, C]


class TransformerOverPages(nn.Module):
    """Transformer encoder over the page sequence with learned page-position
    embeddings; padded pages are masked out via ``src_key_padding_mask``."""

    def __init__(self, dim, n_layers=4, n_heads=8, dropout=0.2, max_pages=512):
        super().__init__()
        self.pos = nn.Embedding(max_pages, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.net = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_dim = dim

    def forward(self, x, page_mask=None):  # x: [B, P, D]
        p = x.size(1)
        pos = self.pos(torch.arange(p, device=x.device)).unsqueeze(0)
        key_padding = None
        if page_mask is not None:
            key_padding = ~page_mask.bool()  # True marks padded pages
        return self.net(x + pos, src_key_padding_mask=key_padding)


class BoundaryHead(nn.Module):
    """Per-page breaking-point classifier (2 classes: continuation / new-doc)."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 2))

    def forward(self, x):
        return self.net(x)


class TypeHead(nn.Module):
    """Per-page document-type classifier (aggregated per segment at inference)."""

    def __init__(self, dim, n_types, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, n_types))

    def forward(self, x):
        return self.net(x)
