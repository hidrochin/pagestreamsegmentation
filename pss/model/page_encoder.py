"""
Shared LayoutXLM page encoder.

Every PSS variant (B0/B1/I1) encodes an individual page with this module and then
differs only in how it models context across pages. LayoutXLM shares the
LayoutLMv2 architecture, so we load it via ``LayoutLMv2Model`` — a *single* encoder
that fuses text + layout + image. This subsumes TABME's two-stream
"LayoutLM (text+layout) + ResNet (image)" design, and (per the TABME ablation)
the image modality is the dominant one, so the integrated visual backbone matters.

Input tensors for one batch of pages (P = #pages in the batch):
    input_ids       [P, T]           SentencePiece/WordPiece ids
    bbox            [P, T, 4]         int boxes in 0..1000 (x0,y0,x1,y1)
    attention_mask  [P, T]            1 for real text tokens
    image           [P, 3, 224, 224]  normalized page image (LayoutLMv2 pixel space)

Output:
    page embeddings [P, output_dim]

Tokenizer/image-processor factories live here too so the data layer and the model
share one source for the backbone name and preprocessing.
"""

import os

import torch
from torch import nn

# Cache downloaded HF weights (LayoutXLM, ~2.8GB) inside the repo instead of the
# user's home cache, so they survive independently of any global cache cleanup and
# don't get silently re-fetched. gitignored (pretrained_models/) — never commit
# these. Respects an explicit HF_HOME if the environment already sets one.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault(
    "HF_HOME", os.path.join(_REPO_ROOT, "pretrained_models", "hf_cache")
)


def build_tokenizer(backbone_name):
    """AutoTokenizer picks the right class per backbone: LayoutXLMTokenizer
    (SentencePiece) for layoutxlm-base, LayoutLMv2Tokenizer (WordPiece) for the
    English layoutlmv2 checkpoint. Both accept ``(words, boxes=...)`` and align
    subword boxes automatically."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(backbone_name)


def build_image_processor(backbone_name):
    """Image preprocessing that matches the pretrained visual backbone. We disable
    the built-in OCR (``apply_ocr=False``) because we bring our own OCR tokens/boxes;
    the processor is used only to resize to 224 and normalize into pixel space."""
    try:  # transformers >= 4.26 renamed FeatureExtractor -> ImageProcessor
        from transformers import LayoutLMv2ImageProcessor as _IP
    except Exception:  # pragma: no cover - older transformers
        from transformers import LayoutLMv2FeatureExtractor as _IP
    return _IP.from_pretrained(backbone_name, apply_ocr=False)


class PageEncoder(nn.Module):
    """LayoutXLM/LayoutLMv2 backbone -> one embedding per page.

    page_embed:
        "cls"              -> [CLS] hidden                              (dim = H)
        "cls_mean"         -> [CLS] ++ mean(text tokens)               (dim = 2H)
        "cls_mean_visual"  -> [CLS] ++ mean(text) ++ mean(visual)      (dim = 3H)
    The visual pooling option exists because vision dominates PSS (TABME ablation).
    """

    def __init__(self, backbone_name, page_embed="cls", freeze=False):
        super().__init__()
        from transformers import LayoutLMv2Model

        self.backbone = LayoutLMv2Model.from_pretrained(backbone_name)
        self.page_embed = page_embed
        self.hidden = self.backbone.config.hidden_size
        if freeze:
            self.freeze()

    def freeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    @property
    def output_dim(self):
        return {
            "cls": self.hidden,
            "cls_mean": 2 * self.hidden,
            "cls_mean_visual": 3 * self.hidden,
        }[self.page_embed]

    def forward(self, input_ids, bbox, attention_mask, image):
        # LayoutLMv2/XLM appends V=49 visual tokens AFTER the T text tokens.
        # The text attention_mask is enough; the model builds the visual mask itself.
        out = self.backbone(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            image=image,
        )
        h = out.last_hidden_state  # [P, T+V, H]
        text_len = input_ids.size(1)
        text = h[:, :text_len]  # [P, T, H]
        cls = text[:, 0]  # [P, H]

        feats = [cls]
        if self.page_embed in ("cls_mean", "cls_mean_visual"):
            m = attention_mask.unsqueeze(-1).to(text.dtype)  # [P, T, 1]
            mean_text = (text * m).sum(1) / m.sum(1).clamp_min(1.0)
            feats.append(mean_text)
        if self.page_embed == "cls_mean_visual":
            visual = h[:, text_len:]  # [P, V, H]
            feats.append(visual.mean(1))

        return torch.cat(feats, dim=-1)  # [P, output_dim]


def build_page_encoder(cfg):
    return PageEncoder(
        backbone_name=cfg.model.backbone,
        page_embed=cfg.model.page_embed,
        freeze=cfg.model.freeze_encoder,
    )
