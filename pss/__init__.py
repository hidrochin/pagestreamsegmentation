"""
PSS — Page Stream Segmentation on LayoutXLM.

Given a PDF that is a stream of pages from several concatenated documents,
segment it into individual documents (boundary detection) and classify each
document by type.

Layers:
- pss.config          : OmegaConf config loading (default + --config + CLI)
- pss.data            : PDF rendering, OCR ingest, TABME-style stream synthesis,
                        PSSDataset / collate
- pss.model           : shared LayoutXLM page encoder + sequence heads + variants
- pss.train / evaluate: PyTorch Lightning training loop and metrics

See the approved plan and papers/ (TABME DocEng'22, Wiedemann & Heyer 2018).
"""

__all__ = ["config", "data", "model"]
