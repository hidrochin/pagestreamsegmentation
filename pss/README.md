# PSS — Page Stream Segmentation on LayoutXLM

Split a PDF that is a *stream* of several concatenated documents into individual
documents (breaking-point detection) and classify each document by type. Built on
a single **LayoutXLM** page encoder (text + layout + image), with a 1D temporal
CNN over the page sequence (TABME-style). See the approved plan and `papers/`
(TABME DocEng'22; Wiedemann & Heyer 2018) for the design rationale.

## Model variants (`model.variant`)
- **B0** — per-page LayoutXLM (no context). Baseline / floor.
- **B1** — page-pair transition classifier. Baseline.
- **I1** — LayoutXLM page encoder → 1D temporal CNN (or Transformer) over pages →
  joint boundary + type heads. **Primary, shippable.**

## Install
Modernized stack: `torch==2.6.0`, `torchvision==0.21.0`, `transformers>=4.53,<4.54`
(ships `LayoutLMv2Model` + `LayoutXLMTokenizer`/`ImageProcessor`), `pytorch-lightning>=2.5.1`.
```bash
pip install -r requirements.txt          # includes sentencepiece + PyMuPDF
# LayoutLMv2/XLM visual backbone (detectron2) — build from source (needs a CUDA toolkit):
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
A CUDA GPU is required for training (the LayoutXLM visual backbone needs it). If the
detectron2 source build fails against torch 2.6, torch 2.4.x is the safest fallback
(`torch==2.4.1` / `torchvision==0.19.1`, still compatible with transformers 4.53).

## Data prep
The corpus is your **single, type-labeled documents**. Build the on-disk layout
under `datasets/pss/` (see `pss/data/__init__.py` for the exact JSON schemas):

1. **Render + OCR** each document's pages into `docs/<doc_id>.json` (+ images under
   `images/<doc_id>/`). Wire your OCR model into `pss/data/ocr_ingest.py::ingest_ocr`,
   then use the `make_word/make_page/make_doc/save_doc` builders. PDF rasterization
   is in `pss/data/render.py`.
2. **Synthesize streams** (TABME Algorithm 1 — concatenate docs into folders,
   Poisson lengths, document-level split):
   ```bash
   python -m pss.data.synthesize --root datasets/pss --lambda 5 \
       --n_train 100000 --n_val 5000 --n_test 5000 --seed 42
   ```
   This writes `folders/{train,val,test}/*.json`, `preprocessed_files_*.txt`, and
   `class_names.txt`. Set `model.n_types` to `len(class_names.txt)`.

## Train
```bash
python -m pss.train --config=configs/pss.yaml                      # I1 (primary)
python -m pss.train --config=configs/pss.yaml model.variant=B0     # baseline
python -m pss.train --config=configs/pss.yaml model.variant=B1     # baseline
python -m pss.train --config=configs/pss.yaml model.seq_head.type=transformer  # I1 ablation
```
Adam @ lr 5e-5, ≤30 epochs, early-stop patience 5 on `val_bd_f1` (TABME protocol).
Best checkpoint + TB logs under `{workspace}/`. Any config leaf is CLI-overridable.

## Evaluate (full stream stitching)
```bash
python -m pss.evaluate --config=configs/pss.yaml eval_mode=test \
    pretrained_model_file=pss_runs/i1_layoutxlm/checkpoints/last.ckpt
```
Reports breaking-point **P/R/F1 + kappa** (page level), **MNDD + STP** (stream
level), and **document-type macro-F1** (scored on true segments).

## Sanity checks
- Overfit ~50 synthetic folders (loss→0, bd F1→1) to prove wiring.
- Also evaluate on public **TABME**/**Tobacco800** to compare against published
  numbers (F1 ~0.95 / acc ~0.92) before trusting your own held-out set.

## Layout
```
pss/config.py                 OmegaConf config (defaults + --config + CLI)
pss/data/                     render, ocr_ingest, synthesize (TABME), dataset, collate
pss/model/page_encoder.py     shared LayoutXLM encoder (+ tokenizer/image-proc factories)
pss/model/sequence_heads.py   TemporalCNN / Transformer over pages; boundary + type heads
pss/model/variants.py         B0 / B1 / I1 assembly + loss
pss/lightning_module.py       training/validation loop + window-level metrics
pss/train.py, pss/evaluate.py entry points
pss/metrics.py                P/R/F1, kappa, MNDD, STP, type macro-F1
configs/pss.yaml              primary experiment config
```
