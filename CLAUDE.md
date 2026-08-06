# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repo does **Page Stream Segmentation (PSS)**: given a PDF that is a sequential
stream of pages from several concatenated documents, it (a) finds where each
document **starts/ends** (per-page breaking-point detection) and (b) **classifies
each document by type**. Input to the model is OCR output — (text, bounding box)
pairs — plus the rendered page image. Everything lives in the self-contained
[`pss/`](pss/) package; the model is built on a single **LayoutXLM** page encoder.

> Origin: this repository started as **BROS** (NAVER Corp.) and was repurposed. The
> BROS backbone (`bros/`), the FUNSD/SROIE key-information-extraction heads, and the
> SPADE/entity-linking code were removed. `LICENSE`/`NOTICE` retain the required
> Apache-2.0 attribution.

## Environment & commands

The stack is modernized (see [requirements.txt](requirements.txt)): `torch==2.6.0`,
`torchvision==0.21.0`, `transformers>=4.53,<4.54`, `pytorch-lightning>=2.5.1`. A CUDA
GPU is required for real training/eval — LayoutXLM's visual backbone is
**detectron2**, which has no prebuilt wheels for torch 2.6 and must be built from
source. (`pss/train.py` has a CPU fallback for smoke tests, but detectron2 is still
imported for the image path.)

```bash
pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/detectron2.git'   # visual backbone
```

Everything is driven by a config file + OmegaConf CLI overrides:

```bash
# Synthesize training streams from a corpus of single, type-labeled documents
python -m pss.data.synthesize --root datasets/pss --lambda 5 \
    --n_train 100000 --n_val 5000 --n_test 5000 --seed 42

# Train (I1 is the default/primary variant)
python -m pss.train --config=configs/pss.yaml
python -m pss.train --config=configs/pss.yaml model.variant=B0     # baseline
python -m pss.train --config=configs/pss.yaml model.seq_head.type=transformer  # I1 ablation

# Evaluate a checkpoint (full stream stitching)
python -m pss.evaluate --config=configs/pss.yaml eval_mode=test \
    pretrained_model_file=pss_runs/i1_layoutxlm/checkpoints/last.ckpt
```

Any config leaf is CLI-overridable with dot-notation, e.g. `train.batch_size=8
model.variant=B1 data.max_pages=48`. There is no unit-test suite; "evaluation" means
running `pss.evaluate` and reading the metrics it prints. Code style: `isort` + `black`.

## Data preparation

Build the corpus under `data.root` (default `datasets/pss/`). Schemas are documented
in [pss/data/__init__.py](pss/data/__init__.py). The on-disk layout:

```
datasets/pss/
  images/<doc_id>/page_000.png ...          rendered page images
  docs/<doc_id>.json                        single, type-labeled documents (OCR words + boxes)
  folders/<split>/<folder_id>.json          synthesized streams (page references, not copies)
  class_names.txt                           document types, one per line
  preprocessed_files_{train,val,test}.txt   index: "<folders/split/fid.json>\t<n_pages>" per line
```

- **Render + OCR** each source document to `docs/<doc_id>.json` (+ page images).
  [pss/data/render.py](pss/data/render.py) rasterizes PDFs (PyMuPDF).
  [pss/data/ocr_ingest.py](pss/data/ocr_ingest.py) is the adapter to plug in **your**
  OCR model — `ingest_ocr()` is a stub you must wire; then use the
  `make_word/make_page/make_doc/save_doc` builders.
- **Synthesize streams** with [pss/data/synthesize.py](pss/data/synthesize.py) (TABME
  Algorithm 1): concatenate whole docs into "folders" with **Poisson-distributed**
  lengths, sampled without repetition, **split at the document level** (90/5/5) to
  avoid leakage. The first page of each source doc gets `boundary=1`; every page
  inherits the doc's `type`. Set `model.n_types` to `len(class_names.txt)`.

`PSSDataset` reads `preprocessed_files_{mode}.txt` to find folders, then cuts each
stream into overlapping sliding windows.

## Architecture

The model is assembled from a **shared page encoder** + a **context model over pages**
+ **two heads**. The `model.variant` config value selects the assembly:

| variant | context over pages | meaning |
|---------|--------------------|---------|
| `B0` | none | per-page LayoutXLM → boundary/type from `e_i` alone (floor) |
| `B1` | page pair | boundary from `[e_{i-1}, e_i, |Δ|, e_{i-1}·e_i]`; type from `e_i` |
| `I1` | 1D temporal CNN (or Transformer) | encoder → context over the page sequence → joint boundary + type (**primary**) |

- **[pss/model/page_encoder.py](pss/model/page_encoder.py)** — the shared encoder.
  `PageEncoder` loads LayoutXLM via `LayoutLMv2Model.from_pretrained(...)` (there is
  **no** `LayoutXLMModel` class — LayoutXLM *is* the LayoutLMv2 architecture) and pools
  one embedding per page. `page_embed` ∈ {`cls`, `cls_mean`, `cls_mean_visual`} controls
  whether mean-pooled text and/or visual tokens are concatenated (dim = H / 2H / 3H).
  Also exports `build_tokenizer` (AutoTokenizer → SentencePiece for layoutxlm) and
  `build_image_processor` (`LayoutLMv2ImageProcessor`, `apply_ocr=False`).
- **[pss/model/sequence_heads.py](pss/model/sequence_heads.py)** — `TemporalCNN`
  (Conv1d over the page axis, `padding_mode="circular"`, ~5 layers, kernel 3, dropout
  0.2 — TABME's design), `TransformerOverPages` (ablation), and `BoundaryHead` /
  `TypeHead`.
- **[pss/model/variants.py](pss/model/variants.py)** — `PSSModel` + `build_model(cfg)`.
  Works on a unified batch dict `[B, P, ...]`; `encode()` folds `[B,P,...]→[B*P,...]`
  through the encoder. Loss = class-weighted boundary CrossEntropy (`ignore_index=-100`,
  minority up-weighted by `boundary_pos_weight`) + `type_loss_weight` · type CE.
- **[pss/lightning_module.py](pss/lightning_module.py)** — `PSSLightningModule`.
  `training_step`/`validation_step`; validation accumulates window logits in `self._val`
  and `on_validation_epoch_end` logs `val_bd_f1` (the monitored metric), P/R, kappa,
  type acc. Optimizer: Adam @ `train.lr` (TABME used 5e-5).
- **[pss/train.py](pss/train.py)** / **[pss/evaluate.py](pss/evaluate.py)** — entry
  points. Training early-stops on `val_bd_f1` (patience `train.early_stop_patience`).
  Eval stitches overlapping windows back per stream (logits averaged over overlaps)
  before scoring.
- **[pss/metrics.py](pss/metrics.py)** — boundary P/R/F1 + Cohen's kappa (page level),
  MNDD + STP (stream level), document-type macro-F1 (scored on true segments).

### Config resolution

`pss/config.py::get_config()` merges three OmegaConf layers: built-in
`default_config()` → the `--config` yaml → CLI dotlist overrides. `_validate` checks
the variant/backbone/seq-head enums; `_derive` sets `save_weight_dir` /
`tensorboard_dir` from `workspace` and **divides `train`/`val` batch size by the GPU
count** (config batch size is global). The primary experiment config is
[configs/pss.yaml](configs/pss.yaml).

### Conventions (important)

- **Bounding boxes**: LayoutXLM wants **4 ints** `[x0,y0,x1,y1]` scaled to `0..1000`.
  `PSSDataset._norm_boxes` scales pixel boxes; the tokenizer inserts the special-token
  boxes automatically (`[CLS]`→`[0,0,0,0]`, `[SEP]`→`[1000,1000,1000,1000]`).
- **Image**: `[3,224,224]`, produced by the image processor. LayoutLMv2/XLM appends 49
  visual tokens **after** the text tokens; the encoder slices `[:, :text_len]` before
  pooling text/CLS.
- **Labels**: `boundary_labels` ∈ {0,1} (1 = first page of a new document);
  `type_labels` = document-type id (or `-100` when unknown). Padded pages (from
  `collate_streams` equalizing the page count across a batch) get `page_mask=0` and
  label `-100`, which every loss/metric ignores.
- **Sliding windows**: long streams are cut into windows of `data.max_pages` with
  stride `data.window_stride` (overlap = `max_pages − window_stride`); eval re-stitches.
- **Precision**: Lightning 2.x uses precision **strings** (`"16-mixed"`, `"32-true"`),
  not the old int `16`/`32`.
