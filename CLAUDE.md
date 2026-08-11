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
`torchvision==0.21.0`, `transformers>=4.53,<4.54`, `pytorch-lightning>=2.5.1`,
`matplotlib>=3.8` (for `pss.analyze`). Training/eval works on **CUDA, Apple Silicon
(MPS), or CPU** — `pss/train.py::_select_accelerator` and the device-selection lines
in `pss/evaluate.py`/`pss/infer.py` all prefer CUDA, then fall back to MPS, then CPU.
LayoutXLM's visual backbone is **detectron2**, which has no prebuilt wheels for
torch 2.6 and must be built from source on every platform (CUDA or Apple Silicon).

**Use `uv` for all environment/package management in this repo** (not raw
pip/venv). A persistent `.venv` lives at the repo root and is gitignored — reuse
it across sessions rather than recreating it, so the ~2.8GB detectron2 build and
LayoutXLM download aren't repeated:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
# --no-build-isolation is required so the detectron2 build sees the already-installed torch
CC=clang CXX=clang++ ARCHFLAGS="-arch arm64" \
    uv pip install --python .venv/bin/python --no-build-isolation \
    'git+https://github.com/facebookresearch/detectron2.git'   # Apple Silicon
# on CUDA machines: drop the CC/CXX/ARCHFLAGS env vars, keep --no-build-isolation
```

On Apple Silicon, the **I1+transformer** variant additionally needs
`PYTORCH_ENABLE_MPS_FALLBACK=1` set — one op used by `nn.TransformerEncoder`
(`_nested_tensor_from_mask_left_aligned`) isn't implemented for MPS yet and falls
back to CPU for just that op; everything else runs on MPS.

HF model weights (LayoutXLM, ~2.8GB) are cached under `pretrained_models/hf_cache/`
**inside the repo** — `pss/model/page_encoder.py` sets `HF_HOME` there by default
(unless the environment already sets one). This directory is gitignored and must
never be committed, but keeping it repo-local means it survives independently of
the user's home cache and isn't silently re-downloaded after cache cleanups.

Everything is driven by a config file + OmegaConf CLI overrides:

```bash
# Synthesize training streams from a corpus of single, type-labeled documents
python -m pss.data.synthesize --root datasets/pss --lambda 5 \
    --n_train 100000 --n_val 5000 --n_test 5000 --seed 42

# Train (I1 is the default/primary variant). Checkpoints + TB logs land under
# {workspace}/checkpoints/{run_id}/ and {workspace}/tensorboard_logs/{run_id}/ —
# run_id defaults to a timestamp so repeated runs never clobber each other's
# checkpoints. Two Lightning checkpoints are kept per run (best-by-val_bd_f1 and
# last), each auto-exported to a plain-PyTorch .pth sibling (see pss.export_pth
# below); both paths print when training finishes.
python -m pss.train --config=configs/pss.yaml
python -m pss.train --config=configs/pss.yaml model.variant=B0     # baseline
python -m pss.train --config=configs/pss.yaml model.seq_head.type=transformer  # I1 ablation

# Evaluate a checkpoint (full stream stitching)
python -m pss.evaluate --config=configs/pss.yaml eval_mode=test \
    pretrained_model_file=pss_runs/i1_layoutxlm/checkpoints/<run_id>/last.ckpt

# Visual model-behavior diagnostics: boundary + document-type confusion matrices,
# per-type P/R/F1 bars, written as PNGs under analyze.out_dir (default {workspace}/analysis)
python -m pss.analyze --config=configs/pss.yaml eval_mode=test \
    pretrained_model_file=pss_runs/i1_layoutxlm/checkpoints/<run_id>/last.ckpt

# Export a Lightning .ckpt to a plain-PyTorch .pth state dict (no lightning/optimizer
# state, no "net." key prefix) — also run automatically at the end of pss.train
python -m pss.export_pth --config=configs/pss.yaml \
    pretrained_model_file=pss_runs/i1_layoutxlm/checkpoints/<run_id>/last.ckpt

# Ingest a raw corpus that already has OCR sidecar files (see Data preparation)
python -m pss.data.ingest_raw --config configs/raw_sources.yaml --out_root datasets/pss

# Production inference: fixed-width sliding window, no ground truth
python -m pss.infer --config=configs/pss.yaml \
    pretrained_model_file=pss_runs/i1_layoutxlm/checkpoints/<run_id>/last.ckpt \
    infer.stream=stream.json infer.window_pages=10 infer.window_stride=6
```

Any config leaf is CLI-overridable with dot-notation, e.g. `train.batch_size=8
model.variant=B1 data.max_pages=48`. There is no unit-test suite; "evaluation" means
running `pss.evaluate` and reading the metrics it prints (`pss.analyze` complements
this with visual per-type/per-boundary-class diagnostics, useful for spotting which
document types the model handles poorly). Code style: `isort` + `black`.

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
  `make_word/make_page/make_doc/save_doc` builders. If OCR was already run and saved
  to per-image sidecar files instead, `ocr_ingest.parse_sidecar_txt(path)` reads the
  `x0,y0,x1,y1<TAB>text`-per-line format directly (no model call needed).
- **[pss/data/ingest_raw.py](pss/data/ingest_raw.py)** converts a raw
  `<raw_root>/<source>/{images/,ocr/}` corpus straight into `docs/`+`images/`: images
  are grouped into multi-page documents by filename prefix (`<doc>_<page>.jpg`), and
  each source path is labeled with a `type` from a small declarative
  [configs/raw_sources.yaml](configs/raw_sources.yaml) (`path` glob → `type`) — the
  raw folder name itself is never assumed to be the type. Re-running is
  **incremental**: `<out_root>/.ingest_manifest.json` tracks each doc's source file
  mtimes, so unchanged documents are skipped and only new/modified ones are
  (re)converted — safe to rerun after adding a new `sources:` entry or dropping more
  images into an existing folder.
- **Synthesize streams** with [pss/data/synthesize.py](pss/data/synthesize.py) (TABME
  Algorithm 1): concatenate whole docs into "folders" with **Poisson-distributed**
  lengths, sampled without repetition, **split at the document level** (90/5/5) to
  avoid leakage. The first page of each source doc gets `boundary=1`; every page
  inherits the doc's `type`. Set `model.n_types` to `len(class_names.txt)`.
  `--mix_strategy` controls how documents are chosen per folder: `stratified`
  (default) deliberately draws from multiple source types per folder so streams
  realistically mix document types instead of relying on chance; `random` is the
  original TABME-style uniform draw from the whole split pool.

`PSSDataset` reads `preprocessed_files_{mode}.txt` to find folders, then cuts each
stream into overlapping sliding windows via
[pss/data/windowing.py](pss/data/windowing.py)`::build_stream_windows` — the same
function `pss/infer.py` uses to window a live, unlabeled stream at inference time, so
training and production windowing can never drift apart. Page → tensor encoding
(tokenizer + image processor) is likewise shared, in
[pss/data/page_codec.py](pss/data/page_codec.py)`::encode_page`.

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
  (Conv1d over the page axis, zero-padded, ~5 layers, kernel 3, dropout 0.2 —
  TABME's design; `page_mask` is re-applied after every layer so a padded page's
  bias-only activations can never leak into a real page's receptive field),
  `TransformerOverPages` (ablation, masks padding via `src_key_padding_mask`), and
  `BoundaryHead` / `TypeHead`.
- **[pss/model/variants.py](pss/model/variants.py)** — `PSSModel` + `build_model(cfg)`.
  Works on a unified batch dict `[B, P, ...]`; `encode()` folds `[B,P,...]→[B*P,...]`
  through the encoder. Loss = class-weighted boundary CrossEntropy (`ignore_index=-100`,
  minority up-weighted by `boundary_pos_weight`) + `type_loss_weight` · type CE.
- **[pss/lightning_module.py](pss/lightning_module.py)** — `PSSLightningModule`.
  `training_step`/`validation_step`; validation accumulates window logits in `self._val`
  and `on_validation_epoch_end` logs `val_bd_f1` (the monitored metric), P/R, kappa,
  type acc. Optimizer: Adam @ `train.lr` (TABME used 5e-5).
- **[pss/train.py](pss/train.py)** / **[pss/evaluate.py](pss/evaluate.py)** — entry
  points. Training early-stops on `val_bd_f1` (patience `train.early_stop_patience`)
  and, once `trainer.fit` returns, auto-exports both the best and last checkpoint to
  `.pth` via `pss/export_pth.py`. Eval stitches overlapping windows back per stream
  before scoring; `evaluate.py::load_weights` raises on any checkpoint/model
  mismatch by default (`allow_partial_load=true` to load anyway) — important during
  architecture sweeps, where a silent partial load would produce misleading numbers
  instead of an error. `evaluate.py::collect_predictions` is the shared
  stitching/prediction routine reused by `pss/analyze.py`, so its diagnostics always
  match what `pss.evaluate` scores.
- **[pss/analyze.py](pss/analyze.py)** — visual diagnostics for a trained
  checkpoint: boundary confusion matrix, document-type confusion matrix, per-type
  P/R/F1 bars (PNGs under `analyze.out_dir`). Built on top of
  `evaluate.py::collect_predictions`, not a separate scoring path.
- **[pss/export_pth.py](pss/export_pth.py)** — `export_state_dict(net, ckpt_path,
  out_path=None)` converts a Lightning `.ckpt` to a plain-PyTorch `.pth` state dict
  (no optimizer/Lightning state, no `"net."` key prefix) — loadable with
  `net.load_state_dict(torch.load(path, weights_only=True))` without a
  pytorch-lightning dependency. Called automatically by `pss/train.py`; also runnable
  standalone against any past checkpoint.
- **[pss/stitch.py](pss/stitch.py)** — `StreamAccumulator`: per-absolute-page-index
  logit averaging over overlapping windows, shared by `pss/evaluate.py` (scores
  against ground truth) and `pss/infer.py` (production, no labels).
- **[pss/infer.py](pss/infer.py)** — production inference over a live, unlabeled page
  stream, fixed to a `infer.window_pages`-wide forward pass (deployment hardware
  constraint). Any window shorter than that width — a stream's tail, or a whole
  stream under one window — is padded with all-zero tensors + `page_mask=0`
  (`pad_window`), **not** by repeating the last real page: a repeated page would
  inject fabricated content into the model's context, whereas a masked pad slot is a
  path the model is already trained to ignore (see Conventions below).
- **[pss/metrics.py](pss/metrics.py)** — boundary P/R/F1 + Cohen's kappa (page level),
  MNDD + STP (stream level), document-type macro-F1 (scored on true segments).

### Config resolution

`pss/config.py::get_config()` merges three OmegaConf layers: built-in
`default_config()` → the `--config` yaml → CLI dotlist overrides. `_validate` checks
the variant/backbone/seq-head enums (including that `seq_head.n_heads` divides the
encoder output dim for the transformer seq-head) and warns if
`infer.window_pages != data.max_pages`; `_derive` sets `run_id` (a timestamp, unless
passed explicitly), `save_weight_dir` / `tensorboard_dir` from `workspace/…/run_id`,
and **divides `train`/`val` batch size by the GPU count** (config batch size is
global). The primary experiment config is [configs/pss.yaml](configs/pss.yaml).

### Conventions (important)

- **Bounding boxes**: LayoutXLM wants **4 ints** `[x0,y0,x1,y1]` scaled to `0..1000`.
  `pss/data/page_codec.py::norm_boxes` scales pixel boxes; the tokenizer inserts the
  special-token boxes automatically (`[CLS]`→`[0,0,0,0]`, `[SEP]`→`[1000,1000,1000,1000]`).
- **Image**: `[3,224,224]`, produced by the image processor. LayoutLMv2/XLM appends 49
  visual tokens **after** the text tokens; the encoder slices `[:, :text_len]` before
  pooling text/CLS.
- **Labels**: `boundary_labels` ∈ {0,1} (1 = first page of a new document);
  `type_labels` = document-type id (or `-100` when unknown).
- **Padding = `page_mask=0`, always** — never repeat a real page to fill a slot.
  `collate_streams` pads the page axis to a batch's max page count this way, and
  `pss/infer.py::pad_window` pads a short/tail inference window to a fixed
  `infer.window_pages` width the same way; padded pages get label `-100` and are
  ignored by every loss/metric (and are safe through LayoutLMv2/XLM — the 49 visual
  tokens stay attendable even under an all-zero text mask).
- **Sliding windows**: long streams are cut into windows of `data.max_pages` with
  stride `data.window_stride` (overlap = `max_pages − window_stride`) via
  `pss/data/windowing.py::build_stream_windows`; eval/infer re-stitch with
  `pss/stitch.py::StreamAccumulator`. For production inference, set
  `infer.window_pages` to the same value trained with (`data.max_pages`) — the model
  has only ever seen that context length.
- **Precision**: Lightning 2.x uses precision **strings** (`"16-mixed"`, `"32-true"`),
  not the old int `16`/`32`.
- **Accelerator selection**: `train.accelerator="gpu"` resolves CUDA → Apple MPS →
  CPU (`pss/train.py::_select_accelerator`; `pss/evaluate.py`/`pss/infer.py` mirror
  the same CUDA→MPS→CPU device-selection order). MPS forces full precision
  (`"32-true"`, single device, no DDP) since CUDA-style mixed precision doesn't
  apply; see the `PYTORCH_ENABLE_MPS_FALLBACK=1` note above for the I1+transformer
  variant on Apple Silicon.
- **Checkpoints**: two Lightning checkpoints kept per run under
  `{workspace}/checkpoints/{run_id}/` — `best-epoch??-f1?.????.ckpt` (highest
  `val_bd_f1`) and `last.ckpt` — each auto-exported to a plain-PyTorch `.pth`
  sibling (`pss/export_pth.py`) when training finishes.
