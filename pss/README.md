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
Use **uv** for environment/package management, not raw pip/venv:
```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt   # includes sentencepiece + PyMuPDF
# LayoutLMv2/XLM visual backbone (detectron2) — build from source, needs --no-build-isolation
# so it can see the already-installed torch:
CC=clang CXX=clang++ ARCHFLAGS="-arch arm64" \
    uv pip install --python .venv/bin/python --no-build-isolation \
    'git+https://github.com/facebookresearch/detectron2.git'   # Apple Silicon
# on CUDA machines: drop CC/CXX/ARCHFLAGS, keep --no-build-isolation
```
Training/eval runs on **CUDA, Apple Silicon (MPS), or CPU** (`pss/train.py`
auto-detects; `PYTORCH_ENABLE_MPS_FALLBACK=1` is needed for the I1+transformer
variant on Apple Silicon — one op falls back to CPU). If the detectron2 source
build fails against torch 2.6, torch 2.4.x is the safest fallback (`torch==2.4.1` /
`torchvision==0.19.1`, still compatible with transformers 4.53).

HF model weights are cached under `pretrained_models/hf_cache/` inside the repo
(gitignored — never commit them) instead of the user's home cache, so they persist
across environment rebuilds.

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
Checkpoints + TB logs land under `{workspace}/checkpoints/{run_id}/` and
`{workspace}/tensorboard_logs/{run_id}/` — `run_id` defaults to a timestamp so
repeated runs never overwrite each other. Two checkpoints are kept per run: best
(by `val_bd_f1`) and last, each auto-exported to a plain-PyTorch `.pth` sibling
(no Lightning dependency needed to load it — see **Export** below). Any config
leaf is CLI-overridable.

## Evaluate (full stream stitching)
```bash
python -m pss.evaluate --config=configs/pss.yaml eval_mode=test \
    pretrained_model_file=pss_runs/i1_layoutxlm/checkpoints/<run_id>/last.ckpt
```
Reports breaking-point **P/R/F1 + kappa** (page level), **MNDD + STP** (stream
level), and **document-type macro-F1** (scored on true segments). Checkpoint
loading is strict by default — a `model.variant`/`page_embed`/`seq_head.type`
mismatch raises instead of silently partial-loading (`allow_partial_load=true` to
override).

## Analyze (confusion matrices, per-type P/R/F1)
```bash
python -m pss.analyze --config=configs/pss.yaml eval_mode=test \
    pretrained_model_file=pss_runs/i1_layoutxlm/checkpoints/<run_id>/last.ckpt
```
Writes `boundary_confusion.png`, `type_confusion.png`, `type_prf.png` under
`analyze.out_dir` (default `{workspace}/analysis`) — which document types the
model confuses, and whether boundary errors skew toward false positives or false
negatives. Scores the exact same predictions as `pss.evaluate`.

## Export (.ckpt -> .pth)
```bash
python -m pss.export_pth --config=configs/pss.yaml \
    pretrained_model_file=pss_runs/i1_layoutxlm/checkpoints/<run_id>/last.ckpt
```
Also run automatically at the end of `pss.train`. Strips Lightning/optimizer state
and the `"net."` key prefix, so the result loads with plain
`net.load_state_dict(torch.load(path, weights_only=True))`.

## Sanity checks
- Overfit ~50 synthetic folders (loss→0, bd F1→1) to prove wiring.
- Also evaluate on public **TABME**/**Tobacco800** to compare against published
  numbers (F1 ~0.95 / acc ~0.92) before trusting your own held-out set.

## Decoder-based track (experimental)

[pss/decoder/](decoder/README.md) fine-tunes an open-weight decoder LLM
(Unsloth + TRL LoRA — `Qwen3-8B-Instruct` text-only, or `Qwen2.5-VL-7B-Instruct`
text+vision) to do PSS, based on and extending Heidenreich et al.
(arXiv:2408.11981, `papers/2408.11981v1.pdf`), who found fine-tuned decoder
LLMs beat every encoder baseline on PSS by a wide margin. It reads the exact
same `docs/*.json`/`folders/*.json` and scores with the exact same
`pss/metrics.py` as the pipeline above, but is an **experimental alternative**
— separate venv (`.venv-decoder`, `requirements-decoder.txt`, Unsloth is
CUDA-only), separate config (`pss/decoder/config.py`), separate training loop
(TRL `SFTTrainer`). Full setup/usage/verification in
[pss/decoder/README.md](decoder/README.md).

## Layout
```
pss/config.py                 OmegaConf config (defaults + --config + CLI)
pss/data/                     render, ocr_ingest, synthesize (TABME), dataset, collate, resolve
pss/model/page_encoder.py     shared LayoutXLM encoder (+ tokenizer/image-proc factories)
pss/model/sequence_heads.py   TemporalCNN / Transformer over pages; boundary + type heads
pss/model/variants.py         B0 / B1 / I1 assembly + loss
pss/lightning_module.py       training/validation loop + window-level metrics
pss/train.py, pss/evaluate.py entry points (train / stitched-stream eval)
pss/analyze.py                confusion-matrix/heatmap diagnostics (built on evaluate.py)
pss/export_pth.py             .ckpt -> plain-PyTorch .pth state dict
pss/metrics.py                P/R/F1, kappa, MNDD, STP, type macro-F1
pss/decoder/                  experimental decoder-LLM PSS track (see pss/decoder/README.md)
configs/pss.yaml              primary experiment config
configs/pss_decoder.yaml      primary decoder-track config
```
