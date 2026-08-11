# PSS — Decoder-based LLM track (experimental, alongside the LayoutXLM pipeline)

Fine-tunes an open-weight decoder LLM (via [Unsloth](https://github.com/unslothai/unsloth)
+ TRL LoRA) to do Page Stream Segmentation, based on and extending
Heidenreich et al., *"Large Language Models for Page Stream Segmentation"*
(arXiv:2408.11981 — copy in `papers/2408.11981v1.pdf`), who show fine-tuned
decoder LLMs (Mistral-7B, Phi-3-mini) beating every encoder baseline they
tried by a wide margin on PSS. This track is an **experimental alternative**
next to the primary `pss/` (LayoutXLM B0/B1/I1) pipeline — that pipeline is
untouched and stays the default.

## Why a separate venv, config, and model family

- **Environment**: Unsloth is CUDA-only and pins its own torch/xformers/
  bitsandbytes versions, which may conflict with the encoder pipeline's
  `torch==2.6.0` + source-built detectron2. Everything here runs in its own
  `.venv-decoder`, installed from `requirements-decoder.txt` at the repo root
  — the encoder's `.venv` is never touched.
- **Config**: `pss/decoder/config.py` is a standalone OmegaConf schema (same
  "defaults → yaml → CLI dotlist" pattern as `pss/config.py`), not threaded
  through the encoder's `_validate`/`_derive` — an unrelated model family
  shouldn't be able to break the encoder config's tested assert logic.
- **Training loop**: Unsloth's speed comes from HF `Trainer`/TRL integration,
  not PyTorch Lightning, so this track uses TRL's `SFTTrainer` directly rather
  than `PSSLightningModule`.
- **Reused, unmodified**: the on-disk data (`docs/*.json`, `folders/*.json`,
  `class_names.txt`) and `pss/metrics.py` (P/R/F1, kappa, MNDD, STP, type
  macro-F1 — plain label lists, no model coupling) — both pipelines report
  literally the same metrics on the same data.

## Setup (offline-staged, mirrors the encoder pipeline's pattern in CLAUDE.md)

On a connected machine:
```bash
uv venv .venv-decoder --python 3.11
uv pip install --python .venv-decoder/bin/python -r requirements-decoder.txt
# If that pulls an incompatible torch/xformers combo for your CUDA version,
# prefer Unsloth's own version-matrix installer instead:
# https://docs.unsloth.ai/get-started/installation
```
Pre-download base model weights into the same repo-local, gitignored
`pretrained_models/hf_cache/` the encoder pipeline uses (`HF_HOME` is already
redirected there by `pss/model/page_encoder.py`; if you import only
`pss.decoder.*` — which never imports `pss.model` — set `HF_HOME` yourself
before running, e.g. `export HF_HOME=$(pwd)/pretrained_models/hf_cache`):
```bash
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen3-8B-Instruct')"
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct')"
```
rsync/scp `.venv-decoder`, the HF cache, and this repo to the offline L40 box.
Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` there so nothing tries a network
call.

## Model choice (see CLAUDE.md / the plan this was built from for the research)

The paper used Phi-3-mini and Mistral-7B (2024). Two newer, Apache-2.0,
Unsloth-supported defaults are used instead — set explicitly per track:
- **Text-only**: `unsloth/Qwen3-8B-Instruct` (`decoder.modality=text`)
- **Text+vision**: `unsloth/Qwen2.5-VL-7B-Instruct` (`decoder.modality=vision`)
  — a hypothesis, not a settled result: the paper's own encoder ablation found
  vision-only (DiT) the *strongest single modality*, and explicitly flags
  combining that signal with a decoder as unexplored future work.

Both fit comfortably on a single 48GB L40 with 4-bit QLoRA (the paper trained
7B models on one H100).

## Usage

```bash
# 1. Build the prompt/completion cache (also runs automatically on first
#    train/evaluate call; run standalone to inspect it first)
python -m pss.decoder.dataset --config=configs/pss_decoder.yaml mode=train

# 2. Fine-tune (text modality only until vision's data collator is wired up
#    — see pss/decoder/train.py's NotImplementedError note)
python -m pss.decoder.train --config=configs/pss_decoder.yaml

# 3. Evaluate — same metrics/units as `python -m pss.evaluate`
python -m pss.decoder.evaluate --config=configs/pss_decoder.yaml \
    infer.checkpoint=pss_runs/decoder_i1_qwen3/checkpoints/<run_id>/last eval_mode=test

# 4. Merge the LoRA adapter for production serving (no PEFT/Unsloth needed to load the result)
python -m pss.decoder.export --config=configs/pss_decoder.yaml \
    infer.checkpoint=pss_runs/decoder_i1_qwen3/checkpoints/<run_id>/last

# 5. Production inference over a live, unlabeled stream (same JSON schema pss.infer reads)
python -m pss.decoder.infer --config=configs/pss_decoder.yaml \
    infer.checkpoint=pss_runs/decoder_i1_qwen3/checkpoints/<run_id>/last \
    infer.stream=stream.json
```

`decoder.context_mode`:
- `pair` — reproduces the paper's formulation exactly (prev+current page text
  → one boundary decision), extended to also predict `type` when a boundary
  is predicted. Use this first as a sanity check against the paper's own
  published numbers (see Verification below).
- `window` — one prompt holds up to `data.max_pages` pages and asks for a
  JSON array of per-page `{boundary, type}` in one shot, matching this repo's
  I1 joint-context encoder variant. The paper explicitly left "give the
  decoder more than a page pair" as unexplored future work — this is that
  experiment.

## Verification

**Do this before trusting any result on your own corpus.** Fine-tune
`decoder.modality=text decoder.context_mode=pair` against the paper's own
released benchmark, **TABME++**
(`huggingface.co/datasets/rootsautomation/TABMEpp` — download on a connected
machine and stage over like the model weights), and confirm you land near
their published numbers (Mistral-7B-FT: page F1 0.987, doc F1 0.967, STP 0.80;
Phi-3-mini-FT: page F1 0.973, doc F1 0.933, STP 0.637 — Table 4 of the paper).
This validates the whole pipeline (prompt construction, layout-preserving text
serialization, JSON parsing, vote-stitching, metrics) against a known-good
target — this repo has no unit-test suite (per CLAUDE.md), so this sanity
check *is* the acceptance gate, same convention as the encoder pipeline's
`pss.evaluate`.

Only after that passes, run the comparison this track was actually built for:
`{text, vision} × {pair, window}` on your own synthesized streams, evaluated
with identical metrics against the current best encoder checkpoint (B0/B1/I1)
— let the numbers pick the production candidate.

## Layout

```
pss/decoder/config.py      standalone OmegaConf config (own defaults + --config + CLI)
pss/decoder/prompt.py      layout-preserving text serialization; pair/window prompt
                            templates; JSON output parsing (with the paper's malformed-
                            JSON fallback)
pss/decoder/dataset.py     docs/+folders/ -> {prompt, completion, meta} JSONL cache
pss/decoder/model.py       Unsloth FastLanguageModel/FastVisionModel + LoRA loading
pss/decoder/train.py       TRL SFTTrainer entry point (completion-only loss)
pss/decoder/generate.py    shared batched-generation helper (evaluate.py + infer.py)
pss/decoder/stitch.py      majority-vote stitching across overlapping windows
                            (discrete-label analogue of pss/stitch.py's logit averaging)
pss/decoder/evaluate.py    stitched-stream eval, scored with pss/metrics.py (unmodified)
pss/decoder/infer.py       production inference over a live, unlabeled stream
pss/decoder/export.py      LoRA adapter -> merged HF model dir (safetensors)
configs/pss_decoder.yaml   primary decoder-track config
requirements-decoder.txt   .venv-decoder dependencies
```
