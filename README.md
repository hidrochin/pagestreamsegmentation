# PSS — Page Stream Segmentation on LayoutXLM

Given a PDF that is a **stream of pages from several concatenated documents**, this
project (a) finds where each document **starts and ends** (breaking-point
detection) and (b) **classifies each resulting document by type**. It is built on a
single **LayoutXLM** page encoder (text + layout + image) with a temporal model
over the page sequence.

> This repository began as [BROS](https://github.com/clovaai/bros) (BERT Relying On
> Spatiality, NAVER Corp., AAAI 2022) and has been repurposed: the FUNSD/SROIE
> key-information-extraction code was removed and replaced by the self-contained
> `pss/` package. See **Acknowledgements** and `NOTICE`/`LICENSE` for attribution.

## Approach

Every page is encoded once by LayoutXLM (which shares the LayoutLMv2 architecture,
so it fuses text, layout, and the page image in one encoder). The per-page
embeddings are then modelled across the stream to predict, for each page, whether
it begins a new document, plus its document type. Three variants form a ladder:

- **B0** — per-page, no context (baseline / floor).
- **B1** — page-pair transition classifier (baseline).
- **I1** — page encoder → **1D temporal CNN** over pages (TABME-style) → joint
  boundary + type heads. **Primary, shippable.**

The design and hyper-parameters follow two papers included under `papers/`:
**TABME** ("Tab this Folder", DocEng '22) and **Wiedemann & Heyer** (2018). A key
finding — vision is the dominant modality for PSS — is what makes LayoutXLM's
integrated visual backbone worth its cost.

## Quickstart

Use `uv` for environment/package management (see [CLAUDE.md](CLAUDE.md) for the
full uv commands, including the Apple Silicon detectron2 build flags):

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt   # torch 2.6 / transformers 4.53 / lightning 2.5
# LayoutLMv2/XLM's visual backbone (build from source):
uv pip install --python .venv/bin/python --no-build-isolation \
    'git+https://github.com/facebookresearch/detectron2.git'
```

Then follow **[pss/README.md](pss/README.md)** for the full workflow: prepare a
corpus of single, type-labeled documents, synthesize training streams
(`python -m pss.data.synthesize`), train (`python -m pss.train`), evaluate
(`python -m pss.evaluate`), and inspect confusion matrices/per-type metrics
(`python -m pss.analyze`). Training/eval runs on CUDA, Apple Silicon (MPS), or CPU.

### Decoder-based track (experimental)

[pss/decoder/](pss/decoder/) fine-tunes an open-weight decoder LLM (Unsloth +
TRL LoRA) to do PSS instead of the LayoutXLM encoder above, based on
Heidenreich et al. 2024 (`papers/2408.11981v1.pdf`), who found fine-tuned
decoder LLMs beat every encoder baseline on PSS. It's an experimental
alternative — separate venv, config, and training loop, same underlying data
and scoring — see [pss/decoder/README.md](pss/decoder/README.md). Requires a
CUDA GPU (Unsloth has no CPU/MPS backend).

## Layout

```
pss/               self-contained PSS package (data, model, train/eval, metrics)
pss/decoder/        experimental decoder-LLM PSS track (see pss/decoder/README.md)
configs/pss.yaml   primary experiment config
configs/pss_decoder.yaml   primary decoder-track config
papers/            reference papers (TABME; Wiedemann & Heyer; Heidenreich et al. decoder-LLM PSS)
requirements.txt   modernized dependency stack (encoder pipeline)
requirements-decoder.txt   decoder-track dependency stack (separate venv)
```

## Acknowledgements

- Derived from **BROS** (NAVER Corp.) — see the original
  [paper](https://arxiv.org/abs/2108.04539) and repository.
- The page encoder is **LayoutXLM / LayoutLMv2**
  ([docs](https://huggingface.co/docs/transformers/model_doc/layoutxlm)).
- Segmentation design follows **TABME** (github.com/aldolipani/TABME) and
  **Wiedemann & Heyer** (arXiv 1710.03006).

## License

```
Copyright 2022-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
