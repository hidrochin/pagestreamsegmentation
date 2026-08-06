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

```bash
pip install -r requirements.txt          # torch 2.6 / transformers 4.53 / lightning 2.5
# LayoutLMv2/XLM's visual backbone (build from source; needs a CUDA toolkit):
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Then follow **[pss/README.md](pss/README.md)** for the full workflow: prepare a
corpus of single, type-labeled documents, synthesize training streams
(`python -m pss.data.synthesize`), train (`python -m pss.train`), and evaluate
(`python -m pss.evaluate`). A CUDA GPU is required for training.

## Layout

```
pss/               self-contained PSS package (data, model, train/eval, metrics)
configs/pss.yaml   primary experiment config
papers/            reference papers (TABME; Wiedemann & Heyer)
requirements.txt   modernized dependency stack
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
