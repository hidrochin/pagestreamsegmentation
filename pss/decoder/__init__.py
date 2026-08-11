"""
Decoder-based PSS track: fine-tunes an open-weight decoder LLM (via Unsloth +
TRL LoRA) to do Page Stream Segmentation, as an experimental alternative next
to the primary LayoutXLM encoder pipeline in ``pss/``. See
``pss/decoder/README.md`` for setup and usage, and ``papers/2408.11981v1.pdf``
(Heidenreich et al., "Large Language Models for Page Stream Segmentation") for
the method this track is based on and extends.

Reads the exact same on-disk data as the encoder pipeline (``docs/<doc_id>.json``,
``folders/<split>/<folder_id>.json`` — see ``pss/data/__init__.py``); only the
"page -> model input" step differs (a text/JSON prompt instead of tokenizer +
image-processor tensors).

Runs in its own venv (``.venv-decoder``, see ``requirements-decoder.txt``) —
modules here must not import ``pss.model`` or ``pss.data.dataset``/``page_codec``
(the encoder stack, which pulls in detectron2/pytorch-lightning and isn't
installed in the decoder venv). Only ``pss.data.resolve``/``pss.data.windowing``
(pure stdlib) and ``pss.metrics`` (plain-array, no torch-model coupling) are
shared with the encoder pipeline.
"""
