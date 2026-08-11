"""
Config loading for the decoder-based PSS track (``pss/decoder/``).

Deliberately a separate schema from ``pss/config.py`` (the encoder pipeline's):
this track fine-tunes a HF causal-LM via Unsloth+TRL instead of LayoutXLM, runs
in its own venv (see ``requirements-decoder.txt``), and its validation rules
don't overlap with the encoder's (``SUPPORTED_BACKBONES``, ``seq_head`` asserts,
etc.) — keeping the schemas apart means neither can accidentally break the
other while both evolve independently.

Same "defaults -> --config <file> -> CLI dotlist" merge as ``pss/config.py``:
    python -m pss.decoder.train --config=configs/pss_decoder.yaml decoder.modality=vision
"""

import os
import time

from omegaconf import OmegaConf

MODALITIES = ("text", "vision")
CONTEXT_MODES = ("pair", "window")  # pair = paper replication; window = joint, this repo's I1-style extension

# Base models known to load via Unsloth's Fast{Language,Vision}Model, one default
# per modality (see CLAUDE.md / pss/decoder/README.md for the rationale).
DEFAULT_TEXT_MODEL = "unsloth/Qwen3-8B-Instruct"
DEFAULT_VISION_MODEL = "unsloth/Qwen2.5-VL-7B-Instruct"


def default_config():
    """Built-in defaults. Values here are the source of truth for the schema;
    the yaml and CLI only override leaves that already exist."""
    return OmegaConf.create(
        {
            "seed": 42,
            "workspace": "./pss_runs/decoder_exp",
            "run_id": None,  # subdir name for this run's checkpoints; None => timestamp
            "data": {
                "root": "./datasets/pss",
                # Only consulted when decoder.context_mode=window; mirrors
                # pss/config.py's data.max_pages/window_stride so encoder and
                # decoder runs can be compared on identical windows.
                "max_pages": 32,
                "window_stride": 24,
            },
            "decoder": {
                "modality": "text",  # text | vision
                "context_mode": "window",  # pair | window
                "base_model": None,  # None => DEFAULT_{TEXT,VISION}_MODEL by modality
                "max_seq_length": 8192,
                "load_in_4bit": True,
                "lora": {"r": 16, "alpha": 16, "dropout": 0.0},  # paper's proven starting point
                # The assistant-turn opener produced by the base model's chat
                # template (e.g. tokenizer.apply_chat_template(..., add_generation_prompt=True)),
                # used by TRL's completion-only collator to find where the
                # completion starts. ChatML (Qwen family, both defaults) uses
                # this string — override if decoder.base_model isn't ChatML-based.
                "chat_response_template": "<|im_start|>assistant\n",
            },
            "train": {
                "batch_size": 16,
                "grad_accum": 1,
                "lr": 3e-5,  # paper
                "max_steps": 20000,  # paper: val metrics start degrading past this
                "warmup_steps": 200,  # paper
                "weight_decay": 0.01,  # paper
                "optimizer": "paged_adamw_8bit",  # paper
                "save_every": 1000,
                "eval_every": 1000,
                "num_workers": 4,
            },
            "infer": {
                "checkpoint": None,  # adapter dir or merged export dir
                "batch_size": 8,
                "max_new_tokens": 512,  # window mode emits one JSON object per page
                "stream": None,  # path to a live stream json ({"doc_id","pages":[...]})
                "image_root": None,  # page["image"] root; defaults to data.root
                "out": None,  # optional output path; defaults to stdout
                # production window width, only used by decoder.context_mode=window
                # (pair mode is inherently a fixed 2-page context). None => data.max_pages.
                "window_pages": None,
                "window_stride": None,  # None => data.window_stride
            },
            "export": {"out": None},  # defaults to {checkpoint}/merged
        }
    )


def _cli_dotlist(argv):
    """Extract OmegaConf dotlist + a --config path from an argv-style list."""
    dotlist = []
    config_file = None
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--config="):
            config_file = tok.split("=", 1)[1]
        elif tok == "--config":
            config_file = argv[i + 1]
            i += 1
        elif tok.startswith("--") and "=" in tok:
            dotlist.append(tok[2:])
        elif "=" in tok:
            dotlist.append(tok)
        i += 1
    return config_file, dotlist


def get_config(cli=None, default_conf_file=None):
    """Build the merged, validated config.

    cli: optional list of tokens (defaults to sys.argv[1:]).
    default_conf_file: optional yaml to merge before CLI (else read --config from cli).
    """
    if cli is None:
        import sys

        cli = sys.argv[1:]

    cfg = default_config()

    config_file, dotlist = _cli_dotlist(cli)
    config_file = default_conf_file or config_file
    if config_file:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_file))
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))

    _validate(cfg)
    _derive(cfg)
    return cfg


def _validate(cfg):
    assert (
        cfg.decoder.modality in MODALITIES
    ), f"decoder.modality must be one of {MODALITIES}"
    assert (
        cfg.decoder.context_mode in CONTEXT_MODES
    ), f"decoder.context_mode must be one of {CONTEXT_MODES}"
    assert cfg.decoder.lora.r >= 1
    assert 1 <= cfg.data.window_stride <= cfg.data.max_pages


def _derive(cfg):
    """Set derived paths/model choice (mirrors pss/config.py::_derive)."""
    if cfg.run_id is None:
        cfg.run_id = time.strftime("%Y%m%d-%H%M%S")
    cfg.save_weight_dir = os.path.join(cfg.workspace, "checkpoints", cfg.run_id)

    if cfg.decoder.base_model is None:
        cfg.decoder.base_model = (
            DEFAULT_VISION_MODEL if cfg.decoder.modality == "vision" else DEFAULT_TEXT_MODEL
        )

    if cfg.infer.window_pages is None:
        cfg.infer.window_pages = cfg.data.max_pages
    if cfg.infer.window_stride is None:
        cfg.infer.window_stride = cfg.data.window_stride
    if cfg.infer.window_pages != cfg.data.max_pages:
        print(
            f"[decoder.config] warning: infer.window_pages ({cfg.infer.window_pages}) "
            f"!= data.max_pages ({cfg.data.max_pages}) — decoder_cache/ examples were "
            "built at the latter width; production inference at a different window "
            "width wasn't seen during training. If this is an intentional "
            "deployment-hardware constraint, ignore this warning."
        )
