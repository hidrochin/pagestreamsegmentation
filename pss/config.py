"""
Config loading for PSS.

Three-layer merge (OmegaConf): built-in defaults -> --config <file> -> CLI dotlist
overrides. Mirrors the resolution style of the original repo's ``utils.get_config``
but self-contained and PSS-specific.

Usage:
    from pss.config import get_config
    cfg = get_config()                       # reads --config / CLI from sys.argv
    cfg = get_config(cli=["model.variant=B0"])

Any leaf can be overridden on the CLI with dot-notation, e.g.:
    python -m pss.train --config=configs/pss.yaml model.variant=I1 train.lr=3e-5
"""

import os
import time

from omegaconf import OmegaConf

# Backbones we support. LayoutXLM shares the LayoutLMv2 architecture (loaded via
# LayoutLMv2Model); "microsoft/layoutlmv2-base-uncased" also works for English-only.
SUPPORTED_BACKBONES = (
    "microsoft/layoutxlm-base",
    "microsoft/layoutlmv2-base-uncased",
)
VARIANTS = ("B0", "B1", "I1")
SEQ_HEAD_TYPES = ("cnn", "transformer")
PAGE_EMBED_MODES = ("cls", "cls_mean", "cls_mean_visual")
TYPE_FROM_MODES = ("context", "page", "page_plus_context")
BOUNDARY_FROM_MODES = ("context", "pairwise", "pairwise_context")


def default_config():
    """Built-in defaults. Values here are the source of truth for the schema;
    the yaml and CLI only override leaves that already exist."""
    return OmegaConf.create(
        {
            "seed": 42,
            "workspace": "./pss_runs/exp",  # checkpoints + tb logs derived from this
            "run_id": None,  # subdir name for this run's checkpoints/logs; None => timestamp
            "model": {
                "backbone": "microsoft/layoutxlm-base",
                "variant": "I1",  # B0 | B1 | I1
                "max_seq_length": 512,  # text tokens per page
                "image_size": 224,  # LayoutLMv2/XLM visual input is 224x224
                "page_embed": "cls",  # cls | cls_mean | cls_mean_visual
                "n_types": 16,  # #document types; must match the dataset
                "type_loss_weight": 1.0,  # lambda on the type loss
                "type_from": "context",  # I1 type head input: context | page | page_plus_context
                "boundary_from": "context",  # I1 boundary head input: context | pairwise | pairwise_context
                "type_class_weights": None,  # per-type CE weights: null | "auto" | [w0,...]
                "type_focal_gamma": 0.0,  # >0 => focal type loss (down-weights majority)
                "boundary_pos_weight": 2.0,  # up-weight the minority breaking-point class
                "freeze_encoder": False,  # freeze LayoutXLM (fast B0) then unfreeze
                "seq_head": {  # used by I1 (context over the page sequence)
                    "type": "cnn",  # cnn (TABME, primary) | transformer (ablation)
                    "n_layers": 5,  # TABME grid-searched 3-6; 5 was best
                    "kernel_size": 3,
                    "dropout": 0.2,
                    "hidden_size": 0,  # 0 => taper channels from encoder hidden dim
                    "residual": False,  # skip-connection + per-page LayerNorm per conv (cnn only)
                    "n_heads": 8,  # transformer head only
                },
            },
            "data": {
                "root": "./datasets/pss",
                "max_pages": 32,  # sliding-window length over a stream
                "window_stride": 24,  # overlap = max_pages - window_stride
                "image_dir": None,  # optional override for rendered page images
            },
            "train": {
                "batch_size": 4,  # streams per batch (global; divided by #gpus)
                "num_workers": 4,
                "lr": 5e-5,  # TABME
                "max_epochs": 30,  # TABME
                "early_stop_patience": 5,  # TABME
                "weight_decay": 0.0,
                "clip_grad_norm": 1.0,
                "accelerator": "gpu",
                "precision": "16-mixed",  # PL 2.x precision string (was int 16)
                "val_interval": 1,
            },
            "val": {"batch_size": 4, "num_workers": 4},
            "infer": {  # pss.infer — fixed-window production inference (no labels)
                "window_pages": 10,  # forward-pass width; match to deployment hardware
                "window_stride": 6,  # overlap = window_pages - window_stride
                "stream": None,  # path to a stream json ({"doc_id","pages":[...]})
                "image_root": None,  # page["image"] root; defaults to data.root
                "out": None,  # optional output path; defaults to stdout
            },
            "analyze": {  # pss.analyze — confusion-matrix/heatmap diagnostics
                "out_dir": None,  # defaults to {workspace}/analysis
            },
            "probe": {  # pss.probe — model-behavior probes (adjacent-emb sim, layer drift)
                "out_dir": None,  # defaults to {workspace}/probe
                "max_streams": 8,  # #per-stream trace figures to emit
            },
            "export": {  # pss.export_pth — .ckpt -> plain torch .pth state dict
                "out": None,  # defaults to pretrained_model_file with .pth extension
            },
        }
    )


def _cli_dotlist(argv):
    """Extract OmegaConf dotlist + a --config path from an argv-style list.

    Accepts both ``--config=foo.yaml`` and ``--config foo.yaml``, and passes
    through ``a.b.c=value`` pairs to OmegaConf.
    """
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
            dotlist.append(tok[2:])  # allow --a.b=c
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
    assert cfg.model.variant in VARIANTS, f"variant must be one of {VARIANTS}"
    assert (
        cfg.model.backbone in SUPPORTED_BACKBONES
    ), f"backbone must be one of {SUPPORTED_BACKBONES}"
    assert cfg.model.seq_head.type in SEQ_HEAD_TYPES
    assert cfg.model.page_embed in PAGE_EMBED_MODES
    assert (
        cfg.model.type_from in TYPE_FROM_MODES
    ), f"model.type_from must be one of {TYPE_FROM_MODES}"
    assert (
        cfg.model.get("boundary_from", "context") in BOUNDARY_FROM_MODES
    ), f"model.boundary_from must be one of {BOUNDARY_FROM_MODES}"
    if cfg.model.type_from == "page_plus_context" and cfg.model.seq_head.hidden_size:
        raise AssertionError(
            "model.type_from=page_plus_context requires model.seq_head.hidden_size=0 "
            "so the context output dim matches the encoder dim"
        )
    tw = cfg.model.type_class_weights
    if tw not in (None, "auto"):
        assert (
            len(tw) == cfg.model.n_types
        ), f"model.type_class_weights must have n_types={cfg.model.n_types} entries"
    assert cfg.model.type_focal_gamma >= 0
    assert cfg.data.max_pages >= 1
    assert 1 <= cfg.data.window_stride <= cfg.data.max_pages
    assert cfg.infer.window_pages >= 1
    assert 1 <= cfg.infer.window_stride <= cfg.infer.window_pages

    if cfg.model.seq_head.type == "transformer":
        encoder_dim = _encoder_output_dim(cfg)
        assert encoder_dim % cfg.model.seq_head.n_heads == 0, (
            f"model.seq_head.n_heads ({cfg.model.seq_head.n_heads}) must divide "
            f"the encoder output dim ({encoder_dim}, from model.page_embed="
            f"{cfg.model.page_embed}) evenly for TransformerOverPages"
        )

    if cfg.infer.window_pages != cfg.data.max_pages:
        print(
            f"[config] warning: infer.window_pages ({cfg.infer.window_pages}) != "
            f"data.max_pages ({cfg.data.max_pages}) — the model was trained with "
            f"{cfg.data.max_pages}-page context; production inference at a "
            "different window width may see degraded accuracy. If this is an "
            "intentional deployment-hardware constraint, ignore this warning."
        )


_BACKBONE_HIDDEN_SIZE = (
    768  # LayoutLMv2/XLM-base hidden size (both SUPPORTED_BACKBONES)
)


def _encoder_output_dim(cfg):
    mult = {"cls": 1, "cls_mean": 2, "cls_mean_visual": 3}[cfg.model.page_embed]
    return mult * _BACKBONE_HIDDEN_SIZE


def _derive(cfg):
    """Set derived paths and split the global batch size across GPUs (if any).

    Checkpoints/logs are nested under ``run_id`` (a timestamp by default, taken
    once when training starts) so successive training runs against the same
    ``workspace`` never overwrite each other's checkpoints — pass ``run_id=...``
    explicitly to reuse/inspect a specific past run's directory."""
    if cfg.run_id is None:
        cfg.run_id = time.strftime("%Y%m%d-%H%M%S")
    cfg.save_weight_dir = os.path.join(cfg.workspace, "checkpoints", cfg.run_id)
    cfg.tensorboard_dir = os.path.join(cfg.workspace, "tensorboard_logs", cfg.run_id)

    n_gpus = _gpu_count() if cfg.train.accelerator == "gpu" else 0
    if n_gpus > 1:
        for mode in ("train", "val"):
            cfg[mode].batch_size = max(1, cfg[mode].batch_size // n_gpus)


def _gpu_count():
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0
