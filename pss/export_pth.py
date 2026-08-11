"""
Export a Lightning ``.ckpt`` to a plain-PyTorch ``.pth`` state dict: no
Lightning/optimizer state and no ``"net."`` key prefix, so it loads with plain
``net.load_state_dict(torch.load(path))`` on the bare ``pss.model.variants.PSSModel``
without depending on pytorch-lightning or ``PSSLightningModule``.

``pss/train.py`` calls ``export_state_dict()`` on its best/last checkpoints when a
training run finishes, so a ``.pth`` sibling is produced automatically. Run this
standalone to (re-)convert any past checkpoint:

    python -m pss.export_pth --config=configs/pss.yaml \
        pretrained_model_file=pss_runs/i1_layoutxlm/checkpoints/<run_id>/last.ckpt

Writes alongside the source checkpoint by default (same basename, .pth extension)
— override with ``export.out``.
"""

import os

import torch

from pss.config import get_config
from pss.evaluate import load_weights
from pss.lightning_module import PSSLightningModule


def export_state_dict(net, ckpt_path, out_path=None):
    """net: a built ``PSSModel`` (``model.variant``/``page_embed``/``seq_head``
    must match the checkpoint) whose weights are overwritten with ckpt_path's
    before saving. Returns the .pth path."""
    load_weights(net, ckpt_path, strict=True)
    out_path = out_path or os.path.splitext(ckpt_path)[0] + ".pth"
    torch.save(net.state_dict(), out_path)
    return out_path


def main():
    cfg = get_config()
    net = PSSLightningModule(cfg).net
    out_path = export_state_dict(net, cfg.pretrained_model_file, cfg.export.out)
    print(f"[export_pth] {cfg.pretrained_model_file} -> {out_path}")


if __name__ == "__main__":
    main()
