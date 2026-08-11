"""
Train a PSS model.

    python -m pss.train --config=configs/pss.yaml                 # I1 (default)
    python -m pss.train --config=configs/pss.yaml model.variant=B0
    python -m pss.train --config=configs/pss.yaml model.seq_head.type=transformer

Checkpoints + TB logs go under {workspace}/checkpoints/{run_id}/ and
{workspace}/tensorboard_logs/{run_id}/, where `run_id` defaults to a timestamp
taken at the start of this run (pss/config.py::_derive) so repeated training runs
against the same workspace never clobber each other's checkpoints. Two Lightning
checkpoints are kept per run: `best-epoch??-f1?.????.ckpt` (highest val_bd_f1) and
`last.ckpt` (final epoch) — both paths are printed when training finishes, and a
plain-PyTorch `.pth` state dict (see pss/export_pth.py) is exported alongside each
one automatically. Best checkpoint is selected on val breaking-point F1; training
stops early after `early_stop_patience` epochs without improvement (TABME
protocol).
"""

import os

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from pss.config import get_config
from pss.data import PSSDataset, collate_streams
from pss.export_pth import export_state_dict
from pss.lightning_module import PSSLightningModule
from pss.model.page_encoder import build_image_processor, build_tokenizer


def _select_accelerator(cfg):
    """Resolve accelerator/devices/strategy/precision from cfg.train.accelerator
    and what's actually available. When accelerator="gpu": prefer CUDA (using the
    configured precision + DDP across GPUs), else fall back to Apple's MPS backend
    (single device — no multi-GPU strategy, and no CUDA-style mixed precision, so
    precision is forced to full), else fall back to CPU."""
    if cfg.train.accelerator != "gpu":
        return "cpu", 1, "auto", "32-true"
    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        return "gpu", n_gpus, ("ddp" if n_gpus > 1 else "auto"), cfg.train.precision
    if torch.backends.mps.is_available():
        return "mps", 1, "auto", "32-true"
    return "cpu", 1, "auto", "32-true"


def main():
    cfg = get_config()
    print(cfg)
    print(f"[train] run_id={cfg.run_id}  checkpoints -> {cfg.save_weight_dir}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed_everything(cfg.seed)

    module = PSSLightningModule(cfg)

    # one tokenizer / image processor shared by both datasets
    tok = build_tokenizer(cfg.model.backbone)
    ip = build_image_processor(cfg.model.backbone)
    train_ds = PSSDataset(cfg, "train", tok, ip)
    val_ds = PSSDataset(cfg, "val", tok, ip)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_streams,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.val.batch_size,
        shuffle=False,
        num_workers=cfg.val.num_workers,
        collate_fn=collate_streams,
        pin_memory=True,
    )

    ckpt = ModelCheckpoint(
        dirpath=cfg.save_weight_dir,
        monitor="val_bd_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="best-epoch{epoch:02d}-f1{val_bd_f1:.4f}",
        auto_insert_metric_name=False,
    )
    early = EarlyStopping(
        monitor="val_bd_f1", mode="max", patience=cfg.train.early_stop_patience
    )

    accelerator, devices, strategy, precision = _select_accelerator(cfg)
    logger = TensorBoardLogger(save_dir=cfg.tensorboard_dir, name="", version="")
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=cfg.train.max_epochs,
        gradient_clip_val=cfg.train.clip_grad_norm,
        precision=precision,
        callbacks=[ckpt, early],
        check_val_every_n_epoch=cfg.train.val_interval,
        default_root_dir=cfg.workspace,
        logger=logger,
    )
    trainer.fit(module, train_loader, val_loader)

    print(f"[train] best checkpoint (val_bd_f1={ckpt.best_model_score}): {ckpt.best_model_path}")
    print(f"[train] last checkpoint: {ckpt.last_model_path}")

    # plain-PyTorch .pth siblings (no Lightning/optimizer state) for each — see
    # pss/export_pth.py. Reloads each checkpoint's weights into `net` before saving
    # so the exported state dict always matches that specific checkpoint file.
    for path in {ckpt.best_model_path, ckpt.last_model_path}:
        if path:
            pth_path = export_state_dict(module.net, path)
            print(f"[train] exported {path} -> {pth_path}")


if __name__ == "__main__":
    main()
