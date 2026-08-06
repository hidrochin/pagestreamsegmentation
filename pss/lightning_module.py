"""
PyTorch Lightning wrapper for PSS.

Training optimizes boundary + type loss (defined in the model). Validation logs
window-level boundary P/R/F1, kappa, and type accuracy as the monitor
(``val_bd_f1``). Full stream-level metrics (MNDD, STP, doc-type macro-F1) are
computed in evaluate.py, which stitches the overlapping windows back into streams.

Optimizer: Adam, lr from config (TABME used Adam @ 5e-5, early stopping).
"""

import torch
from pytorch_lightning import LightningModule
from torch.optim import Adam

from pss import metrics
from pss.model import build_model


class PSSLightningModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.net = build_model(cfg)
        self._val = []

    def forward(self, batch):
        return self.net(batch)

    def training_step(self, batch, batch_idx):
        out = self.net(batch)
        bs = batch["input_ids"].size(0)
        self.log_dict(
            {
                "train_loss": out["loss"],
                "train_bd_loss": out["boundary_loss"],
                "train_ty_loss": out["type_loss"],
            },
            sync_dist=True,
            batch_size=bs,
        )
        return out["loss"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        out = self.net(batch)
        self._val.append(
            {
                "bl": out["boundary_logits"].detach().float().cpu(),
                "by": batch["boundary_labels"].cpu(),
                "tl": out["type_logits"].detach().float().cpu(),
                "ty": batch["type_labels"].cpu(),
            }
        )
        return out["loss"]

    @torch.no_grad()
    def on_validation_epoch_end(self):
        n_types = self.cfg.model.n_types
        bl = torch.cat([v["bl"].reshape(-1, 2) for v in self._val])
        by = torch.cat([v["by"].reshape(-1) for v in self._val])
        tl = torch.cat([v["tl"].reshape(-1, n_types) for v in self._val])
        ty = torch.cat([v["ty"].reshape(-1) for v in self._val])

        prf = metrics.boundary_prf(bl, by)
        kappa = metrics.cohen_kappa(bl, by)
        tacc = metrics.type_accuracy(tl, ty)
        self.log_dict(
            {
                "val_bd_f1": prf["f1"],
                "val_bd_prec": prf["precision"],
                "val_bd_rec": prf["recall"],
                "val_kappa": kappa,
                "val_type_acc": tacc,
            },
            prog_bar=True,
            sync_dist=True,
        )
        self.print(
            f"[val] bd_f1={prf['f1']:.4f} P={prf['precision']:.4f} "
            f"R={prf['recall']:.4f} kappa={kappa:.4f} type_acc={tacc:.4f}"
        )
        self._val = []

    def configure_optimizers(self):
        params = [p for p in self.net.parameters() if p.requires_grad]
        return Adam(
            params, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay
        )
