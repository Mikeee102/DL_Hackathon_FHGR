"""Trainings-Loop (config-getrieben).

Features:
- Mixed Precision auf GPU (CPU-Fallback funktioniert).
- Discriminative LR + zweistufiges Auftauen für Transfer-Modelle.
- Imbalance-Handling per Loss-Wahl oder WeightedRandomSampler (Config).
- Early Stopping auf Val-PR-AUC, bestes Checkpoint wird gespeichert.
- Plottet Trainings-/Val-Kurven nach reports/<run>/.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import CrosswalkDataset
from src.losses import build_loss
from src.metrics import compute_metrics
from src.models.transfer import build_model
from src.utils import Checkpoint, get_logger, load_config, pick_device, save_checkpoint, save_config, set_seed

LOG = get_logger("train")


def _make_loader(cfg, csv_path: Path, train: bool) -> DataLoader:
    ds = CrosswalkDataset(
        csv_path=csv_path,
        image_root=cfg.data.get("image_root"),
        image_size=cfg.data.get("image_size", 224),
        train=train,
    )
    sampler = None
    shuffle = train
    if train and cfg.training.get("sampler") == "weighted":
        labels = ds.labels.numpy().astype(int)
        n_pos = max(int((labels == 1).sum()), 1)
        n_neg = int((labels == 0).sum())
        pos_ratio = float(cfg.training.get("sampler_pos_ratio", 0.3))
        w_pos = pos_ratio / n_pos
        w_neg = (1.0 - pos_ratio) / max(n_neg, 1)
        sample_weights = np.where(labels == 1, w_pos, w_neg)
        num_samples = int(cfg.training.get("sampler_num_samples") or len(labels))
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=num_samples,
            replacement=True,
        )
        shuffle = False
        LOG.info("WeightedRandomSampler: pos_ratio=%.2f, num_samples=%d (n_pos=%d, n_neg=%d)", pos_ratio, num_samples, n_pos, n_neg)

    nw = int(cfg.training.get("num_workers", min(8, os.cpu_count() or 2)))
    use_cuda = torch.cuda.is_available()
    return DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=nw,
        pin_memory=use_cuda,
        persistent_workers=nw > 0,
        drop_last=train,
    )


@torch.no_grad()
def _evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, criterion: torch.nn.Module | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (y_true, y_score, val_loss). val_loss is nan if criterion is None."""
    model.eval()
    scores, targets = [], []
    loss_sum, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        scores.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(y.numpy())
        if criterion is not None:
            y_dev = y.to(device, non_blocking=True)
            loss_sum += criterion(logits, y_dev).item() * x.size(0)
            n += x.size(0)
    val_loss = loss_sum / max(n, 1) if criterion is not None else float("nan")
    return np.concatenate(targets), np.concatenate(scores), val_loss


def _maybe_pos_weight(loader: DataLoader) -> float:
    labels = loader.dataset.labels.numpy().astype(int)
    n_pos = max(int((labels == 1).sum()), 1)
    n_neg = int((labels == 0).sum())
    return n_neg / n_pos


def train(cfg_path: Path) -> int:
    cfg = load_config(cfg_path)
    set_seed(int(cfg.get("seed", 42)))
    device = pick_device()

    run_dir = Path(cfg.training.get("run_dir", "reports/runs")) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    LOG.info("Run-Verzeichnis: %s", run_dir)
    LOG.info("Device: %s", device)

    train_loader = _make_loader(cfg, Path(cfg.data.train_csv), train=True)
    val_loader = _make_loader(cfg, Path(cfg.data.val_csv), train=False)

    # Loss
    loss_kind = cfg.training.loss
    pos_weight = None
    if loss_kind == "bce_pos_weight":
        pos_weight = float(cfg.training.get("pos_weight") or _maybe_pos_weight(train_loader))
        LOG.info("pos_weight = %.3f", pos_weight)
    criterion = build_loss(
        loss_kind,
        pos_weight=pos_weight,
        focal_alpha=cfg.training.get("focal_alpha", 0.25),
        focal_gamma=cfg.training.get("focal_gamma", 2.0),
    ).to(device)

    # Modell
    model = build_model(
        cfg.model.name,
        pretrained=cfg.model.get("pretrained", True),
        dropout=cfg.model.get("dropout", 0.2),
    ).to(device)

    # Optimizer (zweistufig bei Transfer)
    is_transfer = hasattr(model, "param_groups")
    head_only_epochs = int(cfg.training.get("head_only_epochs", 0)) if is_transfer else 0
    if is_transfer and head_only_epochs > 0:
        model.freeze_backbone()
    if is_transfer:
        param_groups = model.param_groups(
            lr_head=float(cfg.training.lr_head),
            lr_backbone=float(cfg.training.lr_backbone),
        )
        optimizer = torch.optim.AdamW(param_groups, weight_decay=float(cfg.training.weight_decay))
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.training.lr_head),
            weight_decay=float(cfg.training.weight_decay),
        )

    total_epochs = int(cfg.training.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    use_amp = device.type == "cuda" and bool(cfg.training.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {"train_loss": [], "val_loss": [], "val_pr_auc": [], "val_f1": []}
    best_metric = -1.0
    epochs_no_improve = 0
    patience = int(cfg.training.get("early_stop_patience", 5))

    for epoch in range(1, total_epochs + 1):
        if is_transfer and head_only_epochs > 0 and epoch == head_only_epochs + 1:
            LOG.info("Backbone auftauen (Epoche %d).", epoch)
            model.unfreeze_backbone()

        model.train()
        running = 0.0
        t0 = time.time()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
        scheduler.step()
        train_loss = running / len(train_loader.dataset)

        y_true, y_score, val_loss = _evaluate(model, val_loader, device, criterion)
        val_metrics = compute_metrics(y_true, y_score, threshold=0.5)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_pr_auc"].append(val_metrics.pr_auc)
        history["val_f1"].append(val_metrics.f1)
        LOG.info(
            "Epoche %3d | train_loss=%.4f val_loss=%.4f PR-AUC=%.4f F1@0.5=%.4f  (%.1fs)",
            epoch, train_loss, val_loss, val_metrics.pr_auc, val_metrics.f1, time.time() - t0,
        )

        if val_metrics.pr_auc > best_metric:
            best_metric = val_metrics.pr_auc
            epochs_no_improve = 0
            save_checkpoint(
                Checkpoint(
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    epoch=epoch,
                    metric=best_metric,
                    extra={"history": history, "cfg_path": str(cfg_path)},
                ),
                run_dir / "best.pt",
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                LOG.info("Early Stopping nach %d Epochen ohne PR-AUC-Verbesserung.", patience)
                break

    # Kurvenplot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoche"); ax[0].legend()
    ax[1].plot(history["val_pr_auc"], label="PR-AUC")
    ax[1].plot(history["val_f1"], label="F1@0.5")
    ax[1].set_title("Val-Metriken"); ax[1].set_xlabel("Epoche"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(run_dir / "training_curves.png", dpi=120)
    plt.close(fig)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    LOG.info("Best Val-PR-AUC: %.4f", best_metric)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    return train(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
