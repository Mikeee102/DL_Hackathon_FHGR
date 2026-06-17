"""Evaluation: Threshold-Tuning auf Val, finale Metriken + Plots auf Test.

Schreibt nach reports/<run>/eval/:
- metrics.json (Val + Test, mit gewähltem Threshold)
- pr_curve.png, roc_curve.png, confusion.png
- errors/ : Top-N False Positives & False Negatives als Bilder
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_curve
from torch.utils.data import DataLoader

from src.data.dataset import CrosswalkDataset
from src.metrics import best_threshold, compute_metrics
from src.models.transfer import build_model
from src.utils import get_logger, load_checkpoint, load_config, pick_device, set_seed

LOG = get_logger("evaluate")


@torch.no_grad()
def _score(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    s, t = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        s.append(torch.sigmoid(model(x)).cpu().numpy())
        t.append(y.numpy())
    return np.concatenate(t), np.concatenate(s)


def _plot_pr(y_true: np.ndarray, y_score: np.ndarray, out: Path) -> None:
    p, r, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(r, p)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("PR-Kurve (Test)")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def _plot_roc(y_true: np.ndarray, y_score: np.ndarray, out: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr); ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC (Test)")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def _plot_cm(cm: list[list[int]], out: Path) -> None:
    arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(arr, cmap="Blues")
    for (i, j), v in np.ndenumerate(arr):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred 0", "pred 1"]); ax.set_yticklabels(["true 0", "true 1"])
    ax.set_title("Confusion (Test)")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def _dominant_hue(img_path: Path) -> str:
    """Grobe Heuristik: prüft, ob gesättigte Pixel eher rot oder gelb sind.

    Gibt 'red', 'yellow' oder 'other' zurück. Nützlich um die Hypothese
    „rote Velostreifen als FP-Quelle" zu prüfen.
    """
    try:
        img = np.array(Image.open(img_path).convert("RGB")).astype(float)
    except (OSError, FileNotFoundError):
        return "unknown"
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx = img.max(axis=-1)
    mn = img.min(axis=-1)
    sat = np.where(mx > 0, (mx - mn) / mx, 0.0)
    bright_sat = (sat > 0.25) & (mx > 80)
    if bright_sat.sum() < 50:
        return "other"
    rs, gs, bs = r[bright_sat], g[bright_sat], b[bright_sat]
    red_mask = (rs > 140) & (gs < 100) & (bs < 100)
    yellow_mask = (rs > 160) & (gs > 140) & (bs < 100)
    n_red, n_yellow = int(red_mask.sum()), int(yellow_mask.sum())
    if n_red > n_yellow and n_red > 30:
        return "red"
    if n_yellow > n_red and n_yellow > 30:
        return "yellow"
    return "other"


def _export_errors(csv_path: Path, y_true: np.ndarray, y_score: np.ndarray, threshold: float, out_dir: Path, top_n: int = 20) -> None:
    import pandas as pd
    df = pd.read_csv(csv_path).copy()
    df["score"] = y_score
    df["pred"] = (y_score >= threshold).astype(int)

    fps = df[(df["label"] == 0) & (df["pred"] == 1)].sort_values("score", ascending=False).head(top_n)
    fns = df[(df["label"] == 1) & (df["pred"] == 0)].sort_values("score", ascending=True).head(top_n)

    for name, sub in (("false_positives", fps), ("false_negatives", fns)):
        target = out_dir / name
        target.mkdir(parents=True, exist_ok=True)
        hues: list[str] = []
        for _, row in sub.iterrows():
            src = Path(row["image_path"])
            h = _dominant_hue(src)
            hues.append(h)
            if src.exists():
                shutil.copy2(src, target / f"score{row['score']:.3f}_{h}_{src.name}")
        sub = sub.copy()
        sub["dominant_hue"] = hues
        sub.to_csv(out_dir / f"{name}.csv", index=False)

    if not fps.empty:
        hue_counts = fps["dominant_hue"].value_counts().to_dict() if "dominant_hue" in fps.columns else {}
        LOG.info("FP Farbverteilung (top %d): %s", top_n, hue_counts)


def evaluate(cfg_path: Path, ckpt_path: Path) -> int:
    cfg = load_config(cfg_path)
    set_seed(int(cfg.get("seed", 42)))
    device = pick_device()

    val_ds = CrosswalkDataset(cfg.data.val_csv, image_size=cfg.data.get("image_size", 224), train=False)
    test_ds = CrosswalkDataset(cfg.data.test_csv, image_size=cfg.data.get("image_size", 224), train=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.get("num_workers", 2))
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.get("num_workers", 2))

    model = build_model(
        cfg.model.name,
        pretrained=False,
        dropout=cfg.model.get("dropout", 0.2),
    ).to(device)
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.model_state)

    LOG.info("Bewerte Val (Threshold-Tuning)…")
    y_val, s_val = _score(model, val_loader, device)
    thr, f1_val = best_threshold(y_val, s_val)
    LOG.info("Bestes F1 auf Val = %.4f @ threshold = %.4f", f1_val, thr)

    val_metrics = compute_metrics(y_val, s_val, threshold=thr)

    LOG.info("Bewerte Test mit fixem Threshold…")
    y_te, s_te = _score(model, test_loader, device)
    test_metrics = compute_metrics(y_te, s_te, threshold=thr)

    # Majority-Baseline (immer 0)
    majority = compute_metrics(y_te, np.zeros_like(s_te), threshold=0.5)

    run_dir = ckpt_path.parent
    out = run_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "threshold": thr,
        "val": asdict(val_metrics),
        "test": asdict(test_metrics),
        "majority_baseline_test": asdict(majority),
        "ckpt": str(ckpt_path),
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOG.info("Test-Metriken: %s", json.dumps(asdict(test_metrics), indent=2))

    _plot_pr(y_te, s_te, out / "pr_curve.png")
    _plot_roc(y_te, s_te, out / "roc_curve.png")
    _plot_cm(test_metrics.confusion, out / "confusion.png")
    _export_errors(Path(cfg.data.test_csv), y_te, s_te, thr, out / "errors")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    args = parser.parse_args()
    return evaluate(args.config, args.ckpt)


if __name__ == "__main__":
    raise SystemExit(main())
