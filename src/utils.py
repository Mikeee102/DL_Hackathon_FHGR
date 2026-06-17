"""Gemeinsame Helfer: Seeding, Config-Laden, Logging, Checkpoints.

Zentral, damit alle Skripte denselben Seed/Logger/Checkpoint-Pfad verwenden und
Runs reproduzierbar bleiben.
"""
from __future__ import annotations

import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


class DictConfig(dict):
    """Minimaler OmegaConf-Ersatz: dict mit Attribut-Zugriff (rekursiv).

    Reicht für unsere Configs (flache YAMLs ohne Interpolation). Vermeidet die
    antlr4-Abhängigkeit von OmegaConf, die in einigen Umgebungen nicht baut.
    """

    def __getattr__(self, key):
        try:
            v = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return DictConfig(v) if isinstance(v, dict) else v

    def get(self, key, default=None):
        v = super().get(key, default)
        return DictConfig(v) if isinstance(v, dict) else v


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Fixiert alle relevanten Zufallsquellen.

    cudnn.deterministic kostet etwas Performance, ist aber für eine Hackathon-
    Abgabe der richtige Trade-off (Reproduzierbarkeit > Speed).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: str | Path) -> DictConfig:
    """YAML-Config laden."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping, got {type(data)}")
    return DictConfig(data)


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Aufgelöste Config neben den Run speichern → Reproduzierbarkeit."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(cfg), fh, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str = "crosswalk", log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
@dataclass
class Checkpoint:
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None
    epoch: int
    metric: float
    extra: dict[str, Any]


def save_checkpoint(ckpt: Checkpoint, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": ckpt.model_state,
            "optimizer_state": ckpt.optimizer_state,
            "epoch": ckpt.epoch,
            "metric": ckpt.metric,
            "extra": ckpt.extra,
        },
        str(path),
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Checkpoint:
    blob = torch.load(str(path), map_location=map_location)
    return Checkpoint(**blob)
