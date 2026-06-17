"""Loss-Funktionen für Imbalance-Handling.

Drei Optionen, per Config umschaltbar:
- BCE mit `pos_weight` (einfachster Hebel, gut wenn Imbalance moderat)
- Focal Loss (fokussiert auf schwere Beispiele, hilft bei harter Imbalance)
- normales BCE (kein Re-Weighting; nur sinnvoll wenn man parallel WeightedSampler nutzt)
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce
        return loss.mean()


def build_loss(kind: str, pos_weight: float | None = None, focal_alpha: float = 0.25, focal_gamma: float = 2.0) -> nn.Module:
    kind = kind.lower()
    if kind == "bce":
        return nn.BCEWithLogitsLoss()
    if kind == "bce_pos_weight":
        if pos_weight is None:
            raise ValueError("bce_pos_weight braucht 'pos_weight'")
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight)))
    if kind == "focal":
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    raise ValueError(f"Unbekannter Loss: {kind}")
