"""Transfer-Learning-Modelle: ResNet18 (Default) und EfficientNet-B0.

Beide ImageNet-pretrained; Klassifikations-Head wird durch einen einzelnen Logit
ersetzt. Wir liefern Hilfen, um den Backbone zu freezen / aufzutauen
(zweistufiges Training: erst Head, dann Backbone mit kleinerer LR).
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def _replace_head_resnet(model: nn.Module, dropout: float) -> nn.Module:
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
    return model


def _replace_head_effnet(model: nn.Module, dropout: float) -> nn.Module:
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
    return model


class TransferModel(nn.Module):
    def __init__(self, arch: str = "resnet18", pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        if arch == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.net = _replace_head_resnet(models.resnet18(weights=weights), dropout)
            self._backbone_params = [p for n, p in self.net.named_parameters() if not n.startswith("fc.")]
            self._head_params = [p for n, p in self.net.named_parameters() if n.startswith("fc.")]
        elif arch == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.net = _replace_head_effnet(models.efficientnet_b0(weights=weights), dropout)
            self._backbone_params = [p for n, p in self.net.named_parameters() if not n.startswith("classifier.")]
            self._head_params = [p for n, p in self.net.named_parameters() if n.startswith("classifier.")]
        else:
            raise ValueError(f"Unbekannte Architektur: {arch}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def freeze_backbone(self) -> None:
        for p in self._backbone_params:
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self._backbone_params:
            p.requires_grad = True

    def param_groups(self, lr_head: float, lr_backbone: float) -> list[dict]:
        """Discriminative LRs: Head bekommt grössere LR als Backbone."""
        return [
            {"params": self._head_params, "lr": lr_head},
            {"params": self._backbone_params, "lr": lr_backbone},
        ]


def build_model(name: str, **kwargs) -> nn.Module:
    if name in ("baseline", "baseline_cnn"):
        from src.models.baseline import BaselineCNN

        return BaselineCNN(**{k: v for k, v in kwargs.items() if k in {"in_channels", "dropout"}})
    if name in ("resnet18", "efficientnet_b0"):
        return TransferModel(arch=name, **{k: v for k, v in kwargs.items() if k in {"pretrained", "dropout"}})
    raise ValueError(f"Unbekanntes Modell: {name}")
