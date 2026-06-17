"""Torch-Dataset + Transforms.

Wir trennen Train- und Eval-Transforms strikt: Eval enthält keinerlei
Zufallsoperation, damit Metriken deterministisch bleiben.

Augmentation-Wahl ist datengetrieben begründet (siehe README):
Luftbilder haben keine kanonische Ausrichtung → volle Rotation + H/V-Flip ist
semantisch zulässig. Crop/Shear meiden wir, damit dünne Linienlabels nicht
„weggeschnitten" werden.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet-Statistiken; passen für RGB-Luftbilder als grobe Normalisierung und
# sind konsistent mit den Pretrained-Backbones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224, train: bool = True) -> transforms.Compose:
    if train:
        # fill=(128,128,128): neutrales Grau statt Schwarz in den
        # Rotationsecken — verhindert, dass randständige Streifen an einem
        # schwarzen Dreieck „kleben" und das Modell auf schwarze Ecken als
        # Negativ-Indikator trainiert.
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(degrees=180, expand=False, fill=(128, 128, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class CrosswalkDataset(Dataset):
    """Liest ein Split-CSV (image_path, label, ...) und liefert (image_tensor, label_float)."""

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path | None = None,
        image_size: int = 224,
        train: bool = True,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        if "image_path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError(f"CSV {csv_path} braucht Spalten 'image_path' und 'label'.")
        self.image_root = Path(image_root) if image_root else None
        self.transform = build_transforms(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        if self.image_root and not path.is_absolute():
            return self.image_root / path
        return path

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img = Image.open(self._resolve(row["image_path"])).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        return x, y

    @property
    def labels(self) -> torch.Tensor:
        return torch.tensor(self.df["label"].astype(float).values, dtype=torch.float32)
