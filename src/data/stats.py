"""Datensatz-Qualitätsreport (Phase 1d).

Schreibt nach reports/data/:
- counts.json: Anzahl pro Klasse & Split, Imbalance-Ratio
- size_check.json: Bildgrössen-Konsistenz, kleinste/grösste Beispiele
- duplicates.csv: perceptual-hash-Duplikate (oder near-Duplikate)
- contact_sheet.png: zufällige Positive/Negative-Thumbnails

Das ist das Material für die Raster-Achse „Datenqualität".
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.utils import get_logger, set_seed

LOG = get_logger("stats")


def _load_splits(split_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        p = split_dir / f"{split}.csv"
        if p.exists():
            out[split] = pd.read_csv(p)
    return out


def count_report(splits: dict[str, pd.DataFrame]) -> dict:
    out: dict = {}
    for name, df in splits.items():
        pos = int((df["label"] == 1).sum())
        neg = int((df["label"] == 0).sum())
        ratio = (neg / pos) if pos else float("inf")
        out[name] = {"n": int(len(df)), "pos": pos, "neg": neg, "neg_per_pos": ratio}
    return out


def size_check(manifest: pd.DataFrame, max_check: int = 500) -> dict:
    sizes: Counter[tuple[int, int]] = Counter()
    sample = manifest.sample(min(max_check, len(manifest)), random_state=0)
    for path in sample["image_path"]:
        try:
            with Image.open(path) as im:
                sizes[im.size] += 1
        except (OSError, FileNotFoundError):
            continue
    return {
        "sample_size": int(len(sample)),
        "distinct_sizes": {f"{w}x{h}": int(c) for (w, h), c in sizes.most_common(10)},
    }


def find_duplicates(manifest: pd.DataFrame, hash_size: int = 8) -> pd.DataFrame:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for path in manifest["image_path"]:
        try:
            h = str(imagehash.phash(Image.open(path), hash_size=hash_size))
        except (OSError, FileNotFoundError):
            continue
        by_hash[h].append(path)
    dupes = [{"phash": h, "paths": paths} for h, paths in by_hash.items() if len(paths) > 1]
    return pd.DataFrame(dupes)


def contact_sheet(manifest: pd.DataFrame, out_path: Path, n_per_class: int = 8) -> None:
    pos = manifest[manifest["label"] == 1].sample(min(n_per_class, (manifest["label"] == 1).sum()), random_state=0)
    neg = manifest[manifest["label"] == 0].sample(min(n_per_class, (manifest["label"] == 0).sum()), random_state=0)
    cols = max(len(pos), len(neg))
    fig, axes = plt.subplots(2, cols, figsize=(cols * 2, 4))
    if cols == 1:
        axes = axes.reshape(2, 1)
    for ax_row, df, title in zip(axes, (pos, neg), ("positive", "negative"), strict=True):
        for i, ax in enumerate(ax_row):
            ax.axis("off")
            if i < len(df):
                try:
                    ax.imshow(Image.open(df.iloc[i]["image_path"]))
                except (OSError, FileNotFoundError):
                    pass
            if i == 0:
                ax.set_title(title, loc="left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Datensatz-Qualitätsreport")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits"))
    parser.add_argument("--out", type=Path, default=Path("reports/data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    splits = _load_splits(args.splits)

    counts = count_report(splits) if splits else count_report({"all": manifest})
    (args.out / "counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    LOG.info("Counts: %s", counts)

    sizes = size_check(manifest)
    (args.out / "size_check.json").write_text(json.dumps(sizes, indent=2), encoding="utf-8")

    dupes = find_duplicates(manifest)
    dupes.to_csv(args.out / "duplicates.csv", index=False)
    LOG.info("Doppelte (per pHash) Gruppen: %d", len(dupes))

    contact_sheet(manifest, args.out / "contact_sheet.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
