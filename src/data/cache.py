"""Pre-Resize + Cache: einmalig alle PNGs auf 224×224 vorresizen.

Auf GPU-Systemen ist die PNG-Dekodierung + Resize der I/O-Bottleneck, nicht die
GPU-Berechnung. Dieses Script resized alle Bilder aus data/raw/{y,n}/ einmalig
auf die Zielgrösse und schreibt sie standardmässig als PNG (verlustfrei; optional
JPEG via --format) nach data/cached/. Die CSV-Manifeste werden dann auf die gecachten Pfade umgeschrieben.

Aufruf::

    python -m src.data.cache --raw data/raw --out data/cached --size 224
    # Danach: re-run ingest + split mit --raw data/cached
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.utils import get_logger

LOG = get_logger("cache")


def cache_images(raw_dir: Path, out_dir: Path, size: int, fmt: str = "PNG") -> int:
    """Resized alle PNGs aus {y,n}/ auf size×size und speichert sie in out_dir."""
    count = 0
    for label_dir in ("y", "n"):
        src = raw_dir / label_dir
        dst = out_dir / label_dir
        if not src.exists():
            LOG.warning("Quellordner %s existiert nicht, übersprungen.", src)
            continue
        dst.mkdir(parents=True, exist_ok=True)
        # macOS AppleDouble-Files (._*.png) sind keine echten Bilder — herausfiltern.
        pngs = sorted(p for p in src.glob("*.png") if not p.name.startswith("._"))
        LOG.info("Resize %s: %d Bilder -> %d×%d %s", label_dir, len(pngs), size, size, fmt)
        for p in tqdm(pngs, desc=label_dir, unit="img"):
            try:
                img = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
            except (OSError, Image.UnidentifiedImageError) as e:
                LOG.warning("Kann %s nicht oeffnen (%s) — uebersprungen.", p.name, e.__class__.__name__)
                continue
            ext = ".png" if fmt.upper() == "PNG" else ".jpg"
            img.save(dst / (p.stem + ext), format=fmt.upper())
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-Resize + Cache fuer schnelleres Training")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/cached"))
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--format", choices=["png", "jpg"], default="png",
                        help="Zielformat: png (verlustfrei) oder jpg (schnellere Dekodierung)")
    args = parser.parse_args()

    n = cache_images(args.raw, args.out, args.size, args.format)
    LOG.info("Gecacht: %d Bilder in %s (%d×%d)", n, args.out, args.size, args.size)
    LOG.info("Jetzt ingest + split mit --raw %s neu laufen lassen, damit manifest.csv die gecachten Pfade hat.", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
