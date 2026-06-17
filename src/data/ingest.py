"""Annotator-Output (ImageFolder, LV95-Koordinaten im Dateinamen) -> manifest.csv.

Datenformat (Phase 1a — bestätigt):
- data/raw/y/*.png  -> label 1 (Fussgängerstreifen vorhanden)
- data/raw/n/*.png  -> label 0
- data/raw/crops/   -> noch nicht annotiert, wird ignoriert.
- Dateiname kodiert LV95-Koordinaten in METERN: ``{E}_{N}.png``
  Beispiel: ``2758012_1191037.png`` (Region Chur).
- Kachelraster 25 m, Auflösung 10 cm/px ⇒ ~250×250 px.

Manifest-Spalten:
    image_path, label, e, n, source

(e, n) statt (lon, lat), weil LV95 metrisch ist — das ist für den Geo-Split
deutlich angenehmer (Distanzen direkt in Metern statt Grad).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.utils import get_logger

LOG = get_logger("ingest")

MANIFEST_COLUMNS = ["image_path", "label", "e", "n", "source"]

# y = positiv (Fussgängerstreifen), n = negativ. crops = unannotiert -> skip.
LABEL_DIRS: dict[str, int] = {"y": 1, "n": 0}
IGNORE_DIRS: set[str] = {"crops"}


def _parse_en(stem: str) -> tuple[float, float] | None:
    """``{E}_{N}`` -> (E, N) in Metern. Gibt None bei Format-Mismatch zurück."""
    parts = stem.split("_")
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def build_manifest(raw_dir: Path) -> tuple[list[dict], dict[str, int]]:
    rows: list[dict] = []
    skipped = {"bad_name": 0, "ignored_dirs": 0, "no_label_dir": 0}

    for label_name, label in LABEL_DIRS.items():
        sub = raw_dir / label_name
        if not sub.exists():
            LOG.warning("Erwarteter Label-Ordner fehlt: %s", sub)
            skipped["no_label_dir"] += 1
            continue
        # macOS AppleDouble-Files (._*.png) sind keine echten Bilder.
        for img in sorted(p for p in sub.glob("*.png") if not p.name.startswith("._")):
            en = _parse_en(img.stem)
            if en is None:
                skipped["bad_name"] += 1
                LOG.warning("Dateiname ohne E_N: %s — übersprungen.", img.name)
                continue
            e, n = en
            rows.append(
                {
                    "image_path": str(img),
                    "label": label,
                    "e": e,
                    "n": n,
                    "source": label_name,  # 'y' / 'n' (woher das Label kommt)
                }
            )

    for ignored in IGNORE_DIRS:
        if (raw_dir / ignored).exists():
            n = sum(1 for _ in (raw_dir / ignored).glob("*.png"))
            skipped["ignored_dirs"] += n
            LOG.info("Ignoriere %s/ (%d unannotierte PNGs).", ignored, n)

    return rows, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="ImageFolder -> manifest.csv")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/manifest.csv"))
    args = parser.parse_args()

    if not args.raw.exists():
        LOG.error("Raw-Verzeichnis %s existiert nicht.", args.raw)
        return 1

    rows, skipped = build_manifest(args.raw)
    if not rows:
        LOG.error("Kein einziges Beispiel gefunden. Liegen y/ und n/ unter %s?", args.raw)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    n_pos = sum(r["label"] == 1 for r in rows)
    n_neg = sum(r["label"] == 0 for r in rows)
    LOG.info(
        "Manifest geschrieben: %s  (n=%d, pos=%d, neg=%d, neg/pos=%.2f)  skipped=%s",
        args.out, len(rows), n_pos, n_neg, (n_neg / n_pos) if n_pos else float("inf"), skipped,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
