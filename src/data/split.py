"""Geografischer Block-Split (LV95-Meter, KEIN zufälliger Split).

Warum kein zufälliger Split:
Luftbild-Kacheln sind räumlich autokorreliert — Nachbarkacheln teilen Strassenzüge,
Beleuchtung, Bebauungsmuster. Zufälliges Sampling liesse diese Information leaken.
Wir würden die Generalisierung auf neue Regionen massiv überschätzen.

Vorgehen:
1. Lege ein metrisches Gitter über die (E, N)-Bounding-Box (Default 500 m Zellen).
2. Weise **ganze Zellen** einem Split zu (nie einzelne Kacheln), gewichtet nach
   Beispielen pro Zelle, sodass das Verhältnis ≈ 70/15/15 stimmt.
3. **Puffer:** Zellen, die direkt an einem anderen Split anliegen, werden
   verworfen, wenn der Abstand kleiner als ``buffer_m`` (Default 25 m = 1 Kachel)
   wäre. Konkret realisieren wir den Puffer dadurch, dass eine "Train"-Zelle nur
   gehalten wird, wenn keine der 8 Nachbarzellen einem anderen Split angehört
   und die Zellgrösse ≥ Puffer ist. Bei 500 m Zellen ist das automatisch erfüllt,
   solange die *Kacheln innerhalb der Zelle* nicht direkt an der Zellkante
   liegen — zusätzlich entfernen wir Kacheln, die näher als ``buffer_m`` an
   der Grenze zu einer fremden Split-Zelle liegen.
4. Logge Klassenverteilung pro Split + Bounding-Boxen der Zellen + Anzahl der
   wegen Puffer verworfenen Kacheln.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.utils import get_logger, set_seed

LOG = get_logger("split")


def _cell_id(e: float, n: float, cell_m: float) -> tuple[int, int]:
    return (math.floor(e / cell_m), math.floor(n / cell_m))


def _greedy_assign_cells(cells_by_size: list[tuple[tuple[int, int], int]], ratios: tuple[float, float, float], n_total: int, seed: int) -> dict[tuple[int, int], str]:
    """Verteilt Zellen gierig auf die drei Buckets, grösste Zellen zuerst."""
    rng = random.Random(seed)
    cells = list(cells_by_size)
    rng.shuffle(cells)
    cells.sort(key=lambda kv: kv[1], reverse=True)

    targets = {"train": ratios[0] * n_total, "val": ratios[1] * n_total, "test": ratios[2] * n_total}
    current = {"train": 0, "val": 0, "test": 0}
    out: dict[tuple[int, int], str] = {}
    for c, size in cells:
        deficit = {k: targets[k] - current[k] for k in targets}
        bucket = max(deficit, key=lambda k: deficit[k])
        out[c] = bucket
        current[bucket] += size
    return out


def _drop_buffer_tiles(df: pd.DataFrame, cell_to_split: dict[tuple[int, int], str], cell_m: float, buffer_m: float) -> tuple[pd.DataFrame, int]:
    """Entfernt Kacheln, deren Mittelpunkt näher als buffer_m an einer Zelle
    mit anderem Split-Label liegt.

    Wir prüfen nur die direkten 8 Nachbarzellen — das reicht, weil weiter
    entfernte Zellen per Konstruktion mindestens cell_m Distanz haben.
    """
    keep = []
    dropped = 0
    for row in df.itertuples(index=False):
        c = (math.floor(row.e / cell_m), math.floor(row.n / cell_m))
        own = cell_to_split[c]
        # Abstand der Kachel zu den vier Zellkanten:
        dx_left = row.e - c[0] * cell_m
        dx_right = (c[0] + 1) * cell_m - row.e
        dy_bot = row.n - c[1] * cell_m
        dy_top = (c[1] + 1) * cell_m - row.n

        # 8 Nachbarn samt der minimalen Distanz, die die Kachel zur jeweiligen
        # Nachbarzelle hätte:
        neighbours = [
            ((c[0] - 1, c[1]),     dx_left),
            ((c[0] + 1, c[1]),     dx_right),
            ((c[0],     c[1] - 1), dy_bot),
            ((c[0],     c[1] + 1), dy_top),
            ((c[0] - 1, c[1] - 1), math.hypot(dx_left, dy_bot)),
            ((c[0] + 1, c[1] - 1), math.hypot(dx_right, dy_bot)),
            ((c[0] - 1, c[1] + 1), math.hypot(dx_left, dy_top)),
            ((c[0] + 1, c[1] + 1), math.hypot(dx_right, dy_top)),
        ]
        too_close = False
        for nc, dist in neighbours:
            other = cell_to_split.get(nc)
            if other is not None and other != own and dist < buffer_m:
                too_close = True
                break
        if too_close:
            dropped += 1
        else:
            keep.append(row)

    return pd.DataFrame(keep, columns=df.columns), dropped


def geo_split(
    manifest: pd.DataFrame,
    cell_m: float = 500.0,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    buffer_m: float = 25.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    cells = [_cell_id(e, n, cell_m) for e, n in zip(manifest["e"], manifest["n"], strict=True)]
    df = manifest.copy()
    df["_cell"] = cells

    sizes_per_cell: dict[tuple[int, int], int] = defaultdict(int)
    for c in cells:
        sizes_per_cell[c] += 1

    cell_to_split = _greedy_assign_cells(list(sizes_per_cell.items()), ratios, len(df), seed)
    df["split"] = df["_cell"].map(cell_to_split)

    df_clean, n_dropped = _drop_buffer_tiles(df, cell_to_split, cell_m=cell_m, buffer_m=buffer_m)
    df_clean = df_clean.drop(columns=["_cell"])

    summary = {"cells": {}, "splits": {}, "buffer_dropped": int(n_dropped), "cell_m": cell_m, "buffer_m": buffer_m}
    for split, sub in df_clean.groupby("split"):
        summary["splits"][split] = {
            "n": int(len(sub)),
            "pos": int((sub["label"] == 1).sum()),
            "neg": int((sub["label"] == 0).sum()),
            "pos_rate": float(sub["label"].mean()),
            "e_min": float(sub["e"].min()),
            "e_max": float(sub["e"].max()),
            "n_min": float(sub["n"].min()),
            "n_max": float(sub["n"].max()),
        }
    summary["cells"] = {f"{c[0]}_{c[1]}": cell_to_split[c] for c in cell_to_split}
    return df_clean, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Geografischer Block-Split (LV95-Meter)")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/splits"))
    parser.add_argument("--cell-m", type=float, default=500.0, help="Zellgrösse in Metern")
    parser.add_argument("--buffer-m", type=float, default=25.0, help="Mindestabstand zwischen Splits in Metern (>= Kachelgrösse)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)
    required = {"image_path", "label", "e", "n"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest fehlt Spalten: {missing}")

    df_split, summary = geo_split(df, cell_m=args.cell_m, buffer_m=args.buffer_m, seed=args.seed)

    for split in ("train", "val", "test"):
        sub = df_split[df_split["split"] == split].drop(columns=["split"])
        sub.to_csv(args.out / f"{split}.csv", index=False)

    # Kompakte Übersicht (ohne die ggf. lange cells-Map) ins Log:
    splits_only = {k: summary[k] for k in ("splits", "buffer_dropped", "cell_m", "buffer_m")}
    (args.out / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("Split-Übersicht: %s", json.dumps(splits_only, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
