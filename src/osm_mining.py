"""OSM Positive-Mining (Phase 1c, optional).

Holt Koordinaten von Fussgängerstreifen aus OSM (`highway=crossing`,
`crossing=marked`) für eine angegebene Bounding-Box. **Exportiert nur
Koordinaten** — lädt keine swisstopo-Tiles automatisch (Nutzungsbedingungen).

Koordinatensysteme:
- Eingabe (BBox) und Ausgabe sind WGS84 (lon/lat). Unser Datensatz ist in
  LV95 (EPSG:2056, Meter); für die Kombination mit OSM-Treffern wäre eine
  Transformation 2056 ↔ 4326 nötig (z. B. via ``pyproj``). Für den
  Geo-Split brauchen wir das nicht — der läuft direkt auf (E, N) in Metern.

Aufruf::

    python -m src.osm_mining --bbox 7.4 46.9 7.5 47.0 --out reports/osm_candidates.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib import request

from src.utils import get_logger

LOG = get_logger("osm")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def fetch_candidates(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> list[dict]:
    query = f"""
    [out:json][timeout:60];
    (
      node["highway"="crossing"]["crossing"="marked"]({min_lat},{min_lon},{max_lat},{max_lon});
      node["highway"="crossing"]["crossing"="zebra"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out body;
    """.strip()
    data = ("data=" + request.quote(query)).encode("utf-8")  # type: ignore[attr-defined]
    req = request.Request(OVERPASS_URL, data=data, method="POST")
    with request.urlopen(req, timeout=120) as resp:
        payload = json.load(resp)
    return [
        {"id": el["id"], "lon": el["lon"], "lat": el["lat"]}
        for el in payload.get("elements", [])
        if "lon" in el and "lat" in el
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"), required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/osm_candidates.csv"))
    args = parser.parse_args()

    min_lon, min_lat, max_lon, max_lat = args.bbox
    LOG.info("Fetching OSM crossings in bbox %s", args.bbox)
    rows = fetch_candidates(min_lon, min_lat, max_lon, max_lat)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "lon", "lat"])
        w.writeheader()
        w.writerows(rows)
    LOG.info("Wrote %d crossings to %s", len(rows), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
