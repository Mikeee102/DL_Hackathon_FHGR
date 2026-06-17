#!/usr/bin/env bash
# End-to-End: cache -> ingest -> split -> stats -> train -> eval
# Erwartet, dass data/raw/{y,n}/ befüllt ist.
#
# Auf GPU: zuerst pip install torch torchvision --index-url .../cu128
# Optional: SKIP_CACHE=1 falls bereits gecacht.

set -euo pipefail

RAW=${RAW:-data/raw}
CACHED=${CACHED:-data/cached}
MANIFEST=${MANIFEST:-data/manifest.csv}
SPLITS=${SPLITS:-data/splits}

# 0) Pre-Resize/Cache (einmalig)
if [ "${SKIP_CACHE:-0}" != "1" ] && [ ! -d "$CACHED/y" ]; then
    echo "[0/6] Pre-Resize + Cache (224x224)"
    python -m src.data.cache --raw "$RAW" --out "$CACHED" --size 224
fi

# Ingest liest aus dem Cache (oder raw, falls kein Cache)
SRC="${CACHED}"
if [ ! -d "$SRC/y" ]; then SRC="$RAW"; fi

echo "[1/6] Ingest"
python -m src.data.ingest --raw "$SRC" --out "$MANIFEST"

echo "[2/6] Geo-Split"
python -m src.data.split --manifest "$MANIFEST" --out "$SPLITS"

echo "[3/6] Daten-Qualitätsreport"
python -m src.data.stats --manifest "$MANIFEST" --splits "$SPLITS" --out reports/data

echo "[4/6] Training (Baseline + ResNet18 Sampler)"
python -m src.train --config configs/baseline_cnn.yaml
python -m src.train --config configs/resnet18_transfer.yaml

echo "[5/6] Training Ablation (ResNet18 Focal)"
python -m src.train --config configs/resnet18_focal.yaml

echo "[6/6] Evaluation"
for run in baseline_cnn resnet18_transfer resnet18_focal; do
    ckpt="reports/runs/${run}/best.pt"
    cfg="configs/${run}.yaml"
    if [ -f "$ckpt" ] && [ -f "$cfg" ]; then
        echo "  Eval: $run"
        python -m src.evaluate --config "$cfg" --ckpt "$ckpt"
    fi
done

echo "Fertig. Outputs unter reports/."
