# Crosswalk Classifier (SwissImage 10 cm)

Binäre Bildklassifikation von **25 m × 25 m**-Kacheln aus SwissImage 10 cm:
Fussgängerstreifen vorhanden (`1`) oder nicht (`0`).

## Resultate

Datensatz: **112'137 Kacheln** aus 6 Regionen (AsconaLocarno, Basel, Bern, Freiburg, Genf, Zürich),
**3'159 Positives / 108'978 Negatives** (Imbalance ~34:1). Geo-Split mit 25 m-Puffer
zwischen Train/Val/Test → **4'011 Kacheln im Puffer gedroppt** (räumliche Leakage stark reduziert, nicht garantiert ausgeschlossen).

| Modell | Imbalance-Handling | Val PR-AUC | **Test PR-AUC** | Test F1 | Test Recall | Test Precision | Threshold (Val) |
|---|---|---:|---:|---:|---:|---:|---:|
| Majority-Class Baseline | — | — | 0.0325 | 0.000 | 0.000 | 0.000 | — |
| Baseline CNN (4 Conv-Blöcke) | WeightedSampler 30 % + BCE | 0.969 | 0.965 | **0.923** | 0.953 | 0.895 | 0.934 |
| ResNet18 Transfer | WeightedSampler 30 % + BCE | 0.966 | 0.966 | 0.915 | 0.953 | 0.881 | 0.794 |
| **ResNet18 Transfer** (Ablation) | **Focal Loss (γ=2, α=0.25)** | **0.977** | **0.976** | 0.917 | 0.945 | 0.890 | 0.451 |

Alle Konfusionsmatrizen, PR-/ROC-Kurven, Trainingsverläufe und Fehler-Thumbnails liegen
unter `reports/runs/<run>/eval/` und `reports/runs/<run>/training_curves.png`.

Bestes Modell: **ResNet18 + Focal Loss** (Test-PR-AUC 0.976). Begründung der Designentscheidungen,
Fehleranalyse und Diskussion entlang der 5 FHGR-Achsen stehen im
**[Projektbericht](reports/PROJEKTBERICHT.md)** (`reports/PROJEKTBERICHT.pdf`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
# GPU (CUDA 12.8, Blackwell/RTX 50xx):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Pipeline

```bash
bash scripts/run_all.sh
```

Schritte einzeln:

```bash
# 0) Pre-Resize/Cache (einmalig, empfohlen auf GPU-Systemen)
#    Resized 250x250 PNGs auf 224x224 — spart PNG-Decode+Resize pro Epoch.
python -m src.data.cache --raw data/raw --out data/cached --size 224

# 1) Annotator-Output -> manifest.csv
python -m src.data.ingest --raw data/cached --out data/manifest.csv

# 2) Geo-Split  (KEIN zufälliger Split)
python -m src.data.split --manifest data/manifest.csv --out data/splits

# 3) Datenqualitätsreport
python -m src.data.stats --manifest data/manifest.csv --splits data/splits --out reports/data

# 4) Training
python -m src.train --config configs/baseline_cnn.yaml
python -m src.train --config configs/resnet18_transfer.yaml
python -m src.train --config configs/resnet18_focal.yaml   # Ablation

# 5) Evaluation
python -m src.evaluate --config configs/resnet18_transfer.yaml --ckpt reports/runs/resnet18_transfer/best.pt
```

## Struktur

```
crosswalk-classifier/
├── configs/                 # alle Hyperparameter (YAML)
├── data/                    # nicht committet (Bilder + Splits)
├── src/
│   ├── data/                # ingest, split, dataset, stats, cache
│   ├── models/              # baseline CNN, transfer (ResNet18)
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics.py
│   └── utils.py
├── scripts/run_all.sh
└── reports/                 # Plots, Metriken, Fehleranalyse (committet, ohne .pt)
    ├── PROJEKTBERICHT.md     # Projektbericht (5 FHGR-Achsen)
    ├── data/                # Datenqualitätsreport
    └── runs/<run>/eval/     # Test-Metriken + Plots pro Modell
```
