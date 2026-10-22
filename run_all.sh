#!/usr/bin/env bash
# End-to-end pipeline for IB-DTA. Run from the repo root.
#
#   ./run_all.sh            quick demo (CPU, ~20 min)
#   ./run_all.sh full       full scale (GPU recommended)
set -euo pipefail

MODE="${1:-quick}"
DATA="${DATA:-davis}"

echo "=== 1/5  property tests ==="
python test_ib.py

echo "=== 2/5  download $DATA ==="
python download_data.py --datasets "$DATA"

echo "=== 3/5  build splits + audit ==="
python splits.py --csv "datasets/$DATA/full.csv" --out "datasets/$DATA"

echo "=== 4/5  train (cold_target) ==="
if [ "$MODE" = "full" ]; then
  python train_ib.py --cfg configs/IBDTA.yaml --data "$DATA" --split cold_target
else
  python train_ib.py --cfg configs/IBDTA.yaml --data "$DATA" --split cold_target \
      --subset 1500 --eval-subset 500 --epochs 8 \
      --set IB.H_DIM 64 PROTEIN.MAX_LEN 400 SOLVER.BATCH_SIZE 16
fi

echo "=== 5/5  experiments + figures ==="
if [ "$MODE" = "full" ]; then
  python run_experiments.py --full
else
  python run_experiments.py --quick
fi

echo
echo "Done. Figures in figures/, tables in results/."
