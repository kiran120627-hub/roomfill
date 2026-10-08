#!/usr/bin/env bash
# After the high-quality (MCMC, base-colour) model finishes: complete, evaluate against the same
# full-resolution vanilla baseline, export to the viewer and verify.
set -euo pipefail
source "$HOME/hn3d/env.sh"
cd "$(dirname "$0")"
R="$HOME/hn3d_work/room/runs"
D="$HOME/hn3d_work/room/dense"
cp "$R/baseline/shell.json" "$R/baseline/scale.json" "$R/hq/"
python 04_complete.py --run "$R/hq" --data "$D" --out "$R/completed_hq" --vis-views 10000 \
    --plan-layout ../results/floorplan/layout.json --plan-door-height 2.05
python 09_eval.py --data "$D" --downscale 1 --runs "$R/full" "$R/hq" "$R/completed_hq"
python 05_geometry.py --base "$R/full" --comp "$R/completed_hq" --measurements ../data/measurements.json \
    --out ../results/room_full/geometry_hq.json
python 06_export.py --run "$R/completed_hq" --shell-run "$R/baseline" --data "$D" --baseline-run "$R/full"
python 08_objects.py --run "$R/completed_hq" --shell-run "$R/baseline" || true
python check_outputs.py --export "$R/completed_hq/export" --viewer ../viewer/room || true
echo POST_HQ_DONE
