#!/usr/bin/env bash
# Final HQ model (extra frames, new COLMAP frame): transfer shell/scale, complete, evaluate, export.
set -euo pipefail
source "$HOME/hn3d/env.sh"
cd "$(dirname "$0")"
R="$HOME/hn3d_work/room/runs"
DX="$HOME/hn3d_work/room/dense_ext"
SRC=$(python -c "import pycolmap,glob;print(max(glob.glob('$HOME/hn3d_work/room/colmap/sparse/*/'),key=lambda m:len(pycolmap.Reconstruction(m).images)))")
python transfer_shell.py --src-model "$SRC" --dst-model "$DX/sparse" --src-run "$R/baseline" --dst-run "$R/hq4"
python 04_complete.py --run "$R/hq4" --data "$DX" --out "$R/completed_hq4" --vis-views 10000 \
    --plan-layout ../results/floorplan/layout.json --plan-door-height 2.05
python 09_eval.py --data "$DX" --downscale 1 --runs "$R/hq4" "$R/completed_hq4"
python 05_geometry.py --base "$R/full" --comp "$R/completed_hq4" --comp-shell-run "$R/hq4" \
    --measurements ../data/measurements.json --out ../results/room_full/geometry_hq4.json
python 06_export.py --run "$R/completed_hq4" --shell-run "$R/hq4" --data "$DX"
echo POST_HQ4_DONE
