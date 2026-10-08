#!/usr/bin/env bash
# One-shot pipeline: video -> poses -> baseline 3DGS -> shell -> scale -> completion -> evaluation
#                    -> geometry -> export -> objects -> report -> viewer.
#
#   ./run_all.sh room data/raw/room.mp4 [END_SECONDS]
#
# Optional inputs picked up automatically:
#   data/heldout/*.jpg        held-out stills (novel-view metrics)
#   data/measurements.json    tape ground truth (dimension error, Chamfer)
#   data/floorplan.jpg        floor plan (Mode A model + blueprint-guided metric scale)
# Each stage is skipped if its output exists (delete ~/hn3d_work/<name> to re-run).
# Env: ITERS (default 15000), DS (downscale, default 2).
set -euo pipefail
source "$HOME/hn3d/env.sh"

NAME="$1"; VIDEO="$2"; END="${3:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$HOME/hn3d_work/$NAME"
ITERS="${ITERS:-15000}"
DS="${DS:-2}"
MEAS="$HERE/data/measurements.json"
PLAN="$HERE/data/floorplan.jpg"
OUT="$HERE/results/$NAME"
cd "$HERE/scripts"
mkdir -p "$OUT"

DENSE="$WORK/dense"
[[ -d "$WORK/images/train" ]] || python 00_ingest.py --video "$HERE/$VIDEO" --work "$WORK" --fps 10 --target 400 ${END:+--end "$END"}
[[ -d "$DENSE/sparse" ]]      || python 01_sfm.py --work "$WORK" --threads 12

R="$WORK/runs"
B="$R/baseline"; C="$R/completed"
[[ -f "$B/metrics.json" ]] || python 02_train.py --data "$DENSE" --out "$B" --iters "$ITERS" --downscale "$DS"
[[ -f "$B/shell.json"   ]] || python 03_shell.py --run "$B" --data "$DENSE"

PLAN_ARGS=()
if [[ -f "$PLAN" ]]; then
  python 10_floorplan.py --plan "$PLAN" --out "$HERE/results/floorplan"
  PLAN_ARGS=(--plan-layout "$HERE/results/floorplan/layout.json")
fi
MEAS_ARGS=(); [[ -f "$MEAS" ]] && MEAS_ARGS=(--measurements "$MEAS")
[[ -f "$B/scale.json"   ]] || python 03b_scale.py --run "$B" --data "$DENSE" "${MEAS_ARGS[@]}" "${PLAN_ARGS[@]}"
PLAN_C=(); [[ -f "$HERE/results/floorplan/layout.json" ]] && PLAN_C=(--plan-layout "$HERE/results/floorplan/layout.json")
[[ -f "$C/metrics.json" ]] || python 04_complete.py --run "$B" --data "$DENSE" --out "$C" --vis-views 10000 "${PLAN_C[@]}"

python 09_eval.py --data "$DENSE" --downscale "$DS" --runs "$B" "$C"
[[ -f "$MEAS" ]] && python 05_geometry.py --base "$B" --comp "$C" --measurements "$MEAS" --out "$OUT/geometry.json"
python 06_export.py --run "$C" --shell-run "$B" --data "$DENSE" --baseline-run "$B"
python 08_objects.py --run "$C" --shell-run "$B" || echo "objects stage failed (non-fatal)"
python 07_report.py --work "$WORK" --out "$OUT" --shell-run baseline --viewer-out "$HERE/viewer/$NAME"

mkdir -p "$HERE/viewer/$NAME"
cp "$C/export/room.json" "$C/export/room.splat" "$C/export/room_honesty.splat" \
   "$C/export/room_baseline.splat" "$C/export/room_splat.ply" "$C/export/room_splat_honesty.ply" "$C/export/room_shell.glb" "$HERE/viewer/$NAME/"
echo "done -> $OUT/RESULTS.md | viewer: python viewer/serve.py, then http://localhost:8766/?scene=$NAME"
