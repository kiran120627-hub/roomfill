"""Stage 5 — geometric accuracy vs ground truth: room dimensions + Chamfer distance.

Ground truth is the tape-measured room box (length x width x height). Both the baseline and
the completed splat are put in the same metric, gravity-aligned room frame (from stage 3/3b)
so the comparison isolates what completion adds.

  accuracy     pred -> GT : mean distance from reconstructed shell points to the true walls
  completeness GT -> pred : mean distance from true wall samples to the nearest reconstructed point
  chamfer                 : mean of the two (metres; lower is better)
  coverage@10cm           : fraction of the true shell within 10 cm of reconstructed geometry

Only points near the shell (within --band of a true wall) count towards accuracy, so desks and
other furniture are not penalised as "wall error".

Usage:
    python scripts/05_geometry.py --base <baseline run> --comp <completed run> \
        --measurements data/measurements.json --out results/room/geometry.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree


def room_frame(sh: dict, s: float):
    up, ex, ey = (np.array(sh[k]) for k in ("up", "ex", "ey"))
    R = np.stack([ex, up, -ey])
    t = np.mean(sh["x"]) * ex + np.mean(sh["y"]) * ey + sh["floor_offset"] * up
    return lambda p: s * (p - t) @ R.T


def gt_box_samples(Lx: float, Lz: float, H: float, step: float = 0.05) -> np.ndarray:
    xs = np.arange(-Lx / 2, Lx / 2 + 1e-9, step)
    zs = np.arange(-Lz / 2, Lz / 2 + 1e-9, step)
    ys = np.arange(0, H + 1e-9, step)
    pts = []
    X, Z = np.meshgrid(xs, zs)
    for y in (0.0, H):                                   # floor, ceiling
        pts.append(np.stack([X.ravel(), np.full(X.size, y), Z.ravel()], 1))
    X, Y = np.meshgrid(xs, ys)
    for z in (-Lz / 2, Lz / 2):
        pts.append(np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1))
    Z, Y = np.meshgrid(zs, ys)
    for x in (-Lx / 2, Lx / 2):
        pts.append(np.stack([np.full(Z.size, x), Y.ravel(), Z.ravel()], 1))
    return np.concatenate(pts)


def dist_to_box(p: np.ndarray, Lx: float, Lz: float, H: float) -> np.ndarray:
    """Distance from points to the nearest face of the (hollow) box surface."""
    d = np.stack([np.abs(p[:, 0] + Lx / 2), np.abs(p[:, 0] - Lx / 2),
                  np.abs(p[:, 1]), np.abs(p[:, 1] - H),
                  np.abs(p[:, 2] + Lz / 2), np.abs(p[:, 2] - Lz / 2)], 1)
    return d.min(1)


def evaluate(pts: np.ndarray, gt: np.ndarray, Lx: float, Lz: float, H: float, band: float) -> dict:
    near = dist_to_box(pts, Lx, Lz, H) < band
    shell_pts = pts[near]
    acc = dist_to_box(shell_pts, Lx, Lz, H)
    comp, _ = cKDTree(shell_pts).query(gt, k=1, workers=-1)
    return {"accuracy_m": float(acc.mean()), "completeness_m": float(comp.mean()),
            "chamfer_m": float(0.5 * (acc.mean() + comp.mean())),
            "coverage_10cm": float((comp < 0.10).mean()),
            "coverage_25cm": float((comp < 0.25).mean()),
            "shell_points": int(near.sum())}


def load_pts(run: Path, to_room) -> np.ndarray:
    p = torch.load(run / "params.pt", map_location="cpu")
    keep = torch.sigmoid(p["opacities"]) > 0.5
    return to_room(p["means"][keep].numpy().astype(np.float64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--comp", required=True, type=Path)
    ap.add_argument("--measurements", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--band", type=float, default=0.30, help="metres from a true wall to count as shell")
    args = ap.parse_args()

    sh = json.loads((args.base / "shell.json").read_text())
    sc = json.loads((args.base / "scale.json").read_text())
    tape = json.loads(args.measurements.read_text())
    s = sc["chosen_scale"]
    to_room = room_frame(sh, s)

    # Our room frame: x spans the shell's "length" axis, z its "width" axis. Assign the tape's
    # longer side to whichever of our axes came out longer.
    ours = sorted([("x", sc["dims_m"]["length"]), ("z", sc["dims_m"]["width"])], key=lambda a: a[1])
    tape_sorted = sorted([tape["room_length_m"], tape["room_width_m"]])
    dims = {ours[0][0]: tape_sorted[0], ours[1][0]: tape_sorted[1]}
    Lx, Lz, H = dims["x"], dims["z"], tape["ceiling_height_m"]
    gt = gt_box_samples(Lx, Lz, H)

    res = {"gt_box_m": {"x": Lx, "z": Lz, "height": H}, "band_m": args.band}
    for name, run in (("baseline", args.base), ("completed", args.comp)):
        res[name] = evaluate(load_pts(run, to_room), gt, Lx, Lz, H, args.band)

    got = sc["dims_m"]
    res["dimensions"] = {
        "ours_m": got, "tape_m": {"length": tape["room_length_m"], "width": tape["room_width_m"],
                                  "height": tape["ceiling_height_m"]},
        # Shell "length" runs along our x axis and "width" along z (see room_frame).
        "abs_error_m": {"x": abs(got["length"] - Lx), "z": abs(got["width"] - Lz),
                        "height": abs(got["height"] - H)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
