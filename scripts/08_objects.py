"""Stage 8 — object-level reconstruction: split the furniture out of the room.

Everything that is confidently observed, inside the room, and not part of the shell
(floor / walls / ceiling) is voxelised in the metric room frame; connected components of
the occupancy grid are the objects (desk+bench units, podium, cupboard, ...).
For each object we export its own Gaussian splat and an oriented bounding box.

Usage:
    python scripts/08_objects.py --run <completed run> --shell-run <baseline run>
Produces in <run>/export/objects/:
    obj_XXX.ply            per-object Gaussian splat (metric, room frame)
    objects_boxes.glb      all oriented boxes, one colour per object
    objects.json           id, centre, size, yaw, height range, point count, size-based label
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from common import save_ply

exp = importlib.import_module("06_export")


def label_for(size: np.ndarray, zmax: float) -> str:
    a, b = sorted(size[:2])[::-1]         # footprint long / short side (m)
    h = size[2]
    if 0.9 < a < 2.2 and 0.25 < b < 0.9 and 0.5 < zmax < 1.1:
        return "desk/bench"
    if a < 1.2 and b < 1.0 and 0.9 < zmax < 1.5:
        return "podium/lectern"
    if h > 1.4 and b < 0.8:
        return "cupboard/shelf"
    if zmax > 2.4 and h < 0.6:
        return "ceiling fixture (fan/light)"
    return "object"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--shell-run", required=True, type=Path)
    ap.add_argument("--voxel", type=float, default=0.05, help="metres")
    ap.add_argument("--wall-margin", type=float, default=0.12)
    ap.add_argument("--min-points", type=int, default=400)
    args = ap.parse_args()

    sh = json.loads((args.shell_run / "shell.json").read_text())
    sc = json.loads((args.shell_run / "scale.json").read_text())
    s = sc["chosen_scale"]
    up, ex, ey = (np.array(sh[k]) for k in ("up", "ex", "ey"))
    R = np.stack([ex, up, -ey])
    t = np.mean(sh["x"]) * ex + np.mean(sh["y"]) * ey + sh["floor_offset"] * up

    p = torch.load(args.run / "params.pt", map_location="cpu")
    pm = exp.transform_params(p, R, t, s)                      # metric room frame, Y up
    xyz = pm["means"].numpy()
    L, W, H = (sc["dims_m"][k] for k in ("length", "width", "height"))
    m = args.wall_margin
    keep = ((torch.sigmoid(pm["opacities"]).numpy() > 0.5) & (pm["tag"].numpy() < 0.5)
            & (np.abs(xyz[:, 0]) < L / 2 - m) & (np.abs(xyz[:, 2]) < W / 2 - m)
            & (xyz[:, 1] > 0.04) & (xyz[:, 1] < H - 0.08))
    idx = np.nonzero(keep)[0]
    P = xyz[idx]

    v = args.voxel
    lo = P.min(0)
    ijk = np.floor((P - lo) / v).astype(int)
    grid = np.zeros(ijk.max(0) + 1, bool)
    grid[tuple(ijk.T)] = True
    grid = ndimage.binary_closing(grid, iterations=1)
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    point_lab = lab[tuple(ijk.T)]

    out = args.run / "export" / "objects"
    out.mkdir(parents=True, exist_ok=True)
    import trimesh
    boxes, objs = [], []
    rng = np.random.default_rng(0)
    for k in range(1, n + 1):
        sel = idx[point_lab == k]
        if len(sel) < args.min_points:
            continue
        q = xyz[sel]
        # Oriented box: yaw from PCA of the floor-plane footprint, robust extents (1-99 pct).
        xz = q[:, [0, 2]] - q[:, [0, 2]].mean(0)
        evals, evecs = np.linalg.eigh(np.cov(xz.T))
        major = evecs[:, -1]
        yaw = float(np.arctan2(major[1], major[0]))
        c, s_ = np.cos(yaw), np.sin(yaw)
        local = xz @ np.array([[c, -s_], [s_, c]])
        lo2, hi2 = np.percentile(local, [1, 99], axis=0)
        ylo, yhi = np.percentile(q[:, 1], [1, 99])
        size = np.array([hi2[0] - lo2[0], hi2[1] - lo2[1], yhi - ylo])
        ctr_local = (lo2 + hi2) / 2
        ctr_xz = q[:, [0, 2]].mean(0) + ctr_local @ np.array([[c, s_], [-s_, c]])
        oid = len(objs)
        sub = {kk: vv[sel] for kk, vv in pm.items()}
        save_ply(out / f"obj_{oid:03d}.ply", sub)
        T = np.eye(4)
        T[:3, :3] = np.array([[c, 0, -s_], [0, 1, 0], [s_, 0, c]])
        T[:3, 3] = [ctr_xz[0], (ylo + yhi) / 2, ctr_xz[1]]
        box = trimesh.creation.box(extents=[size[0], size[2], size[1]], transform=T)
        box.visual.face_colors = np.r_[rng.integers(60, 255, 3), 140]
        boxes.append(box)
        objs.append({"id": oid, "label": label_for(size, float(yhi)), "points": int(len(sel)),
                     "center_m": [round(float(ctr_xz[0]), 3), round(float((ylo + yhi) / 2), 3), round(float(ctr_xz[1]), 3)],
                     "size_m": {"long": round(float(size[0]), 3), "short": round(float(size[1]), 3),
                                "height": round(float(size[2]), 3)},
                     "top_m": round(float(yhi), 3), "yaw_deg": round(float(np.degrees(yaw)), 1)})

    if boxes:
        trimesh.Scene(boxes).export(out / "objects_boxes.glb")
    counts = {}
    for o in objs:
        counts[o["label"]] = counts.get(o["label"], 0) + 1
    summary = {"objects": len(objs), "by_label": counts, "voxel_m": v, "items": objs}
    (out / "objects.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"objects": len(objs), "by_label": counts}, indent=2))


if __name__ == "__main__":
    main()
