"""Stage 3 — fit the room shell (floor, ceiling, 4 Manhattan walls) to the reconstruction.

Works in SfM units; metric scale is applied by 03b_scale.py. Also builds a per-face
coverage grid: which parts of each wall / floor / ceiling were actually observed.
Unobserved cells are the targets for generative completion (stage 4).

Usage:
    python scripts/03_shell.py --run ~/hn3d_work/room/runs/baseline --data ~/hn3d_work/room/dense
Produces:
    <run>/shell.json, <run>/coverage.npz, <run>/shell_topdown.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import load_scene

rng = np.random.default_rng(0)


def load_points(run: Path, max_pts: int = 400_000) -> np.ndarray:
    p = torch.load(run / "params.pt", map_location="cpu")
    keep = (torch.sigmoid(p["opacities"]) > 0.5) & (p["tag"] < 0.5)   # confident, observed only
    pts = p["means"][keep].numpy()
    if len(pts) > max_pts:
        pts = pts[rng.choice(len(pts), max_pts, replace=False)]
    # Drop far outliers (sky through windows, floaters).
    c = np.median(pts, 0)
    d = np.linalg.norm(pts - c, axis=1)
    return pts[d < np.percentile(d, 98)]


def camera_centers_and_up(scene) -> tuple[np.ndarray, np.ndarray]:
    cs, ups = [], []
    for v in scene.train + scene.test:
        c2w = torch.linalg.inv(v.viewmat).numpy()
        cs.append(c2w[:3, 3])
        ups.append(-c2w[:3, 1])          # COLMAP camera y points down
    up = np.mean(ups, 0)
    return np.array(cs), up / np.linalg.norm(up)


def ransac_floor(pts: np.ndarray, up: np.ndarray, cams: np.ndarray, iters: int = 3000):
    h = pts @ up
    below = pts[h < np.median(cams @ up)]
    extent = np.ptp(h)
    thr = 0.008 * extent
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = below[rng.choice(len(below), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n @ up) < np.cos(np.radians(20)):
            continue
        n = n if n @ up > 0 else -n
        inl = np.abs((below - a) @ n) < thr
        if inl.sum() > best_n:
            best_n, best = inl.sum(), (n, a)
    n, a = best
    # Least-squares refine on inliers.
    inl = below[np.abs((below - a) @ n) < thr]
    cen = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - cen, full_matrices=False)
    n2 = vt[-1] if vt[-1] @ up > 0 else -vt[-1]
    return n2, float(cen @ n2), int(len(inl))


def lowest_layer(h: np.ndarray, extent: float, frac: float = 0.25) -> float:
    """Height of the LOWEST strong horizontal layer. Rows of desks form a bigger plane than the
    partly-hidden floor, so 'largest plane' picks desk height; the floor is the lowest strong peak."""
    bw = extent / 250
    cnt, ed = np.histogram(h, np.arange(h.min(), h.max() + bw, bw))
    cnt = np.convolve(cnt, [1, 2, 1], "same")
    strong = np.nonzero(cnt >= frac * cnt.max())[0]
    k = strong.min()
    while k + 1 < len(cnt) and cnt[k + 1] > cnt[k]:      # climb to the local peak
        k += 1
    return float(0.5 * (ed[k] + ed[k + 1]))


def manhattan_angle(xy: np.ndarray, extent: float) -> float:
    """Rotation that makes wall points pile up into sharp axis-aligned histogram peaks."""
    bins = np.linspace(-extent, extent, 400)
    best, best_s = 0.0, -1
    for th in np.radians(np.arange(0, 90, 0.5)):
        c, s = np.cos(th), np.sin(th)
        r = xy @ np.array([[c, -s], [s, c]])
        sc = sum((np.histogram(r[:, k], bins)[0].astype(np.float64) ** 2).sum() for k in (0, 1))
        if sc > best_s:
            best, best_s = th, sc
    return float(best)


def find_walls(coord: np.ndarray, bin_w: float) -> tuple[float, float]:
    """Outermost strong histogram peak on each side = wall position."""
    lo, hi = np.percentile(coord, [0.5, 99.5])
    edges = np.arange(lo - bin_w, hi + 2 * bin_w, bin_w)
    cnt, _ = np.histogram(coord, edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    q1, q3 = np.percentile(coord, [25, 75])
    lo_idx, hi_idx = np.where(mid < q1)[0], np.where(mid > q3)[0]
    lo_strong = lo_idx[cnt[lo_idx] >= 0.35 * cnt[lo_idx].max()]
    hi_strong = hi_idx[cnt[hi_idx] >= 0.35 * cnt[hi_idx].max()]
    return float(mid[lo_strong.min()]), float(mid[hi_strong.max()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--cell", type=float, default=0.02, help="coverage cell size as a fraction of room size")
    args = ap.parse_args()

    scene = load_scene(args.data)
    pts = load_points(args.run)
    cams, up0 = camera_centers_and_up(scene)

    # RANSAC gives a reliable "up" (any big horizontal plane is parallel to the floor) ...
    up, _, floor_inl = ransac_floor(pts, up0, cams)
    # ... but the floor offset is the lowest strong layer below the cameras, not the biggest plane.
    hh = pts @ up
    below_cam = hh[hh < np.median(cams @ up)]
    floor_d = lowest_layer(below_cam, float(np.ptp(hh)))
    e1 = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    h = pts @ up - floor_d
    xy = np.stack([pts @ e1, pts @ e2], 1)
    extent = float(np.percentile(np.linalg.norm(xy - np.median(xy, 0), axis=1), 99))

    th = manhattan_angle(xy - np.median(xy, 0), extent)
    c, s = np.cos(th), np.sin(th)
    R2 = np.array([[c, -s], [s, c]])
    ex, ey = R2.T @ np.stack([e1, e2])          # rotated horizontal axes (world vectors)
    X, Y = pts @ ex, pts @ ey

    cam_h = cams @ up - floor_d
    band = (h > 0.15 * cam_h.mean()) & (h < 1.6 * cam_h.mean())   # skip floor/ceiling points
    bin_w = extent / 150
    x0, x1 = find_walls(X[band], bin_w)
    y0, y1 = find_walls(Y[band], bin_w)

    # Ceiling: a strong horizontal peak above the cameras, if one was observed.
    # Ceiling: the HIGHEST strong horizontal layer above the cameras (fans, tube lights and beams
    # hang below it and would win a "largest peak" vote).
    above = h[h > cam_h.max()]
    ceiling_h, ceiling_observed = None, False
    if len(above) > 0.01 * len(h):
        ceiling_h = -lowest_layer(-above, float(np.ptp(h)), frac=0.08)   # true slab is sparse (plain white); beams sit lower
        ceiling_observed = True
    if ceiling_h is None:
        ceiling_h = float(np.percentile(h, 99.5))   # upper bound from what was seen; refined by scale step

    L, W, H = x1 - x0, y1 - y0, ceiling_h
    cell = args.cell * max(L, W)
    tol = 2.5 * bin_w

    # Coverage grids. Face frames: (u, v) along the face, both starting at the face's corner.
    faces = {
        "floor":   (np.abs(h) < tol,              X - x0, Y - y0, L, W),
        "ceiling": (np.abs(h - H) < tol,          X - x0, Y - y0, L, W),
        "wall_x0": (np.abs(X - x0) < tol,         Y - y0, h,      W, H),
        "wall_x1": (np.abs(X - x1) < tol,         Y - y0, h,      W, H),
        "wall_y0": (np.abs(Y - y0) < tol,         X - x0, h,      L, H),
        "wall_y1": (np.abs(Y - y1) < tol,         X - x0, h,      L, H),
    }
    cov, summary = {}, {}
    for name, (m, u, v, U, V) in faces.items():
        nu, nv = max(1, int(np.ceil(U / cell))), max(1, int(np.ceil(V / cell)))
        g = np.zeros((nv, nu), np.int32)
        iu = np.clip((u[m] / cell).astype(int), 0, nu - 1)
        iv = np.clip((v[m] / cell).astype(int), 0, nv - 1)
        np.add.at(g, (iv, iu), 1)
        observed = g >= 3
        cov[name] = observed
        summary[name] = round(float(observed.mean()), 3)

    shell = {
        "units": "sfm",
        "up": up.tolist(), "ex": ex.tolist(), "ey": ey.tolist(), "floor_offset": floor_d,
        "x": [x0, x1], "y": [y0, y1], "ceiling_h": ceiling_h, "ceiling_observed": ceiling_observed,
        "dims_sfm": {"length": L, "width": W, "height": H},
        "camera_height_sfm": float(np.median(cam_h)),
        "cell": cell, "tol": tol, "floor_inliers": floor_inl,
        "coverage_fraction": summary,
    }
    (args.run / "shell.json").write_text(json.dumps(shell, indent=2))
    np.savez_compressed(args.run / "coverage.npz", **cov)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    sub = rng.choice(len(X), min(60000, len(X)), replace=False)
    ax.scatter(X[sub], Y[sub], s=0.2, c=h[sub], cmap="viridis")
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "r-", lw=2)
    ax.scatter(cams @ ex, cams @ ey, s=4, c="k")
    ax.set_aspect("equal")
    ax.set_title(f"L={L:.3f} W={W:.3f} H={H:.3f} (sfm units)  ceiling_observed={ceiling_observed}")
    fig.savefig(args.run / "shell_topdown.png", dpi=110, bbox_inches="tight")
    print(json.dumps({"dims_sfm": shell["dims_sfm"], "coverage": summary,
                      "ceiling_observed": ceiling_observed}, indent=2))


if __name__ == "__main__":
    main()
