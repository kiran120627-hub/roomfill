"""Carry shell.json / scale.json from one COLMAP frame to another (e.g. after registering extra
frames and re-running bundle adjustment, which moves the gauge slightly).

A similarity transform P' = k R P + t is fitted (Umeyama) between the camera centres both models
share, then every shell quantity is mapped through it:
  axis vectors  ex, ey, up        -> R @ v
  plane offsets along an axis     -> k * offset + t . (R v)
  lengths (dims, heights, cell)   -> k * length
  metres per unit (scale.json)    -> scale / k

Usage:
    python scripts/transfer_shell.py --src-model <old sparse> --dst-model <new sparse> \
        --src-run <run with shell.json, scale.json> --dst-run <new run>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap


def centres(model: Path) -> dict:
    r = pycolmap.Reconstruction(str(model))
    out = {}
    for im in r.images.values():
        p = im.cam_from_world() if callable(getattr(im, "cam_from_world", None)) else im.cam_from_world
        M = np.asarray(p.matrix())
        out[im.name] = -M[:3, :3].T @ M[:3, 3]
    return out


def umeyama(src: np.ndarray, dst: np.ndarray):
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    k = np.trace(np.diag(S) @ D) / (xs ** 2).sum(1).mean()
    t = md - k * R @ ms
    return k, R, t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-model", required=True, type=Path)
    ap.add_argument("--dst-model", required=True, type=Path)
    ap.add_argument("--src-run", required=True, type=Path)
    ap.add_argument("--dst-run", required=True, type=Path)
    args = ap.parse_args()

    a, b = centres(args.src_model), centres(args.dst_model)
    common = sorted(k for k in a if k in b)
    P, Q = np.stack([a[k] for k in common]), np.stack([b[k] for k in common])
    k, R, t = umeyama(P, Q)
    resid = np.linalg.norm((k * (R @ P.T)).T + t - Q, axis=1)
    print(f"common cameras {len(common)}  scale {k:.5f}  residual median {np.median(resid):.4f} max {resid.max():.4f}")

    sh = json.loads((args.src_run / "shell.json").read_text())
    up, ex, ey = (R @ np.array(sh[n]) for n in ("up", "ex", "ey"))
    off = lambda val, v: float(k * val + t @ v)
    out = dict(sh)
    out.update({
        "up": up.tolist(), "ex": ex.tolist(), "ey": ey.tolist(),
        "floor_offset": off(sh["floor_offset"], up),
        "x": [off(x, ex) for x in sh["x"]], "y": [off(y, ey) for y in sh["y"]],
        "ceiling_h": k * sh["ceiling_h"], "camera_height_sfm": k * sh["camera_height_sfm"],
        "cell": k * sh["cell"], "tol": k * sh["tol"],
        "dims_sfm": {n: k * v for n, v in sh["dims_sfm"].items()},
        "transferred_from": str(args.src_run), "transfer_similarity": {"scale": k, "residual_median": float(np.median(resid))},
    })
    args.dst_run.mkdir(parents=True, exist_ok=True)
    (args.dst_run / "shell.json").write_text(json.dumps(out, indent=2))

    sc = json.loads((args.src_run / "scale.json").read_text())
    sc["chosen_scale"] = sc["chosen_scale"] / k
    for e in sc["estimators"].values():
        e["scale"] = e["scale"] / k
    (args.dst_run / "scale.json").write_text(json.dumps(sc, indent=2))
    print("shell + scale written to", args.dst_run)


if __name__ == "__main__":
    main()
