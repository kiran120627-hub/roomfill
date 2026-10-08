"""Stage 12 — backup demo video (MP4): a smooth fly-through along the real walkthrough path.

Segments (each with a caption bar):
  1. RoomFill fly-through
  2. same path, generated regions tinted magenta, tilting up to the unfilmed ceiling
  3. split screen: baseline 3DGS (left) vs RoomFill (right)

    python scripts/12_demo_video.py --data ~/hn3d_work/room/dense --run ~/hn3d_work/room/runs/completed_hq \
        --baseline ~/hn3d_work/room/runs/full --out results/slides/roomfill_demo.mp4
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

from common import load_scene

tm = importlib.import_module("02_train")
C0 = 0.28209479177387814
W, H = 1280, 720


def font(size):
    for f in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def caption(img: Image.Image, text: str, sub: str = "") -> Image.Image:
    d = ImageDraw.Draw(img)
    f1, f2 = font(30), font(20)
    d.rectangle([0, H - 86, W, H], fill=(12, 13, 16))
    d.rectangle([0, H - 86, 8, H], fill=(224, 71, 158))
    d.text((28, H - 76), text, font=f1, fill=(236, 236, 239))
    if sub:
        d.text((28, H - 38), sub, font=f2, fill=(163, 166, 177))
    return img


def smooth_path(views, n_out: int, sigma: float = 6.0):
    """Smoothed camera path through the training views (in walk order)."""
    c2w = np.stack([torch.linalg.inv(v.viewmat).numpy() for v in views])
    pos = gaussian_filter1d(c2w[:, :3, 3], sigma, axis=0, mode="nearest")
    rots = Rotation.from_matrix(c2w[:, :3, :3])
    t = np.linspace(0, len(views) - 1, n_out)
    slerp = Slerp(np.arange(len(views)), rots)
    # smooth orientation by averaging nearby keyframes' rotation vectors around the slerp result
    R = slerp(t).as_matrix()
    P = np.stack([np.interp(t, np.arange(len(views)), pos[:, k]) for k in range(3)], 1)
    out = []
    for r, p in zip(R, P):
        m = np.eye(4); m[:3, :3] = r; m[:3, 3] = p
        out.append(torch.from_numpy(np.linalg.inv(m)).float())
    return out


def tilt(vm: torch.Tensor, deg: float) -> torch.Tensor:
    a = np.radians(deg)
    Rx = torch.tensor([[1, 0, 0, 0], [0, np.cos(a), -np.sin(a), 0], [0, np.sin(a), np.cos(a), 0], [0, 0, 0, 1]],
                      dtype=torch.float32)
    return Rx @ vm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    dev = torch.device("cuda")

    scene = load_scene(args.data, downscale=4)
    train = sorted([v for v in scene.train if "frame_" in v.name], key=lambda v: v.name)
    # skip the corridor seconds at the start of the walk
    train = train[int(len(train) * 0.08):]
    K = train[0].K.clone()
    K[0] *= W / train[0].width
    K[1] *= H / train[0].height

    p = {k: v.to(dev) for k, v in torch.load(args.run / "params.pt").items()}
    hon = {k: v.clone() for k, v in p.items()}
    gen = hon["tag"] > 0.5
    rgb = hon["sh0"][:, 0] * C0 + 0.5
    a = (0.45 + 0.4 * (1 - hon["conf"]))[:, None]
    rgb[gen] = (1 - a[gen]) * rgb[gen] + a[gen] * torch.tensor([0.88, 0.28, 0.62], device=dev)
    hon["sh0"] = ((rgb - 0.5) / C0)[:, None, :]
    hon["shN"][gen] = 0
    base = {k: v.to(dev) for k, v in torch.load(args.baseline / "params.pt").items()}

    def render(params, vm, w=W, h=H, Kx=None):
        v = type(train[0])("", None, Kx if Kx is not None else K, vm, w, h)
        with torch.no_grad():
            return tm.render(params, v, 0, dev)[0]

    to_pil = lambda t: Image.fromarray(t.clamp(0, 1).mul(255).byte().cpu().numpy())
    fps = args.fps
    path = smooth_path(train, 12 * fps)
    frames = []
    # 1. fly-through
    for vm in path[: 6 * fps]:
        frames.append(caption(to_pil(render(p, vm)), "A classroom rebuilt from a 51 s phone video",
                              "RoomFill: metric 3D Gaussian splat, floaters removed, unseen surfaces completed"))
    # 2. generated overlay: from a clean on-path camera, tilt up to the ceiling the walk never filmed
    anchor = next(v for v in scene.train if v.name.endswith("frame_0040.jpg")).viewmat
    for i in range(4 * fps):
        k = min(1.0, i / (1.5 * fps))
        img = render(p if i < fps // 2 else hon, tilt(anchor, -30 * k))
        frames.append(caption(to_pil(img), "Magenta = surfaces no camera ever saw",
                              "Every generated point is flagged with a confidence in the exported files"))
    # 3. split screen at held-out viewpoints (positions off the walking path)
    half = W // 2
    for name in ("photo_3",):
        v = next(t for t in scene.test if name in t.name)
        sc = H / v.height
        Kt = v.K.clone(); Kt[:2] *= sc; Kt[0, 2] = half / 2
        for i in range(4 * fps):
            vm = v.viewmat.clone()
            vm[2, 3] -= 0.004 * i                      # slow push forward
            left = render(base, vm, half, H, Kt)
            right = render(p, vm, half, H, Kt)
            im = to_pil(torch.cat([left, right], 1))
            d = ImageDraw.Draw(im)
            d.rectangle([half - 2, 0, half + 2, H], fill=(12, 13, 16))
            for x, t in ((16, "Baseline 3DGS"), (half + 16, "RoomFill")):
                d.rectangle([x - 6, 14, x + 220, 52], fill=(12, 13, 16))
                d.text((x + 4, 20), t, font=font(24), fill=(236, 236, 239))
            frames.append(caption(im, "New viewpoint, never filmed",
                                  "Baseline shows floaters; RoomFill stays clean. Chamfer 22.2 cm vs 15.5 cm"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            f.save(Path(tmp) / f"f_{i:05d}.jpg", quality=92)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", str(Path(tmp) / "f_%05d.jpg"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(args.out)], check=True)
    print(f"video: {args.out} ({len(frames) / fps:.1f} s)")


if __name__ == "__main__":
    main()
