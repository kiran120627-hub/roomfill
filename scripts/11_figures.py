"""Figures for the slides / write-up, rendered from the real outputs.

    python scripts/11_figures.py --work ~/hn3d_work/room --out results/slides
"""
from __future__ import annotations

import argparse
import importlib
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from common import load_scene

tm = importlib.import_module("02_train")
C0 = 0.28209479177387814
MAGENTA = torch.tensor([0.88, 0.28, 0.62])


def label(img: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("DejaVuSans-Bold.ttf", max(16, img.height // 22))
    except OSError:
        f = ImageFont.load_default()
    pad = img.height // 60
    bbox = d.textbbox((0, 0), text, font=f)
    d.rectangle([pad, pad, pad * 3 + bbox[2], pad * 3 + bbox[3]], fill=(12, 13, 16))
    d.text((pad * 2, pad * 2), text, font=f, fill=(236, 236, 239))
    return img


def to_img(t: torch.Tensor) -> Image.Image:
    return Image.fromarray(t.clamp(0, 1).mul(255).byte().cpu().numpy())


def row(ims: list[Image.Image], gap: int = 8) -> Image.Image:
    h = min(i.height for i in ims)
    ims = [i.resize((round(i.width * h / i.height), h)) for i in ims]
    out = Image.new("RGB", (sum(i.width for i in ims) + gap * (len(ims) - 1), h), (12, 13, 16))
    x = 0
    for i in ims:
        out.paste(i, (x, 0))
        x += i.width + gap
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    runs = args.work / "runs"
    scene = load_scene(args.work / "dense", downscale=2)
    load = lambda r: {k: v.to(dev) for k, v in torch.load(runs / r / "params.pt").items()}

    # 1. Floaters: held-out photo vs baseline vs depth prior (the worst off-path view).
    base, dp = load("baseline"), (load("depthprior") if (runs / "depthprior").exists() else None)
    for v in scene.test:
        if "photo_4" not in v.name:
            continue
        ims = [label(to_img(v.image.float().to(dev) / 255), "Held-out photo")]
        ims.append(label(to_img(tm.colour_align(tm.render(base, v, 3, dev)[0], v.image.float().to(dev) / 255)), "Baseline 3DGS"))
        if dp:
            ims.append(label(to_img(tm.colour_align(tm.render(dp, v, 3, dev)[0], v.image.float().to(dev) / 255)), "+ depth prior"))
        row(ims).save(args.out / "1_floaters_heldout.jpg", quality=92)

    # 2. Honesty: same view, normal vs generated tinted magenta.
    comp = load("completed")
    hon = {k: t.clone() for k, t in comp.items()}
    gen = hon["tag"] > 0.5
    rgb = hon["sh0"][:, 0] * C0 + 0.5
    a = (0.45 + 0.4 * (1 - hon["conf"]))[:, None]
    rgb[gen] = (1 - a[gen]) * rgb[gen] + a[gen] * MAGENTA.to(dev)
    hon["sh0"] = ((rgb - 0.5) / C0)[:, None, :]
    hon["shN"][gen] = 0
    def tilt_up(v, deg: float):
        """Same camera pitched (negative = up, OpenCV y points down): shows ceiling / upper walls."""
        a = np.radians(deg)
        Rx = torch.tensor([[1, 0, 0, 0], [0, np.cos(a), -np.sin(a), 0], [0, np.sin(a), np.cos(a), 0], [0, 0, 0, 1]],
                          dtype=torch.float32)
        return type(v)(v.name, v.image, v.K, Rx @ v.viewmat, v.width, v.height)

    for i, v in enumerate([tilt_up(scene.train[60], -35), tilt_up(scene.train[200], -35), tilt_up(scene.train[330], -35)]):
        row([label(to_img(tm.render(comp, v, 3, dev)[0]), "RoomFill"),
             label(to_img(tm.render(hon, v, 3, dev)[0]), "Generated in magenta")]).save(
            args.out / f"2_honesty_view{i + 1}.jpg", quality=92)

    # 3-5. Copies of existing diagnostic figures.
    faces = runs / "completed" / "faces"
    tiles = [Image.open(faces / f"{n}.png") for n in ("wall_x1", "wall_y0", "ceiling") if (faces / f"{n}.png").exists()]
    if tiles:
        w = max(t.width for t in tiles)
        canvas = Image.new("RGB", (w, sum(t.height for t in tiles) + 10 * len(tiles)), (12, 13, 16))
        y = 0
        for t in tiles:
            canvas.paste(t, (0, y))
            y += t.height + 10
        canvas.save(args.out / "3_completion_textures.jpg", quality=92)
    for src, dst in ((runs / "baseline" / "shell_topdown.png", "4_shell_topdown.png"),
                     (Path(__file__).resolve().parents[1] / "results" / "floorplan" / "floorplan_parsed.png",
                      "5_floorplan_parsed.png")):
        if src.exists():
            shutil.copy(src, args.out / dst)
    print("figures ->", args.out)


if __name__ == "__main__":
    main()
