"""Re-score saved runs on the held-out views (raw + colour-aligned metrics) at one resolution.

Usage:
    python scripts/09_eval.py --data ~/hn3d_work/room/dense --downscale 2 \
        --runs ~/hn3d_work/room/runs/baseline ~/hn3d_work/room/runs/depthprior_b
Writes <run>/eval.json and prints a comparison table.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch

from common import load_scene

train_mod = importlib.import_module("02_train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    args = ap.parse_args()

    device = torch.device("cuda")
    scene = load_scene(args.data, downscale=args.downscale)
    print(f"{'run':24s} {'PSNR':>6s} {'SSIM':>6s} {'LPIPS':>6s} | {'PSNRcc':>6s} {'SSIMcc':>6s} {'LPIPScc':>7s}  n")
    for run in args.runs:
        p = {k: v.to(device) for k, v in torch.load(run / "params.pt").items()}
        m = train_mod.evaluate(p, scene.test, device, run / "eval_renders")
        m["downscale"] = args.downscale
        (run / "eval.json").write_text(json.dumps(m, indent=2))
        a = m["mean"]
        print(f"{run.name:24s} {a['psnr']:6.2f} {a['ssim']:6.3f} {a['lpips']:6.3f} | "
              f"{a['psnr_cc']:6.2f} {a['ssim_cc']:6.3f} {a['lpips_cc']:7.3f}  {len(p['means']):,}")


if __name__ == "__main__":
    main()
