"""Stage 3b — recover metric scale automatically (judges' rooms come without our tape).

Three independent estimators, reported side by side (this is an ablation in itself):
  1. depth   — monocular metric depth (Depth Anything V2 Metric Indoor) vs the splat's rendered
               depth, median ratio over confident pixels in sampled training views.  [primary]
  2. camera  — handheld-capture prior: median camera height above the fitted floor ≈ 1.35 m.
  3. tape    — data/measurements.json, if present (ground truth for our own room only).

Usage:
    python scripts/03b_scale.py --run <run> --data <dense> [--measurements data/measurements.json]
Produces <run>/scale.json with the chosen scale and room dimensions in metres.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from PIL import Image

from common import load_scene

CAMERA_HEIGHT_PRIOR_M = 1.40   # middle of the typical 1.3-1.5 m handheld walkthrough range (not tuned to our tape)
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


@torch.no_grad()
def render_depth(p: dict, v, device) -> tuple[torch.Tensor, torch.Tensor]:
    out, alpha, _ = rasterization(
        means=p["means"], quats=p["quats"], scales=torch.exp(p["scales"]),
        opacities=torch.sigmoid(p["opacities"]), colors=torch.cat([p["sh0"], p["shN"]], 1),
        viewmats=v.viewmat[None].to(device), Ks=v.K[None].to(device),
        width=v.width, height=v.height, sh_degree=3, render_mode="RGB+ED")
    return out[0, ..., 3], alpha[0, ..., 0]


def depth_scale(p: dict, scene, device, n_views: int, model: str) -> dict:
    from transformers import pipeline
    pipe = pipeline("depth-estimation", model=model, device=0)
    idx = np.linspace(0, len(scene.train) - 1, min(n_views, len(scene.train))).astype(int)
    per_view = []
    for i in idx:
        v = scene.train[i]
        d_sfm, alpha = render_depth(p, v, device)
        pred = pipe(Image.fromarray(v.image.numpy()))["predicted_depth"]
        pred = pred.to(device).float().reshape(1, 1, *pred.shape[-2:])
        d_m = F.interpolate(pred, size=(v.height, v.width), mode="bilinear", align_corners=False)[0, 0]
        ok = (alpha > 0.95) & (d_sfm > 1e-4) & (d_m > 0.2) & (d_m < 15)
        if ok.sum() < 1000:
            continue
        per_view.append(float(torch.median(d_m[ok] / d_sfm[ok])))
    s = np.array(per_view)
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med)))
    return {"scale": med, "rel_spread": mad / med, "views_used": len(s)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--measurements", type=Path, default=None)
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--model", default=DEPTH_MODEL)
    ap.add_argument("--plan-layout", type=Path, default=None,
                    help="layout.json from 10_floorplan.py; enables blueprint-guided scale (Mode A + B)")
    args = ap.parse_args()

    device = torch.device("cuda")
    shell = json.loads((args.run / "shell.json").read_text())
    dims = shell["dims_sfm"]
    p = {k: v.to(device) for k, v in torch.load(args.run / "params.pt").items()}
    scene = load_scene(args.data)

    est = {"depth": depth_scale(p, scene, device, args.views, args.model),
           "camera": {"scale": CAMERA_HEIGHT_PRIOR_M / shell["camera_height_sfm"]}}

    # Blueprint-guided: match the fitted shell's two sides to the plan's length / breadth.
    plan = None
    if args.plan_layout and args.plan_layout.exists():
        plan = json.loads(args.plan_layout.read_text())["room"]
        sfm_lw = sorted([dims["length"], dims["width"]])
        plan_lw = sorted([plan["length"], plan["breadth"]])
        ratios = [t / s for t, s in zip(plan_lw, sfm_lw)]
        est["floorplan"] = {"scale": float(np.mean(ratios)),
                            "side_disagreement": float(abs(ratios[0] / ratios[1] - 1))}

    tape = None
    if args.measurements and args.measurements.exists():
        tape = json.loads(args.measurements.read_text())
        # Room length/width are matched order-independently (we don't know which wall is "length").
        sfm_lw = sorted([dims["length"], dims["width"]])
        tape_lw = sorted([tape["room_length_m"], tape["room_width_m"]])
        est["tape"] = {"scale": float(np.mean([t / s for t, s in zip(tape_lw, sfm_lw)]))}

    chosen = est["depth"]["scale"]
    # Sanity gate: if the depth estimate is wildly inconsistent with the handheld prior, distrust it.
    ratio = chosen / est["camera"]["scale"]
    if not 0.6 < ratio < 1.6 or est["depth"]["views_used"] < 5:
        print(f"WARNING: depth scale disagrees with camera prior (ratio {ratio:.2f}); using camera prior")
        chosen = est["camera"]["scale"]

    # Video-only answer is kept for the ablation; with a plan, the plan's scale wins.
    video_only = chosen
    if "floorplan" in est:
        chosen = est["floorplan"]["scale"]

    def metres(s: float) -> dict:
        return {k: round(v * s, 3) for k, v in dims.items()}

    report = {"chosen_scale": chosen, "video_only_scale": video_only,
              "chosen_from": "floorplan" if "floorplan" in est else "video", "estimators": est,
              "dims_m": metres(chosen),
              "dims_m_by_estimator": {k: metres(e["scale"]) for k, e in est.items()},
              "camera_height_m": round(shell["camera_height_sfm"] * chosen, 3),
              "ceiling_observed": shell["ceiling_observed"]}
    if tape:
        got = sorted([report["dims_m"]["length"], report["dims_m"]["width"]])
        want = sorted([tape["room_length_m"], tape["room_width_m"]])
        report["tape"] = tape
        report["dim_error_pct"] = {
            "length_width": [round(100 * abs(g - w) / w, 2) for g, w in zip(got, want)]}
        if "ceiling_height_m" in tape:
            report["dim_error_pct"]["height"] = round(
                100 * abs(report["dims_m"]["height"] - tape["ceiling_height_m"]) / tape["ceiling_height_m"], 2)
        for k, e in est.items():
            g = sorted([dims["length"] * e["scale"], dims["width"] * e["scale"]])
            report.setdefault("dim_error_pct_by_estimator", {})[k] = [
                round(100 * abs(a - b) / b, 2) for a, b in zip(g, want)]

    (args.run / "scale.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
