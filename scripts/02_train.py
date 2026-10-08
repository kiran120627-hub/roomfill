"""Stage 2 — Gaussian-splat training (baseline, and the base for later stages).

Every Gaussian carries a `tag` (0 = observed, 1 = generated). The tag is stored as a
zero-learning-rate parameter so that gsplat's densify / prune keeps it aligned with
the other per-Gaussian tensors.

Usage:
    python scripts/02_train.py --data ~/hn3d_work/room/dense --out ~/hn3d_work/room/runs/baseline
    python scripts/02_train.py --data ~/hn3d_work/datasets/tandt_db/db/playroom --out .../runs/playroom_baseline
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

from common import load_scene, psnr, save_ply, ssim

SH_DEGREE = 3
C0 = 0.28209479177387814  # SH band-0 constant


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / C0


def knn_mean_dist(x: torch.Tensor, k: int = 3, chunk: int = 4096) -> torch.Tensor:
    out = []
    for i in range(0, x.shape[0], chunk):
        d = torch.cdist(x[i:i + chunk], x)
        out.append(d.topk(k + 1, largest=False).values[:, 1:].mean(1))
    return torch.cat(out)


def init_params(points: np.ndarray, colors: np.ndarray, scene_scale: float, device) -> tuple[dict, dict]:
    means = torch.from_numpy(points).float().to(device)
    rgb = torch.from_numpy(colors).float().to(device)
    n = means.shape[0]
    dist = knn_mean_dist(means).clamp_min(1e-7)
    params = {
        "means": means,
        "scales": torch.log(dist)[:, None].repeat(1, 3),
        "quats": F.normalize(torch.rand(n, 4, device=device), dim=-1),
        "opacities": torch.logit(torch.full((n,), 0.1, device=device)),
        "sh0": rgb_to_sh(rgb)[:, None, :],
        "shN": torch.zeros(n, (SH_DEGREE + 1) ** 2 - 1, 3, device=device),
        "tag": torch.zeros(n, device=device),
    }
    params = {k: torch.nn.Parameter(v) for k, v in params.items()}
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
           "sh0": 2.5e-3, "shN": 2.5e-3 / 20, "tag": 0.0}
    optims = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
              for k, lr in lrs.items()}
    return params, optims


def render(params: dict, view, sh_degree: int, device, bg=None, with_depth: bool = False):
    colors = torch.cat([params["sh0"], params["shN"]], 1)
    img, alpha, info = rasterization(
        means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]), colors=colors,
        viewmats=view.viewmat[None].to(device), Ks=view.K[None].to(device),
        width=view.width, height=view.height, sh_degree=sh_degree,
        backgrounds=bg, packed=True, render_mode="RGB+ED" if with_depth else "RGB",
    )
    if with_depth:
        return img[0, ..., :3].clamp(0, 1), alpha[0], info, img[0, ..., 3]
    return img[0].clamp(0, 1), alpha[0], info


def mono_depths(scene, data_dir: Path, downscale: int, device) -> dict[str, torch.Tensor]:
    """Monocular depth for every training view (Depth Anything V2), cached next to the data."""
    cache = data_dir / f"mono_depth_ds{downscale}.pt"
    if cache.exists():
        return torch.load(cache)
    from PIL import Image
    from transformers import pipeline
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", device=0)
    out = {}
    for v in scene.train:
        d = pipe(Image.fromarray(v.image.numpy()))["predicted_depth"].float()
        d = d.reshape(1, 1, *d.shape[-2:])
        d = F.interpolate(d, size=(v.height, v.width), mode="bilinear", align_corners=False)[0, 0]
        out[v.name] = d.half().cpu()
    torch.save(out, cache)
    print(f"mono depth cached for {len(out)} views -> {cache}", flush=True)
    return out


def pearson_depth_loss(pred: torch.Tensor, mono: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """1 - Pearson correlation between rendered and monocular depth (scale/shift invariant)."""
    m = alpha[..., 0] > 0.5
    if m.sum() < 100:
        return pred.sum() * 0
    a, b = pred[m], mono[m]
    a = a - a.mean()
    b = b - b.mean()
    return 1 - (a * b).sum() / (a.norm() * b.norm() + 1e-8)


def colour_align(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Least-squares 3x4 affine colour map pred -> gt for the whole image. Held-out stills come
    from the phone's photo mode (different exposure / tone curve than video mode); this removes
    that global camera difference so the metric measures geometry and texture."""
    P = pred.reshape(-1, 3)
    X = torch.cat([P, torch.ones_like(P[:, :1])], 1)
    A = torch.linalg.lstsq(X, gt.reshape(-1, 3)).solution
    return (X @ A).reshape(pred.shape).clamp(0, 1)


@torch.no_grad()
def evaluate(params: dict, views, device, out_dir: Path | None = None) -> dict:
    import lpips
    lp = lpips.LPIPS(net="vgg", verbose=False).to(device)
    rows = []
    for v in views:
        pred, _, _ = render(params, v, SH_DEGREE, device)
        gt = v.image.to(device).float() / 255
        a, b = pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]
        pc = colour_align(pred, gt)
        c = pc.permute(2, 0, 1)[None]
        rows.append({"name": v.name, "psnr": psnr(pred, gt), "ssim": ssim(a, b).item(),
                     "lpips": lp(a * 2 - 1, b * 2 - 1).item(),
                     "psnr_cc": psnr(pc, gt), "ssim_cc": ssim(c, b).item(),
                     "lpips_cc": lp(c * 2 - 1, b * 2 - 1).item()})
        if out_dir is not None:
            from PIL import Image
            out_dir.mkdir(parents=True, exist_ok=True)
            side = torch.cat([gt, pred], 1).mul(255).byte().cpu().numpy()
            Image.fromarray(side).save(out_dir / (Path(v.name).stem + ".jpg"), quality=92)
    mean = {k: float(np.mean([r[k] for r in rows]))
            for k in ("psnr", "ssim", "lpips", "psnr_cc", "ssim_cc", "lpips_cc")}
    return {"mean": mean, "per_view": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--downscale", type=int, default=1)
    ap.add_argument("--max-gaussians", type=int, default=2_500_000, help="stop densifying past this (8 GB VRAM)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth-prior", type=float, default=0.0,
                    help="weight of the monocular-depth Pearson loss (0 = vanilla 3DGS baseline)")
    ap.add_argument("--opacity-reg", type=float, default=0.0,
                    help="L1 weight on opacities; suppresses semi-transparent floaters")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)

    scene = load_scene(args.data, args.downscale)
    print(f"train={len(scene.train)} test={len(scene.test)} points={len(scene.points)} "
          f"scene_scale={scene.scene_scale:.3f} res={scene.train[0].width}x{scene.train[0].height}", flush=True)

    mono = mono_depths(scene, args.data, args.downscale, device) if args.depth_prior > 0 else None
    params, optims = init_params(scene.points, scene.colors, scene.scene_scale, device)
    strategy = DefaultStrategy(refine_stop_iter=int(args.iters * 0.8), verbose=False)
    strategy.check_sanity(params, optims)
    state = strategy.initialize_state(scene_scale=scene.scene_scale)
    sched = torch.optim.lr_scheduler.ExponentialLR(optims["means"], gamma=0.01 ** (1.0 / args.iters))

    t0 = time.time()
    order = np.random.permutation(len(scene.train))
    for step in range(args.iters):
        v = scene.train[order[step % len(order)]]
        if step % len(order) == len(order) - 1:
            order = np.random.permutation(len(scene.train))
        sh_deg = min(step // 1000, SH_DEGREE)

        if mono is not None:
            pred, alpha, info, depth = render(params, v, sh_deg, device, with_depth=True)
        else:
            pred, alpha, info = render(params, v, sh_deg, device)
        gt = v.image.to(device, non_blocking=True).float() / 255
        strategy.step_pre_backward(params, optims, state, step, info)
        l1 = (pred - gt).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1 - ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        if mono is not None:
            # Strong early (when floaters form), decaying to a light touch for detail.
            w = args.depth_prior * (0.1 + 0.9 * max(0.0, 1 - step / (0.6 * args.iters)))
            loss = loss + w * pearson_depth_loss(depth, mono[v.name].to(device).float(), alpha)
        if args.opacity_reg > 0:
            loss = loss + args.opacity_reg * torch.sigmoid(params["opacities"]).mean()
        loss.backward()

        for opt in optims.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        sched.step()

        # Cap growth to stay inside 8 GB: once over budget, pretend refinement has ended.
        if params["means"].shape[0] > args.max_gaussians:
            strategy.refine_stop_iter = min(strategy.refine_stop_iter, step)
        strategy.step_post_backward(params, optims, state, step, info, packed=True)

        if step % 500 == 0 or step == args.iters - 1:
            print(f"step {step:6d} loss {loss.item():.4f} psnr {psnr(pred.detach(), gt):5.2f} "
                  f"n={params['means'].shape[0]:,} mem={torch.cuda.max_memory_allocated() / 2**30:.2f}GB "
                  f"t={time.time() - t0:.0f}s", flush=True)

    train_time = time.time() - t0
    save_ply(args.out / "splat.ply", params)
    torch.save({k: v.detach() for k, v in params.items()}, args.out / "params.pt")

    metrics = evaluate(params, scene.test, device, args.out / "test_renders")
    metrics.update({"train_time_s": train_time, "num_gaussians": int(params["means"].shape[0]),
                    "iters": args.iters, "train_views": len(scene.train), "test_views": len(scene.test),
                    "downscale": args.downscale, "depth_prior": args.depth_prior,
                    "opacity_reg": args.opacity_reg})
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    m = metrics["mean"]
    print(f"TEST  psnr={m['psnr']:.2f}  ssim={m['ssim']:.4f}  lpips={m['lpips']:.4f}  ({train_time / 60:.1f} min)")


if __name__ == "__main__":
    main()
