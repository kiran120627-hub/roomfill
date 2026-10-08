"""Shared helpers: COLMAP scene loading, train/held-out split, SSIM, PLY export."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class View:
    name: str
    image: torch.Tensor   # H x W x 3 uint8 on CPU (convert with .float() / 255 on device)
    K: torch.Tensor       # 3 x 3
    viewmat: torch.Tensor  # 4 x 4 world -> camera
    width: int
    height: int


@dataclass
class Scene:
    train: list[View]
    test: list[View]
    points: np.ndarray    # N x 3 sparse SfM points
    colors: np.ndarray    # N x 3 in [0, 1]
    scene_scale: float


def _find_model_dir(data_dir: Path) -> Path:
    for cand in (data_dir / "sparse" / "0", data_dir / "sparse"):
        if (cand / "images.bin").exists() or (cand / "images.txt").exists():
            return cand
    raise FileNotFoundError(f"no COLMAP model under {data_dir}/sparse")


def is_heldout(name: str) -> bool:
    return "heldout" in name


def load_scene(data_dir: Path, downscale: int = 1, test_every: int = 8) -> Scene:
    """Load an undistorted (PINHOLE / SIMPLE_PINHOLE) COLMAP scene.

    Split rule: images whose path contains 'heldout' are the test set. If there are none
    (public datasets), every `test_every`-th image is held out — the standard 3DGS protocol.
    """
    import pycolmap

    rec = pycolmap.Reconstruction(str(_find_model_dir(data_dir)))
    img_root = data_dir / "images"

    imgs = sorted(rec.images.values(), key=lambda im: im.name)
    any_heldout = any(is_heldout(im.name) for im in imgs)

    train, test = [], []
    for i, im in enumerate(imgs):
        cam = rec.cameras[im.camera_id]
        model = cam.model.name if hasattr(cam.model, "name") else str(cam.model)
        p = cam.params
        if model.endswith("SIMPLE_PINHOLE"):
            fx = fy = p[0]; cx, cy = p[1], p[2]
        elif model.endswith("PINHOLE"):
            fx, fy, cx, cy = p[:4]
        else:
            raise ValueError(f"camera model {model} is distorted — run colmap image_undistorter first")

        pil = Image.open(img_root / im.name).convert("RGB")
        w, h = pil.size
        if downscale > 1:
            w, h = w // downscale, h // downscale
            pil = pil.resize((w, h), Image.LANCZOS)
        sx, sy = w / cam.width, h / cam.height
        K = torch.tensor([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]], dtype=torch.float32)

        pose = im.cam_from_world() if callable(getattr(im, "cam_from_world", None)) else im.cam_from_world
        vm = torch.eye(4, dtype=torch.float32)
        vm[:3, :4] = torch.from_numpy(np.asarray(pose.matrix(), dtype=np.float32))

        v = View(im.name, torch.from_numpy(np.asarray(pil, dtype=np.uint8).copy()), K, vm, w, h)
        held = is_heldout(im.name) if any_heldout else (i % test_every == 0)
        (test if held else train).append(v)

    pts = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float32)
    cols = np.array([p.color for p in rec.points3D.values()], dtype=np.float32) / 255.0

    centers = np.stack([torch.linalg.inv(v.viewmat)[:3, 3].numpy() for v in train])
    scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
    return Scene(train, test, pts, cols, scene_scale)


# ---------------------------------------------------------------- losses / metrics

def _gauss_window(size: int = 11, sigma: float = 1.5, device="cpu") -> torch.Tensor:
    x = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).expand(3, 1, size, size).contiguous()


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SSIM for B x 3 x H x W images in [0, 1]."""
    w = _gauss_window(device=a.device)
    mu_a = F.conv2d(a, w, padding=5, groups=3)
    mu_b = F.conv2d(b, w, padding=5, groups=3)
    s_aa = F.conv2d(a * a, w, padding=5, groups=3) - mu_a ** 2
    s_bb = F.conv2d(b * b, w, padding=5, groups=3) - mu_b ** 2
    s_ab = F.conv2d(a * b, w, padding=5, groups=3) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    return m.mean()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return 10 * math.log10(1.0 / max(mse, 1e-10))


# ---------------------------------------------------------------- export

def save_ply(path: Path, params: dict[str, torch.Tensor], sh0_only: bool = False) -> None:
    """Standard 3DGS PLY (opens in SuperSplat / antimatter15 viewers) plus `generated` and
    `confidence` fields. sh0_only drops the view-dependent bands (4x smaller file)."""
    from plyfile import PlyData, PlyElement

    means = params["means"].detach().cpu().numpy()
    n = means.shape[0]
    sh0 = params["sh0"].detach().cpu().numpy().reshape(n, -1)                    # N x 3
    shN = params["shN"].detach().cpu().transpose(1, 2).numpy().reshape(n, -1)    # channel-major, 3DGS layout
    if sh0_only:
        shN = shN[:, :0]
    opac = params["opacities"].detach().cpu().numpy().reshape(n, 1)
    scales = params["scales"].detach().cpu().numpy()
    quats = params["quats"].detach().cpu().numpy()
    gen = params["tag"].detach().cpu().numpy().reshape(n, 1)
    conf = (params["conf"] if "conf" in params else torch.ones(n)).detach().cpu().numpy().reshape(n, 1)

    attrs = (["x", "y", "z", "nx", "ny", "nz"]
             + [f"f_dc_{i}" for i in range(sh0.shape[1])]
             + [f"f_rest_{i}" for i in range(shN.shape[1])]
             + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "generated", "confidence"])
    data = np.concatenate([means, np.zeros_like(means), sh0, shN, opac, scales, quats, gen, conf], axis=1)
    arr = np.rec.fromarrays(data.astype(np.float32).T, dtype=[(a, "f4") for a in attrs])
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(path))


def save_splat(path: Path, params: dict[str, torch.Tensor]) -> None:
    """Compact .splat (antimatter15 layout, 32 B/splat): xyz f32, scale f32 (linear), RGBA u8,
    quaternion u8. ~2.4x smaller than PLY; loads in Spark and most web splat viewers.
    Sorted by size x opacity so progressive loaders show the important splats first."""
    C0 = 0.28209479177387814
    xyz = params["means"].detach().cpu().numpy().astype(np.float32)
    sc = np.exp(params["scales"].detach().cpu().numpy()).astype(np.float32)
    rgb = np.clip(params["sh0"].detach().cpu()[:, 0].numpy() * C0 + 0.5, 0, 1)
    a = 1 / (1 + np.exp(-params["opacities"].detach().cpu().numpy()))
    q = params["quats"].detach().cpu().numpy()
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    order = np.argsort(-(sc.prod(1) * a))
    n = len(xyz)
    buf = np.zeros(n, dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    buf["p"], buf["s"] = xyz[order], sc[order]
    buf["c"] = np.concatenate([rgb[order], a[order, None]], 1).__mul__(255).round().clip(0, 255).astype(np.uint8)
    buf["r"] = (q[order] * 128 + 128).round().clip(0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))
