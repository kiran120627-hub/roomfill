"""Stage 6 — export metric, gravity-aligned outputs that open in standard 3D tools.

Room frame (glTF / three.js convention): Y up, floor at y=0, room centred at x=z=0, metres.

Produces in <run>/export/:
    room_splat.ply          Gaussian splat, metric + aligned (SuperSplat, antimatter15, three.js viewers)
    room_splat_honesty.ply  same, generated Gaussians tinted magenta
    room_shell.glb          textured mesh of floor / walls / ceiling (Blender, any glTF viewer)
    room_shell_honesty.glb  same with generated texels tinted
    room.json               dimensions, scale estimators, coverage — the machine-readable answer

Usage:
    python scripts/06_export.py --run <completed run> --shell-run <baseline run>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import save_ply, save_splat

C0 = 0.28209479177387814


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw], -1)


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def transform_params(p: dict, R: np.ndarray, t: np.ndarray, s: float) -> dict:
    """x' = s * R (x - t). View-dependent SH bands are dropped (they would need SH rotation)."""
    out = {k: v.clone() for k, v in p.items()}
    Rt = torch.from_numpy(R).float()
    out["means"] = s * (p["means"] - torch.from_numpy(t).float()) @ Rt.T
    out["scales"] = p["scales"] + float(np.log(s))
    q = torch.nn.functional.normalize(p["quats"], dim=-1)
    out["quats"] = quat_mul(torch.from_numpy(rot_to_quat(R)).float().expand_as(q), q)
    out["shN"] = torch.zeros_like(p["shN"])
    return out


def build_glb(faces_dir: Path, sh: dict, s: float, to_room, honesty: bool, path: Path) -> None:
    import trimesh
    meshes = []
    up, ex, ey = (np.array(sh[k]) for k in ("up", "ex", "ey"))
    (x0, x1), (y0, y1), H, f = sh["x"], sh["y"], sh["ceiling_h"], sh["floor_offset"]
    P = lambda X, Y, h: X * ex + Y * ey + (h + f) * up
    L, W = x1 - x0, y1 - y0
    spec = {   # origin, u-vector, v-vector (world, unnormalised = full extent)
        "floor":   (P(x0, y0, 0), ex * L, ey * W),
        "ceiling": (P(x0, y0, H), ex * L, ey * W),
        "wall_x0": (P(x0, y0, 0), ey * W, up * H),
        "wall_x1": (P(x1, y0, 0), ey * W, up * H),
        "wall_y0": (P(x0, y0, 0), ex * L, up * H),
        "wall_y1": (P(x0, y1, 0), ex * L, up * H),
    }
    for name, (o, U, V) in spec.items():
        tex_p = faces_dir / f"{name}_tex.png"
        if not tex_p.exists():
            continue
        tex = np.asarray(Image.open(tex_p).convert("RGB")).copy()
        if honesty:
            gen = np.asarray(Image.open(faces_dir / f"{name}_gen.png")) > 127
            tex[gen] = (0.45 * tex[gen] + 0.55 * np.array([255, 0, 204])).astype(np.uint8)
        corners = np.array([o, o + U, o + U + V, o + V])
        verts = to_room(corners)
        # Texture row 0 holds the face's v=0 edge. trimesh uses OpenGL UVs (v=0 = image bottom)
        # and flips them when writing glTF, so the face origin maps to UV v=1 (top row).
        uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]])   # double-sided
        vis = trimesh.visual.TextureVisuals(uv=uv, image=Image.fromarray(tex))
        meshes.append(trimesh.Trimesh(vertices=verts, faces=faces, visual=vis, process=False))
    trimesh.Scene(meshes).export(path)


def build_usd(path: Path, faces_dir: Path, sh: dict, to_room, pm: dict, max_points: int = 1_500_000) -> None:
    """room.usdc: textured shell meshes (UsdPreviewSurface) + the splat as a Points prim
    with per-point `generated` and `confidence` primvars."""
    import shutil
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Room")
    stage.SetDefaultPrim(root.GetPrim())
    tex_dir = path.parent / "textures"
    tex_dir.mkdir(exist_ok=True)

    up, ex, ey = (np.array(sh[k]) for k in ("up", "ex", "ey"))
    (x0, x1), (y0, y1), H, f = sh["x"], sh["y"], sh["ceiling_h"], sh["floor_offset"]
    P = lambda X, Y, h: X * ex + Y * ey + (h + f) * up
    L, W = x1 - x0, y1 - y0
    spec = {"floor": (P(x0, y0, 0), ex * L, ey * W), "ceiling": (P(x0, y0, H), ex * L, ey * W),
            "wall_x0": (P(x0, y0, 0), ey * W, up * H), "wall_x1": (P(x1, y0, 0), ey * W, up * H),
            "wall_y0": (P(x0, y0, 0), ex * L, up * H), "wall_y1": (P(x0, y1, 0), ex * L, up * H)}
    for name, (o, U, V) in spec.items():
        tex = faces_dir / f"{name}_tex.png"
        if not tex.exists():
            continue
        shutil.copy(tex, tex_dir / tex.name)
        verts = to_room(np.array([o, o + U, o + U + V, o + V]))
        mesh = UsdGeom.Mesh.Define(stage, f"/Room/Shell/{name}")
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in verts]))
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateDoubleSidedAttr(True)
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                     UsdGeom.Tokens.faceVarying)
        st.Set(Vt.Vec2fArray([(0, 1), (1, 1), (1, 0), (0, 0)]))   # USD v=0 is the image bottom
        mat = UsdShade.Material.Define(stage, f"/Room/Materials/{name}")
        surf = UsdShade.Shader.Define(stage, f"/Room/Materials/{name}/Surface")
        surf.CreateIdAttr("UsdPreviewSurface")
        surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        reader = UsdShade.Shader.Define(stage, f"/Room/Materials/{name}/st")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        img = UsdShade.Shader.Define(stage, f"/Room/Materials/{name}/Tex")
        img.CreateIdAttr("UsdUVTexture")
        img.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./textures/{tex.name}")
        img.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(img.ConnectableAPI(), "rgb")
        mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

    n = len(pm["means"])
    idx = np.random.default_rng(0).permutation(n)[:max_points] if n > max_points else np.arange(n)
    pos = pm["means"][idx].numpy()
    rgb = np.clip(pm["sh0"][idx, 0].numpy() * C0 + 0.5, 0, 1)
    width = 2 * np.exp(pm["scales"][idx].numpy()).mean(1)
    pts = UsdGeom.Points.Define(stage, "/Room/Splat")
    pts.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pos.astype(np.float32)))
    pts.CreateWidthsAttr(Vt.FloatArray.FromNumpy(width.astype(np.float32)))
    pv = UsdGeom.PrimvarsAPI(pts)
    pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec3fArray.FromNumpy(rgb.astype(np.float32)))
    pv.CreatePrimvar("generated", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.vertex).Set(
        Vt.IntArray.FromNumpy((pm["tag"][idx].numpy() > 0.5).astype(np.int32)))
    pv.CreatePrimvar("confidence", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set(
        Vt.FloatArray.FromNumpy(pm["conf"][idx].numpy().astype(np.float32)))
    stage.GetRootLayer().Save()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path, help="completed run (stage 4 output)")
    ap.add_argument("--shell-run", required=True, type=Path, help="run holding shell.json / scale.json")
    ap.add_argument("--baseline-run", type=Path, default=None,
                    help="also export this run (vanilla 3DGS) as room_baseline.splat for the viewer's compare mode")
    ap.add_argument("--data", type=Path, default=None,
                    help="dense COLMAP folder; used to pick a real video camera as the viewer's home view")
    args = ap.parse_args()

    sh = json.loads((args.shell_run / "shell.json").read_text())
    sc = json.loads((args.shell_run / "scale.json").read_text())
    s = sc["chosen_scale"]
    up, ex, ey = (np.array(sh[k]) for k in ("up", "ex", "ey"))
    R = np.stack([ex, up, -ey])          # rows: new x, new y (up), new z  — right-handed
    cx, cy = np.mean(sh["x"]), np.mean(sh["y"])
    t = cx * ex + cy * ey + sh["floor_offset"] * up

    def to_room(pts: np.ndarray) -> np.ndarray:
        return s * (pts - t) @ R.T

    out = args.run / "export"
    out.mkdir(exist_ok=True)
    p = torch.load(args.run / "params.pt", map_location="cpu")
    pm = transform_params(p, R, t, s)
    # Drop stray Gaussians far outside the room (sky / trees through windows, SfM outliers). They are
    # not room content, and their huge depth range breaks web viewers' sort precision.
    d = sc["dims_m"]
    reach = np.array([d["length"], d["height"] * 2, d["width"]])
    keep = (pm["means"].abs() <= torch.from_numpy(reach).float()).all(1)
    keep &= pm["means"][:, 1] >= -0.5 * d["height"]
    pm = {k: v[keep] for k, v in pm.items()}
    p = {k: v[keep] for k, v in p.items()}
    print(f"kept {int(keep.sum()):,} of {len(keep):,} Gaussians inside 2x the room box")
    save_ply(out / "room_splat.ply", pm, sh0_only=True)
    save_splat(out / "room.splat", pm)

    honest = {k: v.clone() for k, v in pm.items()}
    gen = honest["tag"] > 0.5
    rgb = honest["sh0"][:, 0] * C0 + 0.5
    a = (0.45 + 0.4 * (1 - honest["conf"]))[:, None]
    rgb[gen] = (1 - a[gen]) * rgb[gen] + a[gen] * torch.tensor([1.0, 0.0, 0.8])
    honest["sh0"] = ((rgb - 0.5) / C0)[:, None, :]
    save_ply(out / "room_splat_honesty.ply", honest, sh0_only=True)
    save_splat(out / "room_honesty.splat", honest)

    if args.baseline_run:
        pb = torch.load(args.baseline_run / "params.pt", map_location="cpu")
        pb["conf"] = torch.ones(len(pb["means"]))
        pbm = transform_params(pb, R, t, s)
        kb = (pbm["means"].abs() <= torch.from_numpy(reach).float()).all(1)
        save_splat(out / "room_baseline.splat", {k: v[kb] for k, v in pbm.items()})

    faces_dir = args.run / "faces"
    build_glb(faces_dir, sh, s, to_room, False, out / "room_shell.glb")
    build_glb(faces_dir, sh, s, to_room, True, out / "room_shell_honesty.glb")
    build_usd(out / "room.usdc", faces_dir, sh, to_room, pm)

    metrics = json.loads((args.run / "metrics.json").read_text())
    room = {"units": "metres", "frame": "Y-up, floor y=0, room centred at x=z=0",
            "dimensions": sc["dims_m"], "camera_height_m": sc["camera_height_m"],
            "scale_estimators": sc["dims_m_by_estimator"], "ceiling_observed": sh["ceiling_observed"],
            "faces": metrics.get("faces", {}),
            "gaussians": {"total": int(len(p["means"])), "generated": int(gen.sum())}}
    if "dim_error_pct" in sc:
        room["dim_error_pct_vs_tape"] = sc["dim_error_pct"]
    room["scale_from"] = sc.get("chosen_from", "video")

    # Home view: a real training camera (so the first frame is a view the model was fit to),
    # preferring one far from the centre that looks back across the room.
    if args.data:
        from common import load_scene
        scene = load_scene(args.data, downscale=8)
        # Viewpoints come from REAL video cameras (the model is fit there; off-path views show floaters).
        # overview = inside the room, longest free sightline; reverse = best one facing the other way;
        # look up = overview tilted up 35 degrees toward the ceiling the walkthrough barely filmed.
        dims_ = sc["dims_m"]
        hx, hz = dims_["length"] / 2, dims_["width"] / 2
        cands = []
        for v in scene.train:
            c2w = torch.linalg.inv(v.viewmat).numpy()
            pos = to_room(c2w[:3, 3][None])[0]
            if abs(pos[0]) > hx - 1.0 or abs(pos[2]) > hz - 1.0:
                continue
            fwd = R @ c2w[:3, 2]
            fwd = fwd / np.linalg.norm(fwd)
            f = np.array([fwd[0], fwd[2]]) / (np.linalg.norm([fwd[0], fwd[2]]) + 1e-9)
            ts = [((np.sign(f[i]) * h) - pos[2 * i]) / f[i] for i, h in ((0, hx), (1, hz)) if abs(f[i]) > 1e-6]
            reach = min(t for t in ts if t > 0) if ts else 0
            cands.append((reach - 2.0 * abs(fwd[1]), pos, fwd, f))
        cands.sort(key=lambda c: -c[0])
        _, pos0, fwd0, f0 = cands[0]
        # Second view: the open camera that looks most differently from the overview (walkthroughs
        # often never face backwards, so "reverse" may not exist; take the widest real angle).
        open_ = [c for c in cands if c[0] > 0.5 * cands[0][0]] or cands
        rev = min(open_, key=lambda c: float(c[3] @ f0))
        a35 = np.radians(35)
        up_dir = np.array([f0[0] * np.cos(a35), np.sin(a35), f0[1] * np.cos(a35)])
        r3 = lambda x: [round(float(t), 3) for t in x]
        room["views"] = {
            "overview": {"position": r3(pos0), "look_at": r3(pos0 + 3.0 * fwd0)},
            "ceiling": {"position": r3(pos0), "look_at": r3(pos0 + 3.0 * up_dir)},
            "side": {"position": r3(rev[1]), "look_at": r3(rev[1] + 3.0 * rev[2])},
        }
        room["home_camera"] = room["views"]["overview"]
    (out / "room.json").write_text(json.dumps(room, indent=2))
    print(json.dumps(room, indent=2))


if __name__ == "__main__":
    main()
