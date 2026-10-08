"""Stage 4 — generative completion of unseen shell regions, tagged as generated.

For each shell face (floor, ceiling, 4 walls):
  1. Build an observed texture by splatting the colours of confident, observed Gaussians that
     lie on the face plane into a grid.
  2. Mark openings: grid cells where the camera saw *through* the plane (points outside the
     room — windows, open doors). These are never filled.
  3. Inpaint the remaining unobserved cells with LaMa (procedural fill if almost nothing of
     the face was seen, e.g. an unfilmed ceiling).
  4. Lift the filled pixels back into thin, plane-aligned Gaussians with tag=1 and a
     confidence that decays with distance from observed content.

Usage:
    python scripts/04_complete.py --run <baseline run> --data <dense> --out <run>/../completed
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from gsplat import rasterization
from scipy import ndimage

from common import floater_mask, load_scene, save_ply

C0 = 0.28209479177387814
train_mod = importlib.import_module("02_train")


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def face_frames(sh: dict) -> dict:
    """origin, u-axis, v-axis (world vectors, length 1), extent U, V, inward normal."""
    up, ex, ey = (np.array(sh[k]) for k in ("up", "ex", "ey"))
    (x0, x1), (y0, y1), H, f = sh["x"], sh["y"], sh["ceiling_h"], sh["floor_offset"]
    L, W = x1 - x0, y1 - y0

    def P(X, Y, h):  # room coords -> world
        return X * ex + Y * ey + (h + f) * up

    return {
        "floor":   (P(x0, y0, 0), ex, ey, L, W, up),
        "ceiling": (P(x0, y0, H), ex, ey, L, W, -up),
        "wall_x0": (P(x0, y0, 0), ey, up, W, H, ex),
        "wall_x1": (P(x1, y0, 0), ey, up, W, H, -ex),
        "wall_y0": (P(x0, y0, 0), ex, up, L, H, ey),
        "wall_y1": (P(x0, y1, 0), ex, up, L, H, -ey),
    }


def lama_fill(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from simple_lama_inpainting import SimpleLama
    global _LAMA
    if "_LAMA" not in globals():
        _LAMA = SimpleLama()
    out = _LAMA(Image.fromarray(rgb), Image.fromarray((mask * 255).astype(np.uint8)))
    return np.asarray(out)[: rgb.shape[0], : rgb.shape[1]]


def procedural_fill(shape: tuple[int, int], base: np.ndarray, seed: int) -> np.ndarray:
    """Plausible plain surface: base colour + gentle low-frequency shading (ceilings, bare walls)."""
    r = np.random.default_rng(seed)
    noise = ndimage.gaussian_filter(r.standard_normal(shape), sigma=max(shape) / 12)
    noise = noise / (np.abs(noise).max() + 1e-8) * 0.04
    img = np.clip(base[None, None, :] / 255.0 + noise[..., None], 0, 1)
    return (img * 255).astype(np.uint8)


def blueprint_guidance(args, fs: dict, L: float, W: float, px: float, scale_m: float) -> dict:
    """Mode A guiding Mode B: align the parsed floor plan to the fitted shell, then protect the
    plan's doors and windows as openings so completion never paints a wall over them.

    The plan's orientation relative to the reconstruction is unknown (which wall is "west", and
    whether the drawing is mirrored relative to our axes), so all 4 flips are tried and the one whose
    openings land on wall areas the cameras actually SAW THROUGH wins."""
    plan = json.loads(Path(args.plan_layout).read_text())
    Lp, Bp = plan["room"]["length"], plan["room"]["breadth"]
    Lm, Wm = L * scale_m, W * scale_m                      # shell extents in metres (x = ex, y = ey)
    swap = (Lm >= Wm) != (Lp >= Bp)                         # plan length runs along shell y instead of x
    px_m = px * scale_m

    def place(op, flipA, flipB):
        """plan opening -> (shell face, u range in texels)."""
        a, b = op["start_m"], op["end_m"]
        side = op["wall"]
        if not swap:
            ax_len, other_len, sx, sy = Lp, Bp, Lm / Lp, Wm / Bp
            if side in ("west", "east"):                    # wall at x=const, runs along plan z -> shell y
                face = "wall_x0" if (side == "west") != flipA else "wall_x1"
                lo, hi = (Bp - b, Bp - a) if flipB else (a, b)
                return face, lo * sy, hi * sy
            face = "wall_y0" if (side == "north") != flipB else "wall_y1"
            lo, hi = (Lp - b, Lp - a) if flipA else (a, b)
            return face, lo * sx, hi * sx
        sx, sy = Wm / Lp, Lm / Bp                            # plan x -> shell y, plan z -> shell x
        if side in ("west", "east"):
            face = "wall_y0" if (side == "west") != flipA else "wall_y1"
            lo, hi = (Bp - b, Bp - a) if flipB else (a, b)
            return face, lo * sy, hi * sy
        face = "wall_x0" if (side == "north") != flipB else "wall_x1"
        lo, hi = (Lp - b, Lp - a) if flipA else (a, b)
        return face, lo * sx, hi * sx

    def rect(op, flipA, flipB):
        face, lo, hi = place(op, flipA, flipB)
        f = fs[face]
        c0, c1 = int(max(0, lo / px_m)), int(min(f["nu"], np.ceil(hi / px_m)))
        top = args.plan_door_height if op["kind"] == "door" else args.plan_window_top
        bot = 0.0 if op["kind"] == "door" else args.plan_sill
        r0, r1 = int(bot / px_m), int(min(f["nv"], np.ceil(top / px_m)))
        return face, slice(r0, r1), slice(c0, c1)

    scores = {}
    for flipA in (False, True):
        for flipB in (False, True):
            ev = []
            for op in plan["openings"]:
                face, rs, cs = rect(op, flipA, flipB)
                m = fs[face]["opening"][rs, cs]
                ev.append(float(m.mean()) if m.size else 0.0)
            scores[(flipA, flipB)] = float(np.mean(ev)) if ev else 0.0
    best = max(scores, key=scores.get)
    protected = []
    for op in plan["openings"]:
        face, rs, cs = rect(op, *best)
        f = fs[face]
        region = np.zeros_like(f["opening"])
        region[rs, cs] = True
        add = region & ~f["observed"]                      # a seen closed door leaf stays observed
        newly = add & ~f["opening"]                        # not already seen-through by the cameras
        f["opening"] = f["opening"] | add
        f["protected"] = f.get("protected", 0) + float(newly.mean())
        protected.append({"kind": op["kind"], "plan_wall": op["wall"], "face": face,
                          "opening_texels": int(region.sum()),
                          "already_seen_through": int((region & ~newly & ~f["observed"]).sum()),
                          "newly_protected_by_plan": int(newly.sum())})
    out = {"plan": str(args.plan_layout), "swap_axes": bool(swap),
           "orientation": {"flip_along_length": best[0], "flip_along_breadth": best[1]},
           "evidence_seen_through": round(scores[best], 3),
           "evidence_by_orientation": {f"{int(k[0])}{int(k[1])}": round(v, 3) for k, v in scores.items()},
           "openings": protected}
    print("blueprint guidance:", json.dumps(out), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--res", type=int, default=400, help="texture pixels across the longest room side")
    ap.add_argument("--floater-radius", type=float, default=0.5,
                    help="metres around the walked camera path treated as free space (0 = off)")
    ap.add_argument("--vis-views", type=int, default=60, help="training views used for the visibility test")
    ap.add_argument("--vis-downscale", type=int, default=2)
    ap.add_argument("--plan-layout", type=Path, default=None,
                    help="layout.json from 10_floorplan.py: protect the plan's doors/windows during completion")
    ap.add_argument("--plan-door-height", type=float, default=2.1)
    ap.add_argument("--plan-sill", type=float, default=0.9)
    ap.add_argument("--plan-window-top", type=float, default=2.1)
    ap.add_argument("--debug-snap", action="store_true")
    ap.add_argument("--snap", action="store_true",
                    help="snap faces to the seen surface (off: rendered depth on plain walls is too smeared)")
    ap.add_argument("--snap-reach", type=float, default=0.06,
                    help="max face snap distance as a fraction of room size")
    ap.add_argument("--depth-margin", type=float, default=0.15,
                    help="relative depth tolerance for 'the camera saw this wall point'")
    ap.add_argument("--conf-decay", type=float, default=0.08,
                    help="confidence e-folding distance as a fraction of the room size")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "faces").mkdir(exist_ok=True)

    sh = json.loads((args.run / "shell.json").read_text())
    p = torch.load(args.run / "params.pt", map_location="cpu")

    # Camera-path floater pruning: a handheld phone is never inside furniture, so Gaussians within
    # --floater-radius metres of the walked path are floaters (plus large faint haze blobs).
    scale_m = json.loads((args.run / "scale.json").read_text())["chosen_scale"] if (args.run / "scale.json").exists() else 1.0
    path_scene = load_scene(args.data, downscale=8)
    cams = np.stack([torch.linalg.inv(v.viewmat)[:3, 3].numpy() for v in path_scene.train])
    n0 = len(p["means"])
    if args.floater_radius > 0:
        keep = floater_mask(p, cams, args.floater_radius / scale_m, haze_scale=0.3 / scale_m)
        p = {k: v[keep] for k, v in p.items()}
    pruned = n0 - len(p["means"])
    print(f"floater pruning: removed {pruned:,} of {n0:,} Gaussians", flush=True)

    L, W = sh["dims_sfm"]["length"], sh["dims_sfm"]["width"]
    px = max(L, W) / args.res
    tol = sh["tol"]

    # ---- visibility evidence: rendered depth of the observed-only model in sampled train views
    device = torch.device("cuda")
    scene = load_scene(args.data, downscale=args.vis_downscale)
    pd = {k: v.to(device) for k, v in p.items()}
    step = max(1, len(scene.train) // args.vis_views)
    vis_views = []
    with torch.no_grad():
        for v in scene.train[::step]:
            out_, alpha, _ = rasterization(
                means=pd["means"], quats=pd["quats"], scales=torch.exp(pd["scales"]),
                opacities=torch.sigmoid(pd["opacities"]), colors=torch.cat([pd["sh0"], pd["shN"]], 1),
                viewmats=v.viewmat[None].to(device), Ks=v.K[None].to(device),
                width=v.width, height=v.height, sh_degree=3, render_mode="RGB+ED", packed=True)
            vis_views.append((v.viewmat.to(device), v.K.to(device), out_[0, ..., 3], alpha[0, ..., 0],
                              v.image.to(device).float() / 255, v.width, v.height))
    print(f"visibility views={len(vis_views)} at {vis_views[0][5]}x{vis_views[0][6]}", flush=True)

    def classify(centres: np.ndarray, normal: np.ndarray):
        """Per texel: 0 unseen, 1 seen surface, 2 seen-through (opening). Plus best observed colour."""
        X = torch.from_numpy(centres).float().to(device)
        nrm = torch.from_numpy(normal).float().to(device)
        state = torch.zeros(len(X), dtype=torch.int8, device=device)
        through = torch.zeros(len(X), dtype=torch.int32, device=device)
        seen = torch.zeros(len(X), dtype=torch.int32, device=device)
        best = torch.full((len(X),), -1.0, device=device)
        colour = torch.zeros(len(X), 3, device=device)
        for vm, K, D, A, img, w, h in vis_views:
            cam = X @ vm[:3, :3].T + vm[:3, 3]
            z = cam[:, 2]
            u = (K[0, 0] * cam[:, 0] / z + K[0, 2]).round().long()
            v = (K[1, 1] * cam[:, 1] / z + K[1, 2]).round().long()
            ok = (z > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if not ok.any():
                continue
            ui, vi, zi = u[ok], v[ok], z[ok]
            d, a = D[vi, ui], A[vi, ui]
            margin = args.depth_margin * zi + tol
            s_surf = (a < 0.5) | ((d - zi).abs() <= margin)          # ray reached the plane
            s_thru = (a >= 0.5) & (d > zi + margin)                   # ray passed beyond it
            idx = torch.nonzero(ok).squeeze(1)
            seen[idx[s_surf]] += 1
            through[idx[s_thru]] += 1
            # Most head-on, closest view supplies the observed colour.
            cam_c = -vm[:3, :3].T @ vm[:3, 3]
            ray = torch.nn.functional.normalize(X[idx] - cam_c, dim=-1)
            score = (ray @ nrm).abs() / zi
            better = s_surf & (score > best[idx])
            bi = idx[better]
            best[bi] = score[better]
            colour[bi] = img[vi[better], ui[better]]
        state[seen > 0] = 1
        state[(through > 0) & (through >= seen)] = 2
        return state.cpu().numpy(), colour.cpu().numpy()

    def refine_offset(centres: np.ndarray, o: np.ndarray, normal: np.ndarray, reach: float) -> float:
        """Where do the cameras actually see this face? Back-project rendered depth for texels in
        view and take the peak of their signed distance to the plane (n points into the room)."""
        X = torch.from_numpy(centres).float().to(device)
        nrm = torch.from_numpy(normal).float().to(device)
        oo = torch.from_numpy(o).float().to(device)
        sds = []
        for vm, K, D, A, img, w, h in vis_views[::2]:
            cam = X @ vm[:3, :3].T + vm[:3, 3]
            z = cam[:, 2]
            u = (K[0, 0] * cam[:, 0] / z + K[0, 2]).round().long()
            v = (K[1, 1] * cam[:, 1] / z + K[1, 2]).round().long()
            ok = (z > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if not ok.any():
                continue
            d, a = D[v[ok], u[ok]], A[v[ok], u[ok]]
            cam_c = -vm[:3, :3].T @ vm[:3, 3]
            hit = cam_c + (X[ok] - cam_c) * (d / z[ok])[:, None]
            sd = (hit - oo) @ nrm
            sds.append(sd[(a > 0.5) & (sd.abs() < reach)])
        sd = torch.cat(sds).cpu().numpy() if sds else np.zeros(0)
        if len(sd) < 500:
            return 0.0
        bw = reach / 40
        cnt, ed = np.histogram(sd, np.arange(-reach, reach + bw, bw))
        cnt = np.convolve(cnt, [1, 2, 3, 2, 1], "same")
        if args.debug_snap:
            mids = 0.5 * (ed[:-1] + ed[1:])
            print("   sd histogram:", " ".join(f"{m:+.2f}:{c}" for m, c in zip(mids[::3], cnt[::3])), flush=True)
        # The face is the OUTERMOST strong surface (smallest signed distance); the biggest peak is
        # usually furniture in front of it (desks against a wall, benches over the floor, beams).
        strong = np.nonzero(cnt >= 0.3 * cnt.max())[0]
        k = int(strong.min())
        while k + 1 < len(cnt) and cnt[k + 1] > cnt[k]:
            k += 1
        return float(0.5 * (ed[k] + ed[k + 1]))

    new = {k: [] for k in ("means", "rgb", "quat", "conf")}
    report, upper_wall_colours = {}, []
    frames = face_frames(sh)
    ORDER = ["wall_x0", "wall_x1", "wall_y0", "wall_y1", "floor", "ceiling"]
    fs = {}
    for name in ORDER:
        o, du, dv, U, V, n = frames[name]
        nu, nv = max(2, int(U / px)), max(2, int(V / px))
        jj0, ii0 = np.mgrid[0:nv, 0:nu]
        centres = o + ((ii0.ravel() + 0.5) * px)[:, None] * du + ((jj0.ravel() + 0.5) * px)[:, None] * dv
        if args.debug_snap:
            print(name, flush=True)
        shift = refine_offset(centres[::7], o, n, reach=args.snap_reach * max(L, W)) if args.snap else 0.0
        o = o + shift * n                                   # snap the face onto the seen surface
        centres = centres + shift * n
        state, colour = classify(centres, n)
        state, colour = state.reshape(nv, nu), colour.reshape(nv, nu, 3)

        observed = state == 1
        opening = state == 2
        if not name.startswith("wall"):          # shiny floors 'see through' via reflections
            observed |= opening
            opening[:] = False
        # Tiny unseen specks inside observed areas are sampling gaps, not hidden regions.
        unseen = ~observed & ~opening
        small = ndimage.binary_opening(unseen, iterations=2)
        observed |= unseen & ~small
        fs[name] = dict(o=o, du=du, dv=dv, n=n, nu=nu, nv=nv, shift=shift,
                        observed=observed, opening=opening, colour=colour)

    plan_report = blueprint_guidance(args, fs, L, W, px, scale_m) if args.plan_layout else None

    # Walls first so their upper-band colour can seed a fully unseen ceiling.
    for name in ORDER:
        f = fs[name]
        o, du, dv, n, nu, nv, shift = f["o"], f["du"], f["dv"], f["n"], f["nu"], f["nv"], f["shift"]
        observed, opening, colour = f["observed"], f["opening"], f["colour"]
        tex = np.where(observed[..., None], colour, 0)
        # Observed texels with no colour sample are sampling gaps: fill them, but they stay "observed".
        gaps = observed & (colour.sum(-1) <= 1e-6)

        fill_mask = ~observed & ~opening
        frac_obs = float(observed.mean())
        tex8 = (np.clip(tex, 0, 1) * 255).astype(np.uint8)
        if name.startswith("wall") and observed.any():
            top = observed.copy(); top[: int(0.8 * nv)] = False   # rows are v (height) ascending
            if top.any():
                upper_wall_colours.append(np.median(tex8[top], 0))

        if fill_mask.any():
            if frac_obs < 0.05:
                base = (np.median(upper_wall_colours, 0) * 1.08 if upper_wall_colours
                        else np.array([220, 220, 215]))
                filled = procedural_fill((nv, nu), np.clip(base, 0, 255), seed=zlib.crc32(name.encode()))
                method = "procedural"
            else:
                filled = lama_fill(tex8, fill_mask | gaps)
                method = "lama"
            out = np.where((fill_mask | gaps)[..., None], filled, tex8)
        elif gaps.any():
            out, method = np.where(gaps[..., None], lama_fill(tex8, gaps), tex8), "none"
        else:
            out, method = tex8, "none"

        dist = ndimage.distance_transform_edt(~observed) * px if observed.any() else np.full((nv, nu), 1e9)
        conf = np.exp(-dist / (args.conf_decay * max(L, W)))
        if method == "procedural":
            conf = np.minimum(conf, 0.1)

        # Raw (unflipped) texture + masks for the GLB exporter; row index = v, column = u.
        Image.fromarray(out).save(args.out / "faces" / f"{name}_tex.png")
        Image.fromarray((fill_mask * 255).astype(np.uint8)).save(args.out / "faces" / f"{name}_gen.png")
        # Side-by-side figure for the write-up (flip so up is up for walls).
        vis = np.stack([out, np.where(fill_mask[..., None], [255, 0, 200], out).astype(np.uint8)], 0)
        flip = (lambda a: a[::-1]) if name != "floor" and name != "ceiling" else (lambda a: a)
        Image.fromarray(np.concatenate([flip(vis[0]), flip(vis[1])], 1)).save(args.out / "faces" / f"{name}.png")

        jj, ii = np.nonzero(fill_mask)
        if len(ii):
            centres = o + ((ii + 0.5) * px)[:, None] * du + ((jj + 0.5) * px)[:, None] * dv
            R = np.stack([du, dv, np.cross(du, dv)], 1)
            new["means"].append(centres)
            new["rgb"].append(out[jj, ii] / 255.0)
            new["quat"].append(np.repeat(rotmat_to_quat_wxyz(R)[None], len(ii), 0))
            new["conf"].append(conf[jj, ii])
        report[name] = {"snap": round(shift, 4), "plan_protected": round(float(f.get("protected", 0)), 3),
                        "observed": round(frac_obs, 3), "opening": round(float(opening.mean()), 3),
                        "generated": round(float(fill_mask.mean()), 3), "method": method,
                        "new_gaussians": int(len(ii))}
        print(name, report[name], flush=True)

    n_new = sum(len(m) for m in new["means"])
    if n_new:
        cat = lambda k: torch.from_numpy(np.concatenate(new[k])).float()
        rgb_new = cat("rgb")
        g = {
            "means": cat("means"),
            "scales": torch.log(torch.tensor([px * 0.6, px * 0.6, px * 0.05])).repeat(n_new, 1),
            "quats": cat("quat"),
            "opacities": torch.logit(torch.full((n_new,), 0.95)),
            "sh0": ((rgb_new - 0.5) / C0)[:, None, :],
            "shN": torch.zeros(n_new, p["shN"].shape[1], 3),
            "tag": torch.ones(n_new),
        }
        merged = {k: torch.cat([p[k], g[k]]) for k in g}
        merged["conf"] = torch.cat([torch.ones(len(p["means"])), cat("conf")])
    else:
        merged = dict(p); merged["conf"] = torch.ones(len(p["means"]))

    torch.save(merged, args.out / "params.pt")
    save_ply(args.out / "splat.ply", merged)

    # Honesty version: generated Gaussians tinted magenta, stronger where confidence is low.
    honest = {k: v.clone() for k, v in merged.items()}
    gen = honest["tag"] > 0.5
    rgb = honest["sh0"][:, 0] * C0 + 0.5
    a = (0.45 + 0.4 * (1 - honest["conf"]))[:, None]
    tint = torch.tensor([1.0, 0.0, 0.8])
    rgb[gen] = (1 - a[gen]) * rgb[gen] + a[gen] * tint
    honest["sh0"] = ((rgb - 0.5) / C0)[:, None, :]
    honest["shN"][gen] = 0
    save_ply(args.out / "splat_honesty.ply", honest)

    # Evaluate at exactly the resolution the baseline was evaluated at, so numbers compare.
    base_ds = json.loads((args.run / "metrics.json").read_text()).get("downscale", 1)
    scene = load_scene(args.data, downscale=base_ds)
    pd = {k: v.to(device) for k, v in merged.items()}
    metrics = train_mod.evaluate(pd, scene.test, device, args.out / "test_renders")
    metrics.update({"faces": report, "generated_gaussians": n_new, "floaters_pruned": pruned,
                    "blueprint": plan_report,
                    "num_gaussians": int(len(merged["means"]))})
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    m = metrics["mean"]
    print(f"TEST  psnr={m['psnr']:.2f}  ssim={m['ssim']:.4f}  lpips={m['lpips']:.4f}  generated={n_new:,}")


if __name__ == "__main__":
    main()
