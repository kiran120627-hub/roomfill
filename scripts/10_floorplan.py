"""Mode A — floor plan image -> 3D room model (walls, door / window openings, metric dimensions).

Classical, explainable parsing (no training data needed):
  1. walls     = dark, low-saturation pixels; the largest connected component is the wall ring
                 (dimension text and arrows are blue/navy and are excluded by the saturation test)
  2. interior  = the hole the wall ring encloses; its bounding box gives room size in pixels
  3. openings  = runs along each wall where the ring is interrupted; a door if a saturated brown
                 leaf / swing arc sits next to the gap, otherwise a window
  4. scale     = from one known dimension (the plan's annotated length, passed in or read from
                 data/measurements.json); pixels -> metres for everything else
  5. 3D        = walls extruded to ceiling height with door voids (lintel above) and window voids
                 (sill + lintel), plus a floor slab -> GLB, and a machine-readable layout.json

Usage:
    python scripts/10_floorplan.py --plan data/floorplan.jpg --length 9.23 --height 3.5 --out results/floorplan
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) runs of True."""
    d = np.diff(np.r_[0, mask_1d.astype(int), 0])
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--length", type=float, default=None, help="known horizontal interior length (m)")
    ap.add_argument("--measurements", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "measurements.json")
    ap.add_argument("--breadth", type=float, default=None,
                    help="known vertical interior breadth (m); if given, axes are scaled separately "
                         "(hand-drawn / generated plans are often not drawn to scale)")
    ap.add_argument("--height", type=float, default=None, help="wall height (m); default from measurements or 3.0")
    ap.add_argument("--door-height", type=float, default=2.1)
    ap.add_argument("--sill", type=float, default=0.9)
    ap.add_argument("--window-top", type=float, default=2.1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meas = json.loads(args.measurements.read_text()) if args.measurements.exists() else {}
    H = args.height or meas.get("ceiling_height_m", 3.0)
    door_h = meas.get("door_height_m", args.door_height)

    rgb = np.asarray(Image.open(args.plan).convert("RGB")).astype(np.float32)
    hsv = np.asarray(Image.open(args.plan).convert("HSV")).astype(np.float32)
    val, sat = hsv[..., 2], hsv[..., 1]

    # 1. walls: dark + grey (navy annotation text is dark but saturated)
    dark = (val < 120) & (sat < 70)
    lab, n = ndimage.label(ndimage.binary_closing(dark, iterations=2))
    sizes = ndimage.sum(dark, lab, range(1, n + 1))
    ring = lab == (1 + int(np.argmax(sizes)))

    # 2. interior: seal openings by growing the walls by GROW px, take the enclosed hole, then
    #    push its box back out by GROW px (exact for straight walls, and cannot leak through a door).
    GROW = 45
    bridged = ndimage.binary_dilation(ring, iterations=GROW)
    inner = ndimage.binary_fill_holes(bridged) & ~bridged
    lab_i, ni = ndimage.label(inner)
    inner = lab_i == (1 + int(np.argmax(ndimage.sum(inner, lab_i, range(1, ni + 1)))))
    ys, xs = np.nonzero(inner)
    x0, x1, y0, y1 = xs.min() - GROW, xs.max() + 1 + GROW, ys.min() - GROW, ys.max() + 1 + GROW
    w_px, h_px = x1 - x0, y1 - y0

    # wall thickness from the ring just outside the interior
    t_left = int(np.sum(ring[(y0 + y1) // 2, :x0]))
    t_top = int(np.sum(ring[:y0, (x0 + x1) // 2]))
    t_px = int(np.median([t_left, t_top]))

    # 4. scale
    L = args.length or meas.get("room_length_m")
    if L is None:
        raise SystemExit("need one known dimension: pass --length or provide data/measurements.json")
    m_per_px = L / w_px                       # x (length) axis
    B = args.breadth or meas.get("room_width_m")
    my_per_px = (B / h_px) if B else m_per_px  # y (breadth) axis
    W = h_px * my_per_px
    aspect_mismatch = abs((w_px / h_px) / (L / W) - 1) if B else 0.0

    # 3. openings: sample the middle of each wall band; gaps in the ring are openings
    brown = (rgb[..., 0] > rgb[..., 2] + 50) & (rgb[..., 0] > 90) & (sat > 90)
    sides = {
        # name: (1-D ring profile along the wall, axis origin in px, wall line in interior coords)
        "west":  (ring[y0:y1, max(0, x0 - t_px // 2)], y0, "y"),
        "east":  (ring[y0:y1, min(ring.shape[1] - 1, x1 + t_px // 2)], y0, "y"),
        "north": (ring[max(0, y0 - t_px // 2), x0:x1], x0, "x"),
        "south": (ring[min(ring.shape[0] - 1, y1 + t_px // 2), x0:x1], x0, "x"),
    }
    openings = []
    for side, (prof, origin, axis) in sides.items():
        for a, b in runs(~prof):
            s_ax = my_per_px if axis == "y" else m_per_px
            if b - a < 0.25 / s_ax:              # ignore specks under 25 cm
                continue
            # door if a brown leaf / swing arc sits next to the gap (searched one gap-width into the room)
            pad = int(b - a)
            g0, g1 = origin + a, origin + b
            region = {"west": brown[g0:g1, max(0, x0 - t_px): x0 + pad],
                      "east": brown[g0:g1, x1 - pad: x1 + t_px],
                      "north": brown[max(0, y0 - t_px): y0 + pad, g0:g1],
                      "south": brown[y1 - pad: y1 + t_px, g0:g1]}[side]
            kind = "door" if region.mean() > 0.01 else "window"
            openings.append({"wall": side, "kind": kind,
                             "start_m": round(a * s_ax, 3), "end_m": round(b * s_ax, 3),
                             "width_m": round((b - a) * s_ax, 3)})

    # 5. 3D: interior frame, x along "length" (east), z along "breadth" (south), y up; origin = room centre
    import trimesh
    T = t_px * m_per_px
    parts = []

    def box(xa, xb, ya, yb, za, zb, colour):
        b = trimesh.creation.box(extents=[xb - xa, yb - ya, zb - za])
        b.apply_translation([(xa + xb) / 2 - L / 2, (ya + yb) / 2, (za + zb) / 2 - W / 2])
        b.visual.face_colors = colour
        parts.append(b)

    wall_c, lintel_c, floor_c = [214, 214, 210, 255], [200, 200, 196, 255], [150, 152, 150, 255]
    box(-T, L + T, -0.05, 0, -T, W + T, floor_c)
    walls = {"north": ("x", -T, 0.0, L), "south": ("x", W, W + T, L), "west": ("z", -T, 0.0, W), "east": ("z", L, L + T, W)}
    for side, (axis, ta, tb, span) in walls.items():
        cuts = sorted([o for o in openings if o["wall"] == side], key=lambda o: o["start_m"])
        segments = []
        cursor = -T
        for o in cuts:
            segments.append((cursor, o["start_m"], 0.0, H))                      # solid wall up to the opening
            top = door_h if o["kind"] == "door" else args.window_top
            segments.append((o["start_m"], o["end_m"], top, H))                   # lintel
            if o["kind"] == "window":
                segments.append((o["start_m"], o["end_m"], 0.0, args.sill))       # sill
            cursor = o["end_m"]
        segments.append((cursor, span + T, 0.0, H))
        for a, b, ya, yb in segments:
            if b - a <= 1e-6 or yb - ya <= 1e-6:
                continue
            c = wall_c if (ya == 0 and yb == H) else lintel_c
            if axis == "x":
                box(a, b, ya, yb, ta, tb, c)
            else:
                box(ta, tb, ya, yb, a, b, c)

    scene = trimesh.Scene(parts)
    scene.export(args.out / "floorplan_model.glb")

    layout = {
        "source": str(args.plan.name), "units": "metres",
        "frame": "x = length (plan horizontal), z = breadth (plan vertical, down), y up; origin room centre",
        "scale_m_per_px": {"x": m_per_px, "y": my_per_px},
        "scale_from": f"known length {L} m" + (f" and breadth {B} m (axes scaled separately)" if B else ""),
        "drawing_aspect_error": round(aspect_mismatch, 4),
        "room": {"length": round(L, 3), "breadth": round(W, 3), "height": H, "wall_thickness": round(T, 3)},
        "openings": openings,
        "pixels": {"interior_bbox": [int(x0), int(y0), int(x1), int(y1)], "wall_thickness_px": t_px},
    }
    (args.out / "layout.json").write_text(json.dumps(layout, indent=2))

    # Debug overlay for the write-up: interior box, detected openings.
    vis = np.asarray(Image.open(args.plan).convert("RGB")).copy()
    vis[ring] = (0.5 * vis[ring] + 0.5 * np.array([90, 200, 160])).astype(np.uint8)
    for (yy, xx) in ((slice(y0, y0 + 3), slice(x0, x1)), (slice(y1 - 3, y1), slice(x0, x1)),
                     (slice(y0, y1), slice(x0, x0 + 3)), (slice(y0, y1), slice(x1 - 3, x1))):
        vis[yy, xx] = [224, 71, 158]
    Image.fromarray(vis).save(args.out / "floorplan_parsed.png")
    print(json.dumps({k: layout[k] for k in ("room", "openings")}, indent=2))


if __name__ == "__main__":
    main()
