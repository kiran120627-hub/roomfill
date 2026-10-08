"""Smoke-test every deliverable: export files open in their libraries, carry the honesty
attributes, and agree with the reported numbers.

    python scripts/check_outputs.py --export ~/hn3d_work/room/runs/completed/export --viewer viewer/room
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, type=Path)
    ap.add_argument("--viewer", required=True, type=Path)
    ap.add_argument("--floorplan", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "floorplan")
    args = ap.parse_args()
    ex = args.export

    room = json.loads((ex / "room.json").read_text())
    d = room["dimensions"]
    check("room.json dimensions are metric and plausible",
          all(1.5 < d[k] < 30 for k in ("length", "width")) and 2.0 < d["height"] < 6, str(d))
    check("room.json records the scale source", room.get("scale_from") in ("floorplan", "video"), room.get("scale_from"))
    check("room.json has a home camera inside the room",
          "home_camera" in room and abs(room["home_camera"]["position"][0]) < d["length"] / 2
          and abs(room["home_camera"]["position"][2]) < d["width"] / 2)

    from plyfile import PlyData
    ply = PlyData.read(str(ex / "room_splat.ply"))["vertex"]
    names = ply.data.dtype.names
    check("PLY opens and has 3DGS fields", all(n in names for n in ("x", "f_dc_0", "opacity", "scale_0", "rot_0")))
    check("PLY carries generated + confidence attributes", "generated" in names and "confidence" in names)
    gen = np.asarray(ply["generated"]) > 0.5
    check("PLY generated count matches room.json", int(gen.sum()) == room["gaussians"]["generated"],
          f"{int(gen.sum())} vs {room['gaussians']['generated']}")
    xyz = np.stack([ply["x"], ply["y"], ply["z"]], 1)
    inside = (np.abs(xyz[gen, 0]) <= d["length"] / 2 + 0.5) & (np.abs(xyz[gen, 2]) <= d["width"] / 2 + 0.5) \
        & (xyz[gen, 1] > -0.5) & (xyz[gen, 1] < d["height"] + 0.5)
    check("generated Gaussians lie on the room shell (inside the box)", inside.mean() > 0.99, f"{inside.mean():.3f}")
    conf = np.asarray(ply["confidence"])
    check("confidence in [0,1], observed = 1", (conf >= 0).all() and (conf <= 1).all() and (conf[~gen] == 1).all())

    for f in ("room.splat", "room_honesty.splat"):
        n = (ex / f).stat().st_size
        check(f"{f} is a whole number of 32-byte splats", n % 32 == 0 and n // 32 == len(xyz), f"{n // 32} splats")

    import trimesh
    for f in ("room_shell.glb", "room_shell_honesty.glb"):
        sc = trimesh.load(str(ex / f))
        check(f"{f} opens with textured faces", len(sc.geometry) >= 4, f"{len(sc.geometry)} meshes")

    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(str(ex / "room.usdc"))
    pts = UsdGeom.Points(st.GetPrimAtPath("/Room/Splat"))
    pv = UsdGeom.PrimvarsAPI(pts)
    check("USD opens, Y-up, metres", UsdGeom.GetStageUpAxis(st) == "Y" and UsdGeom.GetStageMetersPerUnit(st) == 1.0)
    check("USD splat has generated + confidence primvars",
          pv.HasPrimvar("generated") and pv.HasPrimvar("confidence"))

    objs = ex / "objects" / "objects.json"
    if objs.exists():
        o = json.loads(objs.read_text())
        check("objects.json lists objects with boxes", o["objects"] > 0 and all("size_m" in i for i in o["items"]),
              f"{o['objects']} objects")
        check("objects_boxes.glb opens", len(trimesh.load(str(ex / "objects" / "objects_boxes.glb")).geometry) == o["objects"])

    lay = args.floorplan / "layout.json"
    if lay.exists():
        L = json.loads(lay.read_text())
        check("floor plan layout parsed with a door", any(op["kind"] == "door" for op in L["openings"]),
              f"{L['room']['length']} x {L['room']['breadth']} m")
        check("floor plan GLB opens", len(trimesh.load(str(args.floorplan / "floorplan_model.glb")).geometry) > 4)

    v = args.viewer
    for f in ("room.json", "metrics.json", "room.splat", "room_honesty.splat"):
        check(f"viewer has {f}", (v / f).exists())
    if (v / "room.json").exists():
        check("viewer room.json is the latest export",
              json.loads((v / "room.json").read_text()) == room)

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
