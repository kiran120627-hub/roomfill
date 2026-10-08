"""Stage 7 — collect metrics into the ablation table + figures for the write-up and the viewer.

Rows (each is a saved run under <work>/runs/, missing ones are skipped):
    baseline     vanilla 3DGS                                  (the baseline to beat)
    depthprior   + monocular-depth prior & opacity reg         (floater suppression)
    completed    + room shell completion on top of depthprior  (RoomFill, full method)

Novel-view metrics prefer <run>/eval.json (raw + colour-aligned, written by 09_eval.py) and
fall back to <run>/metrics.json.

Usage:
    python scripts/07_report.py --work ~/hn3d_work/room --out results/room --viewer-out viewer/room
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROWS = [("baseline", "Baseline 3DGS"),
        ("depthprior", "+ depth prior, floater suppression"),
        ("completed", "+ shell completion (RoomFill)")]


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def novel(run: Path):
    m = load(run / "eval.json") or load(run / "metrics.json")
    return m["mean"] if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--viewer-out", type=Path, default=None)
    ap.add_argument("--shell-run", default="depthprior", help="run whose shell.json / scale.json are reported")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = args.work / "runs"

    q = {k: novel(runs / k) for k, _ in ROWS}
    lines = ["# Results", "", "## Novel-view quality on held-out photos", "",
             "Colour-aligned (cc) metrics fit one 3x4 colour map per image before scoring, removing the "
             "photo-mode vs video-mode exposure difference. Raw metrics are reported alongside.", "",
             "| Variant | PSNR | SSIM | LPIPS | PSNR cc | SSIM cc | LPIPS cc |", "|---|---|---|---|---|---|---|"]
    for k, name in ROWS:
        a = q[k]
        if not a:
            continue
        cc = (f"{a['psnr_cc']:.2f} | {a['ssim_cc']:.3f} | {a['lpips_cc']:.3f}" if "psnr_cc" in a else "- | - | -")
        lines.append(f"| {name} | {a['psnr']:.2f} | {a['ssim']:.3f} | {a['lpips']:.3f} | {cc} |")

    shell_run = runs / args.shell_run
    sc = load(shell_run / "scale.json")
    if sc:
        lines += ["", "## Metric room dimensions (m)", "",
                  "| Scale estimator | Length | Width | Height | L / W error vs tape (%) |", "|---|---|---|---|---|"]
        errs = sc.get("dim_error_pct_by_estimator", {})
        for k, d in sc["dims_m_by_estimator"].items():
            e = ", ".join(f"{x:.1f}" for x in errs.get(k, [])) or "-"
            chosen_key = "floorplan" if sc.get("chosen_from") == "floorplan" else (
                "depth" if abs(sc["estimators"]["depth"]["scale"] - sc["chosen_scale"]) < 1e-12 else "camera")
            tag = " (chosen)" if k == chosen_key else (" (reference only)" if k == "tape" else "")
            lines.append(f"| {k}{tag} | {d['length']:.2f} | {d['width']:.2f} | {d['height']:.2f} | {e} |")
        if "tape" in sc:
            t = sc["tape"]
            lines.append(f"| tape (ground truth) | {t['room_length_m']} | {t['room_width_m']} | "
                         f"{t.get('ceiling_height_m', '-')} | - |")

    geo = load(args.out / "geometry.json")
    if geo:
        lines += ["", "## Geometry vs the tape-measured room box", "",
                  "| Variant | Chamfer (cm) | Accuracy (cm) | Completeness (cm) | Coverage @10 cm |", "|---|---|---|---|---|"]
        for k, name in (("baseline", "Baseline 3DGS"), ("completed", "RoomFill")):
            g = geo.get(k)
            if g:
                lines.append(f"| {name} | {100 * g['chamfer_m']:.1f} | {100 * g['accuracy_m']:.1f} | "
                             f"{100 * g['completeness_m']:.1f} | {100 * g['coverage_10cm']:.0f}% |")

    mc = load(runs / "completed" / "metrics.json")
    if mc and mc.get("faces"):
        lines += ["", "## Shell coverage (fraction of each surface)", "",
                  "| Face | Observed | Opening (seen through) | Generated | Method |", "|---|---|---|---|---|"]
        for f, r in mc["faces"].items():
            lines.append(f"| {f} | {r['observed']:.0%} | {r['opening']:.0%} | {r['generated']:.0%} | {r['method']} |")

    (args.out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    for src in [shell_run / "shell_topdown.png"]:
        if src.exists():
            shutil.copy(src, args.out / src.name)
    comp = runs / "completed"
    if (comp / "faces").exists():
        shutil.copytree(comp / "faces", args.out / "faces", dirs_exist_ok=True)
    for k in ("baseline", "completed"):
        for sub in ("eval_renders", "test_renders"):
            if (runs / k / sub).exists():
                shutil.copytree(runs / k / sub, args.out / f"renders_{k}", dirs_exist_ok=True)
                break

    if args.viewer_out:
        def pick(a):   # the viewer shows colour-aligned numbers when available
            return None if not a else {"psnr": a.get("psnr_cc", a["psnr"]), "ssim": a.get("ssim_cc", a["ssim"]),
                                       "lpips": a.get("lpips_cc", a["lpips"])}
        best = q.get("completed") or q.get("depthprior")
        vm = {"quality": {"baseline": pick(q["baseline"]), "completed": pick(best)},
              "quality_is_colour_aligned": bool(q["baseline"] and "psnr_cc" in q["baseline"])}
        if geo:
            vm["geometry"] = {"baseline": geo["baseline"], "completed": geo["completed"]}
        args.viewer_out.mkdir(parents=True, exist_ok=True)
        (args.viewer_out / "metrics.json").write_text(json.dumps(vm, indent=2))


if __name__ == "__main__":
    main()
