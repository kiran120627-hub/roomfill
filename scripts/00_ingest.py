"""Stage 0 — extract sharp, evenly spaced frames from a room walkthrough video.

Usage (inside WSL venv ~/hn3d):
    python scripts/00_ingest.py --video data/raw/room.mp4 --work ~/hn3d_work/room

Produces:
    <work>/images/train/frame_XXXX.jpg   frames used for reconstruction
    <work>/images/heldout/*.jpg          held-out stills copied from data/heldout (never trained on)
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sharpness(img_path: Path) -> float:
    g = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def resize_into(src: Path, dst: Path, max_side: int) -> None:
    img = cv2.imread(str(src))
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--heldout", type=Path, default=ROOT / "data" / "heldout")
    ap.add_argument("--target", type=int, default=220, help="number of training frames to keep")
    ap.add_argument("--fps", type=float, default=6.0, help="candidate extraction rate")
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--start", type=float, default=0.0, help="seconds to skip at the start")
    ap.add_argument("--end", type=float, default=None, help="stop at this time (e.g. before leaving the room)")
    args = ap.parse_args()

    train_dir = args.work / "images" / "train"
    held_dir = args.work / "images" / "heldout"
    for d in (train_dir, held_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-ss", str(args.start),
             *(["-to", str(args.end)] if args.end else []), "-i", str(args.video),
             "-vf", f"fps={args.fps}", "-q:v", "2", str(tmp / "c_%05d.jpg")],
            check=True,
        )
        cands = sorted(tmp.glob("c_*.jpg"))
        if not cands:
            raise SystemExit("ffmpeg produced no frames — check the video path")
        scores = np.array([sharpness(p) for p in cands])

        # Split the timeline into `target` equal windows; keep the sharpest frame in each.
        # This keeps coverage uniform while rejecting motion-blurred frames.
        n = min(args.target, len(cands))
        edges = np.linspace(0, len(cands), n + 1).astype(int)
        keep = [a + int(np.argmax(scores[a:b])) for a, b in zip(edges[:-1], edges[1:]) if b > a]
        for i, idx in enumerate(keep):
            resize_into(cands[idx], train_dir / f"frame_{i:04d}.jpg", args.max_side)

        blur_cut = np.percentile(scores, 10)
        print(f"candidates={len(cands)} kept={len(keep)} "
              f"median_sharpness={np.median(scores):.1f} p10={blur_cut:.1f}")

    if any(p.suffix.lower() == ".heic" for p in args.heldout.glob("*")):
        print("WARNING: .heic photos are skipped — export them as JPG (iPhone: Settings > Camera > Formats > Most Compatible)")
    held = sorted(p for p in args.heldout.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for p in held:
        resize_into(p, held_dir / f"heldout_{p.stem}.jpg", args.max_side)
    print(f"heldout={len(held)}")
    if not held:
        print("WARNING: no held-out photos found — novel-view metrics will fall back to every 8th frame")


if __name__ == "__main__":
    main()
