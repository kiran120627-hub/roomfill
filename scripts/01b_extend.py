"""Stage 1b — densify a weakly covered stretch of the walkthrough without redoing SfM.

The 42-48 s stretch of our video (TV wall, back door) is soft, so the uniform frame picker kept
few frames there and that part of the room reconstructs poorly. This samples that stretch densely,
keeps the sharpest frame per short window, matches the new frames against their neighbours in time
and against the existing frames, registers them into the existing COLMAP model, re-runs bundle
adjustment and writes a new undistorted model.

Usage:
    python scripts/01b_extend.py --work ~/hn3d_work/room --video data/raw/room.mp4 --start 38 --end 51
Produces:
    <work>/images/train/extra_XXXX.jpg, <work>/colmap/sparse_ext/, <work>/dense_ext/
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def run(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def sharpness(p: Path) -> float:
    return float(cv2.Laplacian(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), cv2.CV_64F).var())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--video-end", type=float, default=51.0, help="end time used by 00_ingest (frame-index -> time)")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--per-second", type=float, default=6.0, help="frames kept per second of video")
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--threads", default="12")
    args = ap.parse_args()

    images = args.work / "images"
    train = images / "train"
    col = args.work / "colmap"
    db = col / "database.db"
    for old in train.glob("extra_*.jpg"):
        old.unlink()

    # 1. dense candidates in the stretch, sharpest per window
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        run("ffmpeg", "-loglevel", "error", "-ss", str(args.start), "-to", str(args.end), "-i", str(args.video),
            "-vf", f"fps={args.fps}", "-q:v", "2", str(tmp / "c_%05d.jpg"))
        cands = sorted(tmp.glob("c_*.jpg"))
        scores = np.array([sharpness(p) for p in cands])
        n = max(1, int((args.end - args.start) * args.per_second))
        edges = np.linspace(0, len(cands), n + 1).astype(int)
        keep = [a + int(np.argmax(scores[a:b])) for a, b in zip(edges[:-1], edges[1:]) if b > a]
        new = []
        for i, idx in enumerate(keep):
            img = cv2.imread(str(cands[idx]))
            h, w = img.shape[:2]
            s = args.max_side / max(h, w)
            if s < 1:
                img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            name = f"extra_{i:04d}.jpg"
            cv2.imwrite(str(train / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            new.append(f"train/{name}")
    print(f"extra frames: {len(new)}", flush=True)

    # 2. features for the new images only
    lst = col / "extra_list.txt"
    lst.write_text("\n".join(new) + "\n")
    run("colmap", "feature_extractor", "--database_path", str(db), "--image_path", str(images),
        "--image_list_path", str(lst), "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.single_camera_per_folder", "1", "--SiftExtraction.max_num_features", "12000",
        "--SiftExtraction.num_threads", args.threads, "--SiftExtraction.use_gpu", "0")

    # 3. pairs: new-new neighbours, new vs existing frames from the same time span (+ margin)
    existing = sorted(p.relative_to(images).as_posix() for p in train.glob("frame_*.jpg"))
    t_of = lambda i: i * args.video_end / len(existing)
    near = [f for i, f in enumerate(existing) if args.start - 4 <= t_of(i) <= args.end + 1]
    pairs = set()
    for i, a in enumerate(new):
        for b in new[i + 1:i + 9]:
            pairs.add((a, b))
        for b in near:
            pairs.add((a, b))
    pf = col / "extra_pairs.txt"
    pf.write_text("".join(f"{a} {b}\n" for a, b in sorted(pairs)))
    print(f"pairs: {len(pairs)} (against {len(near)} existing frames)", flush=True)
    run("colmap", "matches_importer", "--database_path", str(db), "--match_list_path", str(pf),
        "--match_type", "pairs", "--SiftMatching.num_threads", args.threads,
        "--SiftMatching.guided_matching", "1", "--SiftMatching.use_gpu", "0")

    # 4. register into the existing model, refine, undistort
    models = sorted(p for p in (col / "sparse").iterdir() if p.is_dir())
    base = max(models, key=lambda m: (m / "images.bin").stat().st_size)
    ext = col / "sparse_ext"
    if ext.exists():
        shutil.rmtree(ext)
    ext.mkdir(parents=True)
    run("colmap", "image_registrator", "--database_path", str(db), "--input_path", str(base),
        "--output_path", str(ext), "--Mapper.abs_pose_min_num_inliers", "15",
        "--Mapper.abs_pose_min_inlier_ratio", "0.15")
    run("colmap", "bundle_adjuster", "--input_path", str(ext), "--output_path", str(ext))
    dense = args.work / "dense_ext"
    if dense.exists():
        shutil.rmtree(dense)
    run("colmap", "image_undistorter", "--image_path", str(images), "--input_path", str(ext),
        "--output_path", str(dense), "--output_type", "COLMAP")
    run("colmap", "model_analyzer", "--path", str(dense / "sparse"))


if __name__ == "__main__":
    main()
