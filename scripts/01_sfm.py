"""Stage 1 — camera poses with COLMAP (train frames + held-out stills in one model).

Matching uses an explicit pair list tuned for handheld video of a room:
  * sequential: every frame against its next `--overlap` frames (survives fast turns
    past plain walls, where frames far apart share nothing),
  * loop closure: every `--loop-step`-th frame against every other one (ties the end
    of the walk back to the start so the room closes into one model),
  * held-out stills against every training frame (they come from new positions).
Held-out stills are registered into the same model so their poses share the frame,
but later stages never train on them.

Usage:
    python scripts/01_sfm.py --work ~/hn3d_work/room
Produces:
    <work>/colmap/sparse/<k>/        COLMAP model(s); the largest is used
    <work>/dense/images, sparse/     undistorted PINHOLE model consumed by the trainer
"""
import argparse
import subprocess
from pathlib import Path


def run(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def write_pairs(images: Path, out: Path, overlap: int, loop_step: int) -> int:
    train = sorted(p.relative_to(images).as_posix() for p in (images / "train").glob("*.jpg"))
    held = sorted(p.relative_to(images).as_posix() for p in (images / "heldout").glob("*.jpg"))
    pairs = set()
    for i, a in enumerate(train):
        for b in train[i + 1:i + 1 + overlap]:
            pairs.add((a, b))
    keys = train[::loop_step]
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            pairs.add((a, b))
    for h in held:
        for t in train:
            pairs.add((h, t))
        for h2 in held:
            if h < h2:
                pairs.add((h, h2))
    out.write_text("".join(f"{a} {b}\n" for a, b in sorted(pairs)))
    print(f"pairs={len(pairs)} train={len(train)} heldout={len(held)}", flush=True)
    return len(pairs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--threads", default="18")
    ap.add_argument("--overlap", type=int, default=20)
    ap.add_argument("--loop-step", type=int, default=6)
    ap.add_argument("--max-features", default="12000")
    args = ap.parse_args()

    images = args.work / "images"
    col = args.work / "colmap"
    db = col / "database.db"
    sparse = col / "sparse"
    if sparse.exists():
        import shutil
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)
    if db.exists():
        db.unlink()

    # One camera per folder: video frames share intrinsics; phone stills differ from video mode.
    run("colmap", "feature_extractor",
        "--database_path", str(db), "--image_path", str(images),
        "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.single_camera_per_folder", "1",
        "--SiftExtraction.max_num_features", args.max_features,
        "--SiftExtraction.num_threads", args.threads,
        "--SiftExtraction.use_gpu", "0")

    pairs = col / "pairs.txt"
    write_pairs(images, pairs, args.overlap, args.loop_step)
    run("colmap", "matches_importer",
        "--database_path", str(db), "--match_list_path", str(pairs), "--match_type", "pairs",
        "--SiftMatching.num_threads", args.threads,
        "--SiftMatching.guided_matching", "1",
        "--SiftMatching.use_gpu", "0")

    # Relaxed registration thresholds: plain classroom walls give few, but valid, matches.
    run("colmap", "mapper",
        "--database_path", str(db), "--image_path", str(images),
        "--output_path", str(sparse),
        "--Mapper.num_threads", args.threads,
        "--Mapper.init_min_num_inliers", "60",
        "--Mapper.abs_pose_min_num_inliers", "15",
        "--Mapper.abs_pose_min_inlier_ratio", "0.15",
        "--Mapper.ba_global_function_tolerance", "0.000001")

    models = sorted(p for p in sparse.iterdir() if p.is_dir())
    if not models:
        raise SystemExit("COLMAP produced no model — footage likely too blurry or textureless")
    # Pick the model with the most registered images (COLMAP may split into several).
    best = max(models, key=lambda m: (m / "images.bin").stat().st_size)
    print(f"models={len(models)} using={best.name}")

    run("colmap", "image_undistorter",
        "--image_path", str(images), "--input_path", str(best),
        "--output_path", str(args.work / "dense"), "--output_type", "COLMAP")

    run("colmap", "model_analyzer", "--path", str(args.work / "dense" / "sparse"))


if __name__ == "__main__":
    main()
