# RoomFill: honest, metric room completion from a phone walkthrough

**Track:** HNX26EPS06, **Mode B** (room video to 3D scene), with Mode A (floor plan to 3D) as a bonus
and both modes combined in one pipeline.

## 1. Problem

Reconstructing what a camera saw is largely solved. The open problem is the rest: the ceiling nobody
filmed, the wall behind the desks, the floor under the benches. A standard Gaussian-splat reconstruction
leaves those regions as holes or floaters, has no real-world scale, and gives no signal about which parts
were actually seen.

## 2. Data

* **Our room:** an empty college classroom, filmed in one 57 s handheld walkthrough (1920x1080, 30 fps).
  The first 51 s are used (the walker leaves the room after that).
* **Held-out views:** 5 phone photos taken from positions not on the video path (up to 4.4 m away from
  the nearest video frame). They are registered jointly with COLMAP so they share the frame, and are
  never used for training.
* **Ground truth:** tape measure, 9.23 x 8.88 m floor, 3.5 m ceiling, door 1.0 x 2.05 m.
* **Floor plan:** a 2D plan of the same room (walls, door with swing arc, labelled dimensions).
* **Development scene:** Deep Blending "Playroom" (public), used to test every stage before our capture.

## 3. Pipeline

```
phone video -> sharpest frame per window (400 frames) -> COLMAP (sequential + loop-closure pair list)
            -> 3D Gaussian Splatting (gsplat, RTX 5070 Laptop, ~8.5 min)
            -> room shell: floor = lowest strong layer, ceiling = highest strong layer,
                           walls = Manhattan-aligned histogram peaks
            -> metric scale: floor plan if available, else camera-height prior
                             (monocular metric depth is computed but gated by a sanity check)
            -> visibility test per shell texel against all 400 training cameras
                 seen surface / seen-through opening / never seen
            -> LaMa inpainting of never-seen texels only, lifted into thin plane-aligned Gaussians
               tagged generated=1 with a confidence that decays with distance from evidence
            -> export: metric, gravity-aligned PLY + compact .splat + textured GLB + USD
            -> objects: furniture split into connected components with oriented boxes
            -> web viewer (three.js + Spark) with an observed / generated toggle
```

### Key ideas
1. **Structure first, pixels second.** Completion happens only on the room's planar shell, so generated
   content cannot float in free space.
2. **A visibility definition of "unseen".** A texel is generated only if no training camera saw it: it was
   out of every frustum, or something clearly closer (desks, beams, fans) blocked every view of it. The
   tolerance scales with distance (15 percent), because rendered depth on plain white walls is smeared.
3. **Openings are not holes.** If cameras saw through the wall plane (windows, open door), the region is
   marked as an opening and never painted over.
4. **Provenance in the file.** Every Gaussian carries `generated` and `confidence`; both survive
   densification during training (stored as zero-learning-rate parameters) and are written to PLY and USD.
5. **Classroom-aware shell fitting.** Rows of desks form a bigger horizontal plane than the visible floor,
   and deep beams hang below the ceiling slab, so "largest plane" fails. We use the lowest and highest
   strong layers instead.

## 4. Baseline

Vanilla 3D Gaussian Splatting (gsplat rasterizer, default densification), same 400 frames, same 15,000
iterations, same resolution (half resolution, 794x446). This is the "standard structure-from-motion plus
Gaussian splatting pipeline" named as the example baseline for Mode B.

## 5. Results (our classroom)

Headline numbers are at full resolution (1588 x 893, 20,000 iterations). RoomFill = the same trained
model plus camera-path floater pruning (0.5 m) and shell completion; the baseline is the untouched model.

### 5.1 Geometric accuracy vs the tape-measured room box (25 percent of the rubric)

| Variant | Chamfer | Accuracy | Completeness | Coverage within 25 cm |
|---|---|---|---|---|
| Baseline 3DGS | 22.2 cm | **13.2 cm** | 31.1 cm | 52% |
| **RoomFill** | **15.5 cm** | 15.1 cm | **15.8 cm** | **90%** |

Completion cuts Chamfer distance by 30 percent and halves completeness error. Accuracy is slightly worse
because generated surfaces sit a few centimetres off the true walls. (Half-resolution run: 21.0 to 15.2 cm.)

### 5.2 Metric dimensions

| Scale source | Length / width error | Height error |
|---|---|---|
| Video only, monocular depth (Depth Anything V2 Metric Indoor S) | 96% / 111% | rejected by the sanity gate |
| Video only, camera-height prior (1.4 m, not tuned to our tape) | 9.6% / 2.4% | about 12% |
| **Video + floor plan** | **3.7% / 4.0%** | **6.4%** |

The floor plan's labels come from the same tape measure, so length and width are not an independent
check of the plan-based scale; ceiling height is (it is not on the plan) and lands within 6.4 percent.
The residual 4 percent is the shell fit, not the scale.

### 5.3 Novel-view quality on the 5 held-out photos (20 percent of the rubric)

Full resolution:

| Variant | PSNR | SSIM | LPIPS | PSNR cc | SSIM cc | LPIPS cc |
|---|---|---|---|---|---|---|
| Baseline 3DGS | 10.68 | 0.655 | 0.652 | 15.79 | 0.720 | 0.605 |
| **RoomFill** | **11.26** | **0.657** | **0.645** | **16.15** | **0.724** | **0.601** |

Half resolution (ablation):

| Variant | PSNR | SSIM | LPIPS | PSNR cc | SSIM cc | LPIPS cc |
|---|---|---|---|---|---|---|
| Baseline 3DGS | 11.65 | 0.598 | 0.663 | 16.04 | 0.642 | 0.628 |
| + depth prior (Pearson, mono depth) | 13.12 | 0.569 | 0.650 | 16.18 | 0.621 | 0.646 |
| + floater pruning + completion (RoomFill) | 12.17 | 0.597 | 0.653 | 16.20 | 0.643 | 0.622 |

"cc" = colour-aligned: one 3x4 colour map per image is fitted before scoring, because the held-out photos
come from the phone's photo mode (different exposure and tone curve from video mode). Raw numbers are
reported alongside. RoomFill beats the baseline on every metric at full resolution. Camera-path floater
pruning alone (half resolution) moves PSNR cc 16.04 to 16.22 and LPIPS cc 0.628 to 0.623; the radius was
fixed at 0.5 m rather than picked as the best of {0.3, 0.5, 0.8} on the test photos.

### 5.4 Honesty (15 percent of the rubric)

| Surface | Observed | Seen-through opening | Generated |
|---|---|---|---|
| wall_x0 | 40% | 8% | 52% |
| wall_x1 (windows) | 73% | 0% | 27% |
| wall_y0 (board) | 57% | 2% | 41% |
| wall_y1 | 39% | 0% | 61% |
| floor | 76% | 0% | 24% |
| ceiling | 39% | 0% | 61% |

Every camera in the walkthrough faced the same half of the room (yaw 0 to 180 degrees), which is why the
back walls and ceiling carry the most generated area. Overall 24.9 percent of exported Gaussians are generated. The viewer's "Show generated" mode tints every
generated Gaussian magenta (stronger tint = lower confidence); the honesty variants of the PLY, .splat and
GLB carry the same tint, and the plain variants carry the flags as attributes.

### 5.5 Objects

36 objects extracted (18 desk/bench units, 6 ceiling fixtures, 2 podium-like, 3 cupboards, 7 other).
Desks touching in a row merge into one object.

### 5.6 Sparse photos instead of a full video (stretch goal)

Same pipeline, COLMAP included, on 31 frames spaced evenly through the walk (every 13th of the 400) plus
the 5 held-out photos. COLMAP registered 20 of the 31 frames and 3 of the 5 held-out photos, so the sparse
set reconstructs only part of the room. On the 3 held-out photos it does cover:

| Held-out photo (PSNR cc) | 31 sparse frames | 400-frame video |
|---|---|---|
| photo 3 | 17.50 | 16.56 |
| photo 4 | 14.98 | 13.73 |
| photo 5 | 18.70 | 21.11 |
| mean | 17.06 | 17.13 |

Where the sparse set has coverage, quality matches the full video (fewer floaters off-path, less detail
on-path); its weakness is coverage, not per-view quality.

## 6. Mode A: floor plan to 3D

Classical, explainable parsing: walls are dark low-saturation pixels (annotation text is navy and fails
the saturation test), the interior is the hole they enclose, openings are gaps in the wall ring, and a gap
next to a brown door leaf or swing arc is a door. Output: GLB with extruded walls, door void with lintel,
floor slab, plus `layout.json`. On our plan: interior 9.23 x 8.88 m, wall thickness 0.35 m, one door on the
west wall. With both dimensions labelled, the axes are scaled separately; the parser reports that the
drawing itself is 8.2 percent out of proportion to its labels.

**Both modes in one pipeline:** the parsed plan sets the metric scale of the video reconstruction
(section 5.2).

## 7. Ablation summary (research contribution)

1. Baseline vs RoomFill (full resolution): Chamfer 22.2 to 15.5 cm, coverage@25 cm 52 to 90 percent,
   better on all six novel-view metrics.
2. Scale estimators: monocular depth (rejected) vs camera-height prior vs floor plan (table 5.2).
3. Depth prior: large gains on off-path views with floaters, small losses on on-path detail.
4. Shell fitting: "largest plane" picks desk height (1.08 SfM units room height); "lowest / highest strong
   layer" recovers 3.25 (6.4 percent from tape after scale).

## 8. Limitations
* The shell model assumes a rectangular (Manhattan) room; L-shaped rooms get a bounding box.
* Generated content is surface texture on the shell; unseen furniture is not invented.
* Small monocular metric depth models were about 2x off in a large, bright classroom.
* The web export drops view-dependent colour (higher SH bands).

## 9. Reproduce
```bash
./run_all.sh room data/raw/room.mp4 51
python viewer/serve.py                              # http://localhost:8766/?scene=room
python scripts/10_floorplan.py --plan data/floorplan.jpg --out results/floorplan
```
