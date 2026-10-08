# RoomFill: judge Q&A cheat sheet

Short answers first. The numbers are the official evaluation (full resolution, same frames for baseline and RoomFill).

## The 30-second pitch

A phone walkthrough of a room goes in. Out comes a 3D model at real-world scale, with the parts no camera saw filled in and marked in magenta. On our Karunya classroom, shape error against a tape measure fell from **22.2 cm to 15.5 cm**, and **90%** of the room surface is within 25 cm of the truth (baseline: **52%**). It runs offline on one laptop GPU, costs **₹0** per run, and uses **no LLM**.

## Key numbers

| What | Baseline 3DGS | RoomFill |
|---|---|---|
| Chamfer distance vs tape box (lower is better) | 22.2 cm | **15.5 cm** (about 30% better) |
| Room surface within 25 cm of truth | 52% | **90%** |
| PSNR on 5 hidden photos (colour-aligned) | 15.79 | **16.15** |
| PSNR raw | 10.68 | **11.26** |
| SSIM (higher is better) | 0.720 | **0.724** |
| LPIPS (lower is better) | 0.605 | **0.601** |

* Room size error vs tape: length 3.7%, width 4.0%, height 6.4%
* Tape: 9.23 × 8.88 m floor, 3.5 m ceiling, door 1.0 × 2.05 m
* Video: 57 s at 1080p, first 51 s used; 400 sharp frames (+78 extra); 5 hidden test photos
* 36 objects found: 18 desk/bench units, 6 fans/lights, 3 cupboards, 2 podiums, 7 other
* Sparse mode: 30 frames vs 400 frames gives 17.06 vs 17.13 PSNR on the views both cover
* Run time on an RTX 5070 laptop: COLMAP about 15 min (CPU), training about 30 min, the rest a few minutes

## Likely questions

**Does it use ChatGPT or any LLM?**
No. There is no language model anywhere. We use COLMAP (classic geometry), 3D Gaussian Splatting (trained fresh for each room), LaMa (an image inpainting model) and Depth Anything V2 (only as a scale sanity check). Everything runs locally, and the API cost is zero.

**What is new here? Isn't this just Gaussian Splatting?**
Gaussian Splatting is stage 3 of 10. Our contribution is stages 7 to 9. A visibility test checks every point on the room's walls, floor and ceiling against all 400 camera views. Only points no camera ever saw get filled, and every filled point is tagged "generated" with a confidence that survives into PLY, GLB and USD. Plain splatting leaves holes and floaters, and it never tells you what was guessed.

**How do you know which parts were never seen?**
For each small patch (texel) of the fitted room shell, we project it into every training camera. Using the rendered depth, we check whether something in front blocked it. If no camera saw it directly, it's "unseen". If a camera saw through it (a door or window), it's an "opening" and we leave it alone.

**How do you get real-world scale? A video has no scale.**
From the floor plan: its labelled 9.23 m length fixes the size. If there's no plan, we fall back to a camera-height prior of about 1.4 m, because people hold phones at roughly chest height. We also tried a depth model, but it was about 2× off in this room. A sanity check caught that and rejected it, and we report that honestly.

**How do you know the results are accurate?**
We tape-measured the room ourselves. Geometry is scored against a box built from those measurements (Chamfer distance and coverage). Image quality is scored on 5 photos taken from new positions that the model never saw in training.

**Why is PSNR only around 16? Other papers report 25+.**
Papers test on frames taken from the same video. We test on still photos taken from new positions with the phone's photo mode. That's much harder, and the colours differ between photo and video mode. That's why we also show colour-aligned numbers and the raw ones. We beat the baseline on all of them.

**What's "colour-aligned"?**
For each test photo we fit one colour correction (a 3×4 matrix) between our render and the photo, the same for baseline and RoomFill. It removes the camera's exposure difference without changing the geometry being judged. The raw numbers are reported too.

**Why are the website numbers slightly different from the slides?**
The website shows a higher-visual-quality display model (more Gaussians, longer training): PSNR 16.26, Chamfer 18.3 cm. The slides use the official evaluation run (16.15, 15.5 cm). Both beat the baseline, and the write-up documents both.

**The website shows "Cover 10 cm" at only about 20%. Why so low?**
10 cm is a very strict threshold for a room reconstructed from a phone video with no depth sensor. At 25 cm we cover 90% vs the baseline's 52%. Chamfer (average error) is the headline metric.

**Is the filled-in ceiling real?**
No, and we say so. It's a plausible completion from LaMa, guided by what the camera did see nearby. That's exactly why every generated point is marked magenta, with lower confidence further from real evidence. Press **G** on the website to see it.

**Can I measure things in the 3D model?**
Yes. Press **T** on the website and click two points; the distance appears in metres, split into vertical (↕) and horizontal (↔). The model is in real metres, so no conversion is needed. Classroom desks read about 1.1 m.

**What are the limitations?**
Rectangular (Manhattan) rooms only. We complete surfaces (walls, floor, ceiling), not furniture hidden behind other furniture. Scale needs a floor plan or the camera-height prior. A fast, blurry walkthrough reduces quality.

**What does the floor-plan mode (Mode A) do?**
It reads the plan image with classic image processing: dark pixels for walls, gaps for doors and windows, and one labelled length for scale. It then extrudes a 3D room model (GLB). In the video pipeline, the same plan sets the scale. Its door lines up 77% with the doorway the video saw through, vs 0% for the other orientations.

**How are the 36 objects found?**
Everything confidently observed that isn't wall, floor or ceiling is turned into a 3D grid of small cubes (voxels). Each connected group becomes one object, with its own splat file and bounding box. Labels come from size (desk/bench, fan/light, cupboard, podium).

**Why no people in the public repo?**
The raw video and test photos show people in the classroom, so they stay private. The code, results and the reconstructed room are public.

**How much would it cost to run for a new room?**
₹0. There are no paid APIs. It needs a laptop with an NVIDIA GPU and takes under an hour, end to end.

**What would you do with more time?**
Non-rectangular rooms (fit more wall planes), completing hidden furniture with a 3D shape prior, and an upload page where anyone can drop in a video.

## Live demo order (about 1 minute)

1. Open `kiran120627-hub.github.io/roomfill/viewer`
2. Drag to look around, scroll to walk.
3. **G**: magenta shows what was generated.
4. **B**: the baseline, with its floaters. Press **B** again to come back.
5. **2**: look up at the ceiling the video never filmed.
6. **T**, then click both ends of a desk: real length in metres. **Esc** to finish.
