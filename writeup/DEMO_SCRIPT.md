# RoomFill demo script (about 3 minutes)

## Before the judges arrive (5 minutes earlier)
1. Laptop plugged in, lid open, power mode on Best performance.
2. Start the viewer (works with no internet):
   ```
   python hacknex-3d/viewer/serve.py
   ```
3. Open http://localhost:8766/?scene=room in Chrome, wait until the classroom appears, press **R**.
4. Open these in tabs or an image viewer, in this order:
   * `results/slides/2_honesty_view1.jpg`
   * `results/slides/1_floaters_heldout.jpg`
   * `results/floorplan/floorplan_parsed.png`
   * `results/room/RESULTS.md`
5. Have the phone with the original classroom video ready.

## 1. The problem (20 s)
> "We took HNX26EPS06 Mode B. From a one-minute phone video of a room, we build a 3D scene you can walk
> through. The hard part is the parts the camera never saw, like the ceiling or the wall behind the desks.
> We fill those in, and we always show exactly which parts we generated."

Show the phone video for 3 seconds.

## 2. Live viewer (60 s)
* Drag to look around, scroll to walk forward.
  > "This is our classroom, reconstructed from that video. 400 frames, camera positions from COLMAP,
  > 3D Gaussian Splatting on this laptop's GPU in about 8 minutes."
* Press **M** for the measurements panel.
  > "The room is 9.60 by 8.55 by 3.28 metres. Tape says 9.23 by 8.88 by 3.5, so we are within 4 percent
  > on the floor and 6.4 percent on height."
* Look up at the ceiling, press **G**.
  > "Magenta is everything no camera ever saw. Every generated point carries a flag and a confidence
  > in the exported file, so nothing is silently made up. 25 percent of the points are generated."
* Press **G** again to go back.

## 3. How it works (30 s)
Show `2_honesty_view1.jpg`.
> "We fit the room's shell: floor, ceiling, four walls. Then for every point on those surfaces we check
> all 400 camera views. If a camera saw it, we keep the real photo colour. If a camera saw through it,
> it is a window or door and we never paint over it. Only if no camera ever saw it do we fill it with
> LaMa inpainting."

## 4. Results vs the baseline (40 s)
Show `RESULTS.md` (geometry table).
> "Against the tape-measured room, Chamfer distance drops from 21 cm to 14.9 cm, and coverage of the
> true walls goes from 57 to 95 percent. On held-out photos we are slightly better than vanilla
> Gaussian splatting on PSNR, SSIM and LPIPS."

Show `1_floaters_heldout.jpg`.
> "From a new viewpoint the baseline is a wall of floaters. Our depth prior removes them."

## 5. Bonus features (20 s)
Show `floorplan_parsed.png`.
> "We also did Mode A: the floor plan parser finds walls, the door and the dimensions and builds a 3D
> model, and it gives the video reconstruction its real-world scale. We split furniture into 37
> objects and export PLY, GLB and USD."

## 6. Honest limitations (10 s)
> "We assume a rectangular room, we complete surfaces rather than inventing hidden furniture, and the
> small monocular depth model was 2x off in this big bright room, so we built a sanity check that
> falls back to a camera-height prior."

## Likely questions
| Question | Answer |
|---|---|
| What is the baseline? | Vanilla 3D Gaussian Splatting (gsplat), same 400 frames, same 15,000 iterations. |
| How do you know what was unseen? | Each wall, floor and ceiling point is projected into all 400 training cameras and compared with the rendered depth there. |
| Why colour-aligned metrics? | The held-out photos are from the phone's photo mode, which has a different exposure from video mode. We report raw numbers too. |
| Does it work on a new room? | Yes, `./run_all.sh <name> <video>` runs the whole pipeline. A floor plan and tape measurements are optional. |
| Where are the files? | `viewer/room/` holds the PLY, .splat and GLB; the USD is in the export folder. |
| What if the room is not rectangular? | It gets a bounding box. That is a stated limitation. |
