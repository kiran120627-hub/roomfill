# Results

## Novel-view quality on held-out photos

Colour-aligned (cc) metrics fit one 3x4 colour map per image before scoring, removing the photo-mode vs video-mode exposure difference. Raw metrics are reported alongside.

| Variant | PSNR | SSIM | LPIPS | PSNR cc | SSIM cc | LPIPS cc |
|---|---|---|---|---|---|---|
| Baseline 3DGS | 11.65 | 0.598 | 0.663 | 16.04 | 0.642 | 0.628 |
| + depth prior, floater suppression | 13.12 | 0.569 | 0.650 | 16.18 | 0.621 | 0.646 |
| + shell completion (RoomFill) | 11.67 | 0.598 | 0.658 | 16.10 | 0.643 | 0.627 |

## Metric room dimensions (m)

| Scale estimator | Length | Width | Height | L / W error vs tape (%) |
|---|---|---|---|---|
| depth | 19.50 | 17.37 | 6.66 | 95.6, 111.2 |
| camera | 9.01 | 8.03 | 3.08 | 9.6, 2.4 |
| floorplan (chosen) | 9.60 | 8.55 | 3.28 | 3.7, 4.0 |
| tape (reference only) | 9.60 | 8.55 | 3.28 | 3.7, 4.0 |
| tape (ground truth) | 9.23 | 8.88 | 3.5 | - |

## Geometry vs the tape-measured room box

| Variant | Chamfer (cm) | Accuracy (cm) | Completeness (cm) | Coverage @10 cm |
|---|---|---|---|---|
| Baseline 3DGS | 21.0 | 13.0 | 29.1 | 20% |
| RoomFill | 14.9 | 15.2 | 14.6 | 30% |

## Shell coverage (fraction of each surface)

| Face | Observed | Opening (seen through) | Generated | Method |
|---|---|---|---|---|
| wall_x0 | 26% | 6% | 68% | lama |
| wall_x1 | 55% | 0% | 45% | lama |
| wall_y0 | 42% | 2% | 56% | lama |
| wall_y1 | 12% | 0% | 88% | lama |
| floor | 72% | 0% | 28% | lama |
| ceiling | 26% | 0% | 74% | lama |
