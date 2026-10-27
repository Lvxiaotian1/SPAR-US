# SPAR-US

Local three-dimensional ultrasound reconstruction from discrete two-dimensional
image frames.

SPAR-US recovers the relative pose of a set of ultrasound frames and
reconstructs a local intensity volume from them. The frames need no acquisition
order, no frame adjacency and no measured probe trajectory: the spatial layout
is initialised from anatomical similarity together with a scan-range prior, and
refined progressively while the intensity field is fitted.

This release provides the optimisation, checkpoint rendering and the
quantitative evaluation used for the reported pose and image metrics.

## Repository layout

```text
.
├── training.py              # initialisation and progressive optimisation
├── demo.py                  # render a checkpoint
├── evaluate.py              # translation, rotation and SSIM
├── model.py                 # intensity field, probe model, checkpoint I/O
├── geometry.py              # rotation conversions
├── config.json              # sampling and optimisation settings
├── data/README.md           # frame-set format
└── examples/phantom/        # synthetic frame set with reference poses
```

## Installation

The implementation is tested with Python 3.10 and PyTorch 2.5.1 on CPU and
CUDA. Evaluation additionally uses SciPy and scikit-image.

```bash
python -m pip install -r requirements.txt
```

## Optimisation

`training.py` holds the fitting procedure:

1. image-pose initialisation from the frame grouping, the pairwise similarity and the scan range;
2. joint initialisation of the anchor frames together with the intensity field;
3. local pose prediction for each newly added frame;
4. refinement of a partial subset around the new frame;
5. periodic global refinement of the field and all incorporated poses.

It is a library module: `train_case` returns the fitted field and the estimated
geometry, and the optimisation settings are read from `config.json`. This
release does not ship a command-line driver for it.

## Checkpoint

A checkpoint holds everything needed to reconstruct one frame set:

```python
checkpoint = torch.load("checkpoint.pt", weights_only=True)
checkpoint["field"]      # 18 tensors: the fitted intensity field
checkpoint["geometry"]   # the estimated frame poses and the imaging geometry
```

| `geometry` key | Contents |
| --- | --- |
| `phi` | `(N, 3)` frame rotations, axis-angle |
| `t` | `(N, 3)` frame translations in millimetres |
| `rotation_format` | `"axis-angle"` |
| `height`, `width` | frame size in pixels |
| `sw`, `sh` | lateral and axial pixel spacing in millimetres |
| `n_frames` | number of frames |

The estimated poses are read from `geometry` and exported by `demo.py` as an
`(N, 3, 4)` array of `[R | t]` matrices. No separate pose file is required.

## Rendering

```bash
python demo.py \
  --checkpoint checkpoint.pt \
  --config config.json \
  --output results
```

`--data` is optional. When the frames are present, the render is also compared
with them and `metrics.csv` is written. Rendering itself depends only on the
checkpoint and the depth range in `config.json`.

| Output | Contents |
| --- | --- |
| `results/frame_00.png` … | rendered eight-bit frames |
| `results/rendered.npy` | floating-point renders, `(N, H, W)` |
| `results/poses.npy` | estimated poses, `(N, 3, 4)` |
| `results/metrics.csv` | per-frame MSE and PSNR, when frames are available |

## Evaluation

```bash
python evaluate.py --data data --results results
```

| Metric | Definition | Aggregation |
| --- | --- | --- |
| Translation error | Euclidean distance between aligned translations, in millimetres | Mean over frames, excluding the alignment frame |
| Rotation error | Geodesic angle between aligned rotations, in degrees | Mean over the same frames |
| SSIM | Similarity between each input frame and its render | Mean over all frames |

Both trajectories are expressed relative to the frame with the smallest
reference z translation, and rotation matrices are orthonormalised before the
error is computed.

## Configuration

| Key | Meaning |
| --- | --- |
| `near`, `far` | depth range sampled along each scanline, in millimetres |
| `sw`, `sh` | lateral and axial pixel spacing used at initialisation |
| `seed` | seed for the field initialisation and the stage ordering |
| `steps_init` | passes over the anchor set at initialisation |
| `steps_stage` | passes over the frame subset in each later stage |
| `partial_views` | maximum size of a partial subset: the new frame plus its nearest incorporated frames |
| `global_every` | global refinement is triggered every this many incorporated frames |
| `lr_field`, `lr_pose` | initial learning rates for the field and the poses |
| `field_step`, `field_decay` | field learning-rate decay period and factor |
| `pose_step`, `pose_decay` | pose learning-rate decay period and factor |

## Example

`examples/phantom` is a synthetic frame set with known reference poses, generated
without clinical data. A fitted checkpoint is included, so the render and the
evaluation can be run directly:

```bash
python demo.py \
  --checkpoint examples/phantom/checkpoint.pt \
  --config examples/phantom/config.json \
  --data examples/phantom/data \
  --output runs/phantom

python evaluate.py --data examples/phantom/data --results runs/phantom
```

The example reaches a mean translation error of about 0.26 mm and a mean rotation
error of about 0.11 degrees over the 15 mm sweep, at an SSIM of about 0.78. The
frame set also carries the grouping and similarity arrays that `training.py`
expects, so the same data can drive a caller-supplied fitting loop. See
[examples/phantom/README.md](examples/phantom/README.md) for details.

## Licence

The code in this repository is released under the MIT licence (see `LICENSE`).
`geometry.py` is derived from PyTorch3D and remains under its BSD-3-Clause
licence, reproduced in `THIRD_PARTY_NOTICES.txt`.

## Acknowledgements

The neural-field architecture and the incremental optimisation follow CF-NeRF
(Yan et al., AAAI 2024). The rotation conversions follow PyTorch3D.
