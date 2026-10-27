# Frame set

A frame set is described by four files:

```text
data/
├── images/
│   ├── 0.png
│   ├── 1.png
│   └── ...                 # 0.png through N-1.png, no gaps
├── class.npy               # (N,) integers in {0, 1, 2}
├── similarity.npy          # (N, N) pairwise similarity
└── poses.npy               # (N, 3, 4) reference poses, evaluation only
```

Only `images/` and `poses.npy` are read by `demo.py` and `evaluate.py`;
`class.npy` and `similarity.npy` are the initialisation inputs of `training.py`.

## images/

One grayscale PNG per frame. File names must run from `0.png` to `N-1.png`
without gaps, and all frames must share the same width and height. Pixel values
are read as eight-bit grayscale and normalized to `[0, 1]`.

The images are optional for rendering: `demo.py` reads them only when image
metrics are requested. `evaluate.py` reads them for SSIM.

## class.npy

One integer label per frame, using exactly the values `0`, `1` and `2`. The
groups are ordered along the sweep direction, and the first frame of each group,
in file order, is that group's anchor.

## similarity.npy

A square matrix of pairwise frame similarity. It is used for the within-group
ordering and for splitting the middle group around its anchor; only the relative
order of the entries matters.

## poses.npy

Reference poses in the `(N, 3, 4)` layout, with a right-handed rotation matrix in
the first three columns and a translation in millimetres in the last column.
This file is read only by `evaluate.py`; the fitting procedure never reads it.

## Coordinate conventions

Local `x` runs across the image, local `y` runs along image depth and local `z`
is the plane normal. Each pose maps local plane coordinates into the common
reconstruction frame. Translations are in millimetres.
