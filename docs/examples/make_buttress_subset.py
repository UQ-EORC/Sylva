"""Cut the two small tree clouds that the buttress example uses.

    python docs/examples/make_buttress_subset.py CAXH_T2_raycloud.ply CAXH_T4_raycloud.ply

The inputs are raycloudtools ray clouds of two tropical trees from the
destructive-harvest data of Burt et al. 2021 (trees T2 and T4 of the ``CAXH``
set). Each cloud holds a single tree and no neighbours; the first also has a
sparse scatter of ground returns around its foot. The first tree has a flanged
base, the second a round one.

Both outputs are in a local frame with the stem axis at (x, y) = (0, 0) and the
foot of the tree at z = 0, so that z is the height above ground that the
buttress functions take:

* ``data/buttress_tree.laz``: the whole of the first input (T2), thinned to one
  point per ``BASE_VOXEL`` m up to ``BASE_HEIGHT`` m, where the buttress
  functions look, and per ``CROWN_VOXEL`` m above it;
* ``data/round_tree.laz``: the lowest ``HEIGHT`` m within ``RADIUS`` m of the
  stem of the second input (T4), thinned to ``BASE_VOXEL``.

The echoes keep no attributes. The foot is the lowest echo within 1.5 m of the
stem, and the stem axis is the median of the echoes 2.5 to 3.5 m above it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import sylva
from sylva import PointCloud, filters

BASE_HEIGHT = 6.0    # below this the buttress functions read the points (m)
BASE_VOXEL = 0.01    # point spacing up to BASE_HEIGHT (m)
CROWN_VOXEL = 0.05   # and above it, for the whole tree
RADIUS = 4.0         # the round tree: horizontal reach from the stem (m)
HEIGHT = 8.0         # and the height of it kept above its foot (m)
OUT = Path(__file__).parent / "data"

DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("t", "<f8"), ("nx", "<f4"),
                  ("ny", "<f4"), ("nz", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1")])


def read_raycloud(path: Path) -> np.ndarray:
    """The echo of every ray, as an (n, 3) array in the source frame."""
    with open(path, "rb") as f:
        n = None
        for line in iter(f.readline, b""):
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
            if line.startswith(b"end_header"):
                break
        offset = f.tell()
    rays = np.memmap(path, dtype=DTYPE, mode="r", offset=offset, shape=(n,))
    return np.column_stack([rays["x"], rays["y"], rays["z"]]).astype(np.float64)


def stem_frame(xyz: np.ndarray) -> tuple[float, float, float]:
    """The stem axis (x, y) and the foot z of a single-tree cloud."""
    foot = np.percentile(xyz[:, 2], 0.05)
    band = (xyz[:, 2] > foot + 2.5) & (xyz[:, 2] < foot + 3.5)
    # The densest 0.5 m cell of the band is on the trunk; the median of the band's
    # echoes within a metre of it is the axis.
    counts, xe, ye = np.histogram2d(xyz[band, 0], xyz[band, 1], bins=40)
    i, j = np.unravel_index(counts.argmax(), counts.shape)
    guess = np.array([(xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2])
    ring = band & (np.hypot(xyz[:, 0] - guess[0], xyz[:, 1] - guess[1]) < 1.0)
    cx, cy = np.median(xyz[ring, :2], axis=0)
    near = np.hypot(xyz[:, 0] - cx, xyz[:, 1] - cy) < 1.5
    return cx, cy, xyz[near, 2].min()


def cut(path: str, out: Path, whole: bool) -> None:
    xyz = read_raycloud(Path(path))
    cx, cy, foot = stem_frame(xyz)
    local = xyz - [cx, cy, foot]
    if whole:
        low = local[:, 2] < BASE_HEIGHT
        parts = [filters.voxel_downsample(PointCloud(local[low]), BASE_VOXEL),
                 filters.voxel_downsample(PointCloud(local[~low]), CROWN_VOXEL)]
        cloud = PointCloud.concatenate(parts)
    else:
        keep = (np.hypot(local[:, 0], local[:, 1]) < RADIUS) & (local[:, 2] < HEIGHT)
        cloud = filters.voxel_downsample(PointCloud(local[keep]), BASE_VOXEL)
    sylva.write(cloud, out)
    print(f"{Path(path).name}: stem axis ({cx:.2f}, {cy:.2f}), foot z {foot:.2f} in the source frame; "
          f"{len(cloud):,} points -> {out.name} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    cut(sys.argv[1], OUT / "buttress_tree.laz", whole=True)
    cut(sys.argv[2], OUT / "round_tree.laz", whole=False)
