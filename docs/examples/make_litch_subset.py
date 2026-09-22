"""Cut the small LITCH tile that the examples and docs use.

    python docs/examples/make_litch_subset.py /path/to/LITCH_2021_raycloud_inside.ply

The input is a raycloudtools ray cloud of the TERN Litchfield Savanna
SuperSite 1 ha plot (Northern Territory, scanned 2021 with a RIEGL VZ-2000i,
many upright scan positions registered into one plot frame). The plot data are
TERN's; this tile is redistributed with the documentation so the examples run
on real savanna instead of a synthetic scene.

Every ray is clipped to a 20 m x 20 m x full-height tile:

* a ray ending inside keeps its echo;
* a ray passing through, and a ray that returned nothing at all, become
  pulses with no return (they measure free space);
* a ray missing the tile is dropped.

Origins therefore sit on the tile boundary, not at the scanner, which is what
ray-traced products inside the tile need. Outputs, in a local frame with the
tile corner at (0, 0) and the ground near z = 0:

* ``data/litch_tile.laz``: echoes thinned to one point per ``POINT_VOXEL`` m,
  with the ray cloud's ``alpha`` kept as ``intensity`` and a ``classification``
  of 2 (ground) or 4 (vegetation);
* ``data/litch_tile_shots.parquet``: every ``RAY_STRIDE``-th clipped ray as
  pulses, misses included, echoes carrying the same ``classification``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import sylva
from sylva import PointCloud, Shots, filters, ground

TILE = (40.0, 60.0, -60.0, -40.0)  # xmin, xmax, ymin, ymax in the plot frame (m)
POINT_VOXEL = 0.05
GROUND_HEIGHT = 0.15  # echoes at or below this height above the DTM are ground
RAY_STRIDE = 40
CHUNK = 1 << 22
OUT = Path(__file__).parent / "data"

DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("t", "<f8"), ("nx", "<f4"),
                  ("ny", "<f4"), ("nz", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1")])


def clip_to_tile(rays: np.ndarray):
    """Clip rays to the tile in x and y.

    Returns entry points, exit or echo points, whether the ray ended inside
    (an echo) rather than passing through (a miss), and which input rays were
    kept.
    """
    end = np.column_stack([rays["x"], rays["y"], rays["z"]])
    origin = end + np.column_stack([rays["nx"], rays["ny"], rays["nz"]]).astype(np.float64)
    d = end - origin
    lo, hi = np.array([TILE[0], TILE[2]]), np.array([TILE[1], TILE[3]])

    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (lo - origin[:, :2]) / d[:, :2]
        t1 = (hi - origin[:, :2]) / d[:, :2]
    parallel = d[:, :2] == 0
    inside = (origin[:, :2] >= lo) & (origin[:, :2] <= hi)
    near = np.where(parallel, np.where(inside, -np.inf, np.inf), np.minimum(t0, t1))
    far = np.where(parallel, np.where(inside, np.inf, -np.inf), np.maximum(t0, t1))
    tmin = np.maximum(near.max(axis=1), 0.0)
    tmax = np.minimum(far.min(axis=1), 1.0)

    # Unbounded rays (alpha 0: the pulse returned nothing) are kept as well --
    # their far point gives the direction, and they are what free space is
    # measured from.
    keep = tmax > tmin
    origin, end, d = origin[keep], end[keep], d[keep]
    tmin, tmax = tmin[keep], tmax[keep]
    ends_inside = (tmax >= 1.0) & (rays["a"][keep] > 0)
    return (origin + tmin[:, None] * d,
            origin + np.minimum(tmax, 1.0)[:, None] * d,
            ends_inside, keep)


def main(path: str) -> None:
    OUT.mkdir(exist_ok=True)
    with open(path, "rb") as fh:
        offset = fh.read(2000).index(b"end_header\n") + len(b"end_header\n")
    rays = np.memmap(path, dtype=DTYPE, mode="r", offset=offset)

    starts, stops, hit, amp = [], [], [], []
    for i in range(0, len(rays), CHUNK):
        chunk = np.asarray(rays[i:i + CHUNK])
        a, b, inside, kept = clip_to_tile(chunk)
        if len(a) == 0:
            continue
        starts.append(a)
        stops.append(b)
        hit.append(inside)
        amp.append(chunk["a"][kept])
    origin = np.vstack(starts)
    end = np.vstack(stops)
    hit = np.concatenate(hit)
    amp = np.concatenate(amp)
    print(f"{len(rays):,} rays in, {len(origin):,} cross the tile, {hit.mean():.0%} end inside")

    shift = np.array([TILE[0], TILE[2], 0.0])
    shift[2] = np.percentile(end[hit][:, 2], 0.5)

    pts = PointCloud(end[hit] - shift, {"intensity": amp[hit].astype(np.uint16)})
    pts = filters.voxel_downsample(pts, POINT_VOXEL)

    # Ground / vegetation per echo, so the voxeliser can leave the terrain out.
    # The progressive morphological filter is used because the cloth filter
    # hangs up on the grass layer of this tile (see the examples index).
    dtm = ground.make_dtm(ground.classify_ground_pmf(pts), 0.5, bounds=(0, 0, 20, 20))
    height = (end - shift)[:, 2] - dtm.sample((end - shift)[:, 0], (end - shift)[:, 1])
    echo_class = np.where(height <= GROUND_HEIGHT, 2, 4).astype(np.uint8)
    pts = pts.with_attrs(classification=np.where(
        pts.z - dtm.sample(pts.x, pts.y) <= GROUND_HEIGHT, 2, 4).astype(np.uint8))
    sylva.write(pts, OUT / "litch_tile.laz")
    print("points:", f"{len(pts):,}", f"{(OUT / 'litch_tile.laz').stat().st_size / 1e6:.1f} MB")

    pick = np.arange(0, len(origin), RAY_STRIDE)
    vec = origin[pick] - end[pick]
    ray = PointCloud(end[pick] - shift, {
        "nx": vec[:, 0].astype(np.float32), "ny": vec[:, 1].astype(np.float32),
        "nz": vec[:, 2].astype(np.float32),
        "alpha": np.where(hit[pick], 1.0, 0.0).astype(np.float32),
        "classification": echo_class[pick]})
    shots = Shots.from_ray_cloud(ray)
    shots.save(OUT / "litch_tile_shots.parquet")
    print("shots:", shots, f"{(OUT / 'litch_tile_shots.parquet').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
