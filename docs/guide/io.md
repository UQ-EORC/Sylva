# Point clouds and files

## PointCloud

A `PointCloud` holds an `(N, 3)` float64 array `xyz` and a dict `attrs` of
length-`N` arrays. Functions return new clouds and never change their inputs.
Indexing with a mask, integer array or slice subsets the points together with
their attributes.

```python
import numpy as np
import sylva

cloud = sylva.read("plot.laz")
cloud                                   # PointCloud(n=12,345,678, attrs=['classification', 'intensity', ...])
lo, hi = cloud.bounds
low = cloud[cloud.z < lo[2] + 2.0]      # mask indexing keeps every attribute
cloud = cloud.with_attrs(scan_id=np.zeros(len(cloud), np.int32))
cloud = cloud.without("user_data", "point_source_id")
merged = sylva.PointCloud.concatenate([a, b])     # keeps attributes common to both
```

Sylva writes and reads these attributes:

| Attribute | Set by | Meaning |
|---|---|---|
| `classification` | `ground.classify_ground_*` | ASPRS codes, 2 = ground, 1 = other |
| `height` | `ground.normalize_height` | height above the DTM (m) |
| `scan_id` | `registration.merge_scans` | index of the source scan |
| `tree_id` | written by you: `cloud.with_attrs(tree_id=labels)` | segmentation label, -1 unassigned |
| `nx`, `ny`, `nz` | ray-cloud PLY | vector from each point to its sensor |

Functions that need heights take `height_attr="height"` and fall back to z
when the attribute is missing. Only rely on that fallback for clouds that
are already flattened with `ground.flatten`.

## Formats

`sylva.read` and `sylva.write` choose the format from the extension:

| Extension | Read | Write | Notes |
|---|---|---|---|
| `.las` `.laz` | yes | yes | ASPRS LAS 1.4; LAZ is LASzip (Isenburg 2013). Every standard dimension. Extra bytes read and written with their types, so `height` and `tree_id` round-trip. `point_format` (default 6) and `scale` (default 1 mm) on write. |
| `.ply` | yes | yes | ASCII or binary; every vertex property becomes an attribute. raycloudtools ray clouds (Lowe & Stepanas 2021) read as ordinary clouds with `nx ny nz`. |
| `.xyz` `.txt` `.asc` `.pts` `.csv` | yes | yes | Delimiter detected; header line gives names; PTS count line skipped. Written with a header, 0.1 mm precision. |
| `.rxp` | yes | no | RIEGL, needs RiVLib (below). Points in the scanner frame. |
| `.parquet` | `Shots.load` | `Shots.save` | Pulse data, see [Pulse data](pulses.md). |

Rasters (DTM, CHM) are `sylva.Raster` objects, written with
`to_ascii_grid` (`.asc`) or `to_geotiff` (`.tif`, needs `rasterio`:
`pip install sylva-rs[geotiff]`). Row 0 of `Raster.data` is the southern
edge, and both writers flip it to north-up.

The format specifications are listed on the [references](../references.md)
page.

## RIEGL data

RiVLib is proprietary, so Sylva loads RIEGL's `libscanifc` at run time
instead of shipping it. Download RiVLib from RIEGL and point `RIVLIB_PATH`
at the extracted folder; `sylva.io.find_rivlib()` shows what will be used.

```python
from sylva import io

pts = io.read_rxp("ScanPos001/SINGLESCANS/200101_120000.rxp",
                  min_range=0.5, echoes="all", shot_stride=4)
pts.attrs.keys()        # amplitude, reflectance, deviation, echo_type, gps_time
```

- `shot_stride=n` keeps every n-th pulse with all its echoes. This is the
  right way to thin data that will be used as pulses.
- `stride=n` keeps every n-th echo. It splits multi-echo pulses, so use it
  only for point work.
- `deviation` measures how distorted the returned pulse shape is. High values
  mark mixed pixels and edge hits. Filtering on it, for example
  `pts[pts.attrs["deviation"] < 20]`, is a common clean-up. The right
  threshold depends on the scanner, so look at the histogram first.

A RiSCAN PRO project gives the scan positions and their matrices:

```python
project = sylva.read_riscan_project("plot.RiSCAN")
project.names                            # ['ScanPos001', ...]
for pos in project.with_scans():         # positions with an .rxp and a SOP
    cloud = pos.read(shot_stride=4)      # project coordinates
    shots = pos.read_shots(fill_missing=True)   # pulses, misses reconstructed
```

The POP (project to global) is often a geocentric transform with offsets of
thousands of kilometres. It is only applied when you pass `pop=project.pop`.
Process in project coordinates and transform the final products (tree
positions, rasters). Many viewers and tools store coordinates as float32,
which at those offsets keeps only about half a metre.
