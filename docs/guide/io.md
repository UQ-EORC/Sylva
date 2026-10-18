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
| `.rdbx` | yes | no | RIEGL RDB 2, the database RiSCAN PRO makes of an imported scan; needs librdb (below). Every attribute as a column; the processing settings as metadata. |
| `.parquet` | `Shots.load` | `Shots.save` | Pulse data, see [Pulse data](pulses.md). |

Rasters (DTM, CHM) are `sylva.Raster` objects, written with
`to_ascii_grid` (`.asc`, with the CRS as WKT in a `.prj` beside it, which
`from_ascii_grid` reads back) or `to_geotiff` (`.tif`, float32 with NaN as
nodata, written by Sylva itself). Row 0 of `Raster.data` is the southern
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

### MTA resolution: converting like RiSCAN PRO

Scanners that fire at megahertz rates (VZ-400i, VZ-600i, VZ-2000i, ...) have
several pulses in flight at once, so the scanner cannot tell which pulse an
echo belongs to: it may sit at its measured range, or one pulse interval
(about 68 m on a VZ-600i at 2200 kHz) further out, or two. RiSCAN PRO's import
resolves this "multiple time around" (MTA) ambiguity with `rimta5`, drops the
echoes it cannot place, applies a few filters and only then writes the
`.rdbx`. Plain `libscanifc` reading gives you none of that: every echo lands in
the first zone, far returns fold back into the first 68 m, and the weak,
uncertain echoes stay.

Sylva does the same conversion itself when a small shim over RiVLib's C++
class is present. Build it once from the SDK (RiVLib is linked statically; the
result stays on your machine):

```bash
RIVLIB=/path/to/rivlib-2_6_0-x86_64-linux-gcc11 crates/sylva-rs/rivlib-shim/build.sh
```

It lands next to `libscanifc` and `sylva.io.find_rivlib_shim()` shows what will
be used (`SYLVA_RIVLIB_SHIM` overrides). From then on `read_rxp`, `sylva
convert scan.rxp scan.laz` and everything else reading `.rxp` resolve MTA by
default (`mta="auto"`); `mta="off"` is the old behaviour and `mta="on"` fails
loudly when the shim is missing.

What happens, per scan line and in this order:

1. every laser shot (also the 12% without echoes) comes from RiVLib with its
   exact time and beam; echoes are attached to the pulse whose time they carry
   and bring their range under each zone hypothesis;
2. the zone is chosen by neighbourhood consistency: the pulse-position
   modulation makes a wrong hypothesis of two neighbouring echoes land on
   different ranges while the right one agrees, so neighbours vote, then the
   remaining echoes are compared with the resolved ones, then with a wide range
   histogram of them;
3. the echo is placed on its pulse at the resolved range with that pulse's beam
   and timestamp, and its reflectance recomputed for that range;
4. echoes with amplitude 0 are dropped, so are echoes below 2.5 dB within 18 m
   (the scanner's dynamic trigger), echoes outside `min_range`/`max_range`,
   zone-1 echoes inside the window notch, and MTA-uncertain echoes unless
   `keep_uncertain=True` (then flagged in `mta_uncertain_point`);
5. `return_number` and `number_of_returns` are recounted over what survived.

The three passes gather evidence per echo and zone (votes, anchor votes,
heat-map density). What turns that into a decision is, by default, a small
learned model: 300 boosted trees trained on rimta5's own output of a VZ-600i
scan (`mta_model="default"`). `mta_model="rule"` uses the hand-written
thresholds instead, and a path loads a model you trained yourself with
`scripts/train_mta.py` from a scan that has both the `.rxp` and RiSCAN's
`.rdbx` (set `SYLVA_MTA_FEATURES=<file>` while reading the `.rxp` to dump
the per-echo evidence the script learns from).

Measured against rimta5 6.9.6 on that scan (ScanPos001, Pilliga 2026, 52.6 M
echoes), on scan lines the model never saw: 98.8% of the points are the same
set and the zone agrees for 99.93% of the common points (the hand-written rule:
98.1% and 99.7%). Coordinates of common points agree to the 0.25 mm the `.rdbx`
stores. What remains is mostly echoes rimta5 drops as MTA-uncertain that look
well supported from the outside, about 1% of the points, and 0.07% of
rimta5's points that RiVLib never delivers at all. Not reproduced: the dynamic
window analysis and the near-range window correction (rimta5 applies them only
when its window model succeeds, which it did not on the reference scans). The
model is specific to the scanner and scan program it was trained on; for other
instruments train your own or use the rule.

Extra attributes with MTA resolution: `mta_zone`, `return_number`,
`number_of_returns`, `scan_line_index` (RiSCAN's convention: negative, -1 is
the first line), `shot_index_line`, `mirror_facet`.

### RDB databases

When RiSCAN PRO imports a scan it writes an RDB 2 database
(`project.rdb/SCANS/ScanPos001/SINGLESCANS/<scan>/<scan>.rdbx`): the echoes
of the `.rxp` after MTA resolution and the import filters, as typed columns,
with the settings used stored beside them as JSON metadata. Sylva reads these
through RIEGL's `librdb`, loaded at run time like RiVLib. Download the RDB SDK
("rdblib") from RIEGL, point `RDBLIB_PATH` at the extracted folder, and
`sylva.io.find_rdblib()` shows what will be used.

```python
path = "project.rdb/SCANS/ScanPos001/SINGLESCANS/200101_120000/200101_120000.rdbx"
info = io.rdbx_info(path)
info.point_count, list(info.attributes)       # 50_862_352, ['riegl.xyz', 'riegl.reflectance', ...]
info.json("riegl.mta_settings")               # {'zone_width': 68.1, 'modulation_depth': 19.3, 'zone_count': 6}

pts = io.read_rdbx(path, attributes=["reflectance", "deviation", "return_number", "gps_time"],
                   filter="riegl.deviation < 20")
sylva.write(io.read_rdbx(path), "ScanPos001.laz")   # everything, as a LAZ
```

Every attribute comes back as a column in the database's own type (`int16`
widened to `int32`, `uint64` to `int64`). The ones LAS has a place for take
the LAS names, so a written LAS/LAZ carries them as standard dimensions:
`riegl.target_index` is `return_number`, `riegl.target_count`
`number_of_returns`, `riegl.timestamp` `gps_time`, `riegl.class`
`classification`, `riegl.id` `id`. The rest keep their RDB names without the
`riegl.` namespace (`reflectance`, `amplitude`, `deviation`, `mta_zone`,
`mta_uncertain_point`, `dyntrig_uncertain_point`, ...), a vector attribute
as `name_0`, `name_1`, .... `sylva info scan.rdbx` prints the schema and the
metadata names from the index alone, and `sylva convert scan.rdbx scan.laz`
converts a whole database.

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
