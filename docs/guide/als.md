# Airborne lidar tiles

Airborne surveys (ALS from aircraft, UAV lidar from drones) cover whole
landscapes and are delivered as hundreds of LAS/LAZ tiles. Together they are
far larger than memory, and no tile can be processed on its own: a ground
filter or a DTM cell at the edge of a tile needs the points on the other side
of it. `sylva.als` handles such collections the way lidR's `LAScatalog`
does (Roussel et al. 2020): a catalogue of the tiles is built from their
headers alone, the area is processed in chunks that each carry a buffer of
points from their neighbours, and the results are put back together so that
the tile edges do not show.

```python
from sylva import als

cat = als.catalog("tiles/")            # headers only: fast for any number of tiles
print(cat.report())
ground = als.classify_ground(cat, "ground/")
dtm = als.dtm(ground, resolution=1.0)
chm = als.chm(ground, resolution=0.5)
dtm.to_geotiff("dtm.tif")
```

## The catalogue

`als.catalog(path)` takes a directory (searched for `*.la[sz]`, case
insensitively), a file or a list of either, and reads each file's header:
its extent, point count, point format, LAS version, coordinate quantisation,
CRS (an EPSG code from the WKT or GeoTIFF records when they name one) and
whether a spatial index (a `.lax` file, or COPC) exists. No point is read,
so a catalogue of a thousand tiles takes a second or so.

The catalogue checks the tiling before anything is processed.
`cat.issues()` lists every problem as `(kind, message)`, and `cat.report()`
prints them under a summary:

```text
ALS catalogue: 4 tiles, 1,256,899 points
  extent   x 0.00 .. 100.00, y 0.00 .. 100.00, z -0.24 .. 33.47
  area     0.0100 km² covered by tile extents (100.0 x 100.0 m bounding box)
  density  125.69 points/m²
  tiles    50.0 m across (median), 289,958 to 325,473 points
  format   LAS 1.4 format 6 (4)
  index    0 of 4 with a spatial index
  crs      EPSG:32755
  checks   no problems found
```

| Issue | Meaning |
|---|---|
| `missing`, `unreadable` | a file asked for does not exist or its header cannot be read; nothing is processed until it is fixed |
| `empty` | tiles with no points |
| `mixed_crs`, `no_crs` | tiles in different coordinate systems (or some with none), or none declared; Sylva never reprojects |
| `mixed_point_format` | attributes missing from some formats (GPS time, colour) are dropped where tiles meet |
| `mixed_scale` | tiles quantise coordinates differently |
| `overlap` | tile extents overlap by more than `tolerance`; the points in the overlap are counted twice |
| `gap` | holes enclosed by tiles (a concave outline of the survey is not a hole) |

`tolerance` (1 m by default) is how far tile extents may overlap, or fall
short of each other, before it counts: tiles whose points stop a few
centimetres short of the tile edge are not a gap. `cat.validate()` raises
if there is any issue; `cat.read(bounds)` reads the points in a box from
whichever tiles hold them.

## Chunks and buffers

Work is divided into chunks: one per tile by default, or squares of
`chunk_size` metres on a regular grid. Each chunk has a *core*, the area
it is responsible for, and a *buffer* of `buffer` metres around it:

- Reading a chunk opens only the files whose extent overlaps its buffered
  box, streams through them, and keeps only the points inside the box.
- The processing sees the core and buffer points together, so a ground
  filter or DTM cell at the edge of the core has the neighbouring points it
  needs.
- Point outputs keep only the core points. With one chunk per tile the core
  is the tile's own file, so every point is written exactly once even if
  tiles overlap; on a grid a point belongs to the chunk whose half-open
  square `[xmin, xmax) x [ymin, ymax)` holds it.
- Raster outputs are computed on one grid shared by the whole catalogue
  (its south-west corner is the catalogue's minimum snapped down to a
  multiple of the resolution), and each cell of the result is taken from the
  chunk whose core holds its centre.

The buffer must be wider than whatever the processing looks across: the
widest gap in the ground for a DTM (under a building or a dense crown, since
the gaps are filled from the nearest measured cells), about half the largest
non-ground object for a ground filter, the search radius for a noise
filter. 20 m, the default, suits most forests; with it the DTM of a set of
tiles is the DTM of all the tiles merged into one cloud, cell for cell, away
from the outer edge of the survey (this is tested).

## Memory and workers

Chunks are processed in parallel, `workers` at a time (one per CPU by
default). How many can be in memory at once is worked out from the header
point counts: each chunk's expected number of points (from the share of each
tile's extent inside its buffered box) times about 256 bytes per point, set
against the memory budget of [`sylva.limits`](../api/limits.md). If the
largest chunk alone does not fit, the run is refused with a message saying
so and suggesting a smaller `chunk_size`; otherwise fewer workers are used
when fewer fit. `SYLVA_MEM_BUDGET` (GB) or `limits.set_budget` change the
budget.

Results never depend on the number of workers: each chunk is processed on
its own, deterministically, and results are assembled in chunk order.
Progress shows through [`sylva.progress`](../api/progress.md), with
`progress.bar()` in scripts and by default in the command line.

## What runs on a catalogue

| Function | Output | Single-cloud equivalent |
|---|---|---|
| `als.classify_ground(cat, out, method="csf" or "pmf")` | classified tiles | `ground.classify_ground_csf`, `classify_ground_pmf` |
| `als.dtm(cat, resolution, method=...)` | `Raster` | `ground.make_dtm` |
| `als.chm(cat, resolution, dtm="auto")` | `Raster` | `ground.make_chm` of a normalised cloud |
| `als.normalize(cat, out, replace_z=False)` | tiles with `height` (or z replaced) | `ground.normalize_height`, `ground.flatten` |
| `als.filter(cat, out, method="ror" or "sor")` | tiles without noise (or noise classified 7) | `filters.radius_outlier_removal`, `statistical_outlier_removal` |
| `als.retile(cat, out, size, buffer=0)` | new square tiles | |
| `als.decimate(cat, out, method="random", "voxel" or "highest")` | thinned tiles | `filters.random_subsample` |
| `als.write_tiles(cloud, out, size)` | tiles from one cloud | `io.write` |
| `als.grid_metrics(cat, resolution)`, `als.plot_metrics(cat, plots)` | area-based metrics as rasters or a plot table ([Area-based ALS metrics](als_metrics.md)) | `als.cloud_metrics` |
| `als.find_trees(cat, out=None, method=...)` | trees, and tiles with tree ids | `als.segment_trees`; see [Airborne trees](als_trees.md) |

Functions that write tiles write one per chunk into the output directory,
in the point format, quantisation and CRS of the input tile, and return the
catalogue of what they wrote, so that steps chain. They refuse to overwrite
an input tile. On a single tile, with the same parameters, each gives what
the single-cloud function gives on that tile (away from its edge, for the
rasters and heights).

A few behaviours differ from the single-cloud functions because a survey is
not a plot:

- `classify_ground` leaves points already classified as noise (7 or 18) out
  of the filter, and keeps their class; `last_returns=True` lets only last
  returns be ground.
- `dtm` and `chm` leave NaN the cells of chunks with fewer than 3 ground
  points, and raise if no chunk has any.
- `chm` with `dtm="auto"` normalises each chunk with a DTM made from its own
  ground points and buffer; `dtm=None` uses z as it is (for tiles already
  normalised with `replace_z=True`, or to make a surface model).
- The `"sor"` noise filter compares each point's mean neighbour distance with
  the mean and spread over its tile and buffer, so its threshold can differ
  slightly between tiles; `"ror"`, the default, is purely local.
- Decimation needs no buffer; a voxel or cell that straddles two tiles
  keeps a point in each.

## Your own function: `als.apply`

`als.apply(cat, fn)` runs any function on every chunk as `fn(cloud,
chunk)`. `cloud` has a boolean `buffer` attribute (True for the buffer
points) and `chunk` is the [`Chunk`](../api/als.md) with its core, buffered
box and name. What `fn` returns decides what `apply` returns:

- a `PointCloud`: the buffer points are removed (by the `buffer` attribute,
  which any subset of `cloud` keeps) and the chunks stacked, or with `out`
  written as one file per chunk;
- a `Raster`: the chunks are joined into one raster on the catalogue grid;
  each chunk's raster must share the resolution and lie on that grid
  (bounds snapped to multiples of the resolution do);
- anything else: a list in chunk order, None for chunks with no points of
  their own.

```python
import numpy as np
from sylva import als, ground

def canopy_above_2m(cloud, chunk):
    x0, y0, x1, y1 = chunk.outer
    grid = (np.floor(x0 / 5) * 5, np.floor(y0 / 5) * 5, x1, y1)
    return ground.make_chm(cloud, 5.0, bounds=grid, min_height=2.0)

normalised = als.normalize(cat, "normalised/", replace_z=True)
cover = als.apply(normalised, canopy_above_2m, chunk_size=200.0, buffer=5.0)

counts = als.apply(cat, lambda cloud, chunk: int((~cloud.attrs["buffer"]).sum()))
```

An exception in `fn` stops the run (no new chunks start) and is raised as
it is.

## A worked example on synthetic tiles

[`synthetic.als_flight`](../api/synthetic.md) flies a simulated scanner
over a synthetic forest, so every step can be checked against the truth:

```python
import numpy as np
from sylva import als, synthetic

rng = np.random.default_rng(0)
trees = [(x, y, 0.3, h) for x, y, h in
         zip(rng.uniform(5, 95, 40), rng.uniform(5, 95, 40), rng.uniform(10, 25, 40))]
scene = synthetic.forest(trees, size=100.0, ground_points=100, margin=0.0)
flight = synthetic.als_flight(scene, altitude=80.0, speed=10.0, line_spacing=40.0,
                              pulse_rate=50_000, bounds=(0, 0, 100, 100))
cat = flight.write_tiles("tiles", size=50.0, epsg=32755)       # 4 tiles, 1.26 M points

ground = als.classify_ground(cat, "ground", method="csf", cloth_resolution=1.0)
dtm = als.dtm(ground, resolution=1.0)
chm = als.chm(ground, resolution=0.5)
truth = synthetic.terrain_height(*dtm.cell_centers())
print(f"DTM error: {np.abs(dtm.data - truth).mean():.3f} m mean")    # 0.068 m
print(f"tallest canopy: {np.nanmax(chm.data):.1f} m")              # 29.9 m
als.normalize(ground, "normalised", replace_z=True)
```

The whole run takes about ten seconds on a laptop.

## The synthetic scanner

`synthetic.als_flight` simulates an airborne survey closely enough for the
catalogue, and later ray-based canopy and change detection, to be tested
against a known truth:

- **Flight plan.** Parallel lines `line_spacing` apart, alternating in
  direction (the first along `heading`, clockwise from north), at constant
  `altitude` (above z = 0) and `speed`, each extended by `margin` beyond the
  area, with `turn_time` seconds between lines while the laser is off.
- **Scanner.** `pulse_rate` pulses per second; a mirror sweeping
  `scan_angle` either side of nadir at `scan_rate` sweeps per second,
  either oscillating (back and forth, a zigzag on the ground) or rotating
  (a polygon: always the same way, parallel lines).
- **Beam.** A cone of `beam_divergence` (mrad) sampled by 1, 7 or 19
  equal-energy sub-beams. A sub-beam stops at the first scene point it
  passes within `target_radius` of, or at the analytic terrain of
  `synthetic.terrain_height` (the scene's own ground points are ignored,
  so the true DTM is known exactly).
- **Returns.** Hits closer in range than `min_separation` form one return
  at their energy-weighted mean range; returns with less than
  `detection_threshold` of the pulse energy are lost; at most `max_returns`
  are kept, nearest first, with `return_number` and `number_of_returns`.
  Each return has `gps_time`, LAS `scan_angle`, an `intensity` from the
  energy, the reflectance of what was hit and the range, `point_source_id`
  (the line), the true `classification` (2 ground, 4 leaf, 5 wood) and
  `tree_id`.
- **Trajectory.** `flight.trajectory` is a table (`time`, `x`, `y`, `z`,
  `roll`, `pitch`, `heading`, `line`) sampled at `trajectory_rate` while
  the laser is on.

The geometry is exact, which is what ray-based methods need (see
[Canopy structure from airborne lidar](als_canopy.md)). Frames are map
x east, y north, z up and body x forward, y right, z down; a body vector `b`
points along `M Rz(heading) Ry(pitch) Rx(roll) b` in the map (roll positive
right wing down, pitch positive nose up, `M` from north-east-down to
east-north-up). The beam leaves the scanner, at the trajectory position,
along `b = (0, sin a, cos a)` for mirror angle `a`, and every return of a
pulse lies on that axis: `point = position(gps_time) + range * direction`.
`flight.sensor_positions(gps_time)` gives the positions (linear
interpolation within a line, exact because lines are straight and flown at
constant speed), so `point - position` is each return's beam direction. The
LAS `scan_angle` is `a - roll`, the angle from the vertical including roll,
as the LAS specification defines it. Line `k` (from 0) starts at
`start_time + k (T + turn_time)`, `T` being the time to fly a line, and
fires a pulse every `1 / pulse_rate` s from its start; all returns of a
pulse share its `gps_time`. Random draws (attitude phases, range noise)
come from NumPy's `default_rng(seed)`, so a seed gives the same flight on
any number of threads.

## Command line

```bash
sylva als-catalog tiles/                          # report; --strict exits 1 on problems
sylva als-ground tiles/ ground/ --method csf
sylva als-dtm ground/ dtm.tif --resolution 1
sylva als-chm ground/ chm.tif --resolution 0.5
sylva als-normalize ground/ normalised/ --replace-z
```

Each takes `--chunk-size`, `--buffer` and `--workers`; see
[Command line](cli.md).

## Checked against lidR

lidR's bundled example files (`Megaplot.laz`, `Topography.laz`,
`MixedConifer.laz`) were cut into 50 m tiles with `als.retile` and processed
as catalogues, and the results compared cell by cell with lidR 4.3 run on
the whole files at 1 m:

| File | Product | Agreement |
|---|---|---|
| Megaplot | DTM, `method="tin"` vs `rasterize_terrain(tin())` | identical |
| Topography | DTM, TIN, 5 m in from the edge | median difference 0.1 mm, 96.8 % of cells within 1 mm |
| MixedConifer | DTM, TIN, 5 m in from the edge | within 5 mm: lidR rounds its DTM to the files' 1 cm z resolution |
| all three | surface model, `chm(dtm=None)` vs `rasterize_canopy(p2r())` | identical where no point lies on a horizontal cell edge; 97 to 100 % of cells overall |

The surface models differ only in which cell a point exactly on a
horizontal cell edge belongs to: Sylva puts it in the cell to the north
(the cell whose half-open range holds it), lidR (through terra) in the cell
to the south. Moving such points by 0.1 µm makes every cell agree. Canopy
height models from normalised clouds differ by a few centimetres more on
steep ground, because lidR samples the DTM at each point differently (Sylva
interpolates bilinearly between cell centres). lidR's TIN DTM
(`use_class = 2`) was used; by default lidR also counts water (class 9) as
ground.

## Limitations

- Every file a chunk overlaps is decompressed in full to find the points in
  the chunk's box; spatial indexes (`.lax`, COPC) are reported but not yet
  used. With one chunk per tile and a buffer, each tile is decoded by itself
  and its neighbours, so about nine times.
- Sylva never reprojects: tiles in different CRS are reported, not
  converted.
- Waveform and NIR point formats are written as the nearest format without
  them (4 as 1, 5 as 3, 8 and 10 as 7, 9 as 6).
