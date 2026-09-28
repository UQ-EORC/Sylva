# Area-based ALS metrics

The area-based approach to forest inventory relates field plots to
statistics of the airborne lidar returns over the same area (the height
percentiles, the canopy cover, the spread of the heights) and then
predicts across the whole survey from a raster of the same statistics.
`sylva.als` computes the standard set of lidR's `stdmetrics`
(Roussel et al. 2020) on a grid over a catalogue of tiles and for plots,
in the Rust core, with the chunks and buffers of the
[catalogue engine](als.md):

```python
from sylva import als

cat = als.catalog("ground/")                         # tiles with ground classified
grid = als.grid_metrics(cat, 20.0, ["zq95", "zmean", "cover"])
grid["zq95"].to_geotiff("zq95.tif")
plots = als.plot_metrics(cat, [(512.0, 330.0), (640.5, 402.0)], radius=11.28)
plots.to_csv("plots.csv")
```

`als.grid_metrics` (also named `als.pixel_metrics`, as in lidR) returns a
dict of [`Raster`](../api/pointcloud.md), one per metric, on the grid
that `als.dtm` and `als.chm` use at the same resolution.
`als.plot_metrics` returns a table with one row per plot.
`als.cloud_metrics(cloud)` gives the same metrics for one cloud in memory.

## The metrics

The metrics are computed from the heights above ground `z` of the
returns kept (see [Heights](#heights)); `n` is their number and `m` their
mean. The definitions are those of lidR's `stdmetrics_z`, `stdmetrics_i`,
`stdmetrics_rn` and `entropy` (lidR 4, `R/metrics_stdmetrics.R`),
reproduced from the R code.

| Metric | Definition |
|---|---|
| `n` | number of returns |
| `zmax`, `zmean` | maximum and mean height |
| `zsd` | standard deviation with `n - 1` (NaN for one return) |
| `zskew` | `(S3 / n) / (S2 / n)^1.5`, with `Sk = sum((z - m)^k)`: the moment estimator, SciPy's `skew(z, bias=True)` |
| `zkurt` | `n S4 / S2^2`: Pearson's kurtosis (3 for a normal distribution, not the excess), SciPy's `kurtosis(z, fisher=False, bias=True)` |
| `zentropy` | normalised Shannon index of the heights in bins of `entropy_bin` m (see below) |
| `pzabovezmean` | percentage of returns above the mean |
| `pzabove2` | percentage of returns above `threshold` m (the name carries the threshold: `pzabove2.5`) |
| `zq5` .. `zq95` | height quantiles at 5 % steps, linear between order statistics (R's type 7, NumPy's default) |
| `zpcum1` .. `zpcum9` | cumulative percentage of returns in the lowest 1 .. 9 tenths of `[0, zmax)` |
| `cover` | percentage of first returns above `cover_break` m |
| `gap_fraction` | share (0 to 1) of first returns at or below `cover_break` |
| `itot`, `imax`, `imean`, `isd`, `iskew`, `ikurt` | sum, maximum, mean, standard deviation, skewness and kurtosis of `intensity` |
| `ipground` | percentage of the total intensity from ground returns (class 2) |
| `ipcumzq10` .. `ipcumzq90` | percentage of the total intensity from returns at or below the 10, 30, 50, 70 and 90 % height quantiles |
| `p1th` .. `p5th` | percentage of returns that are first, second, .. fifth returns |
| `pground` | percentage of ground returns (class 2) |

**Entropy.** lidR's `entropy(z, by)` bins the heights from 0 to
`ceiling(zmax / by) * by` in steps of `by`, takes the proportion `p` of
returns in each of the `k` bins and divides the Shannon index by that of a
uniform distribution over the same bins: `-sum(p ln p) / ln(k)`, so it is
between 0 (every return in one bin) and 1 (as many in each). It is NaN when
`zmax < 2 by` or any height is negative, as in lidR, so it is common to
leave out the returns below 0 (`min_height=0`). The bins are half-open,
`[a, b)`, as R's `findInterval` makes them, so a return exactly on the top
edge (at `zmax` when it is a whole number of bins) is not counted.

**Cumulative deciles.** `zpcum` uses lidR's breaks `seq(0, zmax, zmax / 10)`
with the same half-open bins, so returns at `zmax` itself, and any below 0,
are left out of the percentages; with `zmax <= 0` they are all 0.

**Cover and gap fraction.** These use first returns only
(`return_number == 1`), the usual estimate of canopy cover from ALS: the
share of pulses whose first return came from above the break. They are not
part of lidR's `stdmetrics`.

In a cell or plot with no returns, `n` is 0 and every other metric NaN
(grid cells without returns are NaN throughout). `als.metric_names()` lists
the names in order. Tiles in any LAS point format have intensity, return
numbers and classes, so a catalogue always gives every metric;
`als.cloud_metrics` leaves out those of attributes a cloud does not have.

## Heights

`dtm` says where the heights come from, as for `als.chm`:

| `dtm` | Heights |
|---|---|
| `"auto"` (default) | z minus a DTM made per chunk from its ground points (class 2) and buffer, at `dtm_resolution` |
| a `Raster` | z minus that DTM, sampled bilinearly |
| `None` | z itself: tiles normalised with `als.normalize(..., replace_z=True)` |
| `"height"` (any other string) | the named attribute: tiles written by `als.normalize` without `replace_z` |

`min_height` leaves out returns lower than it (all of them, including for
`n` and `cover`); `drop_noise=True` (the default) leaves out returns
classified as noise (7 or 18), which would otherwise set `zmax` from a bird
or a multipath echo. Returns whose height is NaN (outside a DTM) are left
out.

## Grids and tile edges

Each return belongs to the cell of the catalogue grid that holds it: the
grid's south-west corner is the catalogue's minimum snapped down to a
multiple of `resolution`, and cells are half-open, `[x0, x0 + resolution)`.
A chunk computes the cells lying wholly within its core grown by 1.25
cells, from all of the returns in each (the buffer is raised to at least 1.5
cells so that they are all read), and each cell of the result is taken from
the chunk whose core holds its centre. Before anything is summed, a cell's
returns are put in a fixed order (height, then x and y, then the
attributes). So:

- a cell that straddles two tiles gets its returns from both, and has the
  value it would have if the tiles were one file;
- the result is the same, bit for bit, whatever `chunk_size` and `workers`
  are (with `dtm="auto"`, where the chunks' DTMs agree, as they do away from
  the edge of the survey with a buffer wider than the gaps in the ground).

Both are tested: a 7 m grid across 30 m tiles equals the metrics of each
cell's points in the merged cloud, and a run with one chunk per tile on one
worker equals one with 17 m chunks on three workers, to the bit. `zmax`
equals `als.chm` at the same resolution in every cell whose highest
return is at or above the ground.

## Plots

`als.plot_metrics(cat, plots, radius=...)` takes circular plots as an
`(N, 2)` array of centres and a radius (one for all or one per plot), or,
without `radius`, polygons in any form [`sylva.masks`](masking.md) accepts:
a layer from `masks.read_polygons("plots.shp")`, `Polygon` objects with
holes, or `(K, 2)` vertex arrays, one plot per feature. Points on a plot's
boundary are inside it. The plots are grouped by the tiles they overlap and
each group reads only those tiles, and only the points in the group's box
(grown by `buffer` for `dtm="auto"`, to make the DTM), so a thousand plots
over a large survey read a few tiles each.

The result, a `PlotMetrics`, holds a `plot` column (the position in the
input), an `id` column when `ids` are given, and one column per metric:
`t["zq95"]`, `t.row(0)`, `t.to_pandas()`, `t.to_csv(path)`.

```python
from sylva import als, masks

layer = masks.read_polygons("plots.shp")
t = als.plot_metrics(cat, layer, ids=[f.properties["plot_id"] for f in layer],
                     metrics=["zq95", "zmean", "cover", "zentropy"], min_height=0.0)
```

## Your own metrics

`func` replaces the built-in set with any Python function of a cell's or a
plot's returns, as lidR's `pixel_metrics(las, ~f(Z, Intensity))` does. It
receives a [`PointCloud`](../api/pointcloud.md) whose z is the height above
ground, with the tiles' attributes, in the fixed order above, and returns a
dict of name to number (or a single number, named `value`):

```python
import numpy as np

def canopy(c):
    top = c.z[c.z > 2.0]
    return {"hmean": top.mean() if len(top) else np.nan,
            "vci": np.var(top) if len(top) > 1 else np.nan}

mine = als.grid_metrics(cat, 20.0, func=canopy)          # dict of two rasters
```

The chunks, buffers and heights are those of the built-in path, so the
tile edges do not show here either, but the function runs once per cell in
Python, one chunk at a time. With a simple function this costs about a
quarter of a millisecond per cell: negligible for a 20 m grid (reading the
tiles dominates), but on the example below at 1 m (10,000 cells) it takes
3.0 s where the whole built-in set takes 0.7 s, and the gap grows with the
area.

## A worked example on synthetic tiles

The forest of the [catalogue example](als.md#a-worked-example-on-synthetic-tiles),
flown and classified the same way:

```python
import numpy as np
from sylva import als, synthetic

rng = np.random.default_rng(0)
trees = [(x, y, 0.3, h) for x, y, h in
         zip(rng.uniform(5, 95, 40), rng.uniform(5, 95, 40), rng.uniform(10, 25, 40))]
scene = synthetic.forest(trees, size=100.0, ground_points=100, margin=0.0)
flight = synthetic.als_flight(scene, altitude=80.0, speed=10.0, line_spacing=40.0,
                              pulse_rate=50_000, bounds=(0, 0, 100, 100))
cat = flight.write_tiles("tiles", size=50.0, epsg=32755)
ground = als.classify_ground(cat, "ground", method="csf", cloth_resolution=1.0)

m = als.grid_metrics(ground, 20.0, ["zmax", "zq95", "cover", "zentropy"], min_height=0.0)
print(np.round(m["zq95"].data[0], 1))            # [18.3 17.5 15.9 18.4  0.1  0. ]
t = als.plot_metrics(ground, [(25, 25), (75, 25), (25, 75), (75, 75)], radius=15.0,
                     metrics=["n", "zmax", "zq95", "cover"], min_height=0.0,
                     ids=["SW", "SE", "NW", "NE"])
```

| Plot | `n` | `zmax` (m) | `zq95` (m) | `cover` (%) |
|---|---|---|---|---|
| SW | 97,440 | 27.98 | 19.02 | 23.8 |
| SE | 96,126 | 29.87 | 18.35 | 18.4 |
| NW | 80,555 | 18.57 | 0.11 | 1.4 |
| NE | 96,603 | 27.51 | 18.62 | 19.7 |

The north-west plot holds no tree centre, only the edge of a crown from
outside it, hence its low `zq95` and cover. The metrics take under a
second each on a laptop; on 20 million returns in 16 tiles, the full set on
a 20 m grid takes 3 s on 8 threads (9 s on one), and 2,000 plots of
11.28 m radius 4 s.

## Command line

```bash
sylva als-metrics ground/ metrics/ --resolution 20 --metrics zq95,zmean,cover --format tif
sylva als-plot-metrics ground/ plots.shp plots.csv --id-field plot_id --min-height 0
sylva als-plot-metrics ground/ centres.csv plots.csv --radius 11.28
```

`als-metrics` writes one raster per metric, `<metric>.asc` (or `.tif` with
`--format tif`, which needs `rasterio`), into the output directory.
`als-plot-metrics` reads plot polygons (`.shp`, `.geojson`) or a CSV of
centres with columns `x`, `y` and optionally `radius` and `id`, and writes
the table as CSV (empty cells for NaN). Both take `--dtm DTM.asc`,
`--normalized` (z is height) or `--height-attribute height`, and
`--dtm-resolution`, `--threshold`, `--cover-break`, `--entropy-bin`,
`--min-height`, `--keep-noise`, `--metrics`, `--buffer` and `--workers`;
`als-metrics` also takes `--chunk-size`. See [Command line](cli.md).

## Checked against

- **NumPy and SciPy.** On random clouds with intensities, return numbers
  and classes, every metric agrees to 1e-12 with NumPy (`mean`, `std` with
  `ddof=1`, `quantile`) and SciPy (`stats.skew(bias=True)`,
  `stats.kurtosis(fisher=False, bias=True)`), and `zentropy` and `zpcum`
  with a line-by-line transcription of lidR's R code (`seq`,
  `findInterval`, `fast_table`). Analytic cases (heights 0, 1, ..., 10 have
  entropy 1, deciles 10, 20, ..., 90 % and kurtosis 1.78) are exact.
- **The merged cloud.** Grid cells and plots on
  `synthetic.als_flight` tiles equal the metrics of the points of the
  merged cloud inside them, including cells and plots across tile edges.
- **lidR itself** was not run for this page: R is not part of the test
  environment. The definitions follow the lidR source; the quantiles use
  NumPy's interpolation, which differs from R's type 7 by at most a few
  units in the last place.

## Limitations

- `func` runs one chunk at a time in Python and ignores `workers`.
- Only the standard set is built in; lidR's `stdmetrics_pulse` (pulse
  counts) and `stdshapemetrics` (eigenvalue shape of the points) are not.
- Plots are not buffered for their own sake: a plot's metrics use exactly
  the returns inside it. A group of plots over the same tiles is read as
  one box, so plots scattered widely over the same few tiles read those
  tiles in full.
