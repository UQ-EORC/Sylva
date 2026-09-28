# Airborne trees

`sylva.als` finds individual trees in airborne lidar: their tops, their
crowns, and which points belong to which tree, for one cloud or for a whole
catalogue of tiles. The algorithms are those of lidR (Roussel et al. 2020),
reproduced from its implementations where the papers leave a detail open,
and the catalogue runs through the same chunk engine as the rest of
[`sylva.als`](als.md), so a crown that crosses a tile edge is found once and
whole.

```python
from sylva import als

ground = als.catalog("ground/")                   # tiles with ground classified
trees = als.find_trees(ground, out="labelled/",   # each point with its tree id
                       method="dalponte2016",
                       window=als.LinearWindow(0.0, 0.2, 2.0, 20.0))
trees.to_csv("trees.csv")                         # id, x, y, height, crown_area, n_points
trees.to_geojson("crowns.geojson")                # crown polygons
```

| Sylva | lidR | What |
|---|---|---|
| `als.locate_trees(chm or cloud, window, hmin, shape)` | `locate_trees(x, lmf(ws, hmin, shape))` | tree tops |
| `als.segment_crowns(chm, tops, "dalponte2016")` | `dalponte2016(chm, ttops, th_tree, th_seed, th_cr, max_cr)()` | crowns on a CHM by region growing |
| `als.segment_crowns(chm, tops, "watershed")` | ForestTools' `mcws(treetops, CHM, minHeight)` | crowns by marker-controlled watershed |
| `als.li2012(cloud, dt1, dt2, R, Zu, hmin, speed_up)` | `segment_trees(las, li2012(...))` | point-based segmentation |
| `als.segment_trees(cloud, method=...)` | `segment_trees(las, ...)` then `crown_metrics(las, geom = "convex")` | all of it on one cloud |
| `als.find_trees(catalog, out=..., method=...)` | the same on a `LAScatalog` | all of it on a catalogue |
| `als.crown_hull(xy, "convex" or "concave")` | `st_convex_hull`, `concaveman` | crown outlines |

## A worked example

[`synthetic.crown_forest`](../api/synthetic.md) builds a stand whose trees
have closed, convex crowns of known size, which
[`synthetic.als_flight`](../api/synthetic.md) flies over, so every result
can be checked against the truth:

```python
import numpy as np
from sylva import als, synthetic

trees = synthetic.stand(100, size=100.0, min_spacing=3.0, heights=(10, 25), seed=11)
scene = synthetic.crown_forest(trees, size=100.0, ground_points=100, margin=0.0, seed=11)
flight = synthetic.als_flight(scene, pulse_rate=4_000, line_spacing=30.0,
                              bounds=(0, 0, 100, 100), seed=11)
cat = flight.write_tiles("tiles", size=50.0, epsg=32755)      # 4 tiles, 20 returns/m²

found = als.find_trees(cat, out="labelled", method="dalponte2016",
                       window=als.LinearWindow(0.0, 0.2, 2.0, 20.0), max_cr=20)
print(len(found), "trees; tallest", found.height.max().round(1), "m")
truth = synthetic.forest_trees(scene)          # tops, heights and crowns of the real trees
```

The tiles carry their true `classification`, so the heights above ground
come from a DTM made from the ground returns (`dtm="auto"`, the default).
On this stand 68 % of the trees are found (76 % of those whose crown is at
least a quarter exposed), 41 % of the tops found are over-segmentation, the
heights of those found are 0.16 m low on average, and the crown areas are
8 % smaller than those of the trees' returns in the median; the section
[Validation](#validation) has the numbers across stands and point
densities.

## Tree tops

`als.locate_trees` is the local maximum filter of Popescu and Wynne (2004),
lidR's `lmf`. A site, which is a CHM cell (its centre, with the cell's
value) or a point (with its height), is a tree top when it is at least
`hmin` high and no other site in its window is higher. The window is a disc
of diameter `ws` centred on the site (`shape="circular"`, the default) or a
square of side `ws`. `ws` can be:

- a number (m), the same everywhere;
- `als.LinearWindow(intercept, slope, min, max)`: `clip(intercept + slope
  h, min, max)` for a site `h` m high, evaluated in the Rust core;
- any function of height taking and returning arrays, e.g. `lambda h: 0.1
  * h + 3` as in lidR's documentation. For one cloud it is evaluated at
  every site; for a catalogue it is sampled every centimetre of height and
  interpolated linearly, which differs from the function only where it
  jumps.

Of equal-height maxima that lie in each other's windows only one is kept:
lidR keeps whichever it happens to tag first, which depends on its spatial
index and threads; Sylva keeps the first in order of x, then y, so that
results do not depend on the chunking or the thread count. On a CHM the
windows are measured in whole cells from the site, so they do not depend on
where the raster starts.

## Crowns

**Dalponte and Coomes (2016)**, `method="dalponte2016"`, grows each crown
from its top on the CHM, as lidR's `C_dalponte2016` does, which is what
Sylva follows line by line. The image is swept repeatedly (by columns from
west to east and, within a column, from south to north, lidR's matrix
order, skipping the outermost cells). Each crown cell adds any of its four
neighbours that is not in a crown at the start of the sweep and

- is higher than `th_tree` (2 m),
- is higher than `th_seed` (0.45) times the top's CHM value,
- is higher than `th_cr` (0.55) times the crown's mean height,
- is at most 5 % above the top, and
- lies fewer than `max_cr` cells (10) from the top in x and in y.

Cells added in a sweep count from the next one; a cell claimed by two
crowns in one sweep goes to the later claim, and the mean height counts it
for both, as in lidR. Sweeps repeat until no crown grows. `max_cr` is in
cells, not metres: at 0.5 m cells the default allows crowns up to about 5 m
from the top.

**Marker-controlled watershed** (Meyer and Beucher 1990), `method=
"watershed"`, floods the CHM from the tops, highest cells first (Meyer's
algorithm on the inverted CHM, over 8-connected cells higher than
`th_tree`, without watershed lines): each cell joins the crown whose flood
reaches it first, ties in height taken in the order they were reached.
This is the approach of ForestTools' `mcws`. Tops on cells not higher than
`th_tree` grow nothing.

**Li et al. (2012)**, `method="li2012"`, works on the points, as lidR's
`LAS::segment_trees` does. The points are taken from the highest down; the
highest left starts a tree (its set P, with an empty set N), and every
point left within `speed_up` of that top, from the highest down, joins P or
N by its smallest horizontal distances `d1` to P and `d2` to N. A local
maximum (the highest point within a disc of diameter `R`: lidR passes `R`
as its window size) joins N if `d1 > dt` or `d2 < d1 < dt`, P otherwise,
with `dt = dt2` above `Zu` and `dt1` below; any other point joins P if `d1
<= d2`. P becomes the tree and N is left for the next ones, until the
highest point left is below `hmin`. lidR's "dummy" point in N lies 100 m
beyond the cloud's corner, so no point within `speed_up` of a top is ever
nearer to it; Sylva leaves N empty, which gives the same result for any
`speed_up` under 141 m. Equal heights are taken in order of x, then y.

`als.segment_trees` puts it together for one cloud: a CHM at `resolution`
(the highest point per cell, on a grid aligned with multiples of the
resolution, with two empty cells around the cloud so that the outermost
crowns can grow), tops by `locate_trees` on it (or on the points with
`tops_from="points"`), crowns by `segment_crowns`, and each point takes the
crown of its cell. `smooth=k` mean-filters the CHM over `(2k + 1)²` cells
first (lidR's examples smooth with a 3 x 3 mean, `smooth=1`); a top's
reported height is still the unsmoothed cell value. `method="li2012"`
labels the points directly and a tree's top is its highest point;
`method="tops"` stops at the tops. Points lower than `min_point_height`
(0.5 m) belong to no tree, so ground returns under a crown are not
labelled with it (lidR's CHM methods label every point in a crown's
cells); for `li2012` this cannot change the
labels of higher points, which are all taken before them.

Each tree comes back with its top (`x`, `y`, `height`), its crown outline
(`crowns`, from its points, convex by default), the outline's area
(`crown_area`, NaN without a crown of three points) and its number of
points; `tree_id` gives each point's tree (0 for none). Ids are 1, 2, ...
in order of the tops' x, then y.

### Crown outlines

`hull="convex"` (the default) is the convex hull of a tree's points, as
lidR's `crown_metrics(geom = "convex")`. `hull="concave"` is the
characteristic shape of Duckham et al. (2008): from the Delaunay
triangulation of the points, the outline's edges longer than `concavity`
metres (2 by default) are removed, longest first, as long as the outline
stays a simple polygon. It follows notches in a crown that the convex hull
bridges; its area is never larger. (lidR's `geom = "concave"` uses the
`concaveman` algorithm, a different construction, so the outlines differ.)

## Catalogues

`als.find_trees(catalog, ...)` runs the segmentation of `segment_trees`
on every chunk with its buffer, on the CHM grid shared by the whole
catalogue. The rules that make the result independent of the chunking:

- **A tree belongs to the chunk that holds its top**: a CHM-cell top to the
  chunk whose core is nearest the cell's centre (the rule by which raster
  outputs are joined), a point top to the chunk whose core holds the point.
  It is reported once, with the crown that chunk finds for it, which
  includes points from the buffer.
- **Ids come from the whole catalogue.** After every chunk has been
  segmented, the tops of all the trees are ordered by x, then y, and
  numbered 1, 2, ...; the ids do not depend on the tiling or the workers.
- **Labelled tiles.** With `out`, the tiles are read a second time (their
  cores only) and written with each point's tree id in `attribute`
  (`tree_id` by default, a 32-bit integer; 0 for none). A point is written
  by the chunk whose core holds it and takes the tree whose crown holds it
  in that chunk, identified by its top's exact coordinates, wherever the
  top is.

With a buffer wide enough to hold a crown and every top and crown that
competes with it, the trees and labels are those of `segment_trees` on all
the tiles merged into one cloud, to the bit. A rule of thumb is half the
largest window plus two crown diameters for the CHM methods, and twice
`speed_up` for `li2012`; the default is 30 m. A crown that reaches the
inner edge of its chunk's buffer is counted in `Trees.at_edge`, and a point
whose tree was claimed by no chunk in `Trees.unmatched_points`; either
raises a warning suggesting a wider buffer.

Heights above ground come from `dtm`: `"auto"` makes one DTM of the
whole catalogue from the ground points first (as [`als.dtm`](als.md) with
`method="lowest"`), so that a point has the same height in every chunk
that reads it; a `Raster` is subtracted; None takes z as height
(normalised tiles). (A DTM per chunk, as `als.chm` makes, would differ
slightly between neighbouring chunks near their buffers' edges, and a tree
seen from two chunks could then differ.)

## Validation

**Against the definitions.** The Rust and Python tests check the local
maximum filter against a brute-force search over all pairs of points
(fixed and height-dependent windows, discs and squares), the tops of two
analytic cones, the watershed boundary between them (the valley where the
cones meet), the thresholds and the `max_cr` limit of Dalponte's growing,
the separation of two point clusters by `li2012`, the exact nearest
distances its search relies on, and the concave outline of a square with a
notch cut out of it.

**Against known trees.** On `synthetic.crown_forest` stands (ellipsoidal
crowns of radius a quarter of the tree height and length half of it,
heights 10 to 25 m, stems at least 3 m apart) flown by `als_flight` at
three pulse rates, each 1 ha plot was segmented with fixed settings,
`window=LinearWindow(0, 0.2, 2, 20)` (a window a fifth of the height) and
`max_cr=20` for the CHM methods and lidR's defaults for `li2012`, chosen on
a separate stand of 150 trees/ha; the CHM was at 0.5 m, or 1 m below 12
returns/m². A detected top is matched to the tree of the highest return
within 0.75 m of it; a tree is found if at least one top lies on it, and
every other top is a commission (a second top on a tree, or a top on no
tree). "Exposed" trees are those whose returns are the top of the canopy
over at least a quarter of their crown's area. Heights are compared with
the true tree heights, crown areas with the convex hull of the tree's own
returns (the best a hull of labelled returns can do), and "points right"
is the share of tree returns above 0.5 m that carry the id of the top
matched to their tree. Trees within 5 m of the plot edge are left out.

| Trees/ha | Returns/m² | Method | Found (all) | Found (exposed) | Commission | Height bias, RMSE (m) | Crown area: median error (mean abs.) | Points right |
|---|---|---|---|---|---|---|---|---|
| 50 | 16 | tops, points | 82 % | 89 % | 48 % | -0.23, 0.54 | | |
| 50 | 16 | dalponte2016 | 77 % | 83 % | 56 % | -0.11, 0.15 | -8 % (35 %) | 74 % |
| 50 | 16 | watershed | 77 % | 83 % | 56 % | -0.11, 0.15 | -14 % (40 %) | 72 % |
| 50 | 16 | li2012 | 79 % | 86 % | 26 % | -0.17, 0.31 | 0 % (34 %) | 77 % |
| 50 | 64 | tops, points | 77 % | 83 % | 6 % | -0.04, 0.07 | | |
| 50 | 64 | dalponte2016 | 77 % | 83 % | 12 % | -0.04, 0.07 | 0 % (16 %) | 83 % |
| 50 | 64 | watershed | 77 % | 83 % | 12 % | -0.04, 0.07 | 0 % (15 %) | 83 % |
| 50 | 64 | li2012 | 77 % | 83 % | 0 % | -0.09, 0.28 | 0 % (28 %) | 80 % |
| 100 | 20 | tops, points | 68 % | 76 % | 38 % | -0.20, 0.45 | | |
| 100 | 20 | dalponte2016 | 68 % | 76 % | 41 % | -0.19, 0.44 | -8 % (34 %) | 68 % |
| 100 | 20 | watershed | 68 % | 76 % | 41 % | -0.19, 0.44 | -4 % (47 %) | 64 % |
| 100 | 20 | li2012 | 65 % | 72 % | 27 % | -0.16, 0.36 | 0 % (45 %) | 59 % |
| 100 | 80 | tops, points | 62 % | 70 % | 4 % | -0.03, 0.06 | | |
| 100 | 80 | dalponte2016 | 62 % | 70 % | 6 % | -0.04, 0.07 | +12 % (23 %) | 74 % |
| 100 | 80 | watershed | 62 % | 70 % | 6 % | -0.04, 0.07 | +3 % (27 %) | 73 % |
| 100 | 80 | li2012 | 59 % | 67 % | 0 % | -0.10, 0.50 | +27 % (57 %) | 66 % |
| 200 | 26 | tops, points | 54 % | 66 % | 33 % | -0.17, 0.26 | | |
| 200 | 26 | dalponte2016 | 57 % | 70 % | 34 % | -0.20, 0.40 | -11 % (31 %) | 57 % |
| 200 | 26 | watershed | 57 % | 70 % | 34 % | -0.20, 0.40 | -22 % (58 %) | 52 % |
| 200 | 26 | li2012 | 50 % | 62 % | 25 % | -0.15, 0.23 | -1 % (53 %) | 49 % |
| 200 | 106 | tops, points | 48 % | 59 % | 5 % | -0.04, 0.08 | | |
| 200 | 106 | dalponte2016 | 48 % | 59 % | 9 % | -0.04, 0.08 | +19 % (31 %) | 62 % |
| 200 | 106 | watershed | 48 % | 59 % | 9 % | -0.04, 0.08 | +12 % (37 %) | 61 % |
| 200 | 106 | li2012 | 47 % | 56 % | 1 % | -0.12, 0.43 | +41 % (63 %) | 54 % |

The CHM methods share their tops (those of `locate_trees` on the CHM), so
their detection is the same; they differ in the crowns. What the table
shows:

- **Trees that are not tops cannot be found.** The trees missed are, with
  few exceptions, overtopped: a shorter tree whose top lies under the crown
  of a taller neighbour is no local maximum in any surface model. The
  share of such trees grows with stand density, and so the detection falls
  from about 80 % at 50 trees/ha to about 50 % at 200.
- **Point density decides the commission.** With few returns per square
  metre the CHM is ragged, each crown has several local maxima, and a
  window a fifth of the height splits it: at 4 to 7 returns/m² (not in the
  table) 60 to 75 % of the tops are commissions and the crowns are split
  into pieces a third of their size, even though 90 to 100 % of the exposed
  trees are found. Smoothing the CHM (`smooth=1`) and widening the window
  (a third of the height) at 1 m cells brings the commission to 30 to 40 %
  there for the CHM methods. At 60 returns/m² and more it is 0 to 12 %.
- **Heights** from the CHM are within 0.1 m of the truth at high density
  (the highest return is within centimetres of the apex) and 0.2 m low at
  20 returns/m², where the apex is sampled less often. Those of `li2012`
  scatter more (RMSE 0.2 to 0.5 m), from the trees it merges.
- **Crown areas** of the CHM methods are within 10 to 20 % of those of the
  trees' returns in the median, with a mean absolute error of 15 to 60 %;
  in dense stands the crowns found are larger, since the crowns of the
  trees missed are shared among their neighbours. `li2012` with lidR's
  default spacing thresholds (1.5 and 2 m) merges more crowns at high point
  densities.

**Chunking and threads.** A 1 ha stand of 200 trees/ha at 106 returns/m²
(1.06 million returns), written as four 50 m tiles of heights, was
segmented as one cloud and as a catalogue with one chunk per tile (1 and 4
workers) and on grids of 30 m and 70 m chunks (3 and 2 workers), with a
30 m buffer. For all three methods and all four layouts the trees (ids,
tops, heights, areas, point counts and outline vertices) and every point's
label in the written tiles were identical to those of the single cloud,
and no crown reached a buffer's edge.

**Large inputs.** 25 copies of that stand side by side (26.6 million
returns in 25 tiles of 100 m) were segmented with labelled tiles written
in 49 s with `dalponte2016` (2,585 trees) and 64 s with `li2012`, on 4
workers, the whole process never holding more than 1.9 GB.

## Synthetic stands

- `synthetic.stand(n_trees, size, min_spacing, heights, seed)` draws stems
  uniformly in a square, rejecting any closer than `min_spacing` to one
  already placed; `dbh = 0.1 + 0.015 * height`.
- `synthetic.crown_forest(trees, size, shape, crown_radius, crown_length,
  density, ...)` gives each tree a stem up to its crown and a crown
  (`"ellipsoid"` or `"cone"`) of radius `crown_radius * height` and length
  `crown_length * height`, filled uniformly with `density` leaf points per
  m³, on the terrain of `synthetic.forest`. The trees of `synthetic.forest`
  carry their leaves in clusters at the ends of a few limbs, which suits
  terrestrial scanning but gives an airborne sensor several separate
  "crowns" per tree; these have the closed outline that airborne tree
  detection assumes, and a known projected area `pi (crown_radius *
  height)²`.
- `synthetic.forest_trees(scene)` gives the truth of either kind of scene:
  per tree its stem position, its highest point, its height above the
  terrain beneath that point, and the convex hull of its points seen from
  above, with its area.

## Command line

```bash
sylva als-trees ground/ trees.csv --crowns crowns.geojson --labelled labelled/ \
    --method dalponte2016 --window-linear 0 0.2 2 20 --max-cr 20
sylva als-trees normalised/ tops.csv --normalized --method tops --window 5
```

See [Command line](cli.md) for every option.

## Limitations

- The methods find the trees that make a local maximum in the canopy
  surface; overtopped trees are not found, and in dense stands they are
  most of the trees missed.
- Every setting (window, thresholds, `max_cr`, resolution) interacts with
  point density and crown size; the defaults are lidR's, and the settings
  of the validation suited its crowns. Tune them on a plot with known trees
  before a large run.
- `li2012` compares every point near a top with the tree's points so far;
  it is several times slower than the CHM methods.
- A crown is the set of points with its label, and its outline their hull;
  with `hull="concave"` the outline depends on `concavity`.
- Heights are those of the returns: a tree's apex is rarely hit exactly,
  so heights are low by a few centimetres at high density and a few
  decimetres at low density.
