# Masking

A mask is a boolean array with one entry per point, `True` where the point
is kept. `sylva.masks` builds masks from four sources: polygons, rasters,
expressions over the attributes, and the distance to another cloud. Masks
combine with `&`, `|` and `~` and index a cloud directly, so a selection that
draws on several sources is a single line:

```python
from sylva import masks

keep = masks.inside_polygons(cloud, plots) & masks.expression(cloud, "height > 2")
subset = cloud[keep]
```

Each mask function has a `crop_*` companion that returns the kept cloud in
one call; `invert=True` keeps the other points instead.

| Mask | Crop | Keeps points |
|---|---|---|
| `inside_polygons(c, polygons)` | `crop_polygons` | whose x, y fall inside any polygon |
| `raster_mask(c, raster, min, max, values)` | `crop_raster` | whose raster cell passes a range or value test |
| `expression(c, "height > 2")` | `crop_expression`, `c.where(...)` | satisfying a condition on their attributes |
| `near(c, other, distance)` | `crop_near`, `difference` | within (or, for `difference`, beyond) a distance of another cloud |

## A worked example

The synthetic plot below has four trees on a 20 m square, with a 4 m
margin of extra points around it.

```python
import numpy as np
from sylva import ground, masks, synthetic

cloud = synthetic.forest()
cloud = ground.normalize_height(cloud, ground.make_dtm(cloud))

# The plot boundary; a polygon can also be read from a file (below).
plot = masks.Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
cloud = masks.crop_polygons(cloud, plot)                  # 295,068 points

# Vegetation more than 2 m above the ground.
canopy = cloud.where("height > 2 & classification != 2")   # 252,473 points

# Points in the canopy gaps: cells of the CHM lower than 2 m.
chm = ground.make_chm(cloud, resolution=0.5)
in_gap = masks.raster_mask(cloud, chm, max=2)             # 9,031 points

# Low tree points that are under canopy cover.
low = masks.expression(cloud, "0.5 < height < 2 & tree_id > 0")
covered = cloud[low & ~in_gap]

# Change detection: remove tree 1, then find what was lost.
after = cloud.where("tree_id != 1")
lost = masks.difference(cloud, after, 0.05)
np.unique(lost.attrs["tree_id"])                           # array([1])
```

## Polygons

`read_polygons` reads ESRI shapefiles (`.shp`, with the `.dbf` attribute
table and the `.prj` CRS when present) and GeoJSON (`.geojson`, `.json`).
Each record becomes a `MultiPolygon` with its `properties`; holes and
multi-part features are kept as stored.

```python
plots = masks.read_polygons("survey/plots.shp")
plots.crs                                 # the .prj text
plots[0].properties                       # {'PLOT_ID': 7, 'NAME': 'north'}
stand = plots[[f.properties["STAND"] == "B" for f in plots]]

inside = masks.inside_polygons(cloud, stand)
which = masks.polygon_index(cloud, plots)            # index of the plot, -1 outside
```

When `path` is a directory, `layer` names the file to read
(`read_polygons("survey", layer="plots")`).

Polygons can also be given as coordinates: a `(K, 2)` array of vertices is
one polygon, a list of arrays is one polygon each, and `Polygon(exterior,
holes)` and `MultiPolygon(parts)` describe holes and multi-part shapes.

```python
circle = np.column_stack([10 + 5 * np.cos(t), 10 + 5 * np.sin(t)])
ring = masks.Polygon(circle, holes=[circle_small])
masks.crop_polygons(cloud, [square, ring], invert=True)
```

Polygons are closed: a point on an edge or a vertex is inside, a point on
the boundary of a hole is inside, and a point on an edge shared by two
polygons is inside both. The test uses exact orientation predicates, so
points that lie on an edge are classified consistently whatever the edge's
direction. A grid over the polygons' bounding boxes and a band index over
each ring's edges keep the cost per point low with many polygons or with
polygons of many vertices, and points are processed in parallel.

!!! note "No reprojection"
    Polygons are taken in the cloud's frame. `read_polygons` reports the
    file's CRS in `Polygons.crs` but never applies it, and the masks do not
    compare it with the cloud's CRS. Reproject the layer or the cloud first
    when they differ. A cloud in a scanner or project frame needs polygons
    in that frame too.

## Rasters

`raster_mask` looks up the cell under each point and tests its value:
`min` and `max` give an inclusive range and `values` a set of accepted
values (for class rasters). With neither, every point over a valid cell is
kept. Points outside the raster and points over NaN cells are never kept.

```python
masks.raster_mask(cloud, chm, min=10)                 # under tall canopy
masks.crop_raster(cloud, landcover, values=[3, 4])    # two classes of a class raster
masks.crop_raster(cloud, dtm, invert=True)            # points outside the DTM or over its holes
```

Cells include their southern and western edges, as in `Raster.cell_index`.

## Expressions

`expression(cloud, expr)`, `crop_expression` and `PointCloud.where` accept
a condition written in a small language that the Rust core parses and
evaluates; the text is never run as Python.

| Element | Syntax |
|---|---|
| values | `x`, `y`, `z`, any attribute name, numbers (`2`, `-0.5`, `1e3`), `true`, `false` |
| arithmetic | `+ - * /`, unary `-` |
| comparison | `< <= > >= == !=`, chained as in `1.3 <= z < 40` |
| membership | `classification in (3, 4, 5)`, `classification not in (7, 18)` |
| logic | `&` or `and`, `|` or `or`, `!` or `not`, parentheses |

`&` and `|` bind more loosely than comparisons, so
`height > 2 & classification != 2` needs no parentheses (unlike the same
expression in NumPy). Values are compared as float64, and a comparison with
NaN is false except `!=`. Boolean attributes are conditions on their own
(`withheld & z > 1`). Errors name the problem and its position:

```text
ValueError: unknown attribute 'hieght' at position 0 (did you mean 'height'?); the cloud has: x, y, z, classification, height, intensity
    hieght > 2
    ^
```

## Distance to another cloud

`near(cloud, other, distance)` keeps the points within `distance` of any
point of `other`, in 3-D or, with `horizontal=True`, in x and y only.
`difference(cloud, other, distance)` returns the points of `cloud` farther
than `distance` from `other`: between two epochs of the same scene,
`difference(after, before, d)` is what appeared and
`difference(before, after, d)` what was lost. Choose `d` above the
registration error and the point spacing, or noise and density differences
show up as change.

```python
stems_zone = masks.crop_near(cloud, stems, 0.1, horizontal=True)
new = masks.difference(epoch2, epoch1, 0.05)
```

A k-d tree is built on `other` and queried for each point in parallel. The
result does not depend on the number of threads, and clouds of tens of
millions of points on both sides fit in the memory of a workstation.
