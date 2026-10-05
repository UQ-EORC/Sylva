# Interpolation

`sylva.geo.interpolate` moves values between point clouds and rasters:

| Function | From | To | Methods |
|---|---|---|---|
| `transfer_attributes(source, target, names)` | attributes of one cloud | points of another | `"nearest"`, `"idw"`, `"majority"` |
| `grid(cloud, resolution, value)` | z or an attribute | a `Raster` | `"idw"`, `"tin"`, `"natural"` |
| `sample_raster(cloud, raster, name)` | a `Raster` | a new attribute | `"bilinear"`, `"nearest"` |
| `sample_rasters(cloud, {name: raster})` | several rasters | one attribute each | as above |

All of them run in the Rust core in parallel, and their results do not
depend on the number of threads.

## Worked example

The example tile (`docs/examples/data/litch_tile.laz`, 1.55 million points
with a ground classification) is thinned, labelled on the thinned copy, and
the labels are carried back to every original point. The ground points are
then gridded into a DTM that is read back at every point.

```python
import sylva
from sylva import filters, ground
from sylva.geo import interpolate

cloud = sylva.read("docs/examples/data/litch_tile.laz")
thin = filters.voxel_downsample(cloud, 0.2)           # 92,061 points

# Any per-point result computed on `thin` (here its own classification)
# goes back to the full-resolution cloud.
full = interpolate.transfer_attributes(thin, cloud, "classification",
                                       method="majority", k=5)

# A DTM by linear interpolation on a triangulation of the (thinned) ground.
g = filters.voxel_downsample(cloud[ground.ground_mask(cloud)], 0.25)
dtm = interpolate.grid(g, 0.25, method="tin")          # 81 x 81 cells, NaN outside the hull

# Terrain elevation at every point, and height above it.
cloud = interpolate.sample_raster(cloud, dtm, "ground")
height = cloud.z - cloud.attrs["ground"]
```

On this tile the majority vote over five neighbours restores the original
class of 97.7 % of the points; the remainder lie at the boundary between
ground and low vegetation, where the thinned cloud cannot tell them apart.
The transfer takes about 0.2 s on eight cores; twenty million target points
from a one-million-point source take a few seconds.

## Attributes between clouds

`transfer_attributes` builds a k-d tree on the source and processes every
target point independently, so the target can be very large. Distances are
3D. The methods are:

- `"nearest"`: the value of the nearest source point. Works for any dtype,
  including strings, and keeps it.
- `"idw"`: the inverse-distance weighted mean over the `k` nearest source
  points, weights `1 / d**power`. A target that coincides with source points
  takes the mean of their values, and non-finite source values are skipped.
  Numeric attributes only; the result is float64 (float32 for float32
  attributes).
- `"majority"`: the most common value among the `k` nearest source points,
  with ties going to the value of the nearest one. This is the method for
  labels (`tree_id`, `classification`, a wood mask), which must not be
  averaged, and it smooths isolated mislabelled points that `"nearest"`
  would copy.

`max_distance` limits the neighbours to those within that distance. A target
with none gets the fill value: NaN for floating-point results and -1 for
integer and boolean labels, or whatever `fill` is given. When the fill does
not fit the attribute's dtype (-1 in a `uint8` classification), the result
is widened to the smallest signed type that holds both, so pass `fill=0` to
keep `uint8` for LAS output:

```python
labels = interpolate.transfer_attributes(thin, cloud, ["tree_id", "classification"],
                                         method="majority", k=5, max_distance=0.1)
# tree_id: int64, -1 beyond 10 cm; classification: int16, -1 beyond 10 cm

las_ready = interpolate.transfer_attributes(thin, cloud, "classification",
                                            method="majority", max_distance=0.1, fill=0)
# classification: uint8, 0 (never classified) beyond 10 cm
```

## Points to rasters

`grid` interpolates `"z"` or a numeric attribute at the cell centres of a
grid laid out as `ground.make_dtm` lays it out: row 0 at `ymin`, and without
`bounds` a south-west corner snapped to a multiple of the resolution. Points
sharing an x, y are merged into one carrying their mean value.

| Method | Surface | Outside the data |
|---|---|---|
| `"idw"` | weighted mean of the `k` nearest points in x, y; stays within the range of the data; flat spots at the points for `power` above 1 | every cell gets a value unless `max_distance` is set |
| `"tin"` | linear on each Delaunay triangle; exact for a plane; creases along triangle edges | NaN outside the convex hull |
| `"natural"` | natural-neighbour (Sibson) weights; exact for a linear function, smooth except at the data points | NaN outside the convex hull |

`max_distance` sets cells whose centre is farther than that from every point
to NaN, for every method, which keeps interpolation from bridging large
gaps such as the shadow behind a tree.

The triangulation and the natural-neighbour weights come from the
[spade](https://crates.io/crates/spade) crate. A TIN of two million points
over a million cells takes about three seconds on eight cores.

### DTMs by interpolation

`ground.make_dtm` keeps its default, the lowest ground point per cell with
empty cells filled, and accepts `method="tin"`, `"natural"` or `"idw"` to
interpolate the ground points instead. Those surfaces pass through every
ground point, so they sit on the mean of the ground returns rather than
their minimum (on the example tile, a median of 5 cm above the default
DTM), and they carry the noise of the returns unless the ground is thinned
first. `make_dtm` fills cells outside the convex hull of the ground points
from the nearest interpolated cell, so the DTM has no NaN like the default;
call `interpolate.grid` on the ground points to keep them NaN.

```python
g = filters.voxel_downsample(cloud[ground.ground_mask(cloud)], 0.25)
dtm = ground.make_dtm(g, 0.25, method="natural")
```

## Rasters onto points

`sample_raster` adds the raster value at each point's x, y as a float64
attribute. `"bilinear"` interpolates between the four surrounding cell
centres, exactly as `Raster.sample`, and gives NaN if one of them is NaN;
`"nearest"` takes the cell containing the point. Points outside the raster
get NaN, unlike `ground.normalize_height`, which extends the DTM's edge
values. `sample_rasters` does the same for several rasters at once:

```python
chm = ground.make_chm(ground.normalize_height(cloud, ground.make_dtm(cloud)), 0.5)
cloud = interpolate.sample_rasters(cloud, {"ground": dtm, "canopy_height": chm},
                                   method="nearest")
```
