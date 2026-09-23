# Quantitative structure models

```python
from sylva import qsm
tree = cloud[labels == 7]                       # one segmented tree
wood = qsm.wood_points(tree)                    # leaf/wood separation (anisotropy + path passage), thin to 2 cm
model = qsm.build_qsm(wood, base_xy=(t.x, t.y), max_radius=0.6 * t.dbh)
print(model.summary())                          # volume, length, branch orders, DBH
model.to_csv("tree7.csv")                       # cylinder table
model.to_obj("tree7.obj")                       # 3-D mesh (Blender / MeshLab / CloudCompare)
model.to_ply("tree7.ply")                       # binary mesh, faces coloured by branch order
```

`wood_points` uses topology as well as local anisotropy: shortest paths from the base over a kNN graph are traced to one
target per 20 cm cell and every point that three or more of them run
through is wood, so a sparsely scanned trunk -- neither planar nor linear
at the neighbourhood scale, and dropped by the old filter -- is kept
because every path to the crown crosses it. High-anisotropy points then
survive in connected components that touch passage wood or are large,
medium-likelihood points after outlier removal and only next to wood
found already, and the result is dilated by 3 cm. On the harvest cohort
this took the volume rRMSE from 23.9 to 22.3 % (r² 0.969 → 0.978). The
medium step is what recovers bark on big occluded stems (DBH rRMSE 12 %
against 17 % without it) but on small leafy crowns it pulls foliage in
around the stem; pass `medium_threshold=1.0` to skip it for saplings.

In the cylinder stage, points are binned into 10 cm geodesic
shells from the base and clustered within each shell; cluster centres
(circle-fitted where the section is dense) are chained greedily, each node
linking to the nearest unplaced centre of a higher shell and stragglers to
their nearest placed centre, then Taubin-smoothed with the ends pinned. A
trunk that starts by wandering through buttress arms is cut at the first run
of near-vertical cylinders and replaced by a straight stump. Axes are chosen
by subtree top height, then reassigned by subtree volume once radii exist.
Each node's radius is a RANSAC circle in the plane perpendicular to the
skeleton, refitted in a radius-relative band; thick sections seen all round
take the equivalent-area radius of a Fourier contour instead, and fluted
sections a circle cannot explain take that contour's area. On the stem a
quadratic in subtree length through the accepted circles (or the tree
allometry) replaces weak fits and fills gaps, non-increasing; everything
unmeasured, branches included, comes from a pipe model, tapering
linearly with subtree length on straight runs and sharing the parent's
cross-section by subtree length at every fork.

Accuracy against felled trees and against raycloudtools is on the
[QSM benchmark](../benchmarks/qsm.md) page.

## Meshes

`QSM.mesh`, `to_obj` and `to_ply` build one closed tube per cylinder, so a
branch of 50 cylinders is 50 tubes with 100 caps inside it. With
`contiguous=True` each branch is one continuous tube instead: consecutive
cylinders share a ring, the ring frame is carried along the branch so the
facets do not twist, and a shared ring takes the mean of the two radii.

```python
model.to_ply("tree.ply", contiguous=True)
```

The mesh is about half the size, and because nothing is double-covered it
encloses the volume the cylinders describe (within a few per cent, from the
averaged radii at the joints), which the per-cylinder mesh does not. Each
branch is still its own closed surface pushed into its parent, not welded to
it: welding them into a single solid would need a boolean union, which Sylva
does not do.

## Tree metrics

`QSM.metrics()` reads the tree's architecture off the cylinders:

```python
m = model.metrics()
m["height"], m["dbh"], m["stem_volume"], m["branch_volume"]
m["n_branches_by_order"], m["length_by_order"], m["volume_by_order"]
m["crown_base_height"], m["crown"]["projected_area"], m["crown"]["volume"]
m["lean"], m["sweep"], m["path_fraction"], m["median_insertion_angle"]
m["taper_heights"], m["taper_radii"]                # the stem profile
m["measured_volume_fraction"]                       # fitted to points vs filled in
b = model.branches()                                # one row per branch
b["order"], b["length"], b["base_radius"], b["insertion_angle"], b["zenith"]
```

- A branch is one `branch_id` chain. Its insertion angle is measured
  between its direction over the 0.5 m past its first cylinder and the
  parent's direction over 0.5 m either side of the junction. The first
  cylinder only joins the branch to its parent's axis, and at a fork the
  parent cylinder leans towards the branch. Its zenith is that of the chord
  from base to tip.
- The crown base is the lowest first-order branch at least
  `crown_branch_length` (1 m) long. The crown outline is drawn from points
  along every branch cylinder above it.
- `trees.crown_shape()` gives the same shape from any points (a tree's
  segmented points, say): projected area, stacked slice-hull volume and
  surface, and the crown's offset from the stem.
- `measured_volume_fraction` is the share of the volume in cylinders whose
  radius was fitted to points; the rest comes from the taper and pipe-model
  priors. It is a quality flag for the model itself.

Against the 20 synthetic trees with exact wood meshes (simulated leaf-off
scans, eight positions):

| metric | median error | rRMSE |
|---|---|---|
| height | −2.1 % | 3.2 % |
| DBH (15 trees with a circular section at 1.3 m) | −1.1 % | 1.5 % |
| crown projected area | −3.2 % | 6.0 % |
| crown volume | −10.5 % | 14 % |
| branch-segment zenith (length-weighted median) | −0.2° bias | 3.2° MAE |

The crown volume is low for the same reason the wood volume is. The
QSMs recover only about a third of the length of twigs under 1 cm, and
those twigs mark the crown's outer edge.

## A whole plot

`build_plot` is the loop around `build_qsm`: it takes a segmented cloud and
models every tree, thinning, filtering wood, catching the trees that cannot
be fitted and reporting progress as it goes.

```python
from sylva import ground, qsm, trees

cloud = ground.normalize_height(cloud, dtm)
stems = trees.detect_stems(cloud)
labels = trees.segment_trees(cloud, stems)
stems, labels = trees.prune_trees(stems, labels)

plot = qsm.build_plot(cloud, labels, stems)       # {tree_id: QSM}, and why any were skipped
print(len(plot), plot.total_volume, plot.skipped)
plot.to_csv("trees.csv")                          # a row per tree
plot.write_cylinders("qsms/")                     # tree<id>.csv each
plot.write_meshes("meshes/", contiguous=True)     # tree<id>.ply each, fused where buttressed
```

`stems` only supplies the stem centre each model is built around; without it
the centre comes from the tree's own points between 0.5 and 1.5 m. Trees with
fewer than `min_points` points, and trees whose fit fails, land in
`plot.skipped` with the reason rather than stopping the run.

`buttress=True` additionally looks for a flanged base on each tree and meshes
it, so `plot.volume(tree_id)` is the buttress mesh below its top plus the
cylinders above it (see below). It costs a slice-rasterising pass per tree.

`write_meshes` gives one surface per tree: a tree with a buttress is written
fused, so the flanged base and the cylinders above it are one file (named
objects `buttress` and `wood` in OBJ), and every other tree is its cylinder
mesh. `fmt="obj"` for text and named parts, the default `"ply"` for binary
with face colours.

From the command line, on a cloud that carries a `tree_id` attribute (what
`sylva trees --segment` writes):

```bash
sylva qsm-plot plot_trees.laz trees.csv --cylinders qsms/ --meshes meshes/
```

## Buttresses

A cylinder cannot follow a buttressed base. At 1.3 m a large tropical tree can
be a star of flanges that a circle explains only a fraction of, so the QSM
either misses the flanges or spans the gaps between them. Sylva finds
buttresses, then rebuilds them as a closed mesh:

```python
from sylva import qsm, trees

b = trees.detect_buttress(tree)                 # tree: one tree's points with "height"
if b["buttressed"]:
    base = qsm.buttress_mesh(tree, b["centre"], top=b["top"])
    volume = base.total_volume(model)           # mesh below the top + QSM above it
    base.fuse(model).to_ply("tree.ply")         # both as one mesh of the whole tree
```

**Detection** (`trees.detect_buttress`) uses only bark-like points: locally
planar, with a near-horizontal normal. Flanges and round bark are both
vertical surfaces, while grass, shrubs and resprouts clumped around a stem are
not. Two signals then decide:

- how much of the base a circle explains, compared with the round stem higher
  up;
- how many ridges there are: angles around the stem where protrusions persist
  through the lowest metre. Neighbouring flanges are split where persistence
  dips between them.

The top is the lowest height from which a circle explains the stem again.

On 97 harvest trees labelled by eye (Cameroon, Peru, Guyana, Indonesia and
Wytham), it finds 28 of 29 buttressed trees with no false alarms, whether the
points come at 1 or 2 cm. On the 40 largest trees of the Litchfield savanna,
which have no buttresses, it calls 2 buttressed. Both are dense shrub clumps
pressed against the stem.

**Meshing** (`qsm.buttress_mesh`) rebuilds the base volumetrically, so any
shape works:

- Thin slices are rasterised, and a morphological closing plus a flood fill
  gives the solid cross-section.
- Slices are built from the top down. Each section contains the one above,
  and the part of a slice kept is the part connected to it. The core then
  carries down where near the ground only the outsides of the flanges were
  seen.
- A section may widen by no more than `max_flare` metres per metre of drop
  (default 1.0, that is 45°). Flanges flare well within that. Without it, a
  single scan line lying on the ground and ringing the stem encloses
  everything inside it when the outline is filled, and the base comes out
  wearing a slab: on one tropical tree that was 6.6 m² of ground at the foot
  of a 3.2 m² base, 13 % of the buttress volume.
- Where an outline stays open, the seen bark is kept, thickened to the closing
  radius, plus a circle where the points form a good arc.

The stacked sections become a watertight surface (surface nets), closed where
the solid meets the ground and the edge of its raster. A few edges are shared
by four faces where two voxels touch only corner to corner, which some tools
call non-manifold; the surface is still closed, and the volume it encloses
agrees with the reported one. The volume itself is the sum of the slice areas.

**Fusing** (`Buttress.fuse`) puts the base and the cylinders in one mesh of the
whole stem. The buttress replaces the wood below its top: `QSM.above(top_z)`
cuts the model at that plane, cutting through any cylinder that crosses it, and
the two surfaces meet there, so nothing is counted twice and the volume is the
one `total_volume` reports:

```python
fused = base.fuse(model, contiguous=True)
fused.volume, fused.buttress_volume, fused.wood_volume
fused.to_obj("tree.obj")        # objects "buttress" and "wood"
fused.to_ply("tree.ply")        # one mesh, the base in bark brown
```

The two parts stay watertight and separately labelled (`fused.part` per face)
rather than being welded into a single shell: a boolean union of a flanged base
and a few thousand tubes is not something a triangle mesh survives cleanly, and
the seam at the top plane is where the two volumes already meet.

On the 15 Cameroon harvest trees it detects as buttressed, the QSM alone has a
−10.1 % volume bias against the felled volume. With the buttress mesh below
the top, the bias is +2.3 % (RMSE 18.1 % and 17.7 %). Whether the published
felled volumes include the stump below the cut is not recorded, so treat this
as indicative. That run predates the flare limit, which on three of those
trees moves the buttress volume by −1.1 to −5.5 % (so the combined figure by
under a percent); it is worth re-running.
