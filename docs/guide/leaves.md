# Leaves on a QSM

A QSM describes the wood. `sylva.leaves` adds the foliage as flat polygons
whose total area, position and orientation follow what was measured, giving a
whole-tree model for radiative transfer or visualisation.

```python
from sylva import leaves, qsm

wood = leaves.classify_leaf_wood(tree)                  # bool per point
model = qsm.build_qsm(qsm.wood_points(tree), base_xy=(x, y))
foliage = tree[~wood]

angles = leaves.leaf_angle_distribution(foliage)        # from point normals
print(angles.mean_deg, angles.de_wit, angles.chi, angles.g([0.0, 1.0]))

area = leaves.leaf_area_density(foliage, voxel_size=0.25)
mesh = leaves.add_leaves(model, area, angles, leaf_points=foliage,
                         leaf_length=0.08, leaf_width=0.04)
leaves.write_tree_obj("tree.obj", model, mesh)         # objects "wood" and "leaves"
```

## The four steps

**Leaf / wood labels.** `classify_leaf_wood` uses, by default, the graph-based
separation of [Tian and Li (2022)](../references.md), following their GBSeparation code. Shortest paths from the base give every
point a path length and a direction of growth; graph edges are cut where they
are long for their neighbourhood or join points growing in different
directions, which severs leaves from the branch they hang on before any shape
is judged. The cut graph is split into shells of path length at several
scales (0.1–1 m for trees under 15 m, 0.5–3 m above), and a connected piece
of a shell is wood when it runs the whole shell along its growth direction and
is either cylindrical (a circle fits its cross-section) or linear; a piece
thicker than the wood below it on its path is rejected. Wood then spreads
down every path to the base and to close neighbours. Judging a 10–100 cm
*segment* rather than a 5 cm neighbourhood is what makes it work: over a few
centimetres a leaf is as planar as bark.

`method="passage"` is the QSM's own wood filter (path passage plus local
anisotropy, here with a second 10 cm scale). It keeps nearly all the wood and
half the leaves with it, which is what a QSM wants — a cylinder fit suffers
more from missing wood than from stray leaves — and the wrong trade for leaf
work:

| 30 manually labelled tropical trees (Van den Broeck et al. 2025) | accuracy | mIoU | wood recall / precision | leaf recall / precision |
|---|---|---|---|---|
| anisotropy only | 0.75 | 0.56 | 0.68 / 0.57 | 0.77 / 0.84 |
| passage (the QSM filter) | 0.61 | 0.44 | 0.97 / 0.45 | 0.44 / 0.97 |
| passage, two scales | 0.68 | 0.51 | 0.95 / 0.50 | 0.55 / 0.96 |
| graph-based, 0.1–1 m shells | 0.80 | 0.66 | 0.93 / 0.64 | 0.73 / 0.96 |
| **graph-based, 0.5–3 m shells (default for tall trees)** | **0.90** | **0.79** | 0.91 / 0.80 | 0.89 / 0.96 |

As the QSM input the graph-based labels are worse (destructive-harvest volume
rRMSE 26 % against 20 %), so `qsm.wood_points` keeps the passage filter;
`wood_points(method="gbs")` is there to try.

**Leaf angle distribution.** Normals from a PCA over each leaf point's
neighbours ([Vicari et al. 2019](../references.md)) give inclinations (angle between the leaf normal and the vertical),
weighted by the area each point stands for so densely scanned leaves do not
dominate. The result carries the histogram, mean, a two-parameter beta fit
([Goel and Strebel 1984](../references.md)), [Campbell's (1990)](../references.md) ellipsoidal χ, the
nearest [de Wit (1965)](../references.md) type and the projection function `g(zenith)`
([Wilson 1960](../references.md)). `LeafAngleDistribution.from_type("planophile")` gives the analytic
types.

**Leaf area density.** Without pulses, `leaf_area_density` counts the surface
in the points: thinned to one point per cube of side `res`, a surface with
normal *n* crosses (|nx| + |ny| + |nz|) / res² cubes per unit area, so each
survivor stands for res² / (|nx| + |ny| + |nz|). `res` defaults to 3.5 × the
median point spacing. This is a box count with no plateau — smaller cubes
undercount, larger ones overcount at leaf edges — and it only sees foliage the
scanner saw. With pulse data prefer a ray-traced grid,
`LeafAreaGrid.from_voxels(grid, field)`; with a known total (litterfall,
allometry, hemispherical photos) keep the spatial pattern and rescale:
`area.scaled_to(total_m2)`. A plain number instead of a grid does that in one
step: `add_leaves(model, 85.0, angles, leaf_points=foliage)`.

**Insertion.** Each voxel receives leaves until its area is met, centred on
leaf points of that voxel (uniformly inside it if it has none), with normals
drawn from the angle distribution and a uniform azimuth; blades within
`max_branch_distance` of a cylinder point away from it and record that
cylinder. Leaves may intersect each other: no collision test is made, unlike
the non-intersecting insertion of [Åkerblom et al. (2018)](../references.md).

## Leaf shape and size

The blade is a `LeafShape`: a mesh in unit leaf space — `(along, across, up)`
with the base at the origin, the tip at `along = 1` and the greatest width 1
across — plus the length and width it is placed at. The default is the
six-vertex outline at 8 × 4 cm; `shape.area` is the area of one leaf, as is
`single_leaf_area(length, width)`.

Set the size per call, or once for the session:

```python
mesh = leaves.add_leaves(model, area, angles, leaf_points=foliage,
                         leaf_length=0.15, leaf_width=0.06)

leaves.set_default_leaf(length=0.15, width=0.06)   # every later call
leaves.default_leaf()                              # what is in force
```

A custom blade takes any triangle mesh of a single leaf, so a scanned or
modelled one — lobed, curled, or several leaflets — can be used instead. It
is read with the base at the smallest *x*, the tip along +*x*, the blade
across ±*y* and any curl in *z*; `along` then scales with the length and
`across` and `up` with the width. A mesh drawn in metres keeps the size it
was drawn at:

```python
shape = leaves.LeafShape.from_obj("eucalypt_leaf.obj")   # at its own size
shape = shape.resized(length=0.12)                       # or set one
shape = shape.scaled_to(0.004)                           # or an area (m2)
mesh = leaves.add_leaves(model, area, angles, leaf_points=foliage, shape=shape)
leaves.set_default_leaf(shape)                           # or make it the default
```

The leaf count follows from the area to be met divided by the area of one
leaf, so a bigger blade gives fewer leaves for the same leaf area, and a
curled blade counts the area of its triangles, not of its outline.

## How well it works

Against synthetic trees with separate wood and leaf meshes, sampled as points
without occlusion (so the estimators are isolated from visibility):

| | result |
|---|---|
| mean leaf inclination | within 0.5° of the mesh (48–59°), histogram overlap 0.95, correct de Wit type, *G* within 0.01 — and the same from the classified leaf points as from the true ones |
| point-based leaf area, true leaf points | 0.9–1.15 × the mesh area from 5 000 to 80 000 points per m² |
| leaf / wood labels (graph-based) | accuracy 0.81–0.96; 96–99 % of leaf points found at 80–98 % precision; wood precision 0.86–0.97, wood recall 0.24–0.94 by point count (most wood *surface* in these trees is millimetre twigs inside the foliage, which end up as leaf) |
| leaf area after classification | 0.95–1.16 × the mesh area |
| inserted leaves | area met to within one leaf, inclinations as asked, vertical leaf-area profile r = 0.97–1.00 against the mesh |

So the labels, the angle distribution and the placement hold up on clouds
without occlusion. What a real scan adds is visibility: the point-based area
only counts foliage the scanner saw, which is why a ray-traced or
independently known total is the better input for dense crowns.
