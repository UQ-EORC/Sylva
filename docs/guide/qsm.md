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
