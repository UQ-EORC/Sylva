# Tree detection and segmentation

The detect → segment → prune pipeline is scored
against manually segmented reference plots (per-point tree labels) as
instance segmentation: a reference tree counts as found when its
best-overlapping sylva tree has IoU ≥ 0.5; predicted trees outside the
reference coverage are ignored. raycloudtools' `rayextract trees`
([Devereux et al. 2026](../references.md)) output is scored the same way.

| Site (points) | sylva F1, pruned at 3 m | at 5 m | raycloudtools F1 | sylva TP / ref | point accuracy |
|---|---|---|---|---|---|
| LITCH_CHERLET (8 M) | 0.964 | 0.968 | 0.97 | 122 / 127 | 0.977 |
| ROBSON_CHERLET (9 M) | 0.591 | 0.613 | 0.55 | 99 / 149 | 0.626 |
| WYTHAM_CHERLET (14 M) | 0.864 | 0.874 | 0.73 | 149 / 180 | 0.896 |
| OFENTAL_CHERLET (15 M) | 0.594 | 0.622 | 0.73 | 57 / 89 | 0.706 |

Scored in October 2026 with the current defaults on the plot clouds and
reference labels as kept beside the harvest benchmark (the `basis` arrays):
ground by PMF on a 5 cm thinning, a 0.5 m DTM, stems detected with the
defaults, the graph on 5 cm voxels, pruned at 3 m (left) and 5 m (right).
The same figures come out at three different corners of the thinning grid,
which was not so before (below).

## What the October 2026 changes did

Three things changed together in the segmentation, and were measured on all
four plots with this protocol:

1. **The graph is bridged.** A k-nearest-neighbour graph over thinned points
   left crowns disconnected from their trunks wherever a few sparsely sampled
   branches carried the connection, and *which* crowns depended on which point
   each voxel happened to keep. Moving the thinning grid by a fraction of a
   voxel moved F1 by -0.047 at Litchfield, +0.027 at Robson, +0.008 at Wytham
   and -0.014 at Ofental: more than any setting ever tuned here. The graph now
   joins its components wherever two come within `max_edge`, one edge per pair
   of components, so its connectivity is the cloud's at that distance and not
   the thinning's. On a synthetic plot of interlocking crowns that took the
   share of tree points left unlabelled from 31 % to under 1 %.
2. **Stems are seeded along the whole bark shell** from the ground to half a
   metre above breast height, and the ground cut dropped from 0.25 m to 0.1 m
   (0 with a ground classification), so a trunk is labelled to within a hand
   of the ground. Neither changed a single tree's outcome once the grid was
   held still; what the earlier cut was protecting against turned out to be
   the grid artefact above.
3. **The understorey sources are off by default.** Costed as a 10 m tree, or
   by the vegetation height around each source, or by the height of the
   column above it up to the first gap, they found no tree on any plot and
   lost two at Litchfield and four at Wytham. What they take from a crown
   they take through foliage from below, whatever they are costed.

| Plot | before (F1 at 3 m, two grid corners) | after (three grid corners) | after, understorey sources on |
|---|---|---|---|
| Litchfield | 0.924 / 0.877 | 0.964 / 0.964 / 0.964 | 0.956 |
| Robson Creek | 0.562 / 0.589 | 0.591 / 0.591 / 0.591 | 0.589 |
| Wytham | 0.848 / 0.856 | 0.864 / 0.864 / 0.866 | 0.850 |
| Ofental | 0.611 / 0.597 | 0.594 / 0.594 / 0.594 | 0.597 |

Any comparison finer than about 0.03 on a single plot made before these
changes is within the grid noise, including the tuning notes further down,
which are kept as the record of what was tried.

## Scored the way the benchmark scores

[Cherlet et al. (2026)](../references.md) evaluate only the trees that lie at least 90 % inside the test
sub-plot (128, 181, 89 and 150 of them, which is what the `in_plot_th0.90`
folders hold), and a prediction whose best overlap is with an *edge* tree is
neglected rather than counted against precision. Matching is Hungarian on the
IoU matrix, true positive at IoU ≥ 0.5. Scoring against every labelled
instance instead, and calling every edge-tree hit a false positive, makes the
numbers look far worse than they are.

Scored with a port of the benchmark's own evaluation code
([qforestlab/TreeInstSegEval](https://github.com/qforestlab/TreeInstSegEval):
IoU on point sets rounded to 1 cm, Hungarian matching on the evaluated trees,
a true positive at IoU > 0.5), on the test splits, with `min_height` set to the
5 m the references themselves use (6 m at Robson Creek; `prune_trees` keeps
3 m by default). The figures an earlier version of this page gave (0.976,
0.732, 0.667, 0.528) could not be reproduced with that code and are
superseded. The table below is the September 2026 record with the defaults
of that time (`power=6`, a 25 cm cut, understorey sources on, an unbridged
graph); the test-split data it needs is no longer on the machine the
benchmarks run on, so it has not been re-scored after the October changes:

| test split | eval trees | recall | precision | F1 | Rayextract F1 | best F1 of the other four methods |
|---|---|---|---|---|---|---|
| Litchfield (savanna) | 128 | 0.945 | 0.984 | **0.964** | 0.919 | 0.930 (TreeLearn) |
| Wytham (temperate broadleaf) | 181 | 0.702 | 0.751 | **0.726** | 0.570 | 0.540 (TreeLearn) |
| Ofental (conifer) | 89 | 0.753 | 0.827 | **0.788** | 0.744 | 0.497 (TreeLearn) |
| Robson Creek (rainforest) | 150 | 0.513 | 0.475 | **0.494** | 0.480 | 0.247 (ForAINet) |

The other methods' values are those of Cherlet et al. (2026), Table 2
(Rayextract, Treeiso, SSSC, and TreeLearn and ForAINet fine-tuned on all four
plots). Sylva's are computed here with the same protocol, so the comparison is
indicative rather than independent.

`power` (the exponent on graph edge length) was raised from 4 to 6 after a
sweep on the benchmark's validation areas, never on the test splits: F1 there
went 0.514 → 0.563 at Robson Creek, 0.636 → 0.660 at Ofental and
0.581 → 0.588 at Wytham, and Litchfield did not change. On the test splits,
scored once with the setting fixed, it gave 0.479 → 0.494, 0.682 → 0.788,
0.707 → 0.726 and 0.964 → 0.964. Settings that did not help on the validation
areas: `gravity`, `wood_costs` (better at Ofental only), a weaker or no
`height_prior`, and two forms of a crown-size prior from stem allometry
(crown radius proportional to DBH, applied to every path or only to points
two trees compete for), which truncated real crowns or fragmented trees.

**What remains.** Stem detection is nearly complete (one of the 150 Robson
Creek trees is never detected); the losses are in dividing crowns between
neighbours. Most missed trees are shorter than their neighbours and are
absorbed by them: 49 of 76 at Robson Creek, 44 of 57 at Wytham and 21 of 31
at Ofental before the change of `power`.

The reporting threshold matters more than anything else we tuned, because the
references only label trees: at Litchfield, moving it from 3 m to 5 m took F1
from 0.810 to 0.976 under the earlier scoring, without losing a single tree,
since everything it drops is shrub. Keep the threshold you segment with separate from the one you report,
and set the second to match whatever your inventory calls a tree.

**Seeds used to be too wide.** A tree starts from the points within
`seed_radius` of its stem at `seed_height`; at 0.5 m those discs overlap
wherever stems are close, and the stronger seed takes the other's crown. Of
the 102 Wytham trees missed before, 79 had been swallowed by a neighbour
whose stem stood a median of **0.3 m away**: coppice stools and low forks,
not distant trees. Halving `seed_radius` to 0.25 m and `merge_radius` to
0.2 m finds 22 more trees (130 → 152), lifts mean IoU 0.83 → 0.85 and
precision 0.88 → 0.90, and costs Litchfield nothing (135 → 136).

A **sparser graph and a steeper cost** help on top of that: `k` 10 → 6 and
`power` 3 → 4 (the exponent on edge length, so a path pays more for a long
hop between two crowns; since raised to 6, above). Found trees: Litchfield 136, Wytham 152 → 154,
Robson 105 → 110, Ofental 63 → 68, with fewer spurious trees everywhere.

A **finer graph** pays for itself where the stand is dense: the cloud is
thinned to `voxel_size` before the graph is built, and 0.05 → 0.03 m gives
Wytham 154 → 156, Robson 110 → 117 and Ofental 68 → 72, each with *fewer*
false trees and a better IoU, for about 1.7× the time. Litchfield does not
move at any resolution. 0.02 m gains two more at Wytham for 2.5× the time,
which did not seem worth making the default.

Other things tried, none of which helped: a denser graph (`k=20`, Wytham
136), gravity 0.3 (140), wood costs (142), dropping the height prior (139 at
Wytham, 61 at Ofental), a wider height-prior radius (62 at Ofental), no angle
penalty (48 at Ofental, much worse), `max_edge` 0.5 (no change), and no
pruning at all (142), so the limit is not the candidate list.

**Where the remaining loss is.** Robson and Ofental are still the hard ones,
and for different reasons. At Ofental 41 of the 54 missed trees are absorbed
by one neighbour, and the missed trees are short (median 10.1 m) where the
matched ones are tall (16.4 m): suppressed conifers under dominants, whose
stems *are* detected (81 % have a candidate within 0.5 m, median 0.05 m) but
whose crowns are taken by the tree above. Robson keeps 222 reference trees
against 655 candidates, so it over-detects in the understorey as well as
merging. Both point at the cost model rather than at detection or pruning.

Of Robson's 212 spurious trees, **123 sit almost entirely (> 80 % of their
points) on what the reference calls ground** (rainforest understorey, median
6.2 m tall), and only 40 are fragments of a labelled tree. The same tension
as the savanna shrubs: real vegetation the benchmark does not count as a
tree. `prune_trees(min_quality_short=0.15)` trims about a quarter of them for
two found trees.

Two ideas from other systems were implemented and measured, and both made
things worse on this data:

- **A lateral-offset prior.** raycloudtools multiplies every edge by
  `1 + g·(horizontal offset from the seed)²`, which is how it stops a
  dominant reaching sideways into a neighbour. Sylva has it as `gravity`.
  Ofental barely moves (F1 0.667 → 0.674 at g = 1, with mean IoU falling
  0.718 → 0.694) and Robson falls apart: 0.521 → 0.487 → 0.450 → 0.437 for
  g = 0.1, 0.3, 1. In raycloudtools the term multiplies a cost already
  normalised by canopy height; here it stacks on `length⁴`, so a 10 m-wide
  rainforest crown pays about ×31 on every edge.
- **Edge weight as the gap between clusters rather than the distance
  travelled.** This is TLS2trees' ([Wilkes et al. 2023](../references.md)) documented fix for a path that prefers a
  suppressed tree's base. Grouping points into 0.25 m cubes and pricing each
  edge by the closest approach between two groups gives Ofental F1 0.667 →
  0.158 and Robson 0.521 → 0.150. The reason is instructive: with gap
  weights, groups that touch cost nothing, so in a closed canopy the whole
  plot becomes one cheap component and a single seed takes it. TLS2trees
  avoids this by running the gap graph on **wood points only**, with clusters
  from 0.2 m slices, and attaching foliage afterwards from the branch tips of
  the finished skeletons. Done that way it might work; done naively it does
  not, and the code was removed rather than shipped as a knob.
