# Tree detection and segmentation

The detect → segment → prune pipeline is scored
against manually segmented reference plots (per-point tree labels) as
instance segmentation: a reference tree counts as found when its
best-overlapping sylva tree has IoU ≥ 0.5; predicted trees outside the
reference coverage are ignored. raycloudtools' `rayextract trees`
([Devereux et al. 2026](../references.md)) output is scored the same way.

| Plot (points) | F1, pruned at 3 m | at 5 m | trees found / reference | point accuracy | understorey taken | raycloudtools F1 |
|---|---|---|---|---|---|---|
| Litchfield, savanna (8 M) | 0.977 (0.972–0.980) | 0.979 (0.972–0.984) | 124.0 / 127 | 0.981 | 0.26 | 0.97 |
| Robson Creek, rainforest (9 M) | 0.653 (0.641–0.667) | 0.671 (0.654–0.687) | 109.8 / 149 | 0.691 | 0.71 | 0.55 |
| Wytham Woods, temperate broadleaf (14 M) | 0.861 (0.846–0.872) | 0.863 (0.851–0.876) | 149.2 / 180 | 0.893 | 0.33 | 0.73 |
| Ofental, conifer (15 M) | 0.649 (0.646–0.653) | 0.689 (0.678–0.707) | 63.0 / 89 | 0.726 | 0.39 | 0.73 |

Scored in November 2026 with the defaults of version 1.0 on the plot clouds
and reference labels as kept beside the harvest benchmark (the `basis`
arrays): ground by PMF (the protocol's choice; `sylva ground` defaults to
CSF) on a 5 cm thinning with windows of up to 17 m, a 0.5 m DTM, stems
detected and trees segmented with the defaults (a 3 cm graph), pruned at
3 m and at 5 m. The edge cost of the segmentation and the arc test of the
stem detection were replaced after version 1.1.0; on the same four runs
the present code scores 0.977, 0.665, 0.856 and 0.671, which is the same
within the run-to-run range ([below](#the-edge-cost-after-version-110)).

Each sylva figure is the **mean of four runs** of the same plot, with the
range in brackets: the plot as it is, with the thinning grid moved by a
fraction of a voxel, with its points in another order, and rotated by 30°
(next section). *Understorey taken* is the share of the points between 0.1
and 2 m that lie on no reference tree and carry a tree label all the same
(lower is better; [below](#what-the-scores-cannot-see)). The raycloudtools
column is one run, scored the same way. The table is without
`trees.merge_touching`, which `sylva trees` runs after segmentation and
which adds 0.001 to 0.007 to these figures ([below](#merging-trees-that-touch)).

Earlier versions of this page gave one run with a 5 cm graph: 0.964, 0.591,
0.864 and 0.594. Those figures reproduce exactly, and as four-run means the
5 cm graph gives 0.968, 0.597, 0.854 and 0.603. The 3 cm graph, which is
the default, is better on 13 of the 16 paired runs, by 0.057 at Robson
Creek and 0.047 at Ofental, for about 1.6 times the segmentation time
(52 s against 32 s at Ofental).

## One run is not a measurement

Nothing in a result should depend on where the thinning grid sits, on the
order of the points in the file, or on which way the plot faces. All three
move it. Over seven runs of each plot with the 5 cm graph (two grid shifts
of a fraction of a voxel, two shuffles, two rotations and the plot as
given), F1 at 3 m spans:

| Plot | F1 over seven runs | reference trees found in all seven | in some | in none |
|---|---|---|---|---|
| Litchfield | 0.964–0.988 | 119 | 7 | 1 |
| Robson Creek | 0.582–0.625 | 70 | 53 | 26 |
| Wytham | 0.797–0.874 | 119 | 47 | 14 |
| Ofental | 0.576–0.636 | 52 | 17 | 20 |

The count of trees found is steadier than the trees themselves: at Robson
Creek about 100 of the 149 are found in any one of these runs, but only 70
in every one, and a third of the plot's trees are found or missed
according to a nuisance. These are trees whose best overlap sits near the
0.5 threshold, in crowns that two stems contest.

The default 3 cm graph is steadier as well as better. Over the four runs
of the table at the top it spans 0.008, 0.026, 0.026 and 0.007 of F1,
where the 5 cm graph spans 0.012, 0.028, 0.045 and 0.022 over the same
four; 122, 100, 135 and 57 reference trees are found in all four of its
runs and 3, 19, 26 and 10 in some. So:

- a difference between two settings of less than about 0.04 F1 on one plot,
  from one run each, is not evidence. Compare settings as paired runs
  (the same nuisance on both sides) over several nuisances;
- the tuning notes further down, which are single runs, are kept as the
  record of what was tried and not as measurements of that size.

An earlier version of this page said that the scores came out the same at
three corners of the thinning grid. The three corners tested were whole
voxels apart, which is the same grid three times, so the test could not
have failed. The bridged graph (below) did remove the largest effect of
the grid: without it Litchfield loses 0.039 F1 and ranges over 0.886–0.948,
and the other three plots do not change.

## What the scores cannot see

The reference labels trees. Shrubs, saplings, grass and litter carry no
label, so a tree that also takes the vegetation around its base scores
exactly as one that does not. Measured directly, as the share of a matched
tree's points that have no reference tree point within 0.1 m (one run,
5 cm graph):

| Height above ground | Litchfield | Robson Creek | Wytham | Ofental |
|---|---|---|---|---|
| 0.1–0.5 m | 0.87 | 0.69 | 0.74 | 0.73 |
| 0.5–1 m | 0.83 | 0.61 | 0.50 | 0.51 |
| 1–2 m | 0.54 | 0.64 | 0.43 | 0.33 |
| 2–3 m | 0.23 | 0.63 | 0.42 | 0.18 |

Below a metre, most of what a tree is given is not the tree. Part of this
is by design: under `low_height` (0.5 m) a tree keeps what lies within
`low_radius` (1 m) of its stem, grass included. The rest is understorey
that the least-cost paths reach from the stem.

The understorey sources (`understorey_height`, off by default) are the
setting that addresses it, and what they cost and buy is now measured on
both sides (four paired runs, 3 cm graph):

| Plot | trees found: off → 10 m | F1 at 3 m | understorey taken | reference tree points left unlabelled |
|---|---|---|---|---|
| Litchfield | 124.0 → 121.8 | 0.977 → 0.968 | 0.26 → 0.09 | 0.1 % → 0.7 % |
| Robson Creek | 109.8 → 110.2 | 0.653 → 0.657 | 0.71 → 0.64 | 0.1 % → 0.3 % |
| Wytham | 149.2 → 146.2 | 0.861 → 0.851 | 0.33 → 0.22 | 0.3 % → 1.3 % |
| Ofental | 63.0 → 63.0 | 0.649 → 0.657 | 0.39 → 0.28 | 0.4 % → 0.5 % |

They release between 1.7 (Wytham) and 13 (Litchfield) points of understorey
for every point they take from a reference tree, and they cost two trees
at Litchfield and three at Wytham: small trees whose crowns the sources
reach from below and which then fall under the pruning height. Costing the
sources as a 5 m tree instead of 10 m loses the same trees. The default
stays off, because a tree dropped from the table cannot be had back
afterwards while low points can still be filtered; set
`understorey_height=10` where the lower two metres of each tree matter more
than the last two trees in a hundred (stem models, understorey studies).

## What the November 2026 review found

The segmentation was reviewed against these plots in November 2026, each
setting as four paired runs. No defect was found in the segmentation and
its defaults did not change; one was found in stem detection (the arc
test, below) and fixed. What was learned:

**Where the trees are lost.** In one run with the 5 cm graph, 5, 50, 31 and
32 reference trees are missed. Most have a detected stem and a crown
divided wrongly, the tree losing its crown to a neighbour or taking a
neighbour's: 4 of the 5, 47 of the 50, 18 of the 31 and 20 of the 32. The
rest have no stem candidate (1, 2, 13 and 7) or one whose tree ended under
the pruning height (1 at Robson Creek, 5 at Ofental). The missed trees are
shorter than the ones found (9.2 m against 15.6 m at Ofental, 8.7 m
against 13.5 m at Wytham).

**Leaning stems are not detected.** Of the 13 Wytham trees that are missed
for want of a candidate, 12 lean between 25° and 45°, beyond `max_lean_deg`
(25). Raising it to 45 with `link_radius=0.3` finds 8 more trees there, 2
at Ofental and 1 at Robson Creek, and adds 7 to 11 spurious trees on each
of those three plots, so F1 at 3 m does not improve (−0.001 to −0.014). It
is the setting to change for a stand known to lean, not a better default.

**The height prior is the canopy's height.** `height_prior` scales each
tree's costs by one over the 95th percentile of the heights within 1.5 m of
its stem. In a closed stand that is the height of whatever stands over the
stem: at Robson Creek it is within 20 % of the height the tree ends up with
for 22 of 352 candidates, and the candidates that end below 3 m had
competed as trees of 25.7 m (11.1 m at Litchfield, 18.4 m at Wytham, 16.0 m
at Ofental). Taking each tree's own height from a first pass and segmenting
again helps where short trees stand under tall ones of one species and
hurts where everything is tall and interlocked: F1 +0.070 at Ofental,
−0.033 at Robson Creek, −0.006 at Wytham and +0.001 at Litchfield. It was
not adopted.

**`merge_branches` is not free.** The guide and the notebooks run it before
segmenting; the `sylva trees` command does not. With the 3 cm graph it
removes a fifth to two fifths of the spurious trees on every plot (77 → 60
at Robson Creek, 42 → 30 at Ofental), and it merges real trees wherever
stems share a base: 22 of the 149 trees found at Wytham, a coppice (F1
0.861 → 0.798), 5 at each of Robson Creek and Ofental, none at Litchfield.
F1 at 3 m moves by +0.002, +0.016, −0.062 and +0.005. The trees that remain
also take more of the understorey (*understorey taken* rises by 0.05 to
0.19). It is a step for single-stemmed trees with low limbs, not for
coppice or multi-stemmed stands.

**`min_arc_deg` failed real stems on thinned clouds, and was fixed.** The
arc was the longest run of occupied 10° bins. On these clouds (points
about 2.5 cm apart) a stem of 5 cm radius has about three points a bin, a
bin is empty by chance one time in twenty, and an unbroken 130° needs
thirteen in a row: at Litchfield 9 % of well-supported stems under 5 cm
radius passed, and 130° kept 11 % of the plot's trees (8 % at Robson
Creek, 36 % at Wytham, 35 % at Ofental). The arc is now carried across the
empty bins that the point count explains, up to two in a row. With that,
130° keeps 80 %, 56 %, 66 % and 65 % of the reference trees against 98 %,
74 %, 83 % and 71 % without it, at a precision of 0.97, 0.61, 0.90 and
0.73 against 0.98, 0.59, 0.90 and 0.60. What it still removes are stems
that the scans did not see far enough round. It pays at Ofental (F1
0.649 → 0.688) and costs elsewhere (−0.10, −0.07 and −0.10).

**Measured and not adopted.** Giving a seed point that two stems claim to
the nearer bark instead of the better-supported stem (17 to 33 % of seed
points go to a stem other than the nearest): more spurious trees on 14 of
16 runs, F1 −0.029
at Wytham and −0.028 at Ofental. Segmenting a second time from the
candidates that survive pruning: +0.006 overall, 7 runs up and 7 down. The
99th height percentile in place of the top point, and CSF ground in place
of PMF: no difference beyond the run-to-run range.

## The angle penalty and how well the plot is sampled

This section is about the edge cost of versions up to 1.1.0, which the
[present cost](#the-edge-cost-after-version-110) replaced; `angle_rate`
is no longer a setting. What it found, that the best cost depends on how
completely the trees are sampled, holds for the present cost too.

In those versions the cost of an edge rose with its angle from vertical as
`exp(angle_rate × degrees)`. The rate had been 0.046 since the method was
written (a horizontal step costs 63 times a vertical one). Measured in
November 2026 with paired runs, the best rate turns out to depend on how
completely the trees are sampled:

| Data | 0.046 (default) | 0.023 | 0.012 |
|---|---|---|---|
| Litchfield (7 runs) | 0.976 | 0.974 | 0.974 |
| Robson Creek (7 runs) | 0.653 | 0.676 | 0.677 |
| Wytham (7 runs) | 0.852 | 0.860 | 0.862 |
| Ofental (7 runs) | 0.641 | 0.674 | 0.718 |
| Gingin banksia woodland, 291 trees (2 runs) | 0.671 | 0.693 | 0.704 |
| Great Western Woodlands, 16 trees (2 runs) | 0.660 | 0.690 | 0.698 |
| the four reference plots thinned to 5 cm (mean, 3 runs each) | 0.765 | 0.777 | 0.754 |
| the same thinned to 8 cm, graph at 8 cm | 0.612 | 0.608 | 0.566 |
| eight simulated plots, five scan positions each | 0.721 | 0.710 | 0.693 |

F1 with trees under 3 m pruned. The two woodland plots were never used to
choose a setting. The simulated plots are four stand types of
`sylva.synthetic.plot`, two seeds each, scanned with the VZ-400 model.

At their native spacing of about 2.5 cm, every real plot but Litchfield
gains from a lower rate, with fewer spurious trees (77 → 61 at Robson
Creek and 43 → 33 at Ofental at 0.012), the same or a higher overlap of
the trees found, and no more understorey taken. The mechanism is the one the miss
analysis points to: with horizontal steps at 63 times the cost of vertical
ones, the way out along a branch is so dear that a tree standing under it
reaches the outer part more cheaply across a small vertical gap, and takes
it.

The simulated plots and the thinned clouds go the other way, and for a
reason that can be measured. In the real plots 91 to 98 % of a tree's
points are connected to its own base through its own points by links of
5 cm, and nearly all of them by links of 8 cm. In the simulated scans it is 22 to
45 % at 5 cm and 55 to 81 % at 8 cm: seen from five positions, the crowns
are in pieces. A path then has to jump from piece to piece, the length
term can no longer tell a tree's own gap from the gap to its neighbour,
and the preference for what stands above the stem is what is left to hold
a tree together.

So the default was left unchanged, and `angle_rate` made a setting: 0.023 for
plots scanned from many positions and kept at 2 to 5 cm, where it gained
up to 0.03 and lost nothing beyond the run-to-run range; the default for
sparse, coarse or heavily occluded clouds. Raising `power` behaves the same way (10 in place of 6:
+0.012 on the reference plots, +0.02 at Gingin, −0.030 on the simulated
ones) and was left alone for the same reason.

Also measured in this round and not adopted, each as paired runs: a 4 cm
clustering cell in stem detection (+0.011 on the reference plots, −0.08
at Gingin, where it adds a hundred spurious trees); 500 RANSAC iterations
(+0.010, but 250, 1000 and 2000 give +0.003, +0.003 and −0.005: a
different random draw, not a better fit); linking stem circles across
layers where a fit failed (13 more trees found over the four plots, 16 to
25 more spurious);
merging only the stems that join above breast height (joins of coppice
stems and of fragments happen at the same heights); a higher lean limit
alone; a 2 cm graph; and every other graph, seed, prior and pruning
setting, none of which beat the defaults by more than the run-to-run
range. Removing by hand every stem that the reference shows to be a
fragment of another tree raised F1 by 0.02 to 0.05 on the three
closed-canopy plots and recovered no missed tree, which bounds what
better candidate filtering could do.

Two other ways of dividing contested crowns were prototyped and rejected,
both on the reference plots themselves:

- **A capacity from the pipe model.** A stem can carry a crown in
  proportion to its cross-section, so a tree claiming far more than its
  DBH allows was costed up and the plot segmented again. On well-fitted
  stems crown size does go as DBH to the power 1.9 to 2.2 on three plots,
  and the trees that take a neighbour's branches sit at about twice the
  usual ratio; but the scatter of the relation is as large as that, and at
  Robson Creek the power is 0.8 to 1.3. Every strength tried lost F1:
  −0.04 to −0.09 with a fixed DBH² law, −0.06 to no change with the
  plot's own fitted law and a dead zone, and worse the harder it was
  applied.
- **Assignment by random walks.** Each point goes to the stem a random
  walk from it most probably reaches first ([Grady 2006](../references.md)), so a thick
  attachment should outweigh a thin contact. As a replacement for the
  cheapest path it scored 0.87 to 0.93 at Litchfield and 0.12 to 0.15 at
  Ofental: under a closed canopy the branches of neighbours touch at
  every height, and a few metres above the stem bases the probabilities
  blend (the best stem's averaged 0.15 at Ofental). Used only to
  re-decide a band around the cheapest-path boundaries it did less harm
  the narrower the band (−0.12 to −0.02), and in the narrowest it still
  cost Robson Creek 0.020 in each of four runs.

## The edge cost after version 1.1.0

After version 1.1.0 the edge cost (the angle penalty of the section above)
and the arc of a stem circle (until then counted in ten-degree bins) were
replaced. An edge of length `d` now costs
`d ** power / max(bend_floor, cos) ** bend_power`, the cosine being that of
the angle between the edge and a direction that is the vertical turned by
`path_weight` towards the way the path arrived: the form of raycloudtools'
cost for bending a path ([Lowe and Stepanas 2021](../references.md);
[Devereux et al. 2026](../references.md)), which measures from the path
alone. The arc runs from point to point while
neighbours are within 15° of each other, each point standing for a sector
of 10°.

The defaults are `bend_power=4`, `bend_floor=0.4` and `path_weight=0.3`: a
turn across costs at most 39 times going straight on. F1 with trees under
3 m pruned:

| Data | version 1.1.0 | present default | `bend_power=2, bend_floor=0.3` |
|---|---|---|---|
| Litchfield (4 runs) | 0.977 | 0.977 | 0.969 |
| Robson Creek (4 runs) | 0.653 | 0.665 | 0.679 |
| Wytham (4 runs) | 0.861 | 0.856 | 0.859 |
| Ofental (4 runs) | 0.649 | 0.671 | 0.698 |
| Gingin banksia woodland (2 runs) | 0.671 | 0.674 | 0.685 |
| Great Western Woodlands (2 runs) | 0.660 | 0.652 | 0.690 |
| the four reference plots thinned to 5 cm (mean, 3 runs each) | 0.765 | 0.775 | 0.778 |
| the same thinned to 8 cm, graph at 8 cm | 0.612 | 0.609 | 0.600 |
| eight simulated plots, five scan positions each | 0.721 | 0.719 | 0.706 |

Over seven paired runs of each reference plot the present default differs
from version 1.1.0 by +0.008 on average (+0.001 at Litchfield, +0.010 at
Robson Creek, +0.001 at Wytham and +0.021 at Ofental), with three more trees
found and six fewer spurious trees a run; on the 5 cm thinnings by +0.010,
on the 8 cm thinnings by −0.002 and on the simulated plots by −0.002. The
understorey taken is unchanged. A floor of 0.355, which reproduces the 63-fold
penalty of version 1.1.0, changes nothing anywhere; 0.45 gains as much on the
reference plots and loses 0.009 on the simulated ones.

The gentler cost of the last column (a turn costs at most 11 times) does
what the lower `angle_rate` did, for the same reason: it gains where trees
are sampled completely (+0.02 on the reference plots, with 18 fewer
spurious trees a run, and on both woodland plots) and loses where crowns
are in pieces (−0.017 on the simulated plots, −0.005 at 8 cm). It is a
setting, not the default.

Following the path alone, as raycloudtools does (`path_weight=1` with no
floor), was measured with a prototype before the defaults were fixed:
+0.023 on the reference plots and +0.05 at Gingin, −0.09 on the thinned
clouds and −0.06 on the simulated plots. Where a tree is in pieces, the
direction of a path that has just jumped a gap says little, and the
vertical is what is left to hold the tree together.

A first version also took the covered share of the circle from the gaps
between points. It halved the stem candidates on the thinned clouds (F1
−0.05 at 5 cm and −0.22 at 8 cm) while leaving the native clouds where
they were: a stem slice of a dozen points has no two neighbours within
15°. The covered share is the detector's own measure, the share of
ten-degree bins that hold a point, and is unchanged.

## Merging trees that touch

`trees.merge_touching` runs after segmentation. Detection favours recall,
and a candidate that is really part of another tree (a second circle on a
wide trunk, a low fork, a whorl of branches fitted as a stem) is grown like
any other stem and comes away with a piece of its parent's crown. The oracle
test above put the ceiling of removing such stems at 0.02 to 0.05 F1; what
was missing was a way to know them without the reference. Their piece of
crown gives them away: it lies against the parent along much of its extent,
where two separate trees touch only at the edge of their crowns. The step
thins the labelled points above 0.5 m to 5 cm, takes for each pair of trees
the share of the smaller one's points that have a point of the larger
within 10 cm, and merges the smaller into the larger while that share
reaches a threshold, largest share first. Only trees that reach 3 m take
part.

Each figure is F1 with trees under 3 m pruned, as paired runs against the
same pipeline without the step:

| Data (runs) | without | threshold 0.10 | 0.08 | 0.06 |
|---|---|---|---|---|
| Litchfield (7) | 0.976 | 0.977 | 0.977 | 0.977 |
| Robson Creek (7) | 0.653 | 0.654 | 0.657 | 0.662 |
| Wytham (7) | 0.852 | 0.859 | 0.860 | 0.862 |
| Ofental (7) | 0.641 | 0.648 | 0.659 | 0.688 |
| Gingin banksia woodland, 291 trees (2) | 0.671 | 0.691 | 0.704 | 0.722 |
| Great Western Woodlands, 16 trees (2) | 0.660 | 0.675 | 0.682 | 0.682 |
| the four reference plots thinned to 5 cm (mean, 3 each) | 0.765 | 0.768 | 0.769 | 0.773 |
| eight simulated plots (mean) | 0.721 | 0.725 | 0.727 | 0.732 |
| runs below the pipeline without it, of 52 | | none | 1 (one tree lost) | 1 (two trees lost) |

At 0.10, the default, no run on any data set is below the same run
without the step, and no reference tree is lost in any of them: the gain
is spurious trees removed, 77.3 → 76.6 at Robson Creek, 18.9 → 16.1 at
Wytham, 42.7 → 40.7 at Ofental and 194 → 174 at Gingin, by merging two to
seven trees a plot. The trees found and the share of the understorey that
trees take are unchanged in every one of the 28 reference runs, and the
overlap of the found trees with the reference is the same or higher. The two woodland plots were never used to choose a
setting. The step takes one to three seconds on these plots.

Lower thresholds remove more (at 0.06, 12 of Ofental's 43 spurious trees
and 46 of Gingin's 194) and begin to cost real trees where the sampling is
coarse: on the Litchfield cloud thinned to 5 cm one run of three loses a
tree at 0.08 and two at 0.06, one of them a tree whose stem had been
followed through ten layers, merged into a neighbour 1.6 m away with which
its crown interlocks. On clouds at 2 to 3 cm, 0.06 lost no tree in any run
and is worth trying where spurious trees matter more than the last tree.

Two things were learned on the way. Without the 3 m rule the step merged
low clumps into the trees they stood against, which the tree scores did
not see and the understorey figure did (+0.01 to +0.07 of the points below
2 m on no reference tree); both trees of a pair must reach the height for
it to stay where it was. And unlike `merge_branches`, which joins the
stems of a coppice stool, this step left them apart: no Wytham tree was
lost at any of the three thresholds at the plot's own density.

With the edge cost of version 1.1.0, the step added to what `angle_rate`
gained (above), since the two remove different spurious trees. On the same four runs of each reference plot and
two of each woodland plot:

| Plot | neither | `merge_touching` 0.10 | `angle_rate` 0.023 | both |
|---|---|---|---|---|
| Litchfield | 0.977 | 0.978 | 0.971 | 0.972 |
| Robson Creek | 0.653 | 0.655 | 0.674 | 0.675 |
| Wytham | 0.861 | 0.866 | 0.865 | 0.872 |
| Ofental | 0.649 | 0.656 | 0.673 | 0.686 |
| Gingin | 0.671 | 0.691 | 0.693 | 0.711 |
| Great Western Woodlands | 0.660 | 0.675 | 0.690 | 0.698 |

What `angle_rate` costs at Litchfield and on sparse clouds it costs with
or without the step.

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
   the thinning's. (The division of contested crowns still depends on the
   grid; see *One run is not a measurement*.) On a synthetic plot of interlocking crowns that took the
   share of tree points left unlabelled from 31 % to under 1 %.
2. **Stems are seeded along the whole bark shell** from the ground to half a
   metre above `seed_height` (1.5 m by default, so to 2 m), and the ground cut dropped from 0.25 m to 0.1 m
   (0 with a ground classification), so a trunk is labelled to within a hand
   of the ground. Neither changed a single tree's outcome once the grid was
   held still; what the earlier cut was protecting against turned out to be
   the grid artefact above.
3. **The understorey sources are off by default.** Costed as a 10 m tree, or
   by the vegetation height around each source, or by the height of the
   column above it up to the first gap, they found no tree on any plot and
   lost two at Litchfield and four at Wytham. What they take from a crown
   they take through foliage from below, whatever they are costed. (That
   judged them on trees alone; what they keep out of the trees is measured
   [above](#what-the-scores-cannot-see).)

| Plot | before (F1 at 3 m, two grid corners) | after | after, understorey sources on |
|---|---|---|---|
| Litchfield | 0.924 / 0.877 | 0.964 | 0.956 |
| Robson Creek | 0.562 / 0.589 | 0.591 | 0.589 |
| Wytham | 0.848 / 0.856 | 0.864 | 0.850 |
| Ofental | 0.611 / 0.597 | 0.594 | 0.597 |

These are single runs with the 5 cm graph. The *after* column was reported
at the time as identical at three grid corners; those corners were whole
voxels apart and so one grid (see *One run is not a measurement*). Any
comparison finer than about 0.04 on a single plot, here and in the tuning
notes further down, is within the run-to-run range.

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
areas: `gravity`, `wood_costs` (wood and leaf factors on the edge cost, an option
up to version 1.1.0; better at Ofental only), a weaker or no
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
