# What is ported and what is original

Sylva is partly a port. Some of its methods are other people's software
translated into Rust, some are published methods implemented from the
papers, and some were written for Sylva. This page says which is which, so
that credit goes where it is due and so that a result can be traced to the
method behind it. The copyright and licence notices for the ported code are
in [THIRD_PARTY_NOTICES.md](https://github.com/UQ-EORC/Sylva/blob/main/THIRD_PARTY_NOTICES.md),
and each ported function names its source in a comment above it.

## Ported from other software

Written with that software's source as the reference: translated into Rust,
restructured and in places modified. These are not new methods; cite the
original.

| From | What in Sylva |
|---|---|
| [lidR](https://github.com/r-lidar/lidR) (J.-R. Roussel and contributors) | `segment_crowns(method="dalponte2016")` and the logic of `als.li2012` (translations of lidR's C++); the rules and defaults of the local maximum filter in `als.locate_trees`; the standard area-based metrics of `als.metrics` (`stdmetrics`: names, formulas, entropy, cumulative height percentages). |
| [AMAPVox](https://github.com/umr-amap/AMAPVox) (UMR AMAP) | `sylva.voxels`: the ray-traced voxel grid follows AMAPVox, whose path-length and attenuation estimators, leaf angle distributions and laser specifications it uses. |
| [raycloudtools](https://github.com/csiro-robotics/raycloudtools) (CSIRO; [Lowe and Stepanas 2021](references.md), [Devereux et al. 2026](references.md)) | `canopy.density_grid` ([Lowe et al. 2021](references.md)); in `trees.segment_trees`, the form of the edge cost (edge length to a power over a power of the cosine of the bend), the lateral term and the per-tree scaling; in `qsm.build_qsm`, the power-mean section radius, radii in proportion to the length carried (a fork sharing its parent's cross-section by length) and the cap on a fork's summed cross-section. |
| [GBSeparation](https://doi.org/10.5281/zenodo.6837613) (Tian and Li) | `leaves.classify_leaf_wood(method="gbs")`. |
| The CSF reference implementation (Zhang, Qi, Wan and Wang) | The cloth update and constants of `ground.classify_ground_csf`. |
| The [Point Cloud Library](https://pointclouds.org) | The parameters and window sequence of `ground.classify_ground_pmf`. |
| [SciPy](https://github.com/scipy/scipy) | Two distance-transform routines used by the coregistration. |

## Following another design, with its own code

- **`sylva.als`** processes a catalogue of tiles in buffered chunks, the
  design of lidR ([Roussel et al. 2020](references.md)). The engine, the
  DTM, CHM, normalisation, retiling and trajectory code are Sylva's.
- **`trees.segment_trees`** and **`trees.merge_branches`** take the
  least-cost-path approach of raycloudtools' `rayextract trees`
  ([Lowe and Stepanas 2021](references.md);
  [Devereux et al. 2026](references.md)), with its per-tree height scaling
  and lateral term as options, and the form of its cost for bending a
  path. Seeding the trunk surface rather than its axis is the idea of
  [Arbor](https://github.com/r-lidar/arbor) (J.-R. Roussel, r-lidar).
- **`qsm.build_qsm`** builds its skeleton from shells of path length
  split into their connected pieces, the level-set skeleton of
  [Verroust and Lazarus (2000)](references.md) and
  [Xu et al. (2007)](references.md), and fits each axis's taper by the
  Theil–Sen estimator ([Theil 1950](references.md); [Sen 1968](references.md)).
  **`qsm.wood_points`** starts from the path-frequency cue of
  [Vicari et al. (2019)](references.md).

## Implemented from published methods

Written from the papers, without another implementation as the source.
The [references](references.md) page lists each method and where it is
used: circle and cylinder fitting, registration (Kabsch, ICP), the
watershed and crown outlines, gap-fraction and plant-area profiles, leaf
angle distributions, waveform decomposition, and the allometric and
interpolation methods among them.

## Ported from the authors' own earlier software

The coregistration of scan positions (terrain model, stem maps, matching,
pose graph), the RiSCAN PRO project reader and the multi-slice stem
detector began as a Python package by the same authors and were translated
into Rust, with tests that hold the two to the same results. The
pulse-centric `Shots` model comes from the authors' earlier canopy code.

## Written for Sylva

- **Stem detection as it now stands** (`trees.detect_stems`): the bark
  prefilter, the linking of circles across layers, and the arc of a circle
  measured from point to point, with its allowance for sparse sampling
  (the share of the circle covered is the earlier detector's).
- **In the segmentation:** measuring an edge's bend from a direction
  between the vertical and the arriving path, joining the graph's
  components where the cloud joins them, seeding from the real points of
  the bark shell, the rule for the ground layer, the understorey sources as
  implemented, and `trees.merge_touching`.
- **In the QSMs:** joining the pieces of a tree's cloud where they come
  within reach of each other, the linking of the skeleton through the
  points' shortest paths, the choice of axes by the points a subtree holds,
  the length a node carries as the measure of each axis's taper, setting
  aside sections off their own axis's taper, measuring the wood filter's
  paths as the length of path a point supports and growing the wood through
  wood-like points, the weighted cylinder fit, the anchoring of every tree
  of a plot on its measured DBH (`qsm.build_plot`), and the detection and
  meshing of buttresses.
- **Pulse data:** the Parquet format for pulses with their misses, and the
  resolution of multiple-time-around echoes in RIEGL scans.
- **Whole plots and surveys:** exact processing of a terrestrial plot in
  tiles, change between epochs, the fusion of terrestrial and airborne
  data, the scan-quality measures and the registration report.
- **`sylva.synthetic`**: the tree, plot and scanner models used for tests
  and checks.
- **The benchmarks** on the documentation's benchmark pages: the paired-run
  protocol and what it measured.

Where a method on this page is used in published work, cite its source as
well as Sylva.
