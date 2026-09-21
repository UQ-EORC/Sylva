<p align="center"><img src="https://raw.githubusercontent.com/tim-devereux/Sylva/main/docs/assets/icon.png" width="128" alt="Sylva"></p>

# Sylva

Terrestrial laser scanning (TLS) processing for forest ecology and remote
sensing. A Rust core (the `sylva-rs` crate) does the work; Python gets a
numpy-friendly API (`import sylva`) and a `sylva` command.

| Module | What it does |
|---|---|
| `sylva.io` | LAS/LAZ (typed extra bytes), PLY (incl. raycloudtools ray clouds), XYZ/CSV/PTS, RIEGL `.rxp` via RiVLib, RiSCAN project parsing |
| `sylva.filters` | voxel / random / Poisson-disk subsampling, box & cylinder crops, statistical & radius outlier removal, PCA normals, planarity, Euclidean clustering, kNN |
| `sylva.ground` | Cloth Simulation Filter and Progressive Morphological Filter ground classification, DTM, height normalisation, CHM |
| `sylva.trees` | RANSAC circle fitting, stem detection & DBH, taper profiles, graph-based tree segmentation, tree heights, crown metrics, crown shape (stacked-hull volume, asymmetry) |
| `sylva.canopy` | voxel grids, contact-frequency PAD profiles, zenith-ring gap fraction, hinge/Miller LAI, ray-traced density grids from pulse data, Jupp gap-probability profiles with hinge / linear / weighted PAI, clumping index and canopy height |
| `sylva.voxels` | AMAPVox-style ray-traced voxels (port of raycloudtools `rayvoxel`): echo-weighted free / potential path lengths, FPL / PPL / transmittance / Bailey attenuation, analytic or estimated leaf-angle `G`, PAD / LAD / WAD, occlusion, sub-voxel exploration, QSM wood volume, `.vox` export, occlusion profiles and per-tree sampling (is the top real?) |
| `sylva.registration` | Kabsch, point-to-point / point-to-plane (trimmed) ICP, scan merging |
| `sylva.qsm` | cylinder fitting, geodesic skeletonisation, cylinder QSMs with volumes and branch orders, `_trees.txt` export, tree metrics (branch table, taper, lean, sweep, crown, share of the model fitted to points) |
| `sylva.leaves` | leaf / wood labels, leaf angle distribution, leaf area density from points or voxels, leaf meshes placed on a QSM |
| `sylva.Shots` | pulse-centric data (origin, direction, CSR echoes) for ray-based metrics, with a compact Parquet file format that stores misses without far points and streams into the voxeliser |

## Install

Requires Python ≥ 3.10 and a Rust toolchain.

```bash
pip install maturin
maturin develop --release        # builds the extension into the active environment
```

## A first look

```python
import sylva
from sylva import ground, trees, voxels

cloud = ground.classify_ground_csf(sylva.read("plot.laz"))
cloud = ground.normalize_height(cloud, ground.make_dtm(cloud, resolution=0.5))
stems = trees.detect_stems(cloud)                        # positions and DBH
labels = trees.segment_trees(cloud, stems)               # tree_id per point

shots = sylva.Shots.load("plot.parquet")                 # pulses, misses included
grid = voxels.ray_voxelize(shots, 0.25, ground_class=2)  # ray-traced plant area density
```

```bash
sylva ground plot.laz plot_norm.laz --dtm dtm.asc
sylva trees plot_norm.laz -o trees.csv --segment plot_trees.laz
sylva voxel plot.parquet plot.vox --voxel 0.25 --ground-class 2
```

## Documentation

The documentation lives in [`docs/`](https://github.com/tim-devereux/Sylva/blob/main/docs/index.md):

- Guides: [plot workflow](https://github.com/tim-devereux/Sylva/blob/main/docs/guide/quickstart.md), [pulse data and shots files](https://github.com/tim-devereux/Sylva/blob/main/docs/guide/pulses.md),
  [ray-traced voxels](https://github.com/tim-devereux/Sylva/blob/main/docs/guide/voxels.md), [QSMs](https://github.com/tim-devereux/Sylva/blob/main/docs/guide/qsm.md), [command line](https://github.com/tim-devereux/Sylva/blob/main/docs/guide/cli.md)
- [Example notebooks](https://github.com/tim-devereux/Sylva/blob/main/docs/examples/index.md), one per stage, on synthetic data
- Benchmarks: [tree detection](https://github.com/tim-devereux/Sylva/blob/main/docs/benchmarks/trees.md), [QSMs against felled trees](https://github.com/tim-devereux/Sylva/blob/main/docs/benchmarks/qsm.md)
- API reference (generated from the docstrings), [development notes](https://github.com/tim-devereux/Sylva/blob/main/docs/development.md),
  [references](https://github.com/tim-devereux/Sylva/blob/main/docs/references.md)

Build the site with `pip install -e '.[docs]' && mkdocs serve`.

## Licence

MIT
