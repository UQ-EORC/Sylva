# Sylva

Terrestrial laser scanning processing for forest ecology and remote
sensing. A Rust core (the `sylva-rs` crate) holds every computation and file
format; Python gets a numpy-friendly API (`import sylva`) and a `sylva`
command over it. An R package on the same core, with the same functions, is
kept on the `r-package` branch.

| Module | What it does |
|---|---|
| `sylva.io` | LAS/LAZ (typed extra bytes, CRS from and to the WKT or GeoTIFF records), PLY (incl. raycloudtools ray clouds), XYZ/CSV/PTS, RIEGL `.rxp` via RiVLib, RiSCAN project parsing |
| `sylva.coords` | translation, rotation about any axis and point, recentring, reprojection between CRSs (EPSG codes, PROJ strings, WKT; pure Rust, no PROJ install), applying transform files (`transforms.json`, RiSCAN SOPs) to many scans |
| `sylva.filters` | voxel / random / Poisson-disk subsampling, box & cylinder crops, statistical & radius outlier removal, PCA normals, planarity, Euclidean clustering, kNN |
| `sylva.ground` | Cloth Simulation Filter and Progressive Morphological Filter ground classification, DTM (lowest point, or TIN, natural-neighbour or IDW), height normalisation, CHM |
| `sylva.interpolate` | attributes carried between clouds (nearest, inverse distance, majority vote), grids from points (IDW, TIN, natural neighbour), rasters sampled onto points |
| `sylva.masks` | point masks and crops by polygons (shapefile, GeoJSON, with holes), rasters, attribute expressions (`cloud.where("height > 2 & classification != 2")`) and distance to another cloud |
| `sylva.trees` | RANSAC circle fitting, stem detection & DBH, basal area, taper profiles, graph-based tree segmentation, tree heights, crown metrics, crown shape (stacked-hull volume, asymmetry) |
| `sylva.canopy` | voxel grids, contact-frequency PAD profiles, zenith-ring gap fraction, hinge/Miller LAI, ray-traced density grids from pulse data, Jupp gap-probability profiles with hinge / linear / weighted PAI, clumping index and canopy height |
| `sylva.voxels` | AMAPVox-style ray-traced voxels (port of raycloudtools `rayvoxel`): echo-weighted free / potential path lengths, FPL / PPL / transmittance / Bailey attenuation, analytic or estimated leaf-angle `G`, PAD / LAD / WAD, occlusion, sub-voxel exploration, QSM wood volume, `.vox` export, occlusion profiles and per-tree sampling (is the top real?) |
| `sylva.registration` | Kabsch, point-to-point / point-to-plane (trimmed) ICP, scan merging |
| `sylva.coreg` | marker-free coregistration of scan positions from the trees (a port of tlsalign): terrain model, stem maps, global stem-map and reflector matching, robust point-to-plane ICP, pose graph with outlier rejection, recovery of stragglers, joint multi-view refinement, stem agreement report; tilted scans levelled with the scanner attitude; scanner `.PROJ` projects |
| `sylva.qsm` | cylinder fitting, geodesic skeletonisation, cylinder QSMs with volumes and branch orders, a QSM for every tree of a plot anchored on its measured DBH, buttress meshes, `_trees.txt` export, tree metrics (branch table, taper, lean, sweep, crown, share of the model fitted to points) |
| `sylva.leaves` | leaf / wood labels, leaf angle distribution, leaf area density from points or voxels, leaf meshes placed on a QSM |
| `sylva.quality` | scan quality from stems: range noise with the stem shape removed, per-scan registration offsets, mixed-pixel tails |
| `sylva.Shots` | pulse-centric data (origin, direction, CSR echoes) for ray-based metrics, with a compact Parquet file format that stores misses without far points and streams into the voxeliser; pulses that returned nothing rebuilt from the scan pattern or from the returns alone |
| `sylva.synthetic` | synthetic trees, plots and scans with known answers, for examples and tests |

## Install

A conda environment keeps the Python, the Rust toolchain and the C linker
together, which is the easiest way to build the extension:

```bash
conda create -n sylva -c conda-forge python=3.12 rust maturin c-compiler
conda activate sylva
maturin develop --release        # builds the Rust core into the environment
```

`c-compiler` matters: `rustc` links through a program called `cc`, which
conda's `rust` package does not provide, so without it the build stops at
``error: linker `cc` not found``. The repository's `environment.yml` has the
same list plus the test and notebook extras (`conda env create -f
environment.yml`).

Without conda, any Python >= 3.10 with a Rust toolchain (`rustup`), a system
C compiler (`gcc`, `build-essential`, or Xcode's command line tools) and
`pip install maturin` does the same job.

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

```python
from sylva import coords, interpolate, masks

cloud = coords.reproject(cloud, "EPSG:7855")                  # to GDA2020 / MGA zone 55
trees_only = cloud.where("height > 1.3 & tree_id >= 0")       # attribute expression
plot = masks.crop_polygons(cloud, masks.read_polygons("plot.shp"))
full = interpolate.transfer_attributes(thinned, full, ["tree_id"], method="majority")
```

```bash
sylva ground plot.laz plot_norm.laz --dtm dtm.asc
sylva trees plot_norm.laz -o trees.csv --segment plot_trees.laz
sylva coreg survey.PROJ -o survey_coreg/ --merged survey.laz
sylva voxel plot.parquet plot.vox --voxel 0.25 --ground-class 2
sylva reproject plot.laz plot_mga55.laz --to EPSG:7855
sylva transform scan.laz scan_project.laz --matrix ScanPos001.DAT
```

## Documentation

The documentation lives in [`docs/`](https://github.com/UQ-EORC/Sylva/blob/main/docs/index.md):

- Guides: [plot workflow](https://github.com/UQ-EORC/Sylva/blob/main/docs/guide/quickstart.md), [pulse data and shots files](https://github.com/UQ-EORC/Sylva/blob/main/docs/guide/pulses.md),
  [ray-traced voxels](https://github.com/UQ-EORC/Sylva/blob/main/docs/guide/voxels.md), [QSMs](https://github.com/UQ-EORC/Sylva/blob/main/docs/guide/qsm.md),
  [coordinates](https://github.com/UQ-EORC/Sylva/blob/main/docs/guide/coordinates.md), [interpolation](https://github.com/UQ-EORC/Sylva/blob/main/docs/guide/interpolation.md),
  [masking](https://github.com/UQ-EORC/Sylva/blob/main/docs/guide/masking.md), [command line](https://github.com/UQ-EORC/Sylva/blob/main/docs/guide/cli.md)
- [Example notebooks](https://github.com/UQ-EORC/Sylva/blob/main/docs/examples/index.md), one per stage, on a tile of the TERN Litchfield plot and on synthetic data
- [Design notes](https://github.com/UQ-EORC/Sylva/blob/main/docs/dev/rust-core.md): the Rust core, the Python layer over it, and how changes are checked
- Benchmarks: [tree detection](https://github.com/UQ-EORC/Sylva/blob/main/docs/benchmarks/trees.md), [QSMs against felled trees](https://github.com/UQ-EORC/Sylva/blob/main/docs/benchmarks/qsm.md)
- API reference (generated from the docstrings), [development notes](https://github.com/UQ-EORC/Sylva/blob/main/docs/development.md),
  [references](https://github.com/UQ-EORC/Sylva/blob/main/docs/references.md)

Build the site with `pip install -e '.[docs]' && mkdocs serve`.

## Author

Tim Devereux, The University of Queensland.

## Licence

GNU General Public License v3.0 or later; see [LICENSE](LICENSE). Anything
distributed that builds on Sylva carries the same licence.
