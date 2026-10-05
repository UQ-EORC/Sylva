# Sylva

NOTE: This software is still in ALPHA development.

[![PyPI](https://img.shields.io/pypi/v/sylva-rs)](https://pypi.org/project/sylva-rs/)
[![crates.io](https://img.shields.io/crates/v/sylva-rs)](https://crates.io/crates/sylva-rs)
[![Documentation](https://img.shields.io/badge/docs-uq--eorc.github.io%2FSylva-blue)](https://uq-eorc.github.io/Sylva/)

Terrestrial and airborne laser scanning processing for forest ecology and
remote sensing. A Rust core (the `sylva-rs` crate) holds every computation and file
format; Python gets a numpy-friendly API (`import sylva`) and a `sylva`
command over it.

| Module | What it does |
|---|---|
| `sylva.io` | LAS/LAZ (typed extra bytes, CRS from and to the WKT or GeoTIFF records), PLY (incl. raycloudtools ray clouds), XYZ/CSV/PTS, RIEGL `.rxp` via RiVLib, RiSCAN project parsing |
| `sylva.geo.coords` | translation, rotation about any axis and point, recentring, reprojection between CRSs (EPSG codes, PROJ strings, WKT; pure Rust, no PROJ install), applying transform files (`transforms.json`, RiSCAN SOPs) to many scans |
| `sylva.filters` | voxel / random / Poisson-disk subsampling, box & cylinder crops, statistical & radius outlier removal, PCA normals, planarity, Euclidean clustering, kNN |
| `sylva.ground` | Cloth Simulation Filter and Progressive Morphological Filter ground classification, DTM (lowest point, or TIN, natural-neighbour or IDW), height normalisation, CHM |
| `sylva.geo.interpolate` | attributes carried between clouds (nearest, inverse distance, majority vote), grids from points (IDW, TIN, natural neighbour), rasters sampled onto points |
| `sylva.geo.masks` | point masks and crops by polygons (shapefile, GeoJSON, with holes), rasters, attribute expressions (`cloud.where("height > 2 & classification != 2")`) and distance to another cloud |
| `sylva.trees` | RANSAC circle fitting, stem detection & DBH, basal area, taper profiles, graph-based tree segmentation, tree heights, crown metrics, crown shape (stacked-hull volume, asymmetry) |
| `sylva.canopy` | voxel grids, contact-frequency PAD profiles, zenith-ring gap fraction, hinge/Miller LAI, ray-traced density grids from pulse data, Jupp gap-probability profiles with hinge / linear / weighted PAI, clumping index and canopy height |
| `sylva.voxels` | AMAPVox-style ray-traced voxels (port of raycloudtools `rayvoxel`): echo-weighted free / potential path lengths, FPL / PPL / transmittance / Bailey attenuation, analytic or estimated leaf-angle `G`, PAD / LAD / WAD, occlusion, sub-voxel exploration, QSM wood volume, `.vox` export, occlusion profiles and per-tree sampling (is the top real?) |
| `sylva.registration` | Kabsch, point-to-point / point-to-plane (trimmed) ICP, scan merging |
| `sylva.coreg` | marker-free coregistration of scan positions from the trees: terrain model, stem maps, global stem-map and reflector matching, robust point-to-plane ICP, pose graph with outlier rejection, recovery of stragglers, joint multi-view refinement, stem agreement report; tilted scans levelled with the scanner attitude; scanner `.PROJ` projects |
| `sylva.qsm` | cylinder fitting, geodesic skeletonisation, cylinder QSMs with volumes and branch orders, a QSM for every tree of a plot anchored on its measured DBH, buttress meshes, `_trees.txt` export, tree metrics (branch table, taper, lean, sweep, crown, share of the model fitted to points) |
| `sylva.leaves` | leaf / wood labels, leaf angle distribution, leaf area density from points or voxels, leaf meshes placed on a QSM |
| `sylva.quality` | scan quality from stems: range noise with the stem shape removed, per-scan registration offsets, mixed-pixel tails |
| `sylva.change` | change between two epochs of a plot: epoch alignment on stems and ground, tree matching with increments and their uncertainty, plot summaries (growth, mortality, recruitment), point change (C2C, M3C2, DEM of difference, voxel occupancy with occlusion), QSM change by height and branch, each labelled trusted or not |
| `sylva.als` | airborne lidar over tiled areas: catalogues and buffered chunks, ground, DTM, CHM, normalisation, filtering, retiling and thinning over whole areas |
| `sylva.als.metrics` | area-based metrics (the lidR standard set, cover, gap fraction) as rasters or plot tables, or any user function |
| `sylva.als.trees` | tree tops from local maxima, crowns by watershed, Dalponte 2016 or Li 2012, crown outlines and labelled tiles, each tree once across tiles |
| `sylva.als.canopy` | ALS and UAV pulses from the flight trajectory (SBET or text, or estimated), gap-fraction and PAD profiles corrected for beam angle, ray-traced voxels |
| `sylva.fusion` | TLS and ALS together: a plot registered on a survey (canopy and terrain search, robust ICP, residuals and uncertainty), TLS stems linked to ALS trees with the trees under each crown, merged clouds and plant area profiles weighted by sampling, plot values upscaled by regression with leave-one-out checks |
| `sylva.waveform` | full waveforms: LAS 1.3/1.4 wave packets and PulseWaves read and written, Gaussian decomposition into echoes, waveforms to pulses |
| `sylva.Shots` | pulse-centric data (origin, direction, CSR echoes) for ray-based metrics, with a compact Parquet file format that stores misses without far points and streams into the voxeliser; pulses that returned nothing rebuilt from the scan pattern or from the returns alone |
| `sylva.synthetic` | synthetic trees, plots, scans, repeat surveys, airborne flights and waveforms with known answers, for examples and tests |

## Contributing

Feedback, issues, and PRs all welcome. For issues please use [GitHub issues](https://github.com/UQ-EORC/Sylva/issues) (not a personal message) so the community can benefit.

## Install

Requires Python 3.10 or later. The package is on PyPI:
[pypi.org/project/sylva-rs](https://pypi.org/project/sylva-rs/).

```bash
pip install sylva-rs
```

The wheels carry the compiled core for Linux (x86-64 and arm64), macOS (Intel
and Apple silicon) and Windows. The distribution is called `sylva-rs` because
an unrelated project owns the name `sylva` on PyPI; the package you import is
`sylva`, and the two cannot be installed in the same environment.

Using a dedicated environment:

```bash
conda create -n sylva python=3.12
conda activate sylva
pip install sylva-rs
```

Optional extras: `pip install "sylva-rs[examples]"` for the notebooks. Reading RIEGL `.rxp`
files needs RIEGL's RiVLib, which is proprietary and loaded at run time; see
the [install guide](https://uq-eorc.github.io/Sylva/install/).

### Rust

The core is on crates.io as [sylva-rs](https://crates.io/crates/sylva-rs), and
the `sylva` command without Python as
[sylva-cli](https://crates.io/crates/sylva-cli):

```bash
cargo add sylva-rs               # the library: use sylva_rs::...
cargo install sylva-cli          # the sylva command
```

### From source

For development, or to build the extension yourself. A conda environment keeps
the Python, the Rust toolchain and the C linker together:

```bash
git clone https://github.com/UQ-EORC/Sylva.git
cd Sylva
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
from sylva.geo import coords, interpolate, masks

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
sylva als-ground tiles/ ground/ --method csf                  # airborne tiles
sylva als-metrics ground/ metrics/ --resolution 20 --metrics zq95,zmean,cover
sylva als-trees ground/ trees.csv --crowns crowns.geojson
```

```python
from sylva import als, change

cat = als.catalog("ground/")                                  # ALS tiles
metrics = als.grid_metrics(cat, 20.0)                          # rasters, one per metric
found = als.find_trees(cat, method="dalponte2016")             # tops and crowns

m = change.match_trees(stems_2019, stems_2024)                 # two epochs of a plot
inc = change.tree_increments(m, cloud_2019, cloud_2024, labels_2019, labels_2024)
```

## Documentation

The documentation is at [uq-eorc.github.io/Sylva](https://uq-eorc.github.io/Sylva/); its sources are in [`docs/`](https://github.com/UQ-EORC/Sylva/blob/main/docs/index.md):

- Guides: [plot workflow](https://github.com/UQ-EORC/Sylva/blob/main/docs/guide/quickstart.md), [pulse data and shots files](https://github.com/UQ-EORC/Sylva/blob/main/docs/guide/pulses.md),
  [ray-traced voxels](https://github.com/UQ-EORC/Sylva/blob/main/docs/guide/voxels.md), [QSMs](https://github.com/UQ-EORC/Sylva/blob/main/docs/guide/qsm.md),
  [coordinates](https://github.com/UQ-EORC/Sylva/blob/main/docs/guide/coordinates.md), [interpolation](https://github.com/UQ-EORC/Sylva/blob/main/docs/guide/interpolation.md),
  [masking](https://github.com/UQ-EORC/Sylva/blob/main/docs/guide/masking.md), [change detection](https://github.com/UQ-EORC/Sylva/blob/main/docs/guide/change.md),
  [command line](https://github.com/UQ-EORC/Sylva/blob/main/docs/guide/cli.md)
- Airborne lidar: [tiles](https://github.com/UQ-EORC/Sylva/blob/main/docs/guide/als.md), [area-based metrics](https://github.com/UQ-EORC/Sylva/blob/main/docs/guide/als_metrics.md),
  [trees](https://github.com/UQ-EORC/Sylva/blob/main/docs/guide/als_trees.md), [canopy structure](https://github.com/UQ-EORC/Sylva/blob/main/docs/guide/als_canopy.md),
  [full waveforms](https://github.com/UQ-EORC/Sylva/blob/main/docs/guide/waveform.md)
- [Example notebooks](https://github.com/UQ-EORC/Sylva/blob/main/docs/examples/index.md), one per stage, on a tile of the TERN Litchfield plot and on synthetic data
- [Design notes](https://github.com/UQ-EORC/Sylva/blob/main/docs/dev/rust-core.md): the Rust core, the Python layer over it, and how changes are checked
- Benchmarks: [tree detection](https://github.com/UQ-EORC/Sylva/blob/main/docs/benchmarks/trees.md), [QSMs against felled trees](https://github.com/UQ-EORC/Sylva/blob/main/docs/benchmarks/qsm.md)
- API reference (generated from the docstrings), [development notes](https://github.com/UQ-EORC/Sylva/blob/main/docs/development.md),
  [references](https://github.com/UQ-EORC/Sylva/blob/main/docs/references.md)

Build the site with `pip install -e '.[docs]' && mkdocs serve`.

## Authors

- Tim Devereux, The University of Queensland

## Citation

If Sylva contributed to your work, please cite it:

> Devereux, T. (2026). *Sylva: terrestrial and airborne laser scanning
> processing for forest ecology* (version 0.2.1) [Computer software].
> The University of Queensland. https://github.com/UQ-EORC/Sylva

```bibtex
@software{sylva,
  author       = {Devereux, Tim},
  title        = {Sylva: terrestrial and airborne laser scanning processing
                  for forest ecology},
  year         = {2026},
  version      = {0.2.1},
  organization = {The University of Queensland},
  url          = {https://github.com/UQ-EORC/Sylva}
}
```

## Licence

GNU General Public License v3.0 or later; see [LICENSE](LICENSE). Anything
distributed that builds on Sylva carries the same licence.
