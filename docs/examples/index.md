# Example notebooks

One notebook per stage of a TLS workflow, three more on change between
epochs, airborne lidar and full waveforms, one on the synthetic data the
others check themselves against, and one on buttressed trees. Most of them run on a real scan: a
20 × 20 m tile of the TERN [Litchfield Savanna
SuperSite](https://www.tern.org.au) plot in the Northern Territory, scanned in
2021 with a RIEGL VZ-2000i from many positions and registered into one cloud.

| File | What | Size |
|---|---|---|
| `data/litch_tile.laz` | 1.55 M echoes inside the tile, thinned to 5 cm, with `classification` (2 ground, 4 vegetation) and `intensity` | 6 MB |
| `data/litch_tile_shots.parquet` | 626 k pulses, every 40th ray clipped to the tile, misses included | 6 MB |

`make_litch_subset.py` cuts both from the plot's ray cloud. Rays are clipped at
the tile boundary, so a ray that ends inside keeps its echo and a ray that
passes through becomes a pulse with no return. That preserves free space inside
the tile. One scan position falls in the tile's corner and keeps its true
origin, but the rest of the pulses start where they crossed the boundary, so
the tile carries everything point-based while `sylva.synthetic` is kept for the
cases where the answer has to be known: registration with a known transform,
a QSM and leaf area against a tree's own cylinders and leaves, and leaf area
density, leaf angles and gap fraction against a known stand. These scenes
come from `synthetic.tree_model` and `synthetic.plot`, scanned by
`synthetic.scan` with a beam footprint, range noise and mixed pixels
(notebook 17).

The buttress notebook (18) uses two other real clouds instead: single trees from
the destructive-harvest data of Burt et al. (2021), cut by
`make_buttress_subset.py` (see the [data README](data/README.md)).

| File | What | Size |
|---|---|---|
| `data/buttress_tree.laz` | the whole of a 46 m tree with a flanged base: 1.04 M points, 1 cm up to 6 m and 5 cm above | 3 MB |
| `data/round_tree.laz` | the lowest 8 m of a tree with a round stem, 41 k points at 1 cm | 0.1 MB |

What real data shows that a synthetic scene cannot is worth reading. Two
examples the notebooks work through: the cloth simulation filter puts the
terrain metres too high in 8 % of the tile's cells, all of them within about
2 m of the cut edge, because a cloth needs points on both sides to be pulled
down -- so classify ground on the whole plot and crop afterwards; and the
statistical outlier filter removes 4 % of the cloud, two thirds of it below
1 m, because a sparse grass layer looks like noise by that test.

| | Notebook | Data | Covers |
|---|---|---|---|
| 1 | [Point clouds and I/O](01_pointclouds_io.ipynb) | tile | `PointCloud`, attributes, indexing, LAZ / PLY / text |
| 2 | [Filtering](02_filtering.ipynb) | tile | subsampling, crops, outlier removal, planarity / linearity, clustering |
| 3 | [Registration](03_registration.ipynb) | synthetic | two scans of a synthetic plot: Kabsch from targets, point-to-point and point-to-plane ICP, trimming on partial overlap, merging |
| 4 | [Ground and height](04_ground.ipynb) | tile | CSF and PMF ground filters, DTM, height normalisation, CHM |
| 5 | [Trees](05_trees.ipynb) | tile | stem detection and DBH, segmentation, heights, crowns, taper |
| 6 | [QSMs](06_qsm.ipynb) | both | leaf / wood separation, cylinder models, volumes, mesh export |
| 7 | [Canopy structure](07_canopy.ipynb) | both | voxel occupancy, contact-frequency PAD, gap fraction and effective PAI |
| 8 | [Pulses and shots files](08_shots.ipynb) | tile | `Shots`, misses, conversions, the Parquet shots format |
| 9 | [Ray-traced voxels](09_voxels.ipynb) | both | attenuation, PAD / LAD / WAD, leaf angles, wood volume, `.vox`, streaming |
| 10 | [A RIEGL project, end to end](10_riscan_pipeline.ipynb) | full plot | every stage on one hectare straight from the `.rxp` files: read, ground, trees, scan quality, QSMs and leaves, gap profile, voxels, sampling |
| 11 | [Coordinates](11_coordinates.ipynb) | tile | CRS in LAS headers, reprojection and its warnings, translate / rotate / recentre, `apply_transforms` on scans |
| 12 | [Interpolation](12_interpolation.ipynb) | tile | labels from a thinned copy, TIN / natural-neighbour / IDW DTMs, rasters sampled onto points |
| 13 | [Masking](13_masking.ipynb) | tile | polygons from GeoJSON, raster masks, attribute expressions, change between two clouds |
| 14 | [Change detection](14_change.ipynb) | synthetic | two epochs with known changes: alignment on stems and ground, tree matching, DBH increments and their detection limits, plot summary, C2C and M3C2, CHM differences, voxel occupancy, QSM change |
| 15 | [Airborne lidar](15_als.ipynb) | synthetic | a simulated flight written as tiles: catalogue, ground, DTM, CHM, area-based metrics, individual trees against the known stand, pulses from the trajectory, gap profiles and ray-traced PAI against a known layer |
| 16 | [Full waveforms](16_waveform.ipynb) | synthetic | waveforms of known targets, LAS wave packets written and read, Gaussian decomposition against the truth, echoes as pulses for ray-traced voxels |
| 17 | [Synthetic data](17_synthetic.ipynb) | synthetic | tree archetypes and their truth (cylinders, leaves, leaf angles), a mixed stand with understorey and dead wood, a finite-beam scan and where its mixed pixels fall |
| 18 | [Buttresses](18_buttress.ipynb) | two real trees | 3D views of the points, detecting a flanged base, rebuilding it as a closed mesh, the cylinder model it replaces, one fused mesh of the whole tree |

The numbers the notebooks print describe this one tile and are not validation.
Accuracy against reference plots, felled trees and independent instruments is
on the [benchmark pages](../benchmarks/trees.md).

To run them yourself:

```bash
pip install -e '.[examples]'          # matplotlib and jupyter
jupyter lab docs/examples
```

They are generated from `docs/examples/build_notebooks.py`; edit that file and
run it to regenerate and re-execute them (`python docs/examples/build_notebooks.py
05_trees` for one).

The figures share one style, `sylva.mplstyle` in the same folder, which the
first cell of each notebook loads: fixed figure widths and font sizes, the
colour-blind-safe Okabe-Ito palette for categories, viridis for magnitudes,
and a diverging map centred on zero for signed differences. Ground, wood,
leaves and grass keep the same colours in every notebook. Copy the file
alongside a notebook to run it elsewhere.

The tile is derived from TERN data, distributed here for documentation and
teaching. Cite TERN if you use it for anything else, and see
`data/README.md`.

Notebook 10 is different: it reads a whole RiSCAN PRO project, the TERN
Litchfield core hectare (64 VZ-2000i positions, 35 GB), so it needs RiVLib,
the project on disk and about 40 GB of memory, and is run by hand with
`build_riscan_notebook.py` rather than by `build_notebooks.py`. Point
`PROJECT` at your own project and `PLOT` at its extent to run it elsewhere.
