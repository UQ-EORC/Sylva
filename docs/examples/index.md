# Example notebooks

One notebook per stage of a TLS workflow. Most of them run on a real scan: a
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
the tile, but it also puts the pulse origins on the tile edge instead of at the
scanners, which is why the notebooks use the tile for everything point-based
and fall back to `sylva.synthetic` where the answer has to be known:
registration with a known transform, QSM volume against a known taper, and
leaf area against a known scene.

Where the two disagree is worth reading. Two examples from the notebooks: the
cloth simulation filter leaves the terrain metres too high in 8 % of this
savanna tile's cells, where the cloth hangs on the grass layer, while the
morphological filter handles it; and the statistical outlier filter removes 4 %
of the cloud, most of it real grass, because sparse vegetation looks like noise
by that test.

| | Notebook | Data | Covers |
|---|---|---|---|
| 1 | [Point clouds and I/O](01_pointclouds_io.ipynb) | tile | `PointCloud`, attributes, indexing, LAZ / PLY / text |
| 2 | [Filtering](02_filtering.ipynb) | tile | subsampling, crops, outlier removal, planarity / linearity, clustering |
| 3 | [Registration](03_registration.ipynb) | synthetic | Kabsch from targets, point-to-point and point-to-plane ICP, trimming, merging |
| 4 | [Ground and height](04_ground.ipynb) | tile | CSF and PMF ground filters, DTM, height normalisation, CHM |
| 5 | [Trees](05_trees.ipynb) | tile | stem detection and DBH, segmentation, heights, crowns, taper |
| 6 | [QSMs](06_qsm.ipynb) | both | leaf / wood separation, cylinder models, volumes, mesh export |
| 7 | [Canopy structure](07_canopy.ipynb) | both | voxel occupancy, contact-frequency PAD, gap fraction and effective PAI |
| 8 | [Pulses and shots files](08_shots.ipynb) | tile | `Shots`, misses, conversions, the Parquet shots format |
| 9 | [Ray-traced voxels](09_voxels.ipynb) | both | attenuation, PAD / LAD / WAD, leaf angles, wood volume, `.vox`, streaming |

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

The tile is derived from TERN data, distributed here for documentation and
teaching. Cite TERN if you use it for anything else, and see
`data/README.md`.
