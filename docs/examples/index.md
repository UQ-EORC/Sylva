# Example notebooks

One notebook per stage of a TLS workflow. They run on small synthetic scenes
from `sylva.synthetic` (terrain, a few trees with known positions, diameters,
heights and leaf area, and a pseudo-scanner), so they need no data and the
right answer is known. Outputs are saved in the notebooks.

| | Notebook | Covers |
|---|---|---|
| 1 | [Point clouds and I/O](01_pointclouds_io.ipynb) | `PointCloud`, attributes, indexing, LAZ / PLY / text |
| 2 | [Filtering](02_filtering.ipynb) | subsampling, crops, outlier removal, planarity / linearity, clustering |
| 3 | [Registration](03_registration.ipynb) | Kabsch from targets, point-to-point and point-to-plane ICP, trimming, merging |
| 4 | [Ground and height](04_ground.ipynb) | CSF and PMF ground filters, DTM, height normalisation, CHM |
| 5 | [Trees](05_trees.ipynb) | stem detection and DBH, segmentation, heights, crowns, taper |
| 6 | [QSMs](06_qsm.ipynb) | leaf / wood separation, cylinder models, volumes, mesh export |
| 7 | [Canopy structure](07_canopy.ipynb) | voxel occupancy, contact-frequency PAD, gap fraction and effective PAI |
| 8 | [Pulses and shots files](08_shots.ipynb) | `Shots`, misses, conversions, the Parquet shots format |
| 9 | [Ray-traced voxels](09_voxels.ipynb) | attenuation, PAD / LAD / WAD, leaf angles, wood volume, `.vox`, streaming |

To run them yourself:

```bash
pip install -e '.[examples]'          # matplotlib and jupyter
jupyter lab docs/examples
```

They are generated from `docs/examples/build_notebooks.py`; edit that file and
run it to regenerate and re-execute them (`python docs/examples/build_notebooks.py
05_trees` for one).
