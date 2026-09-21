# Pulse data

`sylva.Shots` holds laser pulses rather than points: an origin and a unit
direction per pulse, and a CSR list of echo ranges. A pulse with no echo is a
genuine miss and still says where there was free space, which is what
ray-traced canopy metrics need.

## From RIEGL projects and ray clouds

Pulse data (from `.rxp` or a raycloudtools ray cloud) enables ray-traced
metrics that account for free space. RiSCAN PRO projects give you every scan
position's SOP and the angular scan pattern, which is needed to reconstruct
the no-return pulses that RiVLib's point stream leaves out:

```python
proj = sylva.read_riscan_project("RobsonCreek.RiSCAN")
pos = proj["ScanPos001"]
shots = pos.read_shots()                              # project coordinates (SOP applied)
zen, gap = canopy.gap_fraction_pattern(shots, shots.to_pointcloud().z - pos.sop[2, 3],
                                       pos.pattern, min_height=1.0)
print("PAI", canopy.lai_from_gap_fraction(zen, gap, "hinge"))

full = pos.read_shots(fill_missing=True)             # adds echo-less pulses
grid = canopy.density_grid(full, voxel_size=1.0)
dtm = ground.make_dtm(ground.classify_ground_pmf(full.to_pointcloud()), 0.5)
z, pad = grid.mask_ground(dtm).profile_above_ground(dtm, bin_size=2.0)   # PAD by height
```

## Shots files

`Shots.save("plot.parquet")` is the native way to keep pulse data. It is a
Parquet file with one row per pulse: `scan` (index into the scanner positions
held in the file metadata), `zenith`, `azimuth`, `range` (a list of echo
ranges, empty for a pulse that returned nothing) and one list column per echo
attribute, zstd compressed in row groups of about a million pulses. A miss
costs two angles instead of a far "sky" point with every point attribute, an
echo is a range instead of three coordinates, and nothing has to sit 1000 m
outside the plot. polars, pyarrow, duckdb and R arrow read it directly.

```python
shots = Shots.from_ray_cloud(sylva.read("rays.laz"))   # or pos.read_shots(fill_missing=True)
shots.save("plot.parquet")
Shots.file_info("plot.parquet")                        # counts, echo bounds, scanner positions
shots = Shots.load("plot.parquet")                     # or groups=[0, 1] for part of it
grid = voxels.ray_voxelize("plot.parquet", 0.25, ground_class=2)   # streamed, never fully in memory
```

Angles and ranges are float32 (echo positions return to about 0.01 mm per
100 m; `double=True` is exact). Origins within `origin_tolerance` (1 mm) are
merged into one scanner position without moving any echo, which removes the
per-ray rounding noise ray clouds carry. Mobile and airborne data (more than
65 536 positions) store the origin per pulse as delta-encoded 0.1 mm integers,
which costs a few bits along a smooth trajectory. On a 185 M-ray mobile ray
cloud (97 % misses) the file is 1.1 GB against 2.4 GB of LAZ, and voxelising
it peaks at 3.6 GB of memory instead of 37 GB.
