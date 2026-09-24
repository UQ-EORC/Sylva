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

## Gap probability profiles

`canopy.GapProfile` pools scans into returns and fired pulses by zenith ring,
azimuth sector and height above ground ([Jupp et al. 2009](../references.md)).
Each echo of an *n*-echo pulse counts 1/*n*, the equal weighting of
[Armston et al. (2013)](../references.md). From that it gives plant area
profiles and a plot summary:

```python
from sylva import canopy, io

prof = canopy.GapProfile.empty()                    # 5-70 deg rings, 36 sectors, 0.5 m bins
for pos in project.positions:                      # upright scans
    s = io.read_rxp_shots(pos.rxp, shot_stride=4)  # every 4th pulse, all its echoes
    fired = canopy.fired_pulses_per_ring(s, pos.pattern, prof.zenith_edges, shot_stride=4)
    s = s.transform(pos.sop)
    xyz = s.echo_xyz()
    a, b, c = canopy.fit_ground_plane(xyz, centre=pos.sop[:2, 3], radius=25)
    prof.add_scan(s, xyz[:, 2] - (a * xyz[:, 0] + b * xyz[:, 1] + c), fired_per_ring=fired)
r = prof.report()
r["pai_hinge"], r["pai_linear"], r["mla_linear"], r["clumping"], r["canopy_height"]
r["height"], r["pai_hinge_profile"], r["pavd_hinge"]
```

- **Estimators** (all effective, not corrected for clumping):
  - `hinge`: −1.1 ln P(57.5°), at the angle where G is close to 0.5 for any
    leaf angle ([Wilson 1963](../references.md));
  - `linear`: Jupp's fit of −ln P(θ) against tan θ, which also gives a mean
    leaf angle;
  - `weighted`: [Miller's (1967)](../references.md) integral over the rings
    measured, assuming spherical leaves.
- **Clumping** is the [Lang–Xiang (1986)](../references.md) index over every
  (scan, azimuth sector) segment of the hinge ring. `pai_hinge_corrected`
  divides by it.
- **Fired pulses.** RiVLib's stream drops the pulses that returned nothing,
  so `fired_pulses_per_ring` rebuilds the fired counts (in the scanner
  frame).
  - Every azimuth step fires one pulse per zenith line.
  - Pulses per line are counted on the downward lines (100–125°), where every
    pulse hits the ground and so none is missing.
  - A line on a ring edge is shared between the two rings.
  - Neither the nominal `phi_count` nor a percentile over all lines will do.
    The scanner fires about 1 % more than nominal, and the mirror's angles do
    not sit on the nominal lines, so line bins catch extra pulses. In dense
    canopy nearly every pulse returns, so a few per cent too many fired
    pulses read as gaps and cap the PAI near 3.
  - With `shot_stride` both sides are decimated alike.
  - Shots that already hold their misses (a ray cloud, `Shots.fill_missing`)
    need no `fired_per_ring`.
- **Ground.** `fit_ground_plane` fits a robust plane through the lowest
  point of each grid cell, after [Calders et al. (2014)](../references.md).
  Heights are only used for the profile. Returns below the ground
  model still count, in the lowest bin: an upward pulse cannot hit the
  ground.

On a simulated turbid layer of PAI 3, hinge and weighted recover 3.0 within
2 %, and the linear fit 3.0 for horizontal and vertical foliage (2.84 for
spherical). Its leaf angle is 0°, 90° and 54° against 0°, 90° and 57°.
On eleven TERN plots read from the raw RXPs, the per-scan hinge PAI matches
pylidar-tls-canopy's for the same scans (see the canopy benchmark page).

Upward-looking scans from a 1.5–2 m tripod do not see vegetation at or below
the scanner. In low woodland (mallee, mulga) the hinge PAI is close to zero
while photographs and voxels find 0.5–0.8.
