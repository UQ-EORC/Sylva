# Pulse data

`sylva.Shots` holds laser pulses rather than points: an origin and a unit
direction per pulse, and a CSR list of echo ranges. A pulse with no echo is a
genuine miss and still says where there was free space, which is what
ray-traced canopy metrics need.

## From RIEGL projects and ray clouds

Pulse data (from `.rxp` or a raycloudtools ray cloud) enables ray-traced
metrics that account for free space. RiVLib's point stream holds only the
pulses that returned an echo; `fill_missing` puts the others back, each at
the place it was fired. A RIEGL scanner fires at a fixed rate while its
mirror sweeps the zenith at constant speed, one line per azimuth step, so
within a line the zenith is a linear function of time: a time gap of *n*
intervals between two recorded pulses holds *n* − 1 misses at the
interpolated angles, and from the first recorded pulse of a line the sweep
is continued back to the sky end at the measured angular step (and from
the last pulse on to the other end). The dead time between lines is left
alone. The pulse interval and angular step are measured from the data; the
scan pattern of a RiSCAN PRO project supplies only the zenith limits, which
is why `ScanPosition.read_shots(fill_missing=True)` is the way to do it:

```python
proj = sylva.read_riscan_project("RobsonCreek.RiSCAN")
pos = proj["ScanPos001"]
shots = pos.read_shots()                              # project coordinates (SOP applied)
zen, gap = canopy.gap_fraction_pattern(shots, shots.to_pointcloud().z - pos.sop[2, 3],
                                       pos.pattern, min_height=1.0)
print("PAI", canopy.lai_from_gap_fraction(zen, gap, "hinge"))

full = pos.read_shots(fill_missing=True)             # the echo-less pulses, in place
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

prof = canopy.GapProfile.empty()                    # 5-75 deg rings, 36 sectors, 0.5 m bins
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
  - Shots that already hold their misses (a ray cloud,
    `read_shots(fill_missing=True)`) need no `fired_per_ring`.
  - A pulse that returned nothing still leaves a record in the stream: a
    pseudo echo, or the scanner's own near-range echo 0.1-0.2 m out (a
    VZ-2000i writes one for every such pulse, 2.4 million in a rainforest
    tilt scan). The reader counts a pulse as recorded only from an echo
    beyond `min_range`, so these become misses where they were fired. The
    pulses the instrument's own body and mount block leave the same record
    and count as misses too; they point at the tripod, below the horizon
    (for a tilted scan, behind and beneath it).
    On a VZ-i scanner the echoes are assigned to their pulses by the MTA
    resolution first, where the shim is built: on the raw stream, an echo
    filed under the wrong pulse leaves a false miss, two per cent of a
    Robson Creek scan and not one of them corroborated by the tilted scan
    on the same tripod.
    `Shots.fill_missing_from_timing` runs the same reconstruction on shots
    already in memory that kept their `gps_time`, e.g. a RiSCAN export
    that kept every echo (one exported with a reflectance or deviation
    filter has lost pulses that were not misses).
- **Misses for voxels.** `Shots.fill_missing(pattern)` knows only the shots,
  not their timing, and draws each line's missing pulses at random azimuths.
  The count per ring is right, so ring-averaged gap fraction is served, but
  a miss drawn through foliage that returned every pulse traces free space
  where there was none: over a Banksia woodland plot the voxel PAI came out
  28 % low against the same pulses with their misses in place. Use the
  timing reconstruction (`read_shots(fill_missing=True)`) for anything
  spatially explicit.
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

## Hemispherical photographs and scan images

A scan with its misses is also the hemispherical photograph the scanner took.
`canopy.hemispherical_photo` projects every pulse through a fisheye lens, the
zenith at the centre and north at the top, and counts per pixel the pulses
fired and those that met canopy above a height; `gap` is then a gap fraction
per pixel rather than the sky-or-canopy a thresholded photograph gives, and
`gap_fraction_zenith` and `lai` read the rings off it the way photograph
software does. `hemispherical_photo_from_cloud` renders the same view from
points alone, from any viewpoint, each point painted with its beam footprint:
that is the camera's photograph, and where the scan is sparse it over-reads
the gaps, since it cannot tell a gap the scanner saw through from one it never
sampled.

A worked example on one position of a RiSCAN project, run on a VZ-2000i
scan of a closed rainforest canopy:

```python
from pathlib import Path
import numpy as np
from sylva import PointCloud, canopy, io, riscan

D = Path("ROBSON_2023")
project = riscan.read_riscan_project(D / "00_raw/RobsonCreek.RiSCAN")
pos = project["ScanPos070"]

# 1. The position's pulses, with the ones that returned nothing recovered from
#    the timing. Bounding the fill by the scan pattern's zenith sweep stops it
#    filling sky the scanner never pointed at.
pat = pos.pattern
sweep = (pat["theta_start"], pat["theta_start"] + pat["theta_delta"] * pat["theta_count"])
shots, report = io.read_rxp_shots(pos.rxp, fill_missing=sweep, return_report=True)
print(shots.n_shots, "pulses,", report["n_misses"], "recovered misses")

# 2. Scan images, in the scanner's frame: zenith down, scan lines across.
#    Misses are NaN so the gaps show.
steps = (pat["theta_delta"], pat["phi_delta"])
shots.scan_image("range", steps=steps).save("scan_range.png", log=True)
shots.scan_image("reflectance", cmap="gray").save("scan_refl.png", vmin=-20, vmax=0)

# 3. Level the scan with its SOP and give every echo a height above ground.
#    On steep ground sample the plot DTM instead of fitting a plane.
level = shots.transform(pos.sop)
xyz = level.echo_xyz()
a, b, c0 = canopy.fit_ground_plane(xyz, centre=pos.sop[:2, 3], radius=25)
heights = xyz[:, 2] - (a * xyz[:, 0] + b * xyz[:, 1] + c0)

# 4. The photograph the scanner took. 4500 pixels is one pulse per pixel for
#    a 0.04 degree scan to the horizon.
photo = canopy.hemispherical_photo(level, size=4500, echo_heights=heights,
                                   min_height=1.5, value="reflectance")
photo.save("hemi_binary.png")                   # sky white, canopy black, ground grey
photo.save("hemi_gap.png", kind="gap")          # gap fraction per pixel
photo.save("hemi_refl.png", kind="value", cmap="gray", vmin=-20, vmax=0)
zen, gap = photo.gap_fraction_zenith(np.arange(0, 91, 15))
print("ring gap fraction", np.round(gap, 4), "PAI", photo.lai(), photo.lai("miller"))

# 5. The same view from the coregistered plot cloud, within 30 m of the
#    position. It also fills the cone above 30 degrees that an upright VZ
#    scan never samples.
view = pos.sop[:3, 3]
parts = []
for f in sorted((D / "04_registered/tiles").glob("*.laz")):
    c = io.read(f)
    near = np.linalg.norm(c.xyz - view, axis=1) <= 30
    if near.any():
        parts.append(c.xyz[near])
xyz = np.concatenate(parts)
plot = PointCloud(xyz, {"height": xyz[:, 2] - (a * xyz[:, 0] + b * xyz[:, 1] + c0)})
camera = canopy.hemispherical_photo_from_cloud(plot, origin=view, size=4500, footprint=0.03,
                                               spacing=0.02, min_height=1.5, max_range=30)
camera.save("hemi_cloud_binary.png")
camera.save("hemi_cloud_range.png", kind="range", vmin=2, vmax=30)
```

```
22826727 pulses, 5067 recovered misses
ring gap fraction [   nan 0.0002 0.0004 0.0002 0.0002 0.0309] PAI 9.28 7.11
```

The first ring is NaN because the upright scan does not sample it. Sky is a
few pulses in ten thousand through this canopy, so the hinge estimate sits
at its saturation and the Miller integral, which weights the rings, reads
lower; the cloud rendering reads lower again (3.7), since gaps in the point
coverage count as sky.

- `size` of about twice `max_zenith` over the scan's angular step gives one
  pulse per pixel (4500 for a 0.04 degree scan to the horizon). Coarser
  pools pulses, finer leaves pixels no pulse passed through, grey in the
  binary rendering and NaN in `gap`.
- Lenses: `equidistant` (*r* ∝ θ, what most analysis assumes), `equisolid`
  (*r* ∝ 2 sin θ/2, Sigma 8 mm and Nikon FC-E8), `orthographic`,
  `stereographic`. `max_zenith` sets the rim, to 180 degrees for a full
  sphere; `north` turns the image and `mirror=False` puts east on the right
  as on a map instead of the left as a camera on the ground sees it.
- **The cone above.** An upright VZ scan never samples zenith angles under
  30 degrees. `hemispherical_photo_from_cloud` is the way to a full sky: the
  coregistered plot's points seen from the position, a camera's photograph
  rather than the scanner's, with the tilted scans of the plot in it.
- **Ground.** A pulse that returned only from at or below `min_height` is
  a ground pulse: terrain and understorey, and on a slope the hillside,
  which fills a sector above the horizon from a tripod. `ground_as` says
  how it counts: `"gap"` (default, as `GapProfile` and Jupp do, since the
  path to the return held no canopy), `"canopy"` (as a thresholded
  photograph reads a hillside), or `"unseen"` (left out, so the rings
  describe only the directions that reached the sky). The binary rendering
  draws ground dark grey whichever is chosen. Heights from the plot DTM
  beat a plane on steep ground.
- `zenith_azimuth()` gives every pixel's direction, to mask or weight the
  image yourself.

`Shots.scan_image` lays the pulses out on the scanner's own grid, one row
per zenith step and one column per azimuth step with the sky at the top, and
colours them by a field. The steps are estimated from the pulses, or given
from the scan pattern. Recovered misses land where they were fired, so this
works on shots straight from `read_rxp_shots` with `fill_missing`, whose
misses follow the recorded pulses; `grid="lines"` lays them out instead as
scan line by index along the line, exactly as recorded: range by default, or `n_echoes`, `zenith`,
`azimuth`, any echo attribute (`amplitude`, `reflectance`, `deviation`), or
an array of your own. Pulses that returned nothing are NaN and draw black, so
the gaps show, and the image is a quick check of a scan's coverage, a
registration's range noise, or a reflectance calibration. `scan_lines` finds
the line starts from the zenith turning back against the sweep (with the
pulse timing, when the shots carry `gps_time`, so that a fast scanner's stray
out-of-sequence pulses do not start lines).

```python
s.scan_image("n_echoes", cmap="viridis").save("echoes.png", vmin=0, vmax=3)
s.scan_image("range", grid="lines").save("lines.png", log=True)
```

Four colour maps are built in (`viridis`, `inferno`, `magma`, `gray`, each
with an `_r` reverse); with matplotlib installed any of its maps can be named.
