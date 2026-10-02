# Canopy structure from airborne lidar

Airborne and UAV lidar see the canopy from above, and each pulse is a ray
from the aircraft down through the foliage. The gap fraction, plant area
density (PAD) and plant area index (PAI) that terrestrial scans give from
below can therefore be measured from above as well, provided the returns
are put back on their rays. That needs the position of the scanner when
each pulse left: the flight trajectory. The functions described here are
part of [`sylva.als`](../api/als_canopy.md) and work on one cloud or on a
whole catalogue of tiles, chunk by chunk, as the rest of
[the ALS tools](als.md) do.

```python
from sylva import als, voxels

traj = als.read_trajectory("flight.out", crs="EPSG:32755")   # SBET or a text table
cat = als.catalog("tiles/")                                   # z as measured, ground classified

prof = als.gap_profile(cat, traj, resolution=20.0)            # MacArthur-Horn, beam-angle corrected
height, pad = prof.profile()                                  # PAD by height over the whole area
prof.pai().to_geotiff("pai_20m.tif")

vox = als.ray_voxelize(cat, traj, voxel_size=2.0, buffer=30.0)   # every pulse traced
vox.pai(min_beams=5).to_geotiff("pai_rays_2m.tif")

shots = als.pulses(cat.read((0, 0, 100, 100)), traj)          # Shots, for sylva.voxels directly
grid = voxels.ray_voxelize(shots, 0.5, ground_class=2)
```

## The trajectory

`als.read_trajectory` reads two kinds of file:

- **SBET** (`.out`, `.sbet`, `.sbt`, `.bin`, or `format="sbet"`): the
  binary "smoothed best estimate of trajectory" written by Applanix POSPac
  and most GNSS/INS software, records of 17 little-endian doubles (time,
  latitude, longitude, ellipsoidal height, velocities, roll, pitch,
  platform heading, wander angle, accelerations and angular rates). The
  positions are projected to `crs`, the coordinate system of the points;
  the heading is the platform heading minus the wander angle.
- **Text** tables separated by commas, semicolons or white space, with a
  header naming the columns (`time`/`gps_time`, `x`/`easting`,
  `y`/`northing`, `z`/`height`, optionally `roll`, `pitch`,
  `heading`/`yaw`) or with `columns=` naming them. With `crs`, `x` and `y`
  are read as longitude and latitude.

A `Trajectory` can also be built from arrays, or from a mapping such as
`synthetic.ALSFlight.trajectory`. `Trajectory.positions(gps_time)`
interpolates the sensor position linearly and leaves NaN outside the
trajectory and inside gaps longer than `max_gap` (ten median sample
intervals by default).

Two things commonly go wrong, and both show in the report of `als.pulses`:

- **Clocks.** LAS files often store adjusted standard GPS time (GPS time
  minus 10⁹ s) while SBETs count seconds from the start of the GPS week.
  When no return falls within the trajectory the error message says so and,
  if the clocks look like these two, gives the `time_offset` to pass
  (`als.week_seconds` does the conversion).
- **Heights.** SBET heights are ellipsoidal; points are often orthometric.
  Give `z_offset` (minus the geoid separation). Every multiple-return pulse
  has its returns on one line through the sensor, and
  `report["line_offset_median"]` is the median distance from the
  interpolated sensor position to those lines: centimetres to decimetres
  for a trajectory that fits the points, metres for a time or height
  offset.

## Pulses from discrete returns

`als.pulses(cloud, trajectory)` groups returns sharing a `gps_time` (and
`point_source_id`) into one pulse, starts it at the sensor position at that
time, points it at its farthest return and gives each return its distance
from the sensor as its range. The result is a [`Shots`](pulses.md) object,
so [`voxels.ray_voxelize`](voxels.md) traces airborne pulses exactly as it
traces terrestrial ones. Groups that repeat a `return_number` (two channels
firing at once) are split where the numbering restarts.

Discrete-return data cannot give back everything a pulse did:

| Lost | What Sylva does |
|---|---|
| Returns missing from a pulse (below the detection threshold, or clipped away at a tile edge) | counted from `number_of_returns` (`n_incomplete`, `n_missing_returns`); the pulse keeps what it has, or `drop_incomplete=True` leaves it out |
| Pulses with no return at all (water, dark roofs, absorption) | not in the file. With `fill_missing=True`, a hole of at most `max_fill` pulses in a line's regular firing is filled with echo-less pulses, their directions interpolated between the pulses either side, if the mirror swept steadily across the hole. Longer holes and pulses at the edge of the data cannot be recovered |
| Returns closer than the range resolution | merged by the receiver; nothing to do |
| The energy of each return | not recorded; every return stands for an equal share of its pulse (`1 / number_of_returns`) |

## Gap fraction and plant area density profiles

`als.gap_profile` bins the returns of each grid cell by height above
ground. A pulse has passed a height if some of its weight is below it, so
the gap probability at height `z` is the share of the cell's weighted
returns below `z` ([Armston et al. 2013](../references.md)), and the
transmittance `T` of a layer is the weight below it over the weight at or
below it. Inverting the Beer-Lambert law layer by layer gives the plant
area density profile of [MacArthur & Horn (1969)](../references.md):

```text
PAD(layer) = -ln T / (k̄ dz),     k = G(θ) / cos θ for each return
```

- **Scan angle.** A beam at zenith θ crosses `dz / cos θ` of the layer and
  meets foliage in proportion to `G(θ)`, the projection of the leaf angle
  distribution (`lad=`, spherical by default: G = 0.5). Each return carries
  its own `k`, and a layer divides by the mean `k` of the returns that
  reached it. Without this, returns at 30° read 15 % denser than at nadir.
  θ comes from the trajectory, else from the LAS `scan_angle` (which
  includes roll but not pitch; `angles="scan_angle"`), or is ignored with
  `angles="none"`.
- **Weighting.** `"equal"` (the default) counts each return as
  `1 / number_of_returns` of its pulse; `"first"` uses first returns only,
  as MacArthur and Horn did; `"all"` counts every return as a pulse.
  lidR's `LAD()` ([Bouvier et al. 2015](../references.md)) is
  `weighting="all", angles="none", g=0.5` with lidR's `z0` as
  `min_height`; the test suite reproduces its definition to 1e-12, except
  that a layer nothing passed is given half a pulse (a lower bound on its
  density) where lidR returns NA.
- **Which cell.** An oblique pulse stopped in the canopy leaves its return
  to one side of where it would have met the ground. Counting returns where
  they are picks a cell's pulses partly by their outcome, and wherever the
  pulse density changes across the swath the gap probability is biased
  (by 5 % on the layer below). With a trajectory, every return is counted
  by default in the cell where its beam meets the ground
  (`anchor="ground"`), so that a cell's pulses are chosen by their geometry
  alone.

`ALSProfile` keeps the counts (`weight`, `weight_k`, per layer and cell);
`pad()` gives the density of every layer and cell, `pai()` and `cover()`
rasters, and `profile(mask)`, `pgap(mask)` and `pooled_pai(mask)` the
profile of an area from its pooled counts (an effective profile, which is
lower than the mean of the cells' profiles where the canopy is clumped
between cells). Cells should hold a few hundred pulses: 10 to 30 m for
typical ALS densities.

On a catalogue, each return is counted once, by the chunk whose core holds
it, so the counts are the same cell for cell whatever the chunks and the
number of workers (this is tested), provided the buffer is wider than the
drift of a beam between its returns and the ground (canopy height times the
tangent of the scan angle) and than what the `"auto"` DTM needs.

### Metrics for models of height, cover and biomass

Height percentiles alone leave much of the variation in biomass
unexplained: stands of one height differ in how much plant material they
hold and where it sits. `ALSProfile.metrics()` summarises each cell's
profile as rasters that sit beside the height metrics of
[`als.grid_metrics`](als_metrics.md), and `plot_metrics(plots)` gives the
same metrics for field plots, to train the model on:

```python
prof = als.gap_profile(cat, traj, resolution=2.0)          # fine cells, pooled per plot below
table = prof.plot_metrics(centres, radius=17.84, ids=plot_ids)   # one row per 0.1 ha plot
table.to_csv("plot_profile_metrics.csv")

coarse = als.gap_profile(cat, traj, resolution=25.0)       # the prediction grid
for name, raster in coarse.metrics(strata=5.0).items():
    raster.to_geotiff(f"{name}_25m.tif")
```

| Metric | Meaning |
|---|---|
| `pulses` | weight of the cell (its pulses, with the `"equal"` weighting): how well it is sampled |
| `pai`, `cover` | plant area index above `min_height`, and one minus the gap probability there |
| `fhd` | foliage height diversity, `-Σ p ln p` over the shares `p` of plant area in each layer ([MacArthur & MacArthur 1961](../references.md)) |
| `pad_max`, `height_pad_max` | density of the densest layer, and the height of its middle |
| `height_pad_mean`, `height_pad_sd` | mean and standard deviation of height weighted by plant area |
| `pavd_<a>_<b>` | mean plant area density of each stratum `[a, b)` of `strata` m |
| `pai_above_<h>`, `cover_above_<h>` | plant area index above, and canopy cover at, the bottom of each stratum's lowest layer |

The strata, `fhd` and the cumulative profiles follow the GEDI L2B canopy
products (`pavd_z`, `pai_z`, `cover_z`, `fhd_normal`;
[Dubayah et al. 2020](../references.md)), so ALS metrics can calibrate a
model driven by GEDI over a wider region. A layer belongs to the stratum
its middle falls in, and strata below `min_height` are left out.

The counts of the cells add up exactly, so a plot's metrics are those of
the cells pooled into one column; with cells much smaller than the plot,
that is the plot's own profile. `area` in the table says how much of the
plot the cells cover. Some care for a model:

- **Sampling.** A cell with few pulses gives a noisy profile; use
  `pulses` to drop or weight cells, and a prediction grid no finer than
  the pulse density allows (a few hundred pulses per cell).
- **Grain.** The metrics of a 25 m cell are not those of a 2 m cell
  averaged: the pooled profile is lower where the canopy is clumped. Train
  and predict at the same grain, or pool the training plots as here.
- **Acquisitions.** The gap probability is a ratio of pulses, so it
  depends less on point density than return counts do, and the beam-angle
  correction removes most of the swath effect. Leaf-on and leaf-off
  surveys still differ, and surveys of different sensitivity detect
  different amounts of fine material in the upper canopy.

## Ray-traced voxels

`als.ray_voxelize` reconstructs the pulses and traces them along their
actual beams through a voxel lattice, with the tracer and estimators of
[`sylva.voxels`](voxels.md) (free path length by default; ground returns,
class 2, end a pulse without being hits). The scan angle is then accounted
for by the geometry itself. The result, `ALSVoxels`, holds the fields asked
for (by default `pad_fpl`, `transmittance`, `num_hits_weighted`,
`free_path_length` and `num_beams`), with `profile()` by height above a DTM
and `pai()` as a raster of column sums.

On a catalogue the lattice is anchored at the catalogue's minimum corner,
snapped down to a multiple of `voxel_size`, and spans `z_range` (the header
z range by default). Each chunk reconstructs the pulses of its core and
buffer points and traces them through the columns around its core; each
column is kept from the chunk whose core is nearest its centre. A pulse
crossing a chunk's core from outside it is traced there as long as one of
its returns is in the buffer, which needs a buffer at least as wide as the
height of the grid above the pulse's last return times the tangent of its
zenith (17 m for 30 m at 30°). `ALSVoxels.reach` reports the widest such
distance among the pulses traced, and a warning is given when it exceeds the
buffer; a lower top of `z_range` reduces it. Use z as measured, not heights
above ground: normalising moves each return by a different amount and bends
the rays. Memory is about 0.4 kB per voxel of a chunk while tracing, which
sets the number of chunks traced at once, and 8 bytes per voxel and field
for the result.

With a sufficient buffer the voxels do not depend on the chunks or the
number of workers, to float32 rounding, with one exception: an echo that
lies exactly on a voxel face (tiles quantise to 1 mm, and the lattice is on
whole multiples of the voxel size) can fall on either side depending on the
corner of the chunk's grid. About one voxel in a thousand then differs by
one hit or one pulse; totals are unchanged.

## Without a trajectory

`als.estimate_trajectory(source)` recovers an approximate trajectory from
the returns alone. The line through the first and last returns of a pulse
passes through the scanner, and within a short window (`interval`, 0.5 s)
the platform flies a nearly straight line at nearly constant speed, so a
linear least-squares fit of a moving point `a + b (t - t_c)` to those lines
gives its position. This is the idea of lidR's `track_sensor()`
([Roussel et al. 2020](../references.md), after [Gatziolis & McGaughey
2019](../references.md)), which fits a fixed point per window. Windows far
off the straight line through the others of their flight line are dropped,
and the ends of each line are extrapolated with the line's velocity, by at
most `extend` seconds, to reach the pulses over open ground that have no
multiple returns. A catalogue is read tile by tile.

The result is approximate: it has no attitude, it needs multiple returns
across the swath (few over open ground or sparse canopy), and noise across
the beams biases the position downwards, more so when the returns come from
a narrow fan of angles. On the synthetic flights, with Gaussian noise added
to every return:

| Canopy | Noise | Position error (median, 95th percentile) | Beam direction error (median, 95th percentile) |
|---|---|---|---|
| 160 trees over the whole swath | none | < 1 mm | < 0.001° |
| | 2 cm | 3 cm, 0.18 m | 0.003°, 0.02° |
| | 5 cm | 0.15 m, 0.5 m | 0.014°, 0.10° |
| a 20 m plot under a 60 m swath | 2 cm | 0.25 m, 0.87 m | 0.02°, 0.11° |
| | 5 cm | 1.8 m, 5.2 m | 0.13°, 0.59° |

Beam directions, which is what the canopy methods use, stay within a
fraction of a degree. `rms` (per sample) and `line_offset_median` (from
`als.pulses`) show how well the lines meet. Without multiple returns the
estimate is refused with an error; no trajectory is then better than a
wrong one, and `gap_profile` still runs from the LAS `scan_angle`.

## A worked example

A layer of 5 cm spheres between 5 and 15 m above the ground is a turbid
medium with a known answer: a thin beam is stopped with probability
`n π r²` per metre in any direction, which is the extinction of spherical
leaf angles (G = 0.5) at `PAD = 2 n π r²`. Here PAD is 0.3 m²/m³ and PAI 3:

```python
import numpy as np
from sylva import PointCloud, als, synthetic

rng = np.random.default_rng(1)
n = 935_000
xyz = np.column_stack([rng.uniform(-15, 55, n), rng.uniform(-15, 55, n), rng.uniform(5, 15, n)])
scene = PointCloud(xyz, {"classification": np.full(n, 4, np.uint8)})
flight = synthetic.als_flight(scene, altitude=60.0, line_spacing=20.0, footprint_samples=1,
                              target_radius=0.05, terrain_slope=0.0, bounds=(0, 0, 40, 40))
cat = flight.write_tiles("tiles", size=20.0, epsg=32755)
traj = als.Trajectory.from_dict(flight.trajectory)

prof = als.gap_profile(cat, traj, resolution=10.0, min_height=2.0, buffer=15.0)
inner = np.zeros(prof.shape[1:], bool)
inner[1:3, 1:3] = True                                   # 10-30 m, away from the clipped edge
height, pad = prof.profile(inner)
print(f"PAI {prof.pooled_pai(inner):.2f}, PAD 6-14 m {pad[(height >= 6) & (height < 14)].mean():.3f}")
# PAI 2.99, PAD 6-14 m 0.300

vox = als.ray_voxelize(cat, traj, voxel_size=1.0, z_range=(-1, 19), buffer=15.0)
print(f"{np.nanmedian(vox.pai(min_beams=5).data[15:35, 15:35]):.2f}, reach {vox.reach:.1f} m")
# 3.01, reach 11.9 m
```

The run takes under two seconds. The flight is clipped to the 40 m square,
so pulses whose ground return fell outside it lost their last return; the
cells along that edge read too dense, which is why the example keeps to the
middle.

## Validation

The layer above was flown with a 30° swath, at three densities, with one
sub-beam per pulse (every pulse a ray, one return) and with seven (a
1 mrad footprint whose sub-beams give up to five returns, merged within
1 m). PAI of the central 20 m, pooled, against the truth:

| PAI | Footprint | Trajectory, `equal` | `scan_angle` | `angles="none"` | `first` | `all` | Ray-traced (pooled) |
|---|---|---|---|---|---|---|---|
| 1 | 1 ray | 0.999 | 1.101 | 1.149 | 0.999 | 0.999 | 0.999 |
| 1 | 7 rays | 1.065 | 1.172 | 1.223 | 1.801 | 1.296 | 1.060 |
| 3 | 1 ray | 2.996 | 3.165 | 3.318 | 2.996 | 2.996 | 3.005 |
| 3 | 7 rays | 2.995 | 3.165 | 3.317 | 5.398 | 2.915 | 2.950 |
| 6 | 1 ray | 6.002 | 6.167 | 6.476 | 6.002 | 6.002 | 6.017 |
| 6 | 7 rays | 5.698 | 5.867 | 6.159 | 10.57 | 5.109 | 5.619 |

- With one return per pulse every weighting is the same, and the
  trajectory-based profile and the ray tracer recover the layer to 0.3 %.
  A 45° swath gives the same (within 0.7 %).
- Without the trajectory the LAS scan angle corrects most of the path
  length but not the cell bias (3 to 10 % high, 4 to 14 % with a 45°
  swath); no correction at all reads 8 to 15 % high (10 to 19 %).
- With a real footprint, the equal weighting stays within 7 %: 6.5 % high
  in the sparse layer, where a return stands for a whole sub-beam's share
  whatever energy it carried, and 5 % low in the dense one, which loses
  returns that merge within the range resolution or fall below the
  detection threshold. First returns fire on any part of the footprint and
  read the footprint rather than the leaves (80 % high); counting every
  return as a pulse reads 30 % high at PAI 1 and 15 % low at PAI 6.

The same sphere layer scanned from below at 1.5 m, pulse by pulse against
the spheres, and processed with the terrestrial tools
(`canopy.gap_fraction_zenith` with the hinge angle, and `voxels.ray_voxelize`
pooled over the layer) gives PAI 0.89, 2.90 and 6.89 (hinge) and 1.00, 2.90
and 6.05 (ray-traced) for 1, 3 and 6: the airborne and terrestrial pipelines
agree on the same scene, the hinge ring being the noisiest with only a few
hundred pulses near 57.5°.

On 21 million returns (a 150 m square at 200 kHz), `als.pulses` takes 5 s,
`gap_profile` 2 s and `ray_voxelize` at 1 m 7 s on eight cores, and recover
PAI 3.00 and 3.01.

## Limitations

- Only discrete returns: no waveform decomposition, and the energy of each
  return is not used.
- Pulses that returned nothing are recovered only inside short holes in
  the firing sequence; surveys over water or with many lost pulses read the
  canopy as denser than it is where those pulses fell.
- The trajectory is interpolated linearly, without lever arm or boresight;
  the scanner is assumed to sit at the trajectory position. For SBETs this
  is usually within a metre, which moves beam directions by less than a
  hundredth of a degree at survey heights.
- Multi-channel and multiple-pulses-in-air scanners must keep their
  channels apart in `point_source_id` or `return_number` for the pulses to
  be grouped correctly.
- The point-based profiles assume the ground is the DTM below each return;
  on steep slopes, prefer the ray-traced voxels.
