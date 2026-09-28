# Full waveforms

A full-waveform scanner digitises the received power of every pulse, a
sample every nanosecond or so, instead of reporting only the few returns its
own detector found. The waveform holds more than those returns: echoes too
weak or too close together for the detector, the width of each echo (how
deep the target is along the beam) and its energy. `sylva.waveform` reads
waveforms from LAS and PulseWaves files, decomposes them into Gaussian
echoes (Hofton et al. 2000; Wagner et al. 2006), and turns the echoes into
[`Shots`](../api/shots.md), so that they feed the ray-traced voxels and the
canopy tools like any other pulses.

```python
from sylva import waveform, voxels

print(waveform.info("flight.las"))           # header, descriptors, where the packets are
wf = waveform.read("flight.las")             # all waveforms, or ...
for chunk in waveform.chunks("flight.las", size=100_000):   # ... a bounded amount at a time
    echoes = chunk.decompose()
    shots = echoes.to_shots(chunk)
```

## Files

| Format | Read | Write |
|---|---|---|
| LAS 1.3 / 1.4, point formats 4, 5, 9, 10, packets in an EVLR or a `.wdp` file | yes (also LAZ) | LAS 1.4 format 9, EVLR or `.wdp` |
| PulseWaves 0.3, `.pls` + `.wvs` | yes (uncompressed) | yes |
| RIEGL SDF | no | no |

**LAS.** A waveform point carries the index of a *wave packet descriptor*
(a VLR giving the bits per sample, the number of samples, the sampling
interval in picoseconds and the digitiser gain and offset), the byte offset
and size of its packet, the *return point location* `L` (ps from the first
sample to the point) and a vector `v` (m/ps). The packets are in an extended
VLR of the file or in a `.wdp` file of the same name; `waveform.info` says
which (`packets` is `"internal"`, `"external"` or `"missing"`). Sample `i`
lies at `P + (L - i dt) v`, so `v` points from the point back towards the
scanner: this is how RIEGL's exports and LAStools read it (the
specification's wording leaves the sign open), and it is checked below on a
RIEGL file. Samples are converted to volts with the descriptor's gain and
offset. The returns of one pulse share a packet; consecutive point records
that point to the same packet become one waveform (`dedupe=False` gives one
per record), with `n_records` saying how many there were.

**PulseWaves** (Isenburg 2012) stores pulses (`.pls`: time, an anchor
point, a target point 1000 sampling units along the beam, a descriptor) and
their waves (`.wvs`). A descriptor lists *samplings*, outgoing or returning,
each stored as one or more *segments*; RIEGL records, for example, keep
only the segments around echoes, on two channels of different gain. Each
segment becomes one row of `Waveforms`, with the attributes `kind` (1
outgoing, 2 returning), `channel`, `segment` and `sampling`, and all rows of
a pulse share `pulse`. `read` returns the returning samplings by default
(`kind="outgoing"` or `"all"` for the rest). Lookup tables that turn sample
values into physical units are applied unless `lookup=False`. Where the
descriptor says the optical centre and the anchor coincide (or are a fixed
number of samples apart), the scanner position is known and stored in
`origin`.

**RIEGL SDF** is a proprietary format that only RIEGL's own library can
read, and that library cannot be used by Sylva. Export the waveforms with
RIEGL's software (RiPROCESS or RiANALYZE) to LAS 1.3 or 1.4 with waveform
data packets, and read that file.

**Large files.** Waveforms are read in chunks: `waveform.chunks(path,
size)` yields about `size` point records (LAS) or pulses (PulseWaves) at a
time with the samples they refer to, reading the packets in file order
through one buffered reader. Memory therefore stays bounded whatever the
size of the file. A pulse belongs to the chunk it starts in; records that
continue a chunk's last packet are read with it, so no waveform is split or
repeated (this is tested). `read(path, start, count)` reads one such range. A LAS
file of 4 million waveforms (160 million samples, 550 MB) streams through
`chunks(size=200_000)` and `decompose` in about ten seconds on 8 threads,
with memory held to one chunk.

## The container

`Waveforms` keeps the samples in compressed sparse row form, like
`Shots` keeps echoes, with each waveform's geometry as both formats keep
it: an `anchor` point on the beam, the unit `direction` (away from the
scanner), the time `offset` (ns) from the anchor to the first sample, the
sampling `interval` (ns) and `metres_per_ns`, the range covered per
nanosecond of round-trip time (half the speed of light, `waveform.C_HALF`,
as the file gives it). Sample `k` lies at

    anchor + direction * metres_per_ns * (offset + k * interval)

which `wf.sample_positions()` returns for every sample and `wf[i].positions()`
for one waveform. `origin` is the scanner position where the file gives
it and NaN otherwise (a LAS file never does). `gps_time`, per-waveform
`attrs` and `subset`, `concatenate` and the writers complete it.

## Decomposition

`waveform.decompose(wf)` models each waveform as a constant background plus
one Gaussian per echo, as Wagner et al. (2006) do for small-footprint
airborne scanners:

1. **Noise.** The background starts as the median of the samples and the
   noise as their median absolute deviation times 1.4826, recomputed
   without the samples more than three standard deviations above the
   background until they settle, so that echoes do not inflate either. Both
   are then refined as the mean and standard deviation of the samples
   within three standard deviations of the background, which stays right
   for digitised samples whose median absolute deviation moves in whole
   counts (`waveform.estimate_noise`). `background` and `noise` can be
   given instead.
2. **Smoothing.** A Gaussian kernel of `smooth` ns (1 by default) removes
   the noise that would otherwise make spurious maxima (`waveform.smooth`).
3. **Candidates.** With `peaks="derivative"`, at the zero crossings of the
   smoothed signal's first derivative, the local maxima (Wagner et al.
   2006); with `peaks="inflection"` (the default), at the local minima of
   its second derivative, with the width taken from the inflection points
   on either side (Hofton et al. 2000). For a Gaussian the inflection points
   are one standard deviation from the centre, and two overlapping echoes
   still make two minima of the second derivative after their sum has
   stopped making two maxima, so this finds closer echoes. Only candidates
   higher than `threshold` noise standard deviations (4) and
   `min_amplitude` are kept, at most `max_echoes`.
4. **Fit.** The amplitudes, positions and widths of all candidates are
   fitted together to the raw (unsmoothed) samples by Levenberg-Marquardt
   least squares with an analytic Jacobian, the widths held within
   `min_width` to `max_width` ns.
5. **Clean-up.** Fitted echoes below the threshold are removed and
   coincident ones merged; with `refine` (the default) an echo is added
   where the smoothed residual still exceeds the threshold. The fit is
   repeated until nothing changes.

The result, `Echoes`, has per echo the waveform row, the `time` of its peak
(ns after the first sample), `amplitude`, `width` (standard deviation, ns),
position `xyz` and `range` from the scanner (NaN where the origin is
unknown), and per waveform `stats`: background, noise, rmse of the fit,
number of echoes and iterations. `energy` is the area under each echo.
Waveforms are decomposed in parallel; the result does not depend on the
number of threads (this is tested).

**Backscatter cross-section.** With a Gaussian system pulse the radar
equation gives each echo's cross-section as `sigma = C_cal R^4 P s`, from
its range `R`, amplitude `P` and width `s` (Wagner et al. 2006).
`echoes.cross_section(calibration)` returns it; `calibration=1` gives
relative values, comparable within a survey. `waveform.calibration_constant`
estimates `C_cal` from echoes of reference targets of known reflectance
(extended Lambertian targets, for which `sigma = pi rho R^2 beta^2 cos
alpha`, Wagner 2010).

## From echoes to pulses

`echoes.to_shots(wf)` (or `waveform.to_shots`) makes one shot per pulse
with its echoes sorted by range and the echo attributes `amplitude`,
`width`, `time` and `waveform`. A pulse in which nothing was found stays a
shot without an echo, which the ray tracing counts as free space. A shot
starts at the scanner, from `origin=` (for example positions interpolated
from a trajectory at `wf.gps_time`) or from the file; where neither gives
it, it starts at the pulse's first sample, and the free space it records is
only the digitised window. The shots then go to
[`voxels.ray_voxelize`](../api/voxels.md), the gap profiles of
[`sylva.canopy`](../api/canopy.md) or `Shots.save`.

## A worked example

[`synthetic.waveforms`](../api/synthetic.md) makes the waveforms of known
targets: the system pulse (a Gaussian of `pulse_width` ns) convolved with
the targets along each beam, plus a background and Gaussian noise, and
digitised. Each echo of the `Shots` given is a target, with its peak from
the echo attribute `amplitude` and its depth along the beam from `extent`.
Here the targets are the returns of a simulated airborne survey:

```python
import numpy as np
from sylva import Shots, synthetic, voxels, waveform

rng = np.random.default_rng(0)
trees = [(x, y, 0.3, h) for x, y, h in
         zip(rng.uniform(5, 45, 10), rng.uniform(5, 45, 10), rng.uniform(10, 25, 10))]
scene = synthetic.forest(trees, size=50.0, ground_points=100, margin=0.0)
flight = synthetic.als_flight(scene, altitude=80.0, line_spacing=40.0, pulse_rate=20_000,
                              bounds=(0, 0, 50, 50))

# The returns of a pulse are consecutive and share gps_time.
pts = flight.points
t = pts.attrs["gps_time"]
first = np.flatnonzero(np.r_[True, t[1:] != t[:-1]])
count = np.diff(np.r_[first, len(t)])
origin = flight.sensor_positions(t[first])
ranges = np.linalg.norm(pts.xyz - np.repeat(origin, count, axis=0), axis=1)
last = first + count - 1
direction = (pts.xyz[last] - origin) / ranges[last, None]
targets = Shots(origin, direction, first, count, ranges,
                {"amplitude": pts.attrs["intensity"] / 100.0})

wf, truth = synthetic.waveforms(targets, gps_time=t[first], noise=2.0, seed=1)
wf.write_las("flight_wdp.las")                       # LAS 1.4, format 9, packets in an EVLR

parts = []
for chunk in waveform.chunks("flight_wdp.las", size=50_000):
    echoes = chunk.decompose()
    parts.append(echoes.to_shots(chunk, origin=flight.sensor_positions(chunk.gps_time)))
shots = Shots.concatenate(parts)
grid = voxels.ray_voxelize(shots, 1.0)
```

The flight fires 110,994 pulses at the plot, whose 124,804 returns become
13.3 million samples written to a 33 MB LAS file. Decomposing it finds
118,606 echoes (most of the rest are weak returns below the detection
threshold), each a median 3.6 mm from the range of its simulated return;
the shots, one per pulse with its origin from the trajectory, go straight
into the voxel grid. The whole run takes about six seconds on a laptop.

## Validation

**Synthetic waveforms.** Waveforms of 120 samples at 1 ns with a system
pulse of standard deviation 1.5 ns (3.5 ns full width at half maximum,
0.22 m in range), a background of 10 and Gaussian noise of standard
deviation 1 (so an echo's amplitude is its signal-to-noise ratio), digitised
to integers, with the default parameters. Each row is 4000 waveforms with
one echo at a random sub-sample position:

| amplitude / noise | detected | false echoes per waveform | range bias (mm) | range sd (mm) | amplitude error, median (%) | width error, median (%) |
|---|---|---|---|---|---|---|
| 5 | 0.31 | 0.002 | -0.3 | 38.4 | 16.7 | 12.6 |
| 10 | 1.00 | 0.002 | +0.1 | 21.0 | 5.3 | 6.2 |
| 20 | 1.00 | 0 | -0.1 | 10.0 | 2.7 | 3.2 |
| 50 | 1.00 | 0 | -0.0 | 4.0 | 1.1 | 1.3 |
| 100 | 1.00 | 0 | -0.0 | 2.1 | 0.5 | 0.6 |
| 500 | 1.00 | 0 | -0.0 | 0.4 | 0.1 | 0.1 |

The range is unbiased and its error falls in inverse proportion to the
signal-to-noise ratio; an echo 5 times the noise is at the 4-sigma
threshold, so only a third are detected. `peaks="derivative"` gives nearly
the same figures for single echoes.

Two echoes of equal amplitude (100 times the noise) at a given separation,
1000 waveforms each; "separated" means both were found, each within half the
separation of its true range:

| separation (pulse sd) | separation (m) | separated, `inflection` | separated, `derivative` | separated, `inflection`, `smooth=0.5` | 100 and 30 | 20 and 20 |
|---|---|---|---|---|---|---|
| 1.5 | 0.34 | 0 | 0 | 0.02 | 0 | 0 |
| 1.75 | 0.39 | 0 | 0 | 0.79 | 0.19 | 0.05 |
| 2.0 | 0.45 | 0.98 | 0.10 | 0.91 | 0.98 | 0.68 |
| 2.25 | 0.51 | 1.00 | 1.00 | 1.00 | 1.00 | 0.92 |
| 2.5 | 0.56 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 |
| 3.0 | 0.68 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

(The last two columns use `inflection` and the default smoothing, with
amplitudes of 100 and 30, and of 20 and 20, times the noise.) With the
default 1 ns smoothing, two echoes are separated from about two pulse
standard deviations apart (0.45 m for this pulse); the inflection points
find them where the smoothed sum no longer has two maxima, which
`peaks="derivative"` needs another quarter of a standard deviation to see.
The second derivative of two equal Gaussians has two minima once they are
1.48 standard deviations apart, so less smoothing (`smooth=0.5`) separates
closer echoes, at the cost of splitting some single echoes in two when the
signal is under 20 times the noise (0.24 false echoes per waveform at 10
times). The separated echoes are placed with a median error of 10 mm at two
standard deviations and 2 mm at three.

Extended targets widen the echo: with a target depth `e` (standard
deviation along the beam) the echo's width is `sqrt(pulse_width^2 +
(e / metres_per_ns)^2)`, which the fit recovers:

| depth (m) | echo width (ns) | width error, median (%) | amplitude error, median (%) | energy error, median (%) |
|---|---|---|---|---|
| 0 | 1.50 | 0.6 | 0.5 | 0.5 |
| 0.1 | 1.64 | 0.6 | 0.6 | 0.6 |
| 0.25 | 2.24 | 0.8 | 0.6 | 0.6 |
| 0.5 | 3.66 | 1.2 | 1.0 | 1.2 |
| 1.0 | 6.84 | 2.6 | 1.8 | 3.5 |

**Speed.** 100,000 waveforms of 120 samples with one to four echoes each
decompose in half a second on 8 threads.

**A RIEGL file.** The PulseWaves repository's RIEGL LMS-Q680i sample
(`100429_152240_2535pt_UTM`, LAS 1.4 with a `.wdp`, 2376 waveforms, and
the same flight in PulseWaves) is read in the tests when it can be
downloaded (it is not part of Sylva). The echoes decomposed from its LAS
waveforms lie a median 3.8 cm (90 % within
6.9 cm) from the points RIEGL's own processing wrote into the
file. Its PulseWaves waves agree sample for sample with the LAS packets of
the same pulses, and their first samples lie on the LAS beams to within
the coordinate quantisation (a few millimetres), which checks the sign of
the LAS vector and both readers against each other.

**Round trips.** Waveforms written to LAS (internal and external packets,
8 and 16 bits) and to PulseWaves read back with the same samples (exactly
for integer samples, within half a quantisation step otherwise), sample
positions within the coordinate resolution, and the same times; the LAS
points also read with [`sylva.read`](../api/io.md).

## Limitations

- Compressed PulseWaves (`.plz`, `.wvz`) and compressed LAS wave packets
  are not read; decompress with `pulsezip` first. LAZ files are read, but a
  LAZ decoder reads extended VLRs whole, so keep the packets of large LAZ
  files in a `.wdp`.
- The background is estimated per row. RIEGL-style records that keep only
  the segments around echoes have few background samples per segment:
  give `background` and `noise` explicitly (for instance from a quiet part
  of the survey) when decomposing them.
- The model is a sum of Gaussians. Echoes of very different shape
  (saturated, or from a system pulse far from Gaussian) are fitted with
  more or wider components than there are targets.
- LAS stores no scanner position: without a trajectory, shots start at the
  first sample. The writers keep what the formats can hold: LAS drops
  `origin` and `pulse`; PulseWaves drops the attributes other than
  intensity and classification.
