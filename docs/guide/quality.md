# Scan quality from stems

Between 1 and 3 m a tree stem is the one surface in a forest scan whose shape
is known well enough to measure the scanner against. `sylva.quality.stem_noise`
cuts every stem into thin slices, fits a circle to each slice from all scans
together, and reads every point's radial residual per scan position:

```python
from sylva import quality

scan = quality.scan_ids_from_origins(shots.origin[shots.shot_of_echo()])
q = quality.stem_noise(cloud, scan_id=scan)      # stems detected if not given
q.summary()
# sigma_total, sigma_corrected, sigma_within, sigma_local, tail_fraction,
# registration_rms, registration_max, worst_scan
q.scans["tx"], q.scans["ty"]                     # horizontal offset of each scan (m)
q.residual                                       # per point, for mapping
```

| figure | what it contains |
|---|---|
| `sigma_total` | all points about the shared circle, as the cloud stands: range noise, bark, stem shape and misregistration |
| `sigma_corrected` | the same after moving each scan back by its offset |
| `sigma_within` | one scan's points about their own median in the slice: range noise, bark and stem shape over the arc it saw |
| `sigma_local` | after removing a smooth curve along that arc (constant, first and second harmonics of the angle): range noise and bark only |
| `tx`, `ty` | a scan's horizontal offset from `sum (r - t . n)^2` over its points on every stem, `n` the outward normal; refined by moving the scans back and refitting; relative to the mean of the scans |
| `tail_fraction` | stem points more than 4 σ off: mixed pixels at edges, ghosts |

A whole-stem circle has a floor: stems are not perfect cylinders. On
synthetic trees (exact meshes, a known range noise added along each beam)
the whole-circle spread does not drop below about 7 mm however clean the
points. `sigma_local` takes out the shape along each scan's arc and follows
the added noise, over a floor of 5–9 mm from the synthetic trunks' bark and
HELIOS's own noise:

| added range noise | total (tree 015) | local (tree 015) |
|---|---|---|
| 0 mm | 8.8 mm | 6.0 mm |
| 3 mm | 9.3 mm | 6.5 mm |
| 6 mm | 10.1 mm | 7.5 mm |
| 10 mm | 11.6 mm | 9.9 mm |

Registration comes from the offsets. On one stem a scan's offset is mixed
up with the trunk's own shape where that scan sees it: a 20 × 10 mm shift of
one of four scans came back as 20 × 12 mm on a straight stem (the other
scans within 2.4 mm), but up to 10 mm off on irregular trunks. Over a plot
of many stems at different bearings the shapes average out. A misregistered
vertical component does not show on vertical stems.

On a real plot, CUP_2022 (Cumberland Plain, TERN; 51 VZ-2000i scans read
from the raw RXPs with the curated registrations; every 4th pulse, 1 cm
thinned, 0.5–3.5 m above ground):

| | |
|---|---|
| stems / slices used | 2,053 / 8,123 |
| `sigma_total` / `sigma_corrected` | 9.1 / 8.8 mm |
| `sigma_within` / `sigma_local` | 8.5 / 6.7 mm |
| `tail_fraction` | 2.3 % |
| per-scan offsets | median 4.0 mm, RMS 4.7 mm, worst 13 mm (one scan) |
| split-half agreement of the offsets | 1.1 mm (median; two disjoint halves of the stems) |

The offsets agree to about 1 mm between two independent sets of stems, so
the 4–5 mm is the registration itself, not the method's noise. That puts the
curated registration at about 5 mm, with one scan position to look at.
