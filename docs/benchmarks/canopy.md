# Canopy gap profiles

Eleven TERN plots scanned with a RIEGL VZ-2000i, 36–138 upright scan
positions each. `canopy.GapProfile` reads the raw RXPs: every 4th pulse with
all its echoes, fired pulses rebuilt from the downward zenith lines, and a
ground plane fitted per scan to its downward echoes. It is compared with:

- **[pylidar-tls-canopy](../references.md)'s [Jupp et al.
  (2009)](../references.md) profiles** of the same scans (hinge
  PAI, pooled per plot and per scan);
- **ray-traced 0.5 m voxel PAI** (raycloudtools `rayvoxel`, free-path-length
  attenuation, occlusion traced; not comparable in kind: voxels count all
  plant area, not the upward view);
- **optical LAI** from digital cover (DCP) and hemispherical (DHP)
  photographs, the nearest date to the scan (dates from the RXP names).

| plot | scans | Sylva hinge | pylidar hinge | per-scan r | voxel | clumping | canopy height (m) | photo (days apart) |
|---|---|---|---|---|---|---|---|---|
| ROBSON_2023 | 116 | *saturated* | 4.65 | – | 9.11 | 0.64 | 29.0 | DHP 2.54 (132) |
| WARRA_2023 | 85 | 3.04 | 3.07 | 0.97 | 6.36 | 0.61 | 55.5 | – |
| WOMBAT_2023 | 84 | 3.16 | 3.19 | 0.99 | 5.74 | 0.71 | 29.5 | – |
| CUP_2022 | 51 | 2.55 | 2.45 | 0.99 | 3.25 | 0.56 | 24.5 | DCP 1.67 (85) |
| TUMBA_2022 | 138 | 1.69 | 1.63 | 1.00 | 1.86 | 0.55 | 43.0 | DCP 1.14 (111) |
| BOYAGIN_2021 | 57 | 0.88 | 0.87 | 1.00 | 1.63 | 0.50 | 13.0 | DHP 0.99 (36) |
| ALICE_2021 | 40 | 0.84 | 0.82 | 1.00 | 1.62 | 0.41 | 7.5 | DHP 0.76 (1600) |
| LITCH_2021 | 36 | 0.87 | 0.86 | 1.00 | 1.44 | 0.67 | 20.0 | DCP 1.04 (29) |
| FLETCH_2022 | 36 | 0.24 | 0.23 | 1.00 | 0.85 | 0.42 | 12.5 | – |
| GWW_2021 | 36 | 0.17 | 0.17 | 1.00 | 0.70 | 0.41 | 18.5 | DCP 0.51 (1) |
| CALP_2021 | 36 | 0.01 | 0.01 | 1.00 | 0.67 | 0.65 | 18.0 | DCP 0.78 (826) |

- **Against pylidar**, on the ten plots that are not saturated: r = 0.999,
  median ratio 1.01, and every plot within 4 %. Scan by scan (599 scans),
  r is 0.97–1.00 per plot.
- **Robson Creek** lets less than 0.1 % of pulses through at 57.5°. The
  hinge PAI there is bounded by the number of pulses, not measured, and the
  report flags it (`saturated`).
- **Against the photographs** taken within 120 days (5 plots), Sylva's
  effective hinge PAI gives r = 0.96 at 0.89 × the photo value. Correcting
  for clumping ([Lang & Xiang 1986](../references.md) over scan-sector segments) overshoots it
  (1.76 ×), consistent with the photo values being effective too. The voxel
  PAI is 1.63 × the photos.
- **Low woodland** (Calperum mallee, Alice mulga) is mostly at or below the
  1.5–2 m tripod. Upward-looking gap fraction does not see it: 0.01 at
  Calperum where the photos find 0.78 and the voxels 0.67.
- **pylidar's Jupp linear estimate** (2.7–4.4 on every plot, including ones
  whose hinge PAI is 0.01) is not usable. Sylva's linear fit is close to
  its hinge value in forest and 0.3–0.4 above it in open woodland. There the
  near-horizontal rings cross stems and shrubs close to the ground, which the
  linear model reads as vertical foliage.

Getting the dense plots right took one fix. RiVLib's stream leaves out the
pulses that returned nothing, so fired pulses have to be reconstructed. The
first version took a percentile of pulses per zenith line, which runs about
7 % high: the mirror's angles do not sit on the nominal lines, so some line
bins catch extra pulses. Where nearly every pulse returns, 7 % too many
fired pulses reads as 7 % gap and capped the hinge PAI near 3 (Robson,
Warra, Wombat were 25–40 % low). Counting pulses per line on the downward
lines, where every pulse hits the ground, fixed it.
