# Ray-traced voxels

`sylva.voxels` is a port of the `rayvoxel` tool from the raycloudtools fork
(G. Eaton, CSIRO), which follows AMAPVox. Each pulse is traced through the
grid twice: once whole, for beam counts and the potential path length (the
full chord of every voxel it could have crossed), and once per echo segment
carrying the share of the pulse still travelling, for free path lengths, beam
sections and mean beam angles. Ground echoes and echo-less pulses are traced
but never count as hits.

```python
from sylva import voxels

shots = pos.read_shots(fill_missing=True)            # or Shots.from_ray_cloud(sylva.read("rays.laz"))
grid = voxels.ray_voxelize(
    shots, voxel_size=0.25, dtm=dtm,                 # echoes within 0.2 m of the DTM are ground
    leaf_classes=[4], wood_classes=[5],              # codes of the "classification" echo attribute
    attenuation=["fpl", "ppl"], laser="VZ-400",      # beam-section weighting, transmittance
    inclination=True, occlusion=True,
)
grid.num_hits, grid.path_length, grid.free_path_length   # raw sums, (nz, ny, nx)
grid.attenuation_ppl, grid.pad_fpl, grid.lad_fpl, grid.wad_fpl, grid.transmittance
grid.state                                               # 0 unobserved, 1 occluded, 2 empty, 3 filled
pai = np.nansum(grid.profile("pad_ppl")) * grid.voxel_size

grid.add_wood_volume(qsms)                           # QSM cylinders -> wood_volume (m3 per voxel)
grid.write("plot.vox")                               # AMAPVox voxel space; "plot.txt" for a table
grid.write_iad_csv("plot_iad.csv")                   # per-tree inclination angle distributions
```

Without `laser` or `beam`, the estimators fall back to counts: FPL is the
weighted hits over the free path, transmittance uses the weighted hits over
the weighted beams. An echo counts as its share of the pulse
(`num_hits_weighted`), like the path lengths it is divided by, so a
three-echo pulse is not three whole hits. On the Litchfield core hectare,
the fallback and the beam-section estimate agree within 3 % in PAI.

| Attenuation | Estimate of λ (m⁻¹) |
|---|---|
| `fpl` | intercepted beam section / effective free path length, minus the Pimont et al. (2018) bias term |
| `ppl` | exact maximum-likelihood solve of `Σ_hits s L / (e^{λL} − 1) = Σ_misses s L` over the beams of each voxel (capped at 20) |
| `transmittance` | `−ln(1 − intercepted / entering)` beam section |
| `bailey` | Bailey & Mahaffee (2017) eq. 10 per class, with `G` from triangle facets between neighbouring echoes |

Area density is `λ / G`. `G` comes from an analytic leaf angle distribution
(`lad=`: spherical, uniform, the four de Wit types, ellipsoidal, two-parameter
beta) at each voxel's mean beam zenith, or with `inclination=True` from
inclination angle distributions estimated per tree (`tree_id` echo attribute)
from PCA normals of the echoes and integrated over the tree's own beam
zeniths (Vicari et al. 2019); each voxel uses its predominant tree.

Differences from the C++ tool: input is a `Shots` object, so ground, leaf /
wood and tree labels are plain per-echo arrays, and an echo-less pulse is
traced to the grid edge (or `unbounded_range`) because `Shots` does not keep
the far end of a miss; sums are accumulated in double precision, so runs differ
only where a float32 result rounds the other way (last printed digit); neighbour priors read the grid as it was
before any top-up and leave the outer voxel layer alone instead of padding;
without `tree_id` all echoes share one inclination distribution; echoes under
the DTM surface are ground, not vegetation; `pad_fpl` pairs the plant hits with
the free path of every vegetation segment (rayvoxel uses only the segments
ending in an echo that is neither leaf nor wood); there is no
out-of-core mode, NetCDF output or per-voxel class histogram. Memory is about
0.25 kB per voxel while tracing.

## What the scan saw

With `occlusion=True` every voxel is observed (a pulse went through or ended
in it), occluded (only pulses that had already stopped reached it) or
unreached.

```python
g = voxels.ray_voxelize(shots, 0.25, bounds, dtm=dtm, occlusion=True)
prof = g.occlusion_profile(min_height=0.5)   # per layer: observed / occluded / unreached, mean pulses
prof["total"]                                # the whole canopy space
m = g.observed_map()                         # (ny, nx): observed share of each column
t = voxels.tree_sampling(g, cloud, labels)   # per tree
t["above_observed_fraction"], t["beams_by_quarter"], t["p10_beams"]
```

The canopy space runs from `min_height` above the ground (needs a DTM) up to
the highest filled voxel.

Per tree, the envelope is the stacked layer hulls of the tree's points.
That envelope is observed almost by construction: parts of a crown nobody
saw left no points. So the telling measures are the pulses that reached it,
per quarter of its height, and `above_observed_fraction`: whether the space
up to 2 m over the tree's highest point was seen, as empty or as a
neighbour's crown. If it was not, the tree may continue where the scanner
could not see.

Ray tracing needs the real scanner position of every pulse. Ray clouds
exported without sensor positions put every ray's start at the origin; they
can still be voxelised, but the occlusion they give is that of pulses from
below, not of the scan. `Shots.from_ray_cloud` warns about this case.
