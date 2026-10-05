# Ray-traced voxels

`sylva.voxels` is a port of `rayvoxel`, J. Rivory's unpublished
reimplementation of AMAPVox ([Vincent et al. 2017](../references.md)) on
raycloudtools. Each pulse is traced through the grid voxel by
voxel ([Amanatides & Woo 1987](../references.md)), twice: once whole, for
beam counts and the potential path length (the full chord of every voxel it
could have crossed), and once per echo segment carrying the share of the
pulse still travelling, for free path lengths, beam sections and mean beam
angles. Ground echoes and echo-less pulses are traced
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
| `fpl` | intercepted beam section / effective free path length, minus the [Pimont et al. (2018)](../references.md) bias term, with the beam footprint of [Pimont et al. (2019)](../references.md) |
| `ppl` | exact maximum-likelihood solve of `Σ_hits s L / (e^{λL} − 1) = Σ_misses s L` over the beams of each voxel (capped at 20) |
| `transmittance` | `−ln(1 − intercepted / entering)` beam section |
| `bailey` | [Bailey & Mahaffee (2017)](../references.md) eq. 10 per class, with `G` from triangle facets between neighbouring echoes |

Area density is `λ / G`. `G` comes from an analytic leaf angle distribution
(`lad=`: spherical, uniform, the four [de Wit (1965)](../references.md) types,
[Campbell's (1990)](../references.md) ellipsoidal, [Goel & Strebel's
(1984)](../references.md) two-parameter beta, `lad_params=[mu, nu]` in their
order, e.g. `[2.770, 1.172]` for planophile) at each voxel's mean beam zenith, or with `inclination=True` from
inclination angle distributions estimated per tree (`tree_id` echo attribute)
from PCA normals of the echoes and integrated over the tree's own beam
zeniths ([Vicari et al. 2019](../references.md)); each voxel uses its
predominant tree.

Differences from the C++ tool: input is a `Shots` object, so ground, leaf /
wood and tree labels are plain per-echo arrays, and an echo-less pulse is
traced to the grid edge (or `unbounded_range`) because `Shots` does not keep
the far end of a miss; sums are accumulated in double precision, so runs differ
only where a float32 result rounds the other way (last printed digit); neighbour priors read the grid as it was
before any top-up and leave the outer voxel layer alone instead of padding;
without `tree_id` all echoes share one inclination distribution; echoes under
the DTM surface are ground, not vegetation; `pad_fpl` pairs the plant hits with
the free path of every vegetation segment (rayvoxel uses only the segments
ending in an echo that is neither leaf nor wood); there is no NetCDF output
or per-voxel class histogram, and grids too large for memory are traced block
by block (below) rather than in rayvoxel's shards. Memory is about 0.25 kB per
voxel while tracing.

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

## Block tracing

A whole-grid trace holds about 0.23 kB of accumulators per voxel, so the
grid, not the number of pulses, sets the memory: a 100 x 100 x 108 m plot at
0.1 m is 1.08 billion voxels, some 240 GB. With `block_size` the grid is cut
into blocks of that many voxels and traced a few blocks at a time, in passes
of at most `max_memory` GB of accumulators; each pass streams every pulse
once (from memory, or from a shots file a few row groups at a time) and
traces it into the blocks of the pass that its line passes through. With
`out` the blocks are written to a directory as they finish, and the grid is
never held whole. Use it when the whole trace does not fit (`ray_voxelize`
refuses a grid larger than the memory budget of `sylva.util.limits`), or to keep
a fine grid on disk and read it a part at a time.

```python
from sylva import voxels
from sylva.voxels.blocks import open_blocked

grid = voxels.ray_voxelize(
    "rays.parquet", 0.1, bounds, dtm=dtm, occlusion=True, beam=(0.007, 0.00027),
    block_size=128, out="voxels_0.1m", max_memory=20,   # GB of accumulators per pass
)
grid.block_stats                                       # blocks, passes, peak_voxels held at once
grid.block((3, 4, 0))                                  # one block as a RayVoxelGrid, every field and metric
grid.read((0, 0, 0), (200, 200, 50))                   # any box of voxels
grid.profile("pad_fpl", min_beams=5)                   # layer by layer, a slab of blocks at a time
grid.occlusion_profile(min_height=0.5)
voxels.tree_sampling(grid, cloud, labels)              # holds 5 bytes per voxel
grid.write("voxels_0.1m.vox")                          # the whole grid's rows, slab by slab
grid = open_blocked("voxels_0.1m")                     # later
```

Without `out` the blocks are assembled into one `RayVoxelGrid` (0.13 kB per
voxel, where a whole-grid trace peaks at 0.35 kB), and neighbour priors,
`inclination` and `bailey`, which need the whole grid, work as usual; with
`out` they are refused. The blocks are Parquet files with one row per voxel
and one column per raw field; blocks that no pulse reached are not written.

**Exactness.** A pulse is always walked through the whole grid with the
whole-grid arithmetic, from where it enters the grid, and a block keeps only
the voxels that fall inside it. Every voxel therefore receives exactly the
numbers a whole-grid trace adds to it, including those that depend on the
whole ray: the share of the pulse still travelling after the echoes before
the block (echo weighting), the leaving fraction of the potential path length
solve, the beam section at the voxel's range, and the occlusion ray beyond
the last echo. Whether a pulse reaches a block is decided by a slab test of
its line against the block grown by 10⁻⁶ voxel, far more than the rounding of
any walk, so no pulse is missed. Each block is filled by one thread, pulse
after pulse in file order, so its double-precision sums are added in the same
order whatever the block size or the number of `workers`: blocked results are
identical for every block size and worker count, and identical to the bit to
a whole-grid trace on one thread. A whole-grid trace on several threads adds
to a voxel in the order the threads reach it, so its single-precision results
differ from the blocked ones by one unit in the last place in a few voxels in
a hundred thousand.

On the Tumbarumba core hectare (87.6 million registered pulses, 0.5 m, DTM,
occlusion, beam divergence; 8.64 million voxels), a blocked trace equalled a
one-thread whole-grid trace in all 32 raw fields of every voxel, and the two
`.vox` files were the same bytes. Against the usual multi-threaded whole
trace, 30 fields were identical everywhere, 344 values of the mean-angle sums
differed by one unit in the last place (1.2 x 10⁻⁷ relative), and 291 of the
8.64 million `.vox` rows differed in the last printed digit of the mean
zenith angle.

**Cost.** Each block walks its pulses from the grid edge, and each pass reads
the pulses again, but a voxel is only written by the pulses that reach it. On
the same plot, on 8 shared cores:

| Grid | Voxels | Trace | Passes | Peak memory | On disk |
|---|---|---|---|---|---|
| 0.5 m, whole grid | 8.6 M | 2.9 min | | 3.3 GB | |
| 0.5 m, blocks of 64, assembled | 8.6 M | 2.7 min | 1 | 3.5 GB | |
| 0.25 m, blocks of 64, 4 GB per pass | 69 M | 5.6 min | 4 | 5.5 GB | 3.7 GB |
| 0.1 m, blocks of 128, 20 GB per pass | 1.08 G | 23 min | 13 | 24 GB | 58 GB |

At 0.1 m the whole-grid trace would need about 240 GB. Over the blocked 0.1 m
grid, the occlusion profile took 3 minutes and a layer profile 1.5 minutes.

A block or box read alone has its own origin, so its voxel centres (and
`distance_from_ground`) can differ from the whole grid's in the last bit,
which can reach the last printed digit of a `.vox` row written from a blocked
grid; the occlusion summaries use the whole grid's centres.
