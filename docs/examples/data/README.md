# Example data

A 20 x 20 m tile of the TERN Litchfield Savanna SuperSite 1 ha TLS plot
(Northern Territory, Australia), scanned in 2021 with a RIEGL VZ-2000i from
many positions and registered into one plot cloud.

| File | Contents |
|---|---|
| `litch_tile.laz` | 1,553,677 echoes inside the tile, thinned to one point per 5 cm, with `classification` (2 = ground, 4 = vegetation) and `intensity` (the ray cloud's `alpha`) |
| `litch_tile_shots.parquet` | 625,764 pulses: every 40th ray crossing the tile, clipped to it, misses included, echoes carrying the same `classification` |

Both are cut from the plot's raycloudtools ray cloud by
`../make_litch_subset.py`, into a local frame with the tile's south-west
corner at (0, 0) and the ground near z = 0. Rays are clipped at the tile
boundary, so pulse origins lie on the tile edge, not at the scanners: the tile
is usable for point-based work and for demonstrating ray tracing, but not for
plot-level gap fraction or PAI (see `../index.md`).

The underlying plot data are TERN's (Terrestrial Ecosystem Research Network,
<https://www.tern.org.au>), collected under its ecosystem surveillance
programme. This tile is redistributed with Sylva's documentation so the
examples run on real data. If you use it beyond running the examples, cite
TERN and check the licence terms of the source plot with TERN.
