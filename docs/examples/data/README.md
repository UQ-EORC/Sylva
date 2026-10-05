# Example data

## Litchfield tile

A 20 x 20 m tile of the TERN Litchfield Savanna SuperSite 1 ha TLS plot
(Northern Territory, Australia), scanned in 2021 with a RIEGL VZ-2000i from
many positions and registered into one plot cloud.

| File | Contents |
|---|---|
| `litch_tile.laz` | 1,553,677 echoes inside the tile, thinned to one point per 5 cm, with `classification` (2 = ground, 4 = vegetation) and `intensity` (the ray cloud's `alpha`) |
| `litch_tile_shots.parquet` | 625,764 pulses: every 40th ray crossing the tile, clipped to it, misses included, echoes carrying the same `classification` |

One scan position falls inside the tile (its north-east corner, about 38 % of
the pulses) and keeps its true origin; the remaining pulses are rays from the
rest of the plot, clipped where they enter.

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

## Two harvest trees

`buttress_tree.laz` and `round_tree.laz` are single tropical trees from the
destructive-harvest data of Burt et al. (2021), for the buttress example
(notebook 18). Each source cloud is a raycloudtools ray cloud that holds a single
tree and no neighbours. Both files are in a local frame with the stem axis at
(x, y) = (0, 0) and the foot of the tree at z = 0.

| File | Contents |
|---|---|
| `buttress_tree.laz` | 1,044,390 echoes of the whole of a tree with a flanged base (tree T2 of the `CAXH` set, 46 m tall): thinned to one point per 1 cm up to 6 m, where the buttress functions read the points, and per 5 cm above. It also has about 5,000 sparse ground returns around the foot. Four ridges; flanges that reach about 1.7 m from the stem and end about 1.7 m up |
| `round_tree.laz` | 41,136 echoes of the lowest 8 m, within 4 m of the stem, of a tree with a round stem (tree T4 of the `CAXH` set), thinned to 1 cm |

Both are cut from the ray clouds by `../make_buttress_subset.py`, which says how
the stem axis and the foot are found. The files carry only coordinates (LAS
dimensions that the format fills with defaults are not data).

The underlying data are from Burt, A., Boni Vicari, M., da Costa, A. C. L.,
Coughlin, I., Meir, P., Rowland, L., & Disney, M. (2021), *Royal Society Open
Science* 8(2), 201458, <https://doi.org/10.1098/rsos.201458>. They are
redistributed with Sylva's documentation so the examples run on real trees. If
you use them beyond running the examples, cite that paper and check the licence
terms of the source data.
