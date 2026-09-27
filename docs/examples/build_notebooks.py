"""Generate and execute the example notebooks.

    python docs/examples/build_notebooks.py            # all
    python docs/examples/build_notebooks.py 05_trees   # one

Each notebook is defined below as a list of cells: a string starting with
``#`` followed by a space, or with plain prose, is Markdown when passed
through ``md()``; everything else is code. They are saved with their outputs
so the documentation does not have to execute them.

Most notebooks run on the real TLS tile in ``data/`` (20 x 20 m of the TERN
Litchfield savanna plot, see ``make_litch_subset.py``). Where a known answer
is needed -- registration with a known transform, QSM volume against a known
taper, leaf area against a known scene -- they use a ``sylva.synthetic``
scene instead, and say so.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).parent


class md(str):
    pass


SETUP = """\
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import sylva
from sylva import synthetic

DATA = Path("data")          # the Litchfield tile, cut by make_litch_subset.py
plt.rcParams.update({"figure.dpi": 90, "figure.figsize": (7, 4), "axes.grid": False})"""

TILE = md("""The data: a 20 x 20 m tile of the TERN [Litchfield Savanna
SuperSite](https://www.tern.org.au) plot in the Northern Territory, scanned in
2021 with a RIEGL VZ-2000i from many positions and registered into one cloud.
`litch_tile.laz` holds the echoes inside the tile thinned to 5 cm, with
`classification` (2 ground, 4 vegetation) and `intensity`;
`litch_tile_shots.parquet` holds every 40th pulse, clipped to the tile, misses
included. The plot data are TERN's; `make_litch_subset.py` cuts the tile.""")

NOTEBOOKS: dict[str, list] = {}

NOTEBOOKS["01_pointclouds_io"] = [
    md("""# 1. Point clouds and I/O

`sylva.PointCloud` is an `(N, 3)` float64 array plus named per-point attributes.
It is what readers return and what almost every function takes."""),
    SETUP,
    TILE,
    """\
cloud = sylva.read(DATA / "litch_tile.laz")
print(cloud)
lo, hi = cloud.bounds
print("extent:", np.round(hi - lo, 2), "m")
{k: (v.dtype, v.min(), v.max()) for k, v in cloud.attrs.items()}""",
    md("""LAS files bring their standard dimensions along, whether or not the data
filled them in: `return_number` and `scan_angle` here are artefacts of the
export, while `classification` and `intensity` carry real information. Indexing
with a mask, indices or a slice keeps every attribute aligned, and `with_attrs`
returns a copy with extra columns."""),
    """\
veg = cloud[cloud.attrs["classification"] == 4]
ground_pts = cloud[cloud.attrs["classification"] == 2]
cloud = cloud.with_attrs(range=np.linalg.norm(cloud.xyz - [10, 10, 1.5], axis=1).astype(np.float32))
print(veg, ground_pts, sep="\\n")
print("vegetation:", f"{len(veg) / len(cloud):.0%} of the tile")""",
    """\
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
s = ax[0].scatter(cloud.x[::4], cloud.y[::4], c=cloud.z[::4], s=0.2, cmap="viridis")
ax[0].set(title="top view, coloured by height", xlabel="x (m)", ylabel="y (m)", aspect="equal")
fig.colorbar(s, ax=ax[0], label="z (m)")
slab = (cloud.y > 9) & (cloud.y < 11)
is_ground = cloud.attrs["classification"] == 2
ax[1].scatter(cloud.x[slab & ~is_ground], cloud.z[slab & ~is_ground], s=0.2, c="tab:green", label="vegetation")
ax[1].scatter(cloud.x[slab & is_ground], cloud.z[slab & is_ground], s=0.6, c="tab:brown", label="ground")
ax[1].set(title="a 2 m slice through the tile", xlabel="x (m)", ylabel="z (m)", aspect="equal")
ax[1].legend(markerscale=12, loc="upper center", ncol=2);""",
    md("""## Reading and writing

`sylva.read` / `sylva.write` pick the format from the extension: LAS/LAZ (extra
attributes become typed extra-bytes dimensions), PLY, and delimited text.
RIEGL `.rxp` needs RiVLib, see `sylva.io.read_rxp`. This tile came from a
raycloudtools ray cloud, which `read` also opens -- `nx, ny, nz` then point
from each echo back to the scanner (notebook 8)."""),
    """\
import tempfile
tmp = Path(tempfile.mkdtemp())
small = cloud[::20]
for name in ("tile.laz", "tile.ply", "tile.csv"):
    sylva.write(small, tmp / name)
    back = sylva.read(tmp / name)
    print(f"{name:9s} {(tmp / name).stat().st_size / 1e6:6.2f} MB  {len(back):,} points  attrs: {sorted(back.attrs)}")""",
    """\
laz = sylva.read(tmp / "tile.laz")
print("max coordinate error after LAZ (1 mm scale):", np.abs(laz.xyz - small.xyz).max())
print("classification dtype preserved:", laz.attrs["classification"].dtype,
      " range attribute:", laz.attrs["range"].dtype)""",
    md("Plain arrays come in through `PointCloud.from_array`, naming the columns after x, y, z."),
    """\
arr = np.column_stack([cloud.xyz, cloud.attrs["classification"]])
sylva.PointCloud.from_array(arr, names={3: "classification"})""",
]

NOTEBOOKS["02_filtering"] = [
    md("""# 2. Filtering

Thinning, cropping, outlier removal and local geometry from `sylva.filters`,
on the Litchfield tile."""),
    SETUP + "\nfrom sylva import filters\n\ncloud = sylva.read(DATA / \"litch_tile.laz\")",
    md("""## Subsampling

Registered plot clouds are far denser near the scanners than between them.
Voxel downsampling keeps one point per cell, `min_distance_subsample` enforces
a spacing without the grid pattern, and random thinning keeps the density
gradient as it is."""),
    """\
thin = {
    "voxel 10 cm": filters.voxel_downsample(cloud, 0.10),
    "voxel 10 cm (centroid)": filters.voxel_downsample(cloud, 0.10, method="centroid"),
    "random 20 %": filters.random_subsample(cloud, fraction=0.2),
    "min distance 10 cm": filters.min_distance_subsample(cloud, 0.10),
}
for k, v in thin.items():
    print(f"{k:24s} {len(v):>9,} of {len(cloud):,}")""",
    md("## Crops\n\nA box, a circular subplot, and a range shell around a point."),
    """\
box = filters.crop_box(cloud, (0, 0, 2.0), (20, 20, None))          # everything above 2 m
plot = filters.crop_cylinder(cloud, (10.0, 10.0), radius=5.0)       # a 5 m radius subplot
shell = filters.range_filter(cloud, origin=(10, 10, 1.0), min_range=4.0, max_range=8.0)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
for a, (name, c) in zip(ax, {"crop_box (z > 2 m)": box, "crop_cylinder (r = 5 m)": plot,
                             "range_filter (4-8 m)": shell}.items()):
    a.scatter(cloud.x[::20], cloud.y[::20], s=0.2, c="0.85")
    a.scatter(c.x[::8], c.y[::8], s=0.2, c="C2")
    a.set(title=f"{name}\\n{len(c):,} points", aspect="equal")""",
    md("""## Outliers

Two filters, both deciding from the neighbourhood: statistical removal compares
each point's mean distance to its `k` neighbours against the cloud-wide
distribution, radius removal needs a minimum count inside a ball. On real data there is no
truth to check against, so look at what they actually take. Here the two
disagree completely: the statistical filter removes 4 % of the cloud, most of
it the thin grass layer near the ground, because sparse-but-real vegetation
looks like noise by that test; the radius filter removes 77 genuinely isolated
points. Tune on the part of the cloud you care about, or the filter will eat
the understorey."""),
    """\
thinned = filters.voxel_downsample(cloud, 0.05)
sor = filters.statistical_outlier_removal(thinned, k=8, std_ratio=2.0, return_mask=True)
ror = filters.radius_outlier_removal(thinned, radius=0.25, min_neighbors=4, return_mask=True)
for name, keep in (("statistical (k=8, 2 sd)", sor), ("radius (0.25 m, 4)", ror)):
    drop = ~keep
    print(f"{name:24s} drops {drop.sum():>6,} ({drop.mean():.2%}); "
          f"median height of dropped points {np.median(thinned.z[drop]):5.1f} m vs {np.median(thinned.z):.1f} m overall")""",
    """\
slab = (thinned.y > 9) & (thinned.y < 11)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True, sharey=True)
for a, keep, name in ((ax[0], sor, "statistical (k=8, 2 sd)"), (ax[1], ror, "radius (0.25 m, 4)")):
    drop = slab & ~keep
    a.scatter(thinned.x[slab & keep], thinned.z[slab & keep], s=0.2, c="0.8")
    a.scatter(thinned.x[drop], thinned.z[drop], s=1.2, c="C3")
    a.set(title=f"{name}: {int(drop.sum()):,} dropped in this slice", xlabel="x (m)", ylim=(-0.5, 8))
ax[0].set_ylabel("z (m)")
print("of the points the statistical filter drops,", f"{np.mean(thinned.z[~sor] < 1.0):.0%}",
      "are below 1 m; for the radius filter,", f"{np.mean(thinned.z[~ror] < 1.0):.0%}")""",
    md("""## Local geometry

PCA over the `k` nearest neighbours gives normals and the planarity / linearity
the wood filter uses. On a real stem, bark is locally planar and the trunk
is linear at metre scale, while the grass layer is neither."""),
    """\
stem = filters.crop_cylinder(cloud, (4.8, 7.5), radius=1.2, zmin=0.5, zmax=8.0)
planarity, linearity = filters.planarity_linearity(stem, k=20)
fig, ax = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
for a, v, name in ((ax[0], planarity, "planarity"), (ax[1], linearity, "linearity")):
    sc = a.scatter(stem.x, stem.z, c=v, s=0.6, cmap="magma", vmin=0, vmax=1)
    a.set(title=name, xlabel="x (m)", aspect="equal")
ax[0].set_ylabel("z (m)")
fig.colorbar(sc, ax=ax, shrink=0.8);""",
    md("""## Clustering

`euclidean_clusters` labels connected components of the radius graph. Above the
grass layer the savanna crowns mostly separate, which is the cheap version of
tree segmentation -- notebook 5 does it properly, because crowns that touch
merge here."""),
    """\
above = filters.voxel_downsample(cloud[cloud.z > 2.0], 0.15)
labels = filters.euclidean_clusters(above.xyz, radius=0.4, min_points=200)
print("clusters:", labels.max() + 1, " unassigned points:", int(np.sum(labels < 0)),
      " largest cluster:", int(np.sum(labels == 0)), "points")
plt.scatter(above.x, above.y, c=np.where(labels < 0, np.nan, labels % 10), s=1.0, cmap="tab10")
plt.gca().set(aspect="equal", xlabel="x (m)", ylabel="y (m)", title="crown clusters above 2 m");""",
]

NOTEBOOKS["03_registration"] = [
    md("""# 3. Registration

Aligning scans: `kabsch` for known correspondences (reflector targets), `icp`
for clouds, and `merge_scans` to bring everything into one frame."""),
    SETUP + "\nfrom sylva import filters, registration as reg",
    md("Two 'scans' of the same plot: the second is a different subsample, moved by a known rigid transform that registration should undo."),
    """\
plot = synthetic.forest()
scan_a = filters.random_subsample(plot, fraction=0.3, seed=1)
truth = reg.translation(0.6, -0.4, 0.15) @ reg.rotation_z(6.0)
scan_b = filters.random_subsample(plot, fraction=0.3, seed=2).transform(np.linalg.inv(truth))""",
    md("## Targets: Kabsch\n\nWith matched points (here the four stem bases) the transform is closed-form."),
    """\
targets_a = np.array([[x, y, synthetic.terrain_height(x, y)] for x, y, *_ in synthetic.DEFAULT_TREES])
targets_b = (np.linalg.inv(truth) @ np.c_[targets_a, np.ones(4)].T).T[:, :3]
coarse = reg.kabsch(targets_b, targets_a)
print("error vs truth:", np.abs(coarse - truth).max())""",
    md("## Clouds: ICP\n\nICP refines a starting guess (identity here) as long as it lies within `max_correspondence_distance`. Point-to-plane converges in fewer iterations on surfaces such as ground and stems."),
    """\
def report(name, T, info):
    err = np.linalg.norm((T @ np.linalg.inv(truth))[:3, 3])
    print(f"{name:16s} rmse {info['rmse']:.4f} m  iterations {info['iterations']:3d}  translation error {err * 1000:.1f} mm")

report("point-to-point", *reg.icp(scan_b, scan_a, max_correspondence_distance=1.0))
T, info = reg.icp(scan_b, scan_a, max_correspondence_distance=1.0, method="plane")
report("point-to-plane", T, info)""",
    """\
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
for a, b, title in ((ax[0], scan_b, "before"), (ax[1], scan_b.transform(T), "after ICP")):
    a.scatter(scan_a.x[::4], scan_a.y[::4], s=0.2, c="C0", label="scan A")
    a.scatter(b.x[::4], b.y[::4], s=0.2, c="C3", label="scan B")
    a.set(title=title, aspect="equal")
ax[0].legend(markerscale=20);""",
    md("""## Partial overlap

When one scan sees things the other does not, the unmatched part drags the
solution. `trim` keeps only the closest fraction of correspondences each
iteration. Trimming throws information away, so start it from a coarse
alignment (targets, or an untrimmed run): from far off, the closest pairs are
all on the ground and the solution can slide along it."""),
    """\
rng = np.random.default_rng(0)
extra = sylva.PointCloud(rng.uniform([22, 0, 0], [30, 20, 8], (20000, 3)))      # only scan B sees this
scan_b2 = sylva.PointCloud.concatenate([scan_b.without(*scan_b.attrs), extra.transform(np.linalg.inv(truth))])
start = reg.translation(0.05, -0.04, 0.02) @ coarse                              # targets, a few cm off
report("all pairs", *reg.icp(scan_b2, scan_a, init=start, max_correspondence_distance=3.0))
report("trim 0.7", *reg.icp(scan_b2, scan_a, init=start, max_correspondence_distance=3.0, trim=0.7))""",
    md("## Merging\n\n`merge_scans` applies one transform per scan and records where each point came from."),
    """\
merged = reg.merge_scans([scan_a, scan_b], [np.eye(4), T])
print(merged, np.bincount(merged.attrs["scan_id"]))""",
]

NOTEBOOKS["04_ground"] = [
    md("""# 4. Ground and height

Classify ground returns, interpolate a terrain model, and turn elevations into
heights above ground."""),
    SETUP + "\nfrom sylva import canopy, filters, ground\n\ncloud = filters.voxel_downsample(sylva.read(DATA / \"litch_tile.laz\"), 0.05)",
    md("""## Ground classification

Two filters: the cloth simulation filter (CSF, Zhang et al. 2016) drapes a
cloth over the inverted cloud; the progressive morphological filter (PMF,
Zhang et al. 2003) opens a minimum surface with growing windows.

The tile's own `classification` came from PMF (see `make_litch_subset.py`), so
compare the two directly. CSF puts the terrain metres too high in about 8 % of
cells here, and the map below shows where: a band along the tile boundary. The
cloth is a connected sheet, so it needs points on both sides to be pulled down;
at a cut edge it has nothing to hold it and rides up on the vegetation just
inside. PMF works cell by cell and is unaffected."""),
    """\
csf = ground.ground_mask(ground.classify_ground_csf(cloud))
pmf = ground.ground_mask(ground.classify_ground_pmf(cloud))
for name, m in (("CSF", csf), ("PMF", pmf)):
    print(f"{name}: {m.sum():>8,} ground points ({m.mean():.1%}); "
          f"highest ground point {cloud.z[m].max():5.2f} m")

dtm_csf = ground.make_dtm(cloud.with_attrs(classification=np.where(csf, 2, 1).astype("uint8")), 0.5, bounds=(0, 0, 20, 20))
dtm = ground.make_dtm(cloud.with_attrs(classification=np.where(pmf, 2, 1).astype("uint8")), 0.5, bounds=(0, 0, 20, 20))
diff = dtm_csf.data - dtm.data
print(f"CSF above PMF by more than 0.5 m in {np.mean(diff > 0.5):.1%} of cells, "
      f"95th percentile {np.nanpercentile(diff, 95):.1f} m")""",
    """\
row = int(np.nanargmax(np.nan_to_num(diff).max(axis=1)))       # the worst row of the difference
y0 = dtm.ymin + row * dtm.resolution
slab = (cloud.y > y0 - 0.5) & (cloud.y < y0 + 0.5)
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
im = ax[0].imshow(diff, origin="lower", extent=(dtm.xmin, dtm.xmax, dtm.ymin, dtm.ymax),
                  cmap="OrRd", vmin=0, vmax=6)
ax[0].axhline(y0, color="k", lw=0.8, ls="--")
ax[0].set(title="CSF terrain minus PMF terrain (m)", xlabel="x (m)", ylabel="y (m)")
fig.colorbar(im, ax=ax[0], shrink=0.85)
ax[1].scatter(cloud.x[slab], cloud.z[slab], s=0.3, c="0.8")
for m, name, c, size in ((pmf, "PMF ground", "C0", 2.0), (csf, "CSF ground", "C3", 2.0)):
    ax[1].scatter(cloud.x[slab & m], cloud.z[slab & m], s=size, c=c, label=name)
ax[1].set(title=f"the marked row, y = {y0:.1f} m", xlabel="x (m)", ylabel="z (m)", ylim=(-1, 12))
ax[1].legend(markerscale=6);""",
    md("""The diagnosis is testable: crop the cloud further in and the bad band
should follow the new edge, which is what happens. Distance of the disagreeing
cells from whichever boundary the cloud was cut at:"""),
    """\
for lo, hi, label in ((0, 20, "full tile"), (4, 16, "inner 12 x 12 m"), (7, 13, "inner 6 x 6 m")):
    sub = filters.crop_box(cloud, (lo, lo, None), (hi, hi, None))
    a = ground.make_dtm(ground.classify_ground_csf(sub), 0.5, bounds=(lo, lo, hi, hi)).data
    b = ground.make_dtm(ground.classify_ground_pmf(sub), 0.5, bounds=(lo, lo, hi, hi)).data
    bad = np.argwhere(np.nan_to_num(a - b) > 0.5)
    if not len(bad):
        print(f"{label:16s} no cell differs by more than 0.5 m")
        continue
    xs, ys = lo + bad[:, 1] * 0.5, lo + bad[:, 0] * 0.5
    edge = np.minimum(np.minimum(xs - lo, hi - xs), np.minimum(ys - lo, hi - ys))
    print(f"{label:16s} {len(bad) / a.size:5.1%} of cells disagree, "
          f"median {np.median(edge):.1f} m from the edge, 90th percentile {np.percentile(edge, 90):.1f} m")""",
    md("""So the lesson is not "CSF is worse here": it is to classify ground on
the whole plot and crop afterwards, or to cut tiles with a buffer of a few
metres. Away from the edge the two filters agree to within 0.5 m on every cell
of this tile."""),
    md("## DTM, heights and CHM"),
    """\
cloud = ground.normalize_height(cloud, dtm)          # adds the 'height' attribute
flat = ground.flatten(cloud, dtm)                    # or replace z itself
chm = ground.make_chm(cloud, resolution=0.5)
print("terrain relief across the tile:", round(float(np.nanmax(dtm.data) - np.nanmin(dtm.data)), 2), "m")
print("canopy height p99:", round(float(np.percentile(cloud.attrs["height"], 99)), 1), "m,",
      "max", round(float(cloud.attrs["height"].max()), 1), "m")
print("canopy cover above 2 m:", round(float(canopy.canopy_cover(chm.data, 2.0)), 3))""",
    """\
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for a, r, title, cmap in ((ax[0], dtm, "DTM (m)", "terrain"), (ax[1], chm, "CHM (m)", "YlGn")):
    im = a.imshow(r.data, origin="lower", extent=(r.xmin, r.xmax, r.ymin, r.ymax), cmap=cmap)
    a.set(title=title, xlabel="x (m)"); fig.colorbar(im, ax=a, shrink=0.85)
ax[2].scatter(cloud.x[slab], cloud.attrs["height"][slab], s=0.3, c="C2")
ax[2].set(title="height above ground, 2 m slice", xlabel="x (m)", ylabel="height (m)");""",
    md("""Rasters export to an ESRI ASCII grid, or to GeoTIFF with
`dtm.to_geotiff("dtm.tif", crs="EPSG:28352")` when `rasterio` is installed
(`pip install sylva-rs[geotiff]`). Pass the same `bounds` on every date so a
time series lines up."""),
    """\
chm.to_ascii_grid("chm.asc")
sylva.write(cloud, "tile_normalized.laz")       # keeps 'height' as an extra-bytes dimension
print(sorted(sylva.read("tile_normalized.laz").attrs))""",
]

NOTEBOOKS["05_trees"] = [
    md("""# 5. Trees

Stem detection and DBH, segmentation of the cloud into trees, heights and
crown metrics, on a height-normalised cloud (notebook 4)."""),
    SETUP + """
from sylva import filters, ground, trees

cloud = filters.voxel_downsample(sylva.read(DATA / "litch_tile.laz"), 0.02)
dtm = ground.make_dtm(ground.classify_ground_pmf(cloud), 0.5, bounds=(0, 0, 20, 20))
cloud = ground.normalize_height(cloud, dtm)""",
    md("""## Stems

`detect_stems` fits RANSAC circles in horizontal slices from 1 to 5 m, links
them vertically, and reports DBH at 1.3 m with quality measures. It favours
recall: the candidates below include shrubs and low branches, which the next
steps remove."""),
    """\
stems = trees.detect_stems(cloud)
print(f"{len(stems)} candidates")
print(f"{'id':>3} {'x':>6} {'y':>6} {'dbh':>6} {'cover':>6} {'slices':>6} {'rmse':>6} {'quality':>7}")
for t in stems[:12]:
    print(f"{t.tree_id:>3} {t.x:6.2f} {t.y:6.2f} {t.dbh:6.3f} {t.inlier_fraction:6.2f} "
          f"{t.n_slices:>6} {t.rmse:6.3f} {t.quality:7.2f}")""",
    md("""## Segmentation and pruning

`merge_branches` folds limbs back into their own tree, `segment_trees` grows
each tree from its stem by least-cost paths over a kNN graph, `tree_heights`
measures the result, and `prune_trees` drops what is too short and merges
duplicate stems."""),
    """\
stems, merged_into = trees.merge_branches(cloud, stems)
labels = trees.segment_trees(cloud, stems)
trees.tree_heights(cloud, labels, stems, percentile=99)
stems, labels = trees.prune_trees(stems, labels, min_height=3.0)
crowns = trees.crown_metrics_all(cloud, labels)
print(f"{len(stems)} trees after pruning; {np.mean(labels > 0):.0%} of points assigned")
print(f"{'id':>3} {'dbh':>6} {'height':>7} {'crown area':>11} {'crown base':>11}")
for t in stems[:10]:
    c = crowns.get(t.tree_id, {})
    print(f"{t.tree_id:>3} {t.dbh:6.3f} {t.height:7.1f} {c.get('crown_area', float('nan')):11.1f} "
          f"{c.get('crown_base_height', float('nan')):11.1f}")""",
    """\
m = labels > 0
shuffle = np.random.default_rng(3).permutation(labels.max() + 2)   # neighbouring trees get unlike colours
colour = shuffle[labels] % 20
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].scatter(cloud.x[~m][::10], cloud.y[~m][::10], s=0.2, c="0.88")
ax[0].scatter(cloud.x[m][::4], cloud.y[m][::4], c=colour[m][::4], s=0.3, cmap="tab20")
for t in stems:
    ax[0].add_patch(plt.Circle((t.x, t.y), max(t.dbh / 2, 0.2), fill=False, color="k", lw=1.0))
    ax[0].annotate(str(t.tree_id), (t.x + 0.35, t.y + 0.35), fontsize=7)
ax[0].set(title="tree labels and stem positions", xlabel="x (m)", ylabel="y (m)", aspect="equal")
ax[1].scatter(cloud.x[m][::4], cloud.attrs["height"][m][::4], c=colour[m][::4], s=0.3, cmap="tab20")
ax[1].set(title="side view", xlabel="x (m)", ylabel="height (m)", aspect="equal");""",
    md("""There is no field inventory for this tile, so treat the table as a
demonstration: detection and segmentation are scored against manually
segmented plots (including Litchfield) in
[Benchmarks](../benchmarks/trees.md). Where labels have to be right, correct
them by hand in [Segfix](https://github.com/tim-devereux/segfix), which reads
and writes the `tree_id` column:"""),
    """\
ids = np.where(labels > 0, labels, 0).astype("int32")      # Segfix: 0 = unassigned
sylva.write(cloud.with_attrs(tree_id=ids), "tile_trees.laz")""",
    md("## The tree table\n\n`Tree.as_dict` and the crown metrics make one row per tree."),
    """\
import pandas as pd

table = pd.DataFrame([{**t.as_dict(), **crowns.get(t.tree_id, {})} for t in stems])
table.to_csv("trees.csv", index=False)
table[["tree_id", "x", "y", "dbh", "height", "crown_area", "crown_depth", "quality"]].head(8).round(2)""",
    md("""The candidates with a large DBH but a height of about 3 m are shrub
and grass clumps that the slice fits read as wide stems. Their `quality` is
mostly around 0.1, against 0.5 for the real trees, but not always: one here
scores 0.16, so `prune_trees(..., min_quality_short=0.15)` misses it. A
`max_dbh`, or dropping wide stems that are short, removes them all. Measure
the taper on a tree that detection is confident about instead."""),
    md("""## Basal area

`basal_area` sums the stems' cross-sections at breast height, in m²/ha of the
area given; the tile is 20 × 20 m, and `min_dbh` sets an inventory threshold.
The shrub clumps dominate it, since basal area grows with DBH squared, so drop
the wide, short candidates first."""),
    """\
area = 20.0 * 20.0
real = [t for t in stems if not (t.dbh > 0.4 and t.height < 8)]
print(f"every candidate:         {trees.basal_area(stems, area):5.1f} m2/ha")
print(f"without wide and short:  {trees.basal_area(real, area):5.1f} m2/ha ({len(real)} stems)")
print(f"  and DBH >= 10 cm:      {trees.basal_area(real, area, min_dbh=0.1):5.1f} m2/ha")""",
    md("## Taper and crown shape"),
    """\
big = max(stems, key=lambda t: t.height)
diam = trees.dbh_profile(cloud, (big.x, big.y), heights=np.arange(0.5, 8.0, 0.5))
shape = trees.crown_shape(cloud[labels == big.tree_id], base_xy=(big.x, big.y))
print(f"tree {big.tree_id}: DBH {big.dbh:.3f} m, height {big.height:.1f} m, "
      f"crown volume {shape['volume']:.0f} m3, asymmetry {shape['asymmetry']:.2f}")
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].plot(diam[:, 1], diam[:, 0], "o-")
ax[0].set(xlabel="diameter (m)", ylabel="height (m)", title=f"taper of tree {big.tree_id}")
sel = labels == big.tree_id
ax[1].scatter(cloud.x[sel], cloud.attrs["height"][sel], s=0.3, c="C2")
ax[1].set(xlabel="x (m)", title=f"tree {big.tree_id}", aspect="equal");""",
]

NOTEBOOKS["06_qsm"] = [
    md("""# 6. Quantitative structure models

From the points of one tree to a connected set of cylinders with volumes and
branch orders. This one starts on a synthetic tree, whose stem volume follows
from its taper, so the model can be checked against a known answer; a real
tree from the Litchfield tile follows."""),
    SETUP + "\nfrom sylva import qsm\n\ntree = synthetic.tree(dbh=0.35, height=14.0, seed=4)\nprint(tree)",
    md("""## Leaf / wood separation

`wood_points` combines local anisotropy with a topological cue: points that many
shortest paths from the base to the crown pass through are wood. The synthetic
tree knows which points are wood, so the filter can be checked."""),
    """\
wood = qsm.wood_points(tree, voxel_size=0.02)
true_wood = tree[tree.attrs["classification"] == 5]
from sylva import filters
d, _ = filters.knn(true_wood.xyz, wood.xyz, 1)
print(f"{len(wood):,} wood points kept; {np.mean(d[:, 0] < 0.03):.1%} of them lie on true wood")

fig, ax = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
ax[0].scatter(tree.x, tree.z, s=0.3, c=np.where(tree.attrs["classification"] == 5, "saddlebrown", "yellowgreen"))
ax[0].set(title="all points", aspect="equal", xlabel="x (m)", ylabel="z (m)")
ax[1].scatter(wood.x, wood.z, s=0.3, c="saddlebrown")
ax[1].set(title="wood_points", aspect="equal", xlabel="x (m)");""",
    md("## Cylinder model"),
    """\
model = qsm.build_qsm(wood, base_xy=(0.0, 0.0))
for k, v in model.summary().items():
    print(f"{k:18s} {v:.4f}" if isinstance(v, float) else f"{k:18s} {v}")""",
    md("The stem of the synthetic tree tapers from 1.05 to 0.25 times the breast-height radius over 90 % of the height, so its volume is known:"),
    """\
r0, r1, L = 0.35 / 2 * 1.05, 0.35 / 2 * 0.25, 14.0 * 0.9
true_stem = np.pi * L / 3 * (r0**2 + r0 * r1 + r1**2)
print(f"stem volume: model {model.stem_volume:.3f} m3, truth {true_stem:.3f} m3;  DBH: model {model.dbh:.3f} m, truth 0.35 m")""",
    """\
from matplotlib.collections import PolyCollection

# Each cylinder drawn at its true width: the outline of its side view.
a, b = model.start[:, [0, 2]], model.end[:, [0, 2]]
d = b - a
n = np.c_[-d[:, 1], d[:, 0]] / np.maximum(np.hypot(*d.T), 1e-9)[:, None] * model.column("radius")[:, None]
order = model.column("branch_order").astype(int)
fig, ax = plt.subplots(figsize=(5, 6))
ax.add_collection(PolyCollection(np.stack([a + n, b + n, b - n, a - n], axis=1),
                                 facecolor=[f"C{min(o, 9)}" for o in order],
                                 edgecolor=[f"C{min(o, 9)}" for o in order], lw=0.3))
ax.autoscale_view()
ax.set(aspect="equal", xlabel="x (m)", ylabel="z (m)", title="cylinders at true width, coloured by branch order");""",
    md("""## A real tree, and why to check `measured_volume_fraction`

The same steps on the tallest tree of the Litchfield tile (notebook 5). The
tile is thinned to 5 cm and cut out of a decimated ray cloud, so the trunk and
branches are sampled far more thinly than a single-tree scan: most cylinders
end up interpolated rather than fitted. `metrics()["measured_volume_fraction"]`
says how much of the volume came from real fits, and it is the flag to filter
on before using QSM volumes."""),
    """\
from sylva import filters, ground, trees

plot = filters.voxel_downsample(sylva.read(DATA / "litch_tile.laz"), 0.02)
plot = ground.normalize_height(plot, ground.make_dtm(ground.classify_ground_pmf(plot), 0.5, bounds=(0, 0, 20, 20)))
stems = trees.detect_stems(plot)
stems, _ = trees.merge_branches(plot, stems)
labels = trees.segment_trees(plot, stems)
trees.tree_heights(plot, labels, stems)
stems, labels = trees.prune_trees(stems, labels, min_height=3.0)
tallest = max(stems, key=lambda t: t.height)

points = plot[labels == tallest.tree_id]
points = points[points.attrs["height"] > 0.3]      # leave the grass layer out of the base fit
real = qsm.build_qsm(qsm.wood_points(points, voxel_size=0.02), base_xy=(tallest.x, tallest.y))
print(f"detection: DBH {tallest.dbh:.3f} m, height {tallest.height:.1f} m")
print(f"QSM:       DBH {real.dbh:.3f} m, stem {real.stem_volume:.2f} m3, total {real.total_volume:.2f} m3, "
      f"{real.summary()['n_cylinders']} cylinders")
m = real.metrics()
print(f"fitted to points: {m['measured_volume_fraction']:.0%} of the volume, "
      f"{m['measured_length_fraction']:.0%} of the length")""",
    md("""Two lessons. The DBHs agree (0.316 m against 0.319 m), because breast
height is where the stem is best sampled. The two fractions differ widely:
most of the volume lies in the trunk and the main limbs, which were fitted to
points, but only about a fifth of the length was, so the finer branches come
from the taper and pipe-model priors. The total volume is supported by the
data here; the branch volume and length are not. The height cut keeps the
grass layer out of the base fit; on this tree it changes little, but tussocks
against a trunk can widen the base cylinder. Volumes are validated against felled
trees in [Benchmarks](../benchmarks/qsm.md), on single-tree clouds two orders
of magnitude denser than this tile."""),
    md("## Export\n\nA cylinder table, a raycloudtools-style tree file, and meshes for Blender / CloudCompare."),
    """\
model.to_csv("tree_qsm.csv")
model.to_ply("tree_qsm.ply")        # faces coloured by branch order
model.to_obj("tree_qsm.obj")
qsm.QSM.from_csv("tree_qsm.csv").summary()["n_cylinders"]""",
]

NOTEBOOKS["07_canopy"] = [
    md("""# 7. Canopy structure

Vertical profiles and plant area from a point cloud, and gap fraction from
pulse data. For the rigorous ray-traced version see notebook 9."""),
    SETUP + """
from sylva import canopy, filters, ground

cloud = filters.voxel_downsample(sylva.read(DATA / "litch_tile.laz"), 0.05)
dtm = ground.make_dtm(ground.classify_ground_pmf(cloud), 0.5, bounds=(0, 0, 20, 20))
cloud = ground.normalize_height(cloud, dtm)""",
    TILE,
    md("""## Occupancy and the contact-frequency profile

`voxelize` counts points per voxel. `pad_profile_voxel` turns the fraction of
occupied voxels per layer into plant area density (a simplified Hosoi & Omasa
2006, assuming vertical beams and G = 0.5). Both ignore occlusion and both
depend on the point density, so they describe the cloud as much as the canopy
-- pulse data (notebook 9) is what measures the canopy."""),
    """\
veg = cloud[cloud.attrs["height"] > 0.5]
grid = canopy.voxelize(veg, 0.25)
z, pad = canopy.pad_profile_voxel(veg, voxel_size=0.25)
print("voxel grid (nx, ny, nz):", grid.shape)
print("PAI from voxel occupancy:", round(float(np.nansum(pad) * 0.25), 2),
      "  (the ray-traced plot value for Litchfield is 1.4, see the canopy benchmark)")

hb, counts = canopy.vertical_profile(veg, bin_size=0.5)
fig, ax = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
ax[0].barh(hb, counts, height=0.5, align="edge"); ax[0].set(title="points per 0.5 m", ylabel="height (m)")
ax[1].plot(grid.vertical_profile(), grid.z_levels() - grid.origin[2]); ax[1].set(title="fraction of voxels occupied")
ax[2].plot(pad, z); ax[2].set(title="PAD (m2 m-3)");""",
    md("""The savanna's structure shows in all three: a dense grass and shrub
layer below 2 m, a sparse middle, and a canopy from 8 m to 20 m. The point
profile exaggerates the lower layers, which are metres from the scanners; the
occupancy profile is flatter because a voxel counts once however many points
it holds."""),
    md("""## Canopy cover and the CHM"""),
    """\
chm = ground.make_chm(cloud, resolution=0.5)
for h in (0.5, 2.0, 5.0, 10.0):
    print(f"cover above {h:4.1f} m: {canopy.canopy_cover(chm.data, h):.2f}")""",
    md("""## Gap fraction from pulses

Gap fraction inverts the fraction of pulses that got through at each zenith
angle, so it needs pulses with their scanner origins. The tile's pulse file
cannot serve here: its rays were clipped at the tile boundary, so their origins
sit on the tile edge rather than at a scanner (notebook 8). This part uses a
synthetic scene instead, where the leaf area is known -- see
[Pulse data](../guide/pulses.md) and the [canopy
benchmark](../benchmarks/canopy.md) for whole plots read from RIEGL `.rxp`,
where Sylva's profiles match pylidar-tls-canopy to within 4 %."""),
    """\
scene = synthetic.forest()
shots = synthetic.scan(scene, origin=(10.0, 10.0, 1.5), resolution_deg=0.25)
print(shots, f"- {np.mean(shots.echo_count == 0):.0%} of pulses returned nothing")
print(f"true leaf area index of the scene: {synthetic.leaf_area(scene) / 20**2:.2f} (plus bark)")
echo_height = shots.echo_xyz()[:, 2] - synthetic.terrain_height(*shots.echo_xyz()[:, :2].T)
zen, gap = canopy.gap_fraction_zenith(shots, echo_height, min_height=1.0, zenith_edges=np.arange(0, 95, 5.0))
for method in ("hinge", "miller"):
    print(f"effective PAI ({method}): {canopy.lai_from_gap_fraction(zen, gap, method):.2f}")
plt.plot(zen, gap, "o-"); plt.gca().set(xlabel="zenith (deg)", ylabel="gap fraction", ylim=(0, 1.02));""",
    md("""Both estimators land well below the scene's 0.30, and that is the
point of the exercise: one scan from inside a scene of discrete leaf discs
misses most of the leaf area, and an *effective* PAI is a lower bound on the
real one. Several positions, or the ray-traced voxels of notebook 9, recover
more of it."""),
]

NOTEBOOKS["08_shots"] = [
    md("""# 8. Pulses and shots files

`sylva.Shots` stores pulses rather than points: an origin and direction per
pulse, with a CSR list of echo ranges. Pulses without a return stay in the
data, because they say where there was nothing."""),
    SETUP + "\nfrom sylva import Shots",
    md("""The tile's pulse file came from a raycloudtools ray cloud through
`Shots.from_ray_cloud`, then `save`. Every ray was clipped to the tile, so a
ray that ends inside carries its echo and a ray that passes through (or never
returned) is a miss. That keeps the free-space information inside the tile,
but it also moves the origins onto the tile boundary: these are not pulses
from a scanner, and anything that needs true beam geometry -- gap fraction by
zenith ring, or a scan pattern -- needs the original `.rxp` (see
[Pulse data](../guide/pulses.md))."""),
    """\
shots = Shots.load(DATA / "litch_tile_shots.parquet")
print(shots)
print("misses:", f"{np.mean(shots.echo_count == 0):.0%}",
      " echoes per pulse:", dict(enumerate(np.bincount(shots.echo_count))))
print("echo attributes:", sorted(shots.echo_attrs))""",
    md("The echo arrays are flat; `echo_start` / `echo_count` say which belong to which pulse."),
    """\
s = int(np.flatnonzero(shots.echo_count > 0)[0])
a = shots.echo_start[s]
print("pulse", s, "origin", shots.origin[s].round(2), "direction", shots.direction[s].round(3),
      "range", shots.echo_range[a].round(2))
zen, az = shots.zenith_azimuth()
first = shots.echo_rank() == 0
print("first returns:", int(first.sum()), " later returns:", int((~first).sum()))
print("zenith quartiles:", np.percentile(zen, [25, 50, 75]).round(0),
      "deg - mostly near-horizontal, because a 20 m tile is crossed by rays from the whole plot")
print("pulses with more than one echo:", int(np.sum(shots.echo_count > 1)),
      "- a ray cloud stores one echo per ray, so the multi-echo structure is already gone")""",
    """\
hit = shots.echo_count > 0
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(az[~hit][::4], zen[~hit][::4], s=0.2, c="lightskyblue", label="no return")
ax[0].scatter(az[hit][::4], zen[hit][::4], s=0.2, c="darkgreen", label="return")
ax[0].set(xlabel="azimuth (deg)", ylabel="zenith (deg)", ylim=(130, 0), title="pulses by direction")
ax[0].legend(markerscale=20, loc="lower right")
ax[1].scatter(shots.origin[::20, 0], shots.origin[::20, 1], s=0.3, c="C1")
ax[1].set(title="origins: the tile boundary, plus one real scan position",
          xlabel="x (m)", ylabel="y (m)", aspect="equal");""",
    md("""One scan position does fall inside the tile, in the corner, and keeps
its true origin: 38 % of the pulses here are its. The rest of the interior
scatter is not scan positions but rounding -- a ray cloud stores the vector to
the sensor as float32 next to double coordinates, so origins reconstructed 20 m
away land within a few centimetres of each other rather than exactly on the
scanner. `Shots.save(origin_tolerance=...)` collapses origins that close
together into one position when writing a shots file."""),
    """\
o = shots.origin
interior = (o[:, 0] > 0.05) & (o[:, 0] < 19.95) & (o[:, 1] > 0.05) & (o[:, 1] < 19.95)
uniq, counts = np.unique(np.round(o[interior], 1), axis=0, return_counts=True)
print(f"{interior.mean():.0%} of pulses start inside the tile, at {len(uniq):,} distinct rounded origins")
print("the biggest:", uniq[counts.argmax()], f"with {counts.max():,} pulses - the scan position")
print("the rest hold", f"{np.sort(counts)[-2]:,}", "pulses at most each - float32 rounding")""",
    md("Conversions: `to_pointcloud` gives the echoes as points (with `return_number`, `number_of_returns`, `range`); `from_pointcloud` and `from_ray_cloud` go the other way; `transform`, `subset` and `concatenate` behave as for point clouds."),
    """\
points = shots.to_pointcloud()
downward = shots.subset(zen > 90)
print(points, downward, sep="\\n")
print("mean echo range:", round(float(shots.echo_range.mean()), 2), "m")""",
    md("""## Shots files

`save` writes a Parquet file with one row per pulse: a scan index, two beam
angles and list columns for the echoes. A pulse without a return costs its two
angles and nothing else. Compare with storing the same pulses as a ray cloud,
where every miss has to become a far point carrying all its attributes."""),
    """\
import tempfile
tmp = Path(tempfile.mkdtemp())
shots.save(tmp / "tile.parquet")

miss = shots.echo_count == 0
far = shots.origin[miss] + 100 * shots.direction[miss]
ends = np.vstack([shots.echo_xyz(), far])
starts = np.vstack([shots.origin[shots.shot_of_echo()], shots.origin[miss]])
bound = np.r_[np.ones(shots.n_echoes, np.uint8), np.zeros(miss.sum(), np.uint8)]
off = (starts - ends).astype(np.float32)
sylva.write(sylva.PointCloud(ends, {"sx": off[:, 0], "sy": off[:, 1], "sz": off[:, 2], "bound": bound}),
            tmp / "rays.laz")
for f in ("tile.parquet", "rays.laz"):
    print(f"{f:14s} {(tmp / f).stat().st_size / 1e6:6.2f} MB")""",
    """\
info = Shots.file_info(tmp / "tile.parquet")
print({k: v for k, v in info.items() if k != "scans"})
back = Shots.load(tmp / "tile.parquet")
print("max echo position error after the float32 round trip:",
      f"{np.abs(back.echo_xyz() - shots.echo_xyz()).max() * 1000:.3f} mm")""",
    md("""Any Parquet reader opens the file (polars, pyarrow, duckdb, R arrow).
Row groups can be read one at a time with `Shots.load(path, groups=[...])`, and
`sylva.voxels.ray_voxelize` accepts the path and streams it (notebook 9). For
scans read from `.rxp`, `Shots.fill_missing` reconstructs the pulses RIEGL
leaves out of the point stream; `ScanPosition.read_shots(fill_missing=True)`
does it in the right order."""),
]

NOTEBOOKS["09_voxels"] = [
    md("""# 9. Ray-traced voxels

`sylva.voxels` traces every pulse through a voxel grid and estimates the
attenuation coefficient λ of each voxel from how far pulses got, then plant area
density as λ / G. It follows AMAPVox and the rayvoxel tool; see the
[guide](../guide/voxels.md) for the estimators."""),
    SETUP + "\nfrom sylva import Shots, voxels",
    md("""## Tracing a real tile

The Litchfield pulses, with the echoes' `classification` so the terrain can be
left out of the plant area. `occlusion=True` also records what the pulses never
reached."""),
    """\
shots = Shots.load(DATA / "litch_tile_shots.parquet")
grid = voxels.ray_voxelize(
    shots, voxel_size=0.5, bounds=((0, 0, 0), (20, 20, 18)),
    ground_class=2, leaf_classes=[4], attenuation=["fpl", "ppl"], occlusion=True,
)
print(grid)
print("raw fields:", ", ".join(grid.fields[:10]), "...")""",
    md("""`state` is 0 unobserved, 1 occluded (only reached behind a last echo),
2 empty, 3 filled. Arrays are `(nz, ny, nx)`. Here 97 % of the tile was
observed and 3 % is occluded, which is what many scan positions buy you."""),
    """\
state = grid.state
names = ["unobserved", "occluded", "empty", "filled"]
print({n: f"{np.mean(state == i):.1%}" for i, n in enumerate(names)})
print({k: round(v, 3) for k, v in grid.occlusion_profile()["total"].items()})
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
j = 15                                        # the row of voxels at y = 7.5 m
for a, (v, title, kw) in zip(ax, [
        (state[:, j, :], "state", dict(cmap="viridis", vmin=0, vmax=3)),
        (np.log10(grid.num_beams[:, j, :] + 1), "log10 beams", dict(cmap="magma")),
        (grid.num_hits[:, j, :], "echoes", dict(cmap="Greens", vmax=50))]):
    im = a.imshow(v, origin="lower", **kw); a.set(title=title, xlabel="x voxel", ylabel="z voxel")
    fig.colorbar(im, ax=a, shrink=0.8)""",
    md("""## Attenuation, and where the tile can support an estimate

FPL and PPL agree where voxels are well sampled and diverge where they are not,
which is the useful diagnostic. Before reading any profile, look at how many
beams reached each layer: the tile holds one real scan position (its corner)
plus rays from the rest of the plot clipped at the boundary, so sampling falls
away with height. Layers whose voxels see only a few tens of beams produce
large, meaningless λ -- a voxel with 20 beams, 2 echoes and short path lengths
reads as dense -- and `profile` averages over whichever voxels pass
`min_beams`, so a stricter threshold can *raise* the number rather than
settle it."""),
    """\
beams = np.median(grid.num_beams, axis=(1, 2))
z = grid.z_levels() + 0.25
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
ax[0].loglog(grid.attenuation_fpl[(grid.num_beams >= 200) & (grid.num_hits > 0)],
             grid.attenuation_ppl[(grid.num_beams >= 200) & (grid.num_hits > 0)], ".", ms=1.5, alpha=0.3)
ax[0].plot([1e-3, 40], [1e-3, 40], "k", lw=0.6)
ax[0].set(xlabel="λ FPL (1/m)", ylabel="λ PPL (1/m)", title="the two estimators (≥ 200 beams)")
ax[1].plot(beams, z, "C1")
ax[1].axvline(200, color="k", ls="--", lw=0.8)
ax[1].set(xscale="log", xlabel="beams per voxel (median)", ylabel="z (m)", title="sampling by layer")
for mb in (20, 200):
    ax[2].plot(grid.profile("pad_ppl", min_beams=mb), z, label=f"min_beams={mb}")
ax[2].set(xlabel="PAD (m2 m-3)", title="the same profile, two thresholds"); ax[2].legend()
usable = z[beams >= 200].max()
print(f"median beams per voxel stays above 200 up to {usable:.1f} m, and falls below 20 above "
      f"{z[beams >= 20].max():.1f} m")""",
    md("""Read the profile up to that height and no further. Summed over the
well-sampled layers it comes to about 0.7 -- the grass layer and the lower
canopy, and in the same range as the plot's hinge PAI of 0.87. Summed over
every layer it comes to 6.3, which is the sparsely sampled top of the tile
inventing plant area. A 20 x 20 m cut-out cannot give a plot's plant area
index either way: its rays were clipped at the boundary and the upper canopy is
barely sampled. The [canopy benchmark](../benchmarks/canopy.md) has the plot
values from whole scans, where Sylva's profiles match pylidar-tls-canopy to
within 4 %."""),
    """\
ok = beams >= 200
print("PAI over the well-sampled layers only:",
      round(float(np.nansum(grid.profile("pad_ppl", min_beams=200)[ok]) * grid.voxel_size), 2),
      " over every layer:",
      round(float(np.nansum(grid.profile("pad_ppl", min_beams=200)) * grid.voxel_size), 2))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for a, (v, title) in zip(ax, [(grid.transmittance[:, j, :], "transmittance"),
                              (grid.pad_ppl[:, j, :], "PAD (m2 m-3), same slice")]):
    im = a.imshow(v, origin="lower", cmap="bone" if "trans" in title else "YlGn",
                  vmin=0, vmax=1 if "trans" in title else 3)
    a.set(title=title, xlabel="x voxel", ylabel="z voxel"); fig.colorbar(im, ax=a, shrink=0.8)""",
    md("""## Checking the estimators against a known scene

To see whether the numbers are right, the scene has to be known. Four
synthetic scans of a scene with a known leaf area: the estimate should be the
right size, and low, because the pseudo-scanner sees each leaf disc through
only a few points."""),
    """\
scene = synthetic.forest()
positions = [(3, 3), (17, 3), (3, 17), (17, 17), (10, 10)]
sim = Shots.concatenate([
    synthetic.scan(scene, origin=(x, y, synthetic.terrain_height(x, y) + 1.5), resolution_deg=0.2)
    for x, y in positions])
sim_grid = voxels.ray_voxelize(sim, 0.5, ((0, 0, -0.5), (20, 20, 16.5)), ground_class=2,
                               leaf_classes=[4], wood_classes=[5], attenuation=["fpl", "ppl"])
pai = float(np.nansum(sim_grid.profile("pad_ppl", min_beams=20)) * 0.5)
lai = float(np.nansum(sim_grid.profile("lad_ppl", min_beams=20)) * 0.5)
print(f"PAI {pai:.2f}  LAI {lai:.2f}  true LAI of the scene {synthetic.leaf_area(scene) / 20**2:.2f}")""",
    md("""## Leaf angles

With `inclination=True`, normals of the echoes give an inclination angle
distribution per tree, and G is integrated over it and over the tree's beam
zeniths instead of assuming a spherical distribution. The synthetic leaves are
randomly oriented discs, so the result should be close to spherical with
G ≈ 0.5."""),
    """\
inc = voxels.ray_voxelize(sim, 0.5, ((0, 0, -0.5), (20, 20, 16.5)), ground_class=2,
                          leaf_classes=[4], wood_classes=[5], inclination=True)
for tid, t in inc.tree_iad.items():
    print(f"tree {tid}: leaves {t['liad_de_wit']:12s} G_leaf {t['g_leaf']:.2f}   "
          f"wood {t['wiad_de_wit']:12s} G_wood {t['g_wood']:.2f}")
t = inc.tree_iad[3]
plt.step(np.degrees(t["bin_centres"]), t["liad"], where="mid", label="leaf")
plt.step(np.degrees(t["bin_centres"]), t["wiad"], where="mid", label="wood")
plt.gca().set(xlabel="inclination of the surface normal (deg)", ylabel="fraction", title="tree 3"); plt.legend();""",
    md("## Wood volume, files and streaming\n\nQSM cylinders can be rasterised into the same grid; `write` produces an AMAPVox `.vox` file or a text table; and a shots file is voxelised without being loaded."),
    """\
from sylva import qsm
tree3 = scene[scene.attrs["tree_id"] == 3]
model = qsm.build_qsm(tree3[tree3.attrs["classification"] == 5], base_xy=(8.0, 15.0))
sim_grid.add_wood_volume(model)
print(f"wood volume in the grid {sim_grid.wood_volume.sum():.3f} m3 of {model.total_volume:.3f} m3 in the QSM")

n = grid.write("tile.vox")
print(n, "voxels written to tile.vox")
streamed = voxels.ray_voxelize(DATA / "litch_tile_shots.parquet", 0.5, ((0, 0, 0), (20, 20, 18)),
                               ground_class=2, leaf_classes=[4])
print("streamed from the file:", streamed,
      "- same echo count:", int(streamed.num_hits.sum()) == int(grid.num_hits.sum()))""",
]

NOTEBOOKS["11_coordinates"] = [
    md("""# 11. Coordinates

Coordinate reference systems on point clouds, reprojection, shifts and
rotations, and registration matrices applied to a set of scans, with
`sylva.coords` and the `PointCloud` methods `translate`, `rotate` and
`recentre`. The [guide](../guide/coordinates.md) gives the rules behind
them."""),
    SETUP + "\nfrom sylva import coords, filters\n\ncloud = sylva.read(DATA / \"litch_tile.laz\")",
    md("""## A CRS on the cloud

`sylva.read` takes the CRS of a LAS/LAZ file from its header. The tile is
stored in a local frame with its south-west corner at (0, 0), so its header
names none and `cloud.crs` is None. To have map coordinates to work with,
the tile is placed here in GDA2020 / MGA zone 52 (EPSG:7852), the zone of the
Litchfield plot, near the site. The offset is illustrative: it is not the
plot's surveyed position. A shift by whole metres is exact, so nothing is
lost on the way."""),
    """\
print("CRS read from the header:", cloud.crs)
site = coords.reproject(np.array([[130.7945, -13.1790, 0.0]]), "EPSG:7852", "EPSG:7844")   # longitude, latitude
E0, N0 = np.floor(site[0, :2])
plot = cloud.translate(E0, N0, 20.0)
plot.crs = "EPSG:7852"                     # an EPSG code, a PROJ string or WKT
info = coords.crs_info(plot.crs)
print(plot)
print(f"{info.name}: {info.proj4}")""",
    md("""`sylva.write` stores the CRS in LAS/LAZ files as an OGC WKT record,
which PDAL, LAStools, CloudCompare and QGIS read, and `sylva.read` restores
it. PLY and text files carry no CRS."""),
    """\
import tempfile
tmp = Path(tempfile.mkdtemp())
sylva.write(plot, tmp / "plot_mga.laz")
back = sylva.read(tmp / "plot_mga.laz")
print("CRS read back:", back.crs[:48], "...")
print("the same CRS:", coords.same_crs(back.crs, 7852),
      "| largest coordinate change:", np.abs(back.xyz - plot.xyz).max(), "m")
sylva.write(plot[::50], tmp / "plot.ply")
print("CRS from a PLY file:", sylva.read(tmp / "plot.ply").crs)""",
    md("""## Reprojecting

`coords.transformation` reports, before anything is moved, how one CRS
reaches another: a conversion between projections on one datum is exact, a
Helmert datum change is exact to its published parameters, and a change
between datums with no parameters between them is a *null* transformation
that is not applied."""),
    """\
import pandas as pd

targets = {"EPSG:7844": "GDA2020 latitude, longitude", "EPSG:32752": "WGS 84 / UTM zone 52S",
           "EPSG:28352": "GDA94 / MGA zone 52"}
rows = []
for code, name in targets.items():
    t = coords.transformation(plot.crs, code)
    rows.append({"from EPSG:7852 to": f"{name} ({code})", "kind": t.kind, "exact": t.exact, "changes z": t.changes_z})
pd.DataFrame(rows)""",
    """\
lonlat = coords.reproject(plot, "EPSG:7844")
print(lonlat.crs, "first point:", lonlat.xyz[0].round(7))
again = coords.reproject(lonlat, "EPSG:7852")
print(f"MGA -> latitude, longitude -> MGA: largest error {np.abs(again.xyz - plot.xyz).max() * 1e9:.1f} nm")""",
    md("""To WGS 84 the transformation is a null one: the EPSG definition of
GDA2020 gives no parameters to WGS 84, so latitude and longitude are
carried across unchanged and the datum difference is not applied.
`reproject` still returns coordinates, and says so with an
`ApproximateTransformationWarning`. MGA zone 52 and UTM zone 52S use the
same projection on nearly identical ellipsoids, so the coordinates come out
the same to within a tenth of a millimetre; the warning is the only sign
that they should not have. Where a silent error of this kind
matters, turn the warning into an error with
`warnings.simplefilter("error", coords.ApproximateTransformationWarning)`."""),
    """\
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    utm = coords.reproject(plot, "EPSG:32752")
for w in caught:
    print(f"{w.category.__name__}: {w.message}")
print("largest change from the MGA coordinates:", np.abs(utm.xyz - plot.xyz).max(), "m")""",
    md("""A datum change with published parameters is applied. Suppose the same
numbers had been delivered in GDA94 / MGA zone 52: EPSG again gives no
parameters from GDA94 to GDA2020, but GDA2020 can be written as a PROJ
string carrying the published Helmert (EPSG:8048), and the shift is then
applied. A Helmert change treats z as ellipsoidal height and moves it too;
TLS heights are local or orthometric, so keep the original z afterwards."""),
    """\
gda94 = plot.copy()
gda94.crs = "EPSG:28352"                  # the same numbers, read as GDA94
gda2020 = ("+proj=utm +zone=52 +south +ellps=GRS80 +units=m +no_defs "
           "+towgs84=-0.06155,0.01087,0.04019,-0.0394924,-0.0327221,-0.0328979,0.009994")
print(coords.transformation("EPSG:28352", gda2020).note)
shifted = coords.reproject(gda94, gda2020)
d = shifted.xyz - gda94.xyz
print("shift east, north, up (m):", d.mean(axis=0).round(3),
      f"| its variation across the tile: {d.std(axis=0).max() * 1000:.4f} mm")
print(f"and back to GDA94: largest error {np.abs(coords.reproject(shifted, 'EPSG:28352').xyz - gda94.xyz).max() * 1000:.4f} mm")
shifted.xyz[:, 2] = gda94.z               # keep the heights
shifted.crs = "EPSG:7852"                 # and label the result with the code once shifted""",
    md("""## Shifting, rotating and recentring

Map coordinates are large numbers. Float32 holds about seven significant
digits, so a northing of 8.5 million metres is stored to the nearest metre,
with errors of up to half a metre, and tools or formats that work in float32 lose the detail of the
cloud. `recentre` moves the cloud to a local origin, by default its minimum
corner rounded down to whole metres, and returns the offset;
`translate(*offset)` undoes it exactly."""),
    """\
local, offset = plot.recentre()
print("offset:", offset.tolist(), "| local extent:", local.bounds[0], "to", local.bounds[1])
for name, c in (("map", plot), ("local", local)):
    err = np.abs(c.xyz.astype(np.float32) - c.xyz).max(axis=0)
    print(f"largest float32 rounding of {name:5s} coordinates (x, y, z):", ", ".join(f"{e * 1000:.4g}" for e in err), "mm")""",
    md("""`rotate` takes degrees, counter-clockwise looking down the axis, about
the origin unless `about` names a point. Rotating map coordinates about the
origin swings the plot around the grid's (0, 0), thousands of kilometres
away, so rotate about the plot, or recentre first. The same operations are
available as 4 x 4 matrices, to compose with registration results."""),
    """\
centre = local.xyz.mean(axis=0)
turned = local.rotate(30, about=centre)
restored = turned.rotate(-30, about=centre).translate(*offset)
print("rotated by 30 deg, back, and shifted to the map again: largest error",
      np.abs(restored.xyz - plot.xyz).max(), "m")
M = coords.rotation_matrix(30, about=centre)
print("the same as a matrix: largest difference", np.abs(local.transform(M).xyz - turned.xyz).max(), "m")
swung = plot.rotate(30)
print(f"rotated about the grid origin instead, the tile moves "
      f"{np.linalg.norm(swung.xyz.mean(axis=0) - plot.xyz.mean(axis=0)) / 1000:,.0f} km")

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(local.x[::40], local.y[::40], s=0.2, c="0.7", label="local")
ax.scatter(turned.x[::40], turned.y[::40], s=0.2, c="C0", label="rotated 30 deg about the centre")
ax.set(aspect="equal", xlabel="x (m)", ylabel="y (m)")
ax.legend(markerscale=20, loc="upper right", fontsize=8);""",
    md("""## Applying registration matrices

Registration yields one matrix per scan: RiSCAN SOPs, `.DAT` files, or the
`transforms.json` of `sylva.coreg`. `coords.apply_transforms` puts the scans
into one frame with them and can merge the result. The repository holds no
per-scan files, so three scans are made from the tile: the points within
9 m of three positions, each moved into its own scanner frame by the inverse
of an SOP-like matrix (the scanner 1.5 m above the ground, a heading, and a
1.2 degree tilt). The matrices are written as RiSCAN `.DAT` files and the
scans as LAZ files, named as RiSCAN exports them."""),
    """\
positions = {"ScanPos001": (4.0, 4.0, 25.0), "ScanPos002": (16.0, 5.0, 140.0), "ScanPos003": (10.0, 16.0, 260.0)}
(tmp / "DAT").mkdir()
seen = {}
for name, (x, y, heading) in positions.items():
    sop = (coords.translation_matrix(x, y, 1.5) @ coords.rotation_matrix(heading)
           @ coords.rotation_matrix(1.2, axis="x"))
    seen[name] = filters.range_filter(cloud, origin=(x, y, 1.5), max_range=9.0)
    np.savetxt(tmp / "DAT" / f"{name}.DAT", sop)                               # 4 rows of 4 numbers
    sylva.write(seen[name].transform(np.linalg.inv(sop)), tmp / f"{name}_5cm.laz")   # scanner frame
print(sorted(p.name for p in tmp.glob("ScanPos*")), sorted(p.name for p in (tmp / "DAT").iterdir()))""",
    md("""Each file is matched to a matrix by name: `ScanPos001_5cm.laz` takes
`ScanPos001.DAT`, because the file stem starts with the matrix name followed
by a separator."""),
    """\
files = sorted(tmp.glob("ScanPos*_5cm.laz"))
merged = coords.apply_transforms(files, tmp / "DAT", merge=True)
reference = sylva.PointCloud.concatenate([seen[n] for n in positions])
print(merged)
print("points per scan:", np.bincount(merged.attrs["scan_id"]))
print(f"largest distance from the tile's own coordinates: {np.abs(merged.xyz - reference.xyz).max() * 1000:.2f} mm")

raw = [sylva.read(f) for f in files]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for i, r in enumerate(raw):
    ax[0].scatter(r.x[::30], r.y[::30], s=0.2, c=f"C{i}", label=files[i].stem)
ax[0].set(aspect="equal", title="each scan in its own frame", xlabel="x (m)", ylabel="y (m)")
ax[0].legend(markerscale=20, fontsize=8)
for i in range(len(files)):
    m = merged.attrs["scan_id"] == i
    ax[1].scatter(merged.x[m][::30], merged.y[m][::30], s=0.2, c=f"C{i}")
ax[1].set(aspect="equal", title="after apply_transforms", xlabel="x (m)");""",
    md("""The scans come back to within a millimetre of where they started. The
remaining error is the LAZ files' 1 mm coordinate scale, applied to the
points in their scanner frames before the matrices moved them back; with
clouds and matrices in memory the round trip is exact to float rounding.
`apply_transforms` also takes a mapping of clouds and matrices, a list by
position, a `transforms.json`, or a RiSCAN project, whose SOPs it applies,
and it writes the result with `out=`."""),
    """\
sops = {n: np.loadtxt(tmp / "DAT" / f"{n}.DAT") for n in positions}
scans = {n: seen[n].transform(np.linalg.inv(sops[n])) for n in positions}      # scanner frames, not rounded
in_memory = coords.apply_transforms(scans, sops)
print("in memory: largest error", max(np.abs(c.xyz - seen[n].xyz).max() for c, n in zip(in_memory, positions)), "m")""",
]

NOTEBOOKS["12_interpolation"] = [
    md("""# 12. Interpolation

Moving values between point clouds and rasters with `sylva.interpolate`:
labels computed on a thinned copy carried back to every point, terrain
models interpolated from the ground returns, and rasters read back onto the
points as attributes. The [guide](../guide/interpolation.md) describes the
methods."""),
    SETUP + "\nimport pandas as pd\nfrom sylva import filters, ground, interpolate, trees\n\ncloud = sylva.read(DATA / \"litch_tile.laz\")",
    md("""## Labels from a thinned copy

Much of the work on a plot cloud is done on a thinned copy, and the result
then has to reach every point. The tile's own `classification` (ground and
vegetation, from PMF) is known at every point, so it shows how well a
transfer does: thin to 20 cm, carry the classes back, and compare.
`"nearest"` copies the class of the nearest point of the copy;
`"majority"` takes the most common class of the `k` nearest, which is the
method for labels, since they must not be averaged."""),
    """\
thin = filters.voxel_downsample(cloud, 0.2)
truth = cloud.attrs["classification"]
print(f"{len(thin):,} points in the 20 cm copy, {len(cloud):,} in the tile")
for method, k in (("nearest", 1), ("majority", 5), ("majority", 9)):
    back = interpolate.transfer_attributes(thin, cloud, "classification", method=method, k=k)
    agree = back.attrs["classification"] == truth
    print(f"{method:8s} k = {k}: {agree.mean():.2%} of the points get their own class back")
back = interpolate.transfer_attributes(thin, cloud, "classification", method="majority", k=5)
wrong = back.attrs["classification"] != truth
print(f"of the points that do not, {np.mean(cloud.z[wrong] < 1.0):.0%} lie below 1 m")""",
    md("""The labels that do not come back lie where ground and grass meet,
which a 20 cm copy cannot resolve. On this tile the majority vote does no
better than the nearest point: the classes form large, clean regions, and
the vote only helps where isolated labels are wrong.

The same holds for trees. Segmentation (notebook 5) is one of the costlier
steps and grows with the number of points, so a common pattern is to
segment a 10 cm copy and carry `tree_id` back. Here both run from the same
stems, and the segmentation of the full tile is the reference.
`max_distance` leaves a point with no labelled point within 15 cm
unassigned (-1) rather than copying a distant label."""),
    """\
import time

cloud = ground.normalize_height(cloud, ground.make_dtm(cloud, 0.5, bounds=(0, 0, 20, 20)))
stems, _ = trees.merge_branches(cloud, trees.detect_stems(cloud))
t0 = time.perf_counter()
direct = np.asarray(trees.segment_trees(cloud, stems))
t_full = time.perf_counter() - t0

t0 = time.perf_counter()
coarse = filters.voxel_downsample(cloud, 0.1)
coarse = coarse.with_attrs(tree_id=np.asarray(trees.segment_trees(coarse, stems)))
moved = interpolate.transfer_attributes(coarse, cloud, "tree_id", method="majority", k=5, max_distance=0.15)
t_coarse = time.perf_counter() - t0

tid = moved.attrs["tree_id"]
in_tree = (direct > 0) | (tid > 0)
print(f"segmenting all {len(cloud):,} points: {t_full:.1f} s; "
      f"{len(coarse):,} points and the transfer: {t_coarse:.1f} s")
print(f"the same label as the full segmentation: {np.mean(tid == direct):.1%} of all points, "
      f"{np.mean(tid[in_tree] == direct[in_tree]):.1%} of the points either assigns to a tree")""",
    """\
slab = (cloud.y > 6.5) & (cloud.y < 8.5)
differ = slab & in_tree & (tid != direct)
fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(cloud.x[slab], cloud.attrs["height"][slab], s=0.2, c="0.85")
ax.scatter(cloud.x[differ], cloud.attrs["height"][differ], s=1.0, c="C3", label="label differs")
ax.set(title="2 m slice: where the transferred tree label differs from the full segmentation",
       xlabel="x (m)", ylabel="height (m)", aspect="equal")
ax.legend(markerscale=8, loc="upper right");""",
    md("""## Terrain models by interpolation

`ground.make_dtm` takes the lowest ground point in each cell by default.
With `method="tin"`, `"natural"` or `"idw"` it interpolates the ground
points instead: a linear surface on a Delaunay triangulation, Sibson's
natural-neighbour weights, or inverse-distance weighting. These surfaces
pass through the points, so they follow the ground returns rather than
their lowest members, and they carry the returns' noise unless the ground is
thinned first; here it is thinned to 25 cm."""),
    """\
ground_pts = filters.voxel_downsample(cloud[ground.ground_mask(cloud)], 0.25)
lowest = ground.make_dtm(cloud, 0.25, bounds=(0, 0, 20, 20))
dtms = {m: ground.make_dtm(ground_pts, 0.25, bounds=(0, 0, 20, 20), method=m) for m in ("tin", "natural", "idw")}
rows = []
for m, d in dtms.items():
    diff = d.data - lowest.data
    rows.append({"method": m, "median above lowest (m)": np.nanmedian(diff),
                 "5th percentile": np.nanpercentile(diff, 5), "95th percentile": np.nanpercentile(diff, 95)})
print(f"{len(ground_pts):,} ground points after thinning; grids of {lowest.data.shape[1]} x {lowest.data.shape[0]} cells")
pd.DataFrame(rows).round(3)""",
    md("""`make_dtm` fills the cells outside the ground points' convex hull from
the nearest interpolated cell, so its DTM has no gaps, like the default.
`interpolate.grid` keeps them, and with `max_distance` also leaves out cells
far from any ground point, which keeps the surface from bridging the ground
hidden under stems and shrubs."""),
    """\
holes = interpolate.grid(ground_pts, 0.25, method="tin", bounds=(0, 0, 20, 20), max_distance=0.5)
print(f"cells with no ground point within 0.5 m: {np.isnan(holes.data).mean():.1%}")
ext = (lowest.xmin, lowest.xmax, lowest.ymin, lowest.ymax)
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
im = ax[0].imshow(lowest.data, origin="lower", extent=ext, cmap="terrain")
fig.colorbar(im, ax=ax[0], shrink=0.8); ax[0].set_title("lowest point per cell (m)")
for a, m in zip(ax[1:3], ("tin", "natural")):
    im = a.imshow(dtms[m].data - lowest.data, origin="lower", extent=ext, cmap="RdBu_r", vmin=-0.2, vmax=0.2)
    fig.colorbar(im, ax=a, shrink=0.8); a.set_title(f"{m} minus lowest (m)")
im = ax[3].imshow(holes.data, origin="lower", extent=ext, cmap="terrain")
fig.colorbar(im, ax=ax[3], shrink=0.8); ax[3].set_title("TIN, max_distance 0.5 m")
for a in ax:
    a.set(xlabel="x (m)")""",
    md("""## Rasters onto points

`sample_rasters` reads several rasters at every point in one call and adds
each as an attribute. Sampling the DTM gives height above ground, as
`normalize_height` does. Sampling the CHM gives the canopy height of each
point's column, and the ratio of the two places a point within the canopy
above it."""),
    """\
dtm = ground.make_dtm(cloud, 0.5, bounds=(0, 0, 20, 20))
chm = ground.make_chm(cloud, 0.5, bounds=(0, 0, 20, 20))
pts = interpolate.sample_rasters(cloud, {"ground": dtm, "canopy_height": chm})
h = pts.z - pts.attrs["ground"]
print("largest difference from normalize_height:", np.abs(h - cloud.attrs["height"]).max(), "m")
tall = (pts.attrs["canopy_height"] > 10) & (cloud.attrs["classification"] == 4)
relative = h / pts.attrs["canopy_height"]
print(f"vegetation under canopy taller than 10 m: {tall.sum():,} points, "
      f"{np.mean(relative[tall] < 1 / 3):.0%} of them in the lowest third of their column")

small = ground.make_dtm(cloud, 0.5, bounds=(5, 5, 15, 15))
outside = np.isnan(interpolate.sample_raster(cloud, small, "ground").attrs["ground"])
print(f"a DTM of the inner 10 x 10 m only: {outside.mean():.0%} of the points fall outside it and get NaN")

slab = (cloud.y > 9) & (cloud.y < 11)
fig, ax = plt.subplots(figsize=(10, 4))
sc = ax.scatter(cloud.x[slab], h[slab], c=np.clip(relative[slab], 0, 1), s=0.3, cmap="viridis")
ax.set(title="2 m slice, coloured by height relative to the CHM above", xlabel="x (m)", ylabel="height (m)", aspect="equal")
fig.colorbar(sc, ax=ax, shrink=0.8);""",
]

NOTEBOOKS["13_masking"] = [
    md("""# 13. Masking

Selecting points with `sylva.masks`: by polygons read from a file, by the
raster cell under each point, by an expression over the attributes, and by
the distance to another cloud. A mask is a boolean array, one entry per
point, so masks from different sources combine with `&`, `|` and `~`. The
[guide](../guide/masking.md) has the details."""),
    SETUP + """
import json
from sylva import ground, masks

cloud = sylva.read(DATA / "litch_tile.laz")
cloud = ground.normalize_height(cloud, ground.make_dtm(cloud, 0.5, bounds=(0, 0, 20, 20)))""",
    md("""## Polygons

`read_polygons` reads GeoJSON and ESRI shapefiles. The file below holds two
subplots with a `treatment` property: a rectangle with a circular hole
around the tile's largest stem (excluded, say, for destructive sampling),
and a pentagon. Coordinates are in the tile's frame; polygons are never
reprojected, so they have to be in the cloud's frame."""),
    """\
import tempfile
tmp = Path(tempfile.mkdtemp())
t = np.linspace(0, 2 * np.pi, 33)[:-1]
hole = np.column_stack([4.8 + 1.5 * np.cos(t), 7.5 + 1.5 * np.sin(t)])[::-1].tolist()
features = [
    {"type": "Feature", "properties": {"name": "west", "treatment": "burnt"},
     "geometry": {"type": "Polygon", "coordinates": [[[1, 1], [9, 1], [9, 15], [1, 15], [1, 1]], hole + hole[:1]]}},
    {"type": "Feature", "properties": {"name": "east", "treatment": "control"},
     "geometry": {"type": "Polygon", "coordinates": [[[11, 3], [19, 3], [19, 11], [15, 19], [11, 11], [11, 3]]]}},
]
json.dump({"type": "FeatureCollection", "features": features}, open(tmp / "subplots.geojson", "w"))

subplots = masks.read_polygons(tmp / "subplots.geojson")
print(len(subplots), "features:", [f.properties for f in subplots])
which = masks.polygon_index(cloud, subplots)             # index of the polygon, -1 outside all
for i, f in enumerate(subplots):
    print(f"{f.properties['name']:5s} {np.sum(which == i):>9,} points")
burnt = masks.crop_polygons(cloud, subplots[[f.properties["treatment"] == "burnt" for f in subplots]])
print("burnt subplot:", burnt)""",
    """\
fig, ax = plt.subplots(figsize=(5.5, 5.5))
colours = np.array(["0.85", "C1", "C0"])
ax.scatter(cloud.x[::10], cloud.y[::10], s=0.2, c=colours[which[::10] + 1])
for f in subplots:
    for part in f.parts:
        ax.plot(*part.exterior.T, "k", lw=0.8)
        for h in part.holes:
            ax.plot(*h.T, "k--", lw=0.8)
ax.set(aspect="equal", xlabel="x (m)", ylabel="y (m)", title="points by subplot; the hole stays out");""",
    md("""## Raster masks

`raster_mask` tests the raster cell under each point against a range
(`min`, `max`, inclusive) or a set of `values`. With the CHM it separates
the points under tall canopy from those in the gaps. Points outside the
raster or over NaN cells are never kept."""),
    """\
chm = ground.make_chm(cloud, 0.5, bounds=(0, 0, 20, 20))
under_tall = masks.raster_mask(cloud, chm, min=10)
in_gaps = masks.raster_mask(cloud, chm, max=2)
print(f"under canopy of 10 m or more: {under_tall.mean():.0%} of the points; in cells below 2 m: {in_gaps.mean():.0%}")
low_veg = masks.expression(cloud, "0.3 < height < 2 & classification == 4")
print(f"vegetation between 0.3 and 2 m: {np.mean(under_tall[low_veg]):.0%} of it under tall canopy, "
      f"{np.mean(in_gaps[low_veg]):.0%} in the gaps")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
im = ax[0].imshow(chm.data, origin="lower", extent=(chm.xmin, chm.xmax, chm.ymin, chm.ymax), cmap="YlGn")
fig.colorbar(im, ax=ax[0], shrink=0.8); ax[0].set(title="CHM (m)", xlabel="x (m)", ylabel="y (m)")
m = low_veg & under_tall
ax[1].scatter(cloud.x[low_veg & ~m][::4], cloud.y[low_veg & ~m][::4], s=0.2, c="C1", label="elsewhere")
ax[1].scatter(cloud.x[m][::4], cloud.y[m][::4], s=0.2, c="C2", label="under canopy >= 10 m")
ax[1].set(aspect="equal", title="vegetation 0.3-2 m", xlabel="x (m)"); ax[1].legend(markerscale=20, fontsize=8);""",
    md("""## Expressions

`cloud.where`, `masks.expression` and `masks.crop_expression` take a
condition over `x`, `y`, `z` and the attributes, which the Rust core parses
and evaluates; the text is never run as Python. Comparisons can be chained,
`in` tests membership, and `&` and `|` bind more loosely than comparisons,
so no parentheses are needed where NumPy would need them."""),
    """\
veg = cloud.where("height > 2 & classification != 2")
numpy_way = cloud[(cloud.attrs["height"] > 2) & (cloud.attrs["classification"] != 2)]
print(veg, "| the same as the NumPy mask:", np.array_equal(veg.xyz, numpy_way.xyz))
print("chained:", masks.expression(cloud, "1.3 <= height < 5").sum(),
      "| membership:", masks.expression(cloud, "classification in (3, 4, 5)").sum(),
      "| arithmetic:", masks.expression(cloud, "x + y < 10 & not (height > 1)").sum())
try:
    cloud.where("hieght > 2")
except ValueError as err:
    print(err)""",
    md("""## Distance to another cloud

`near` keeps the points within a distance of another cloud and
`difference` returns those beyond it, which is change detection between
two epochs of the same scene. Here the second epoch is a copy of the tile
with a 3 x 3 m block between 0.5 and 6 m removed, which takes a section of
a stem and the shrubs beside it, as if they had been cut, and every
remaining point moved by 5 mm of random noise, as a
re-survey would. `difference(before, after, d)` should find the block and
nothing else, provided `d` is above the noise and the point spacing."""),
    """\
block = (cloud.x > 14) & (cloud.x < 17) & (cloud.y > 14) & (cloud.y < 17) & (cloud.z > 0.5) & (cloud.z < 6)
rng = np.random.default_rng(1)
kept = cloud[~block]
after = sylva.PointCloud(kept.xyz + rng.normal(0, 0.005, kept.xyz.shape), kept.attrs)
print(f"{block.sum():,} points removed")


def in_block(c):
    return (c.x > 14) & (c.x < 17) & (c.y > 14) & (c.y < 17) & (c.z > 0.5) & (c.z < 6)


for d in (0.01, 0.05, 0.10):
    lost = masks.difference(cloud, after, d)
    print(f"d = {d * 100:4.0f} cm: {len(lost):>9,} points found, {in_block(lost).sum():,} of them in the block")
print("appeared (after against before, 5 cm):", len(masks.difference(after, cloud, 0.05)))""",
    md("""At 1 cm the noise itself reads as change. At 5 cm only points of the
block are found, but not all of them: a removed point within 5 cm of a
point that stayed, on the faces of the block, is matched to that neighbour.
A larger distance loses more of the block's edge, so choose it just above
the registration error and the point spacing."""),
    """\
lost = masks.difference(cloud, after, 0.05)
missed = block & ~masks.near(cloud, lost, 1e-6)
fig, ax = plt.subplots(figsize=(6, 5))
side = (cloud.y > 14) & (cloud.y < 17)
ax.scatter(cloud.x[side & ~block], cloud.z[side & ~block], s=0.2, c="0.8")
ax.scatter(lost.x, lost.z, s=0.4, c="C3", label="found by difference")
ax.scatter(cloud.x[missed], cloud.z[missed], s=0.8, c="C0", label="removed but matched to a neighbour")
ax.set(xlim=(12, 19), ylim=(-0.5, 8), xlabel="x (m)", ylabel="z (m)", title="the removed block, side view")
ax.legend(markerscale=10, fontsize=8);""",
]


def build(name: str, cells: list, execute: bool = True) -> None:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = [nbformat.v4.new_markdown_cell(c) if isinstance(c, md) else nbformat.v4.new_code_cell(c) for c in cells]
    if execute:
        import tempfile
        with tempfile.TemporaryDirectory() as work:      # files the notebooks write stay out of the repo
            (Path(work) / "data").symlink_to(HERE / "data")   # ... but data is read from the repo
            NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": work}}).execute()
    for cell in nb.cells:
        cell.pop("id", None)
        if cell.cell_type == "code":
            cell.metadata.pop("execution", None)
    nbformat.write(nb, HERE / f"{name}.ipynb", version=4)
    print("wrote", name)


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(NOTEBOOKS)
    for name in wanted:
        build(name, NOTEBOOKS[name])
