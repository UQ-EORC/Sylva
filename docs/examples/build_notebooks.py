"""Generate and execute the example notebooks.

    python docs/examples/build_notebooks.py            # all
    python docs/examples/build_notebooks.py 05_trees   # one

Each notebook is defined below as a list of cells: a string starting with
``#`` followed by a space, or with plain prose, is Markdown when passed
through ``md()``; everything else is code. They are saved with their outputs
so the documentation does not have to execute them.

Most notebooks run on the real TLS tile in ``data/`` (20 x 20 m of the TERN
Litchfield savanna plot, see ``make_litch_subset.py``). Where a known answer
is needed -- registration with a known transform, a QSM against a tree's own
cylinders, leaf area and leaf angles against a known stand -- they use
``synthetic.tree_model`` or ``synthetic.plot`` scanned by ``synthetic.scan``
with a finite beam instead, and say so.
Notebooks 14 to 17 (change detection, airborne lidar, full waveforms and the
synthetic data itself) run entirely on synthetic scenes, whose truth every
result is checked against.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).parent


class md(str):
    pass


# The figure style shared by every notebook (sylva.mplstyle): sizes, fonts, the
# colour-blind-safe Okabe-Ito colours as C0 to C7, viridis for magnitudes, and
# DIVERGING, centred on zero, for signed differences. Classes keep one colour
# throughout, and arbitrary labels (tree ids, clusters) cycle through LABELS.
STYLE = """\
plt.style.use("sylva.mplstyle")     # shared figure style; C0 to C7 are the Okabe-Ito colours
GROUND, WOOD, LEAF, GRASS, CONTEXT = "#997A5C", "#4D2B12", "#009E73", "#E69F00", "0.8"   # the same in every notebook
DIVERGING = "RdBu_r"                # for signed differences, centred on zero
LABELS = ListedColormap(["#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#DDCC77",
                         "#CC6677", "#882255", "#AA4499"])   # tree ids and clusters: Paul Tol's muted set
"""

SETUP = """\
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sylva
from sylva import synthetic

DATA = Path("data")          # the Litchfield tile, cut by make_litch_subset.py
""" + STYLE

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
fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
order = np.argsort(cloud.z[::4])                      # highest points drawn last
top = cloud[::4][order]
s = ax[0].scatter(top.x, top.y, c=top.z, s=0.2)
ax[0].set(title="Top view, coloured by height", xlabel="x (m)", ylabel="y (m)", aspect="equal")
fig.colorbar(s, ax=ax[0], label="z (m)", shrink=0.9)
slab = (cloud.y > 9) & (cloud.y < 11)
is_ground = cloud.attrs["classification"] == 2
ax[1].scatter(cloud.x[slab & ~is_ground], cloud.z[slab & ~is_ground], s=0.2, c=LEAF, label="vegetation (class 4)")
ax[1].scatter(cloud.x[slab & is_ground], cloud.z[slab & is_ground], s=0.6, c=GROUND, label="ground (class 2)")
ax[1].set(title="A 2 m slice at y = 9 to 11 m", xlabel="x (m)", ylabel="z (m)", aspect="equal")
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
fig, ax = plt.subplots(1, 3, figsize=(10, 3.7), sharex=True, sharey=True)
for a, (name, c) in zip(ax, {"crop_box (z > 2 m)": box, "crop_cylinder (r = 5 m)": plot,
                             "range_filter (4 to 8 m)": shell}.items()):
    a.scatter(cloud.x[::20], cloud.y[::20], s=0.2, c=CONTEXT)
    a.scatter(c.x[::8], c.y[::8], s=0.2, c="C0")
    a.set(title=f"{name}: {len(c):,} points", aspect="equal", xlabel="x (m)")
ax[0].set_ylabel("y (m)")
fig.suptitle("Points kept (blue) by each crop, top view");""",
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
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True, sharey=True)
for a, keep, name in ((ax[0], sor, "statistical (k = 8, 2 sd)"), (ax[1], ror, "radius (0.25 m, 4 neighbours)")):
    drop = slab & ~keep
    a.scatter(thinned.x[slab & keep], thinned.z[slab & keep], s=0.2, c=CONTEXT, label="kept")
    a.scatter(thinned.x[drop], thinned.z[drop], s=4 if drop.sum() < 100 else 1.2, c="C3", label="dropped")
    a.set(title=f"{name}: {int(drop.sum()):,} dropped", xlabel="x (m)", ylim=(-0.5, 8))
ax[0].set_ylabel("z (m)")
ax[0].legend(markerscale=4, loc="upper right")
fig.suptitle("Points dropped by each outlier filter, in a 2 m slice at y = 9 to 11 m")
print("of the points the statistical filter drops,", f"{np.mean(thinned.z[~sor] < 1.0):.0%}",
      "are below 1 m; for the radius filter,", f"{np.mean(thinned.z[~ror] < 1.0):.0%}")""",
    md("""## Local geometry

PCA over the `k` nearest neighbours gives normals and the planarity / linearity
the wood filter uses. On a real stem, bark is locally planar and the trunk
is linear at metre scale, while the grass layer is neither."""),
    """\
stem = filters.crop_cylinder(cloud, (4.8, 7.5), radius=1.2, zmin=0.5, zmax=8.0)
planarity, linearity = filters.planarity_linearity(stem, k=20)
fig, ax = plt.subplots(1, 2, figsize=(6, 5), sharey=True)
for a, v, name in ((ax[0], planarity, "Planarity"), (ax[1], linearity, "Linearity")):
    sc = a.scatter(stem.x, stem.z, c=v, s=0.6, vmin=0, vmax=1)
    a.set(title=name, xlabel="x (m)", aspect="equal")
ax[0].set_ylabel("z (m)")
fig.colorbar(sc, ax=ax, shrink=0.8, label="value (0 to 1)");""",
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
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(above.x[labels < 0], above.y[labels < 0], s=0.5, c=CONTEXT)
ax.scatter(above.x[labels >= 0], above.y[labels >= 0], c=labels[labels >= 0] % LABELS.N, s=0.5,
           cmap=LABELS, vmin=0, vmax=LABELS.N)
ax.set(aspect="equal", xlabel="x (m)", ylabel="y (m)", title="Clusters above 2 m, one colour each (grey: none)");""",
]

NOTEBOOKS["03_registration"] = [
    md("""# 3. Registration

Aligning scans: `kabsch` for known correspondences (reflector targets), `icp`
for clouds, and `merge_scans` to bring everything into one frame."""),
    SETUP + "\nfrom sylva import filters, registration as reg",
    md("""Two scans of one synthetic plot (`synthetic.plot`: 20 broadleaf and
eucalypt trees on rough, sloping ground with shrubs, grass, logs and stumps),
each from its own position with the model of a RIEGL VZ-400, 5 mm of range
noise and mixed pixels. Each scan sees its own side of every stem and crown,
so the two clouds overlap only in part, as real scans do. The second is
then moved by a known rigid transform, which registration should undo."""),
    """\
p = synthetic.plot(size=20, density=500, archetypes={"broadleaf": 1, "eucalypt": 1}, seed=1)
scans = []
for k, (x, y) in enumerate([(6, 8), (14, 12)]):
    shots = synthetic.scan(p.points, origin=(x, y, 1.5 + p.ground_height(x, y)), resolution_deg=0.08,
                           scanner="vz400", range_noise=0.005, seed=k)
    scans.append(filters.voxel_downsample(shots.to_pointcloud(), 0.05))
truth = reg.translation(0.6, -0.4, 0.15) @ reg.rotation_z(6.0)
scan_a, scan_b = scans[0], scans[1].transform(np.linalg.inv(truth))
d, _ = filters.knn(scans[0].xyz, scans[1].xyz, 1)
print(f"scan A: {len(scan_a):,} points, scan B: {len(scan_b):,} points (5 cm); "
      f"{np.mean(d[:, 0] < 0.1):.0%} of scan B lies within 10 cm of scan A")""",
    md("""## Targets: Kabsch

With matched points the transform is closed-form. Here the targets are
reflectors on the four largest stems at breast height, each located in each
scan with 1 cm of error."""),
    """\
big = np.argsort(p.trees["dbh"])[-4:]
targets_a = np.c_[p.trees["x_bh"][big], p.trees["y_bh"][big], p.trees["z"][big] + 1.3]
rng = np.random.default_rng(0)
targets_b = (np.linalg.inv(truth) @ np.c_[targets_a, np.ones(4)].T).T[:, :3] + rng.normal(0, 0.01, (4, 3))
coarse = reg.kabsch(targets_b, targets_a)


def report(name, T, info=None):
    err = np.linalg.norm((T @ np.linalg.inv(truth))[:3, 3])
    fit = f"rmse {info['rmse']:.4f} m  iterations {info['iterations']:3d}  " if info else ""
    print(f"{name:28s} {fit}translation error {err * 1000:5.1f} mm")


report("targets", coarse)""",
    md("""## Clouds: ICP

ICP refines a starting guess as long as it lies within
`max_correspondence_distance`. From no guess at all (the identity), it gets
within centimetres; point-to-plane converges in fewer iterations, on
surfaces such as ground and stems. From the targets, with a tighter search
distance, it improves on them."""),
    """\
report("point-to-point, identity", *reg.icp(scan_b, scan_a, max_correspondence_distance=1.0))
report("point-to-plane, identity", *reg.icp(scan_b, scan_a, max_correspondence_distance=1.0, method="plane"))
report("point-to-plane, targets", *reg.icp(scan_b, scan_a, init=coarse, max_correspondence_distance=0.3, method="plane"))""",
    md("""## Partial overlap

What one scan sees and the other does not (the far side of a stem, a crown
seen only from below) drags the solution: those points are paired with
whatever lies nearest, which is the wrong surface. `trim` keeps only the
closest fraction of correspondences each iteration. Trimming throws
information away, so start it from a coarse alignment (targets, or an
untrimmed run): from far off, the closest pairs are all on the ground and
the solution can slide along it."""),
    """\
T, info = reg.icp(scan_b, scan_a, init=coarse, max_correspondence_distance=0.3, method="plane", trim=0.7)
report("point-to-plane, trim 0.7", T, info)""",
    """\
fig, ax = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
for a, b, title in ((ax[0], scan_b, "Before: scan B as delivered"), (ax[1], scan_b.transform(T), "After trimmed ICP")):
    for c, colour, name in ((scan_a, "C0", "scan A"), (b, "C1", "scan B")):
        band = (c.z > p.ground_height(c.x, c.y) + 1.0) & (c.z < p.ground_height(c.x, c.y) + 1.6)   # stems at breast height
        a.scatter(c.x[band], c.y[band], s=0.5, c=colour, label=name)
    a.set(title=title, aspect="equal", xlabel="x (m)", xlim=(0, 20), ylim=(0, 20))
ax[0].set_ylabel("y (m)")
ax[1].legend(markerscale=10, loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.suptitle("Points 1 to 1.6 m above the ground, from above: stems and shrubs");""",
    md("""No untrimmed run gets much below a centimetre, wherever it starts:
a fifth of scan B has no counterpart in scan A, and those points pull the
solution towards the wrong surfaces. Trimmed and started from the targets,
ICP comes within a millimetre of the true transform, well inside the 5 mm
range noise."""),
    md("## Merging\n\n`merge_scans` applies one transform per scan and records where each point came from."),
    """\
merged = reg.merge_scans([scan_a, scan_b], [np.eye(4), T])
print(merged.without(*[a for a in merged.attrs if a != "scan_id"]), np.bincount(merged.attrs["scan_id"]))""",
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
from matplotlib.patches import Patch

ext = (dtm.xmin, dtm.xmax, dtm.ymin, dtm.ymax)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
ax[0].imshow(np.where(np.abs(diff) <= 0.5, 1.0, np.nan), origin="lower", extent=ext,
             cmap=ListedColormap([CONTEXT]))                          # the cells where the two agree
im = ax[0].imshow(np.where(np.abs(diff) > 0.5, diff, np.nan), origin="lower", extent=ext,
                  cmap=DIVERGING, vmin=-6, vmax=6)
ax[0].axhline(y0, color="k", lw=0.8, ls="--")
ax[0].set(title="CSF terrain minus PMF terrain", xlabel="x (m)", ylabel="y (m)", aspect="equal")
ax[0].legend(handles=[Patch(color=CONTEXT, label="within 0.5 m"),
                      plt.Line2D([], [], color="k", lw=0.8, ls="--", label="row shown right")],
             loc="lower left")
fig.colorbar(im, ax=ax[0], shrink=0.85, label="difference (m)")
ax[1].scatter(cloud.x[slab], cloud.z[slab], s=0.3, c=CONTEXT, label="all points")
for m, name, c in ((pmf, "PMF ground", "C0"), (csf, "CSF ground", "C3")):
    ax[1].scatter(cloud.x[slab & m], cloud.z[slab & m], s=2.0, c=c, label=name)
ax[1].set(title=f"Side view of the row at y = {y0:.1f} m (1 m thick)", xlabel="x (m)", ylabel="z (m)", ylim=(-1, 12))
ax[1].legend(markerscale=4, loc="upper left");""",
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
fig, ax = plt.subplots(1, 3, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1, 1.15]})
for a, r, title, label in ((ax[0], dtm, "Terrain (DTM)", "elevation (m)"),
                           (ax[1], chm, "Canopy height (CHM)", "height above ground (m)")):
    im = a.imshow(r.data, origin="lower", extent=(r.xmin, r.xmax, r.ymin, r.ymax))
    a.set(title=title, xlabel="x (m)", aspect="equal"); fig.colorbar(im, ax=a, shrink=0.8, label=label)
ax[0].set_ylabel("y (m)")
ax[2].scatter(cloud.x[slab], cloud.attrs["height"][slab], s=0.3, c=LEAF)
ax[2].set(title=f"Heights in the row at y = {y0:.1f} m", xlabel="x (m)", ylabel="height (m)");""",
    md("""Rasters export to an ESRI ASCII grid (with a `.prj` when the CRS is
known), or to GeoTIFF with `dtm.to_geotiff("dtm.tif", crs="EPSG:28352")`.
Pass the same `bounds` on every date so a time series lines up."""),
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
colour = shuffle[labels] % LABELS.N
fig, ax = plt.subplots(1, 2, figsize=(10, 4.8))
ax[0].scatter(cloud.x[~m][::10], cloud.y[~m][::10], s=0.2, c=CONTEXT)
ax[0].scatter(cloud.x[m][::4], cloud.y[m][::4], c=colour[m][::4], s=0.3, cmap=LABELS, vmin=0, vmax=LABELS.N)
for t in stems:
    ax[0].add_patch(plt.Circle((t.x, t.y), max(t.dbh / 2, 0.2), fill=False, color="k", lw=1.0))
    ax[0].annotate(str(t.tree_id), (t.x + 0.35, t.y + 0.35), fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
ax[0].set(title="Trees from above, with stems and their ids (grey: no tree)", xlabel="x (m)", ylabel="y (m)", aspect="equal")
ax[1].scatter(cloud.x[m][::4], cloud.attrs["height"][m][::4], c=colour[m][::4], s=0.3, cmap=LABELS, vmin=0, vmax=LABELS.N)
ax[1].set(title="The same trees from the side", xlabel="x (m)", ylabel="height above ground (m)", aspect="equal");""",
    md("""There is no field inventory for this tile, so treat the table as a
demonstration: detection and segmentation are scored against manually
segmented plots (including Litchfield) in
[Benchmarks](../benchmarks/trees.md). Where labels have to be right, correct
them by hand in [Segfix](https://github.com/UQ-EORC/Segfix), which reads
and writes the `tree_id` column:"""),
    """\
sylva.write(cloud.with_attrs(tree_id=labels.astype("int32")), "tile_trees.laz")
#  segfix opens this as it is: answer "Unassigned" when it asks what -1 means""",
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
    md("""## Taper and crown shape

`dbh_profile` fits a circle to a thin slice at each height, using the points
within `search_radius` of the stem (0.75 m by default). On the tallest tree
one slice fails: at 2 m the search radius takes in a shrub beside the trunk,
and the circle is fitted through both, 1.3 m wide. A search radius of 0.4 m,
still well above the trunk's radius, leaves the shrub out. The widening at
7 m is real: the stem forks there."""),
    """\
big = max(stems, key=lambda t: t.height)
heights = np.arange(0.5, 8.0, 0.5)
diam = trees.dbh_profile(cloud, (big.x, big.y), heights=heights)
tight = trees.dbh_profile(cloud, (big.x, big.y), heights=heights, search_radius=0.4)
shape = trees.crown_shape(cloud[labels == big.tree_id], base_xy=(big.x, big.y))
print(f"tree {big.tree_id}: DBH {big.dbh:.3f} m, height {big.height:.1f} m, "
      f"crown volume {shape['volume']:.0f} m3, asymmetry {shape['asymmetry']:.2f}")
print("diameter at 2 m:", round(float(diam[3, 1]), 2), "m with a 0.75 m search radius,",
      round(float(tight[3, 1]), 2), "m with 0.4 m")

fig, ax = plt.subplots(1, 3, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1, 0.9]})
ax[0].plot(100 * diam[:, 1], diam[:, 0], "o-", c="C0", label="search radius 0.75 m (default)")
ax[0].plot(100 * tight[:, 1], tight[:, 0], "s--", c="C1", ms=3.5, label="search radius 0.4 m")
ax[0].set(xlabel="stem diameter (cm)", ylabel="height above ground (m)", title=f"Taper of tree {big.tree_id}")
ax[0].legend(loc="center right")
r = np.hypot(cloud.x - big.x, cloud.y - big.y)
at2 = (np.abs(cloud.attrs["height"] - 2.0) < 0.05) & (r < 1.0)
ax[1].scatter(cloud.x[at2] - big.x, cloud.y[at2] - big.y, s=3, c="k")
for radius, colour in ((0.75, "C0"), (0.4, "C1")):
    ax[1].add_patch(plt.Circle((0, 0), radius, fill=False, ec=colour, ls="--", lw=1.2))
ax[1].set(aspect="equal", xlim=(-1, 1), ylim=(-1, 1), xlabel="x from the stem (m)", ylabel="y from the stem (m)",
          title="The slice at 2 m, from above,\\nwith the two search radii")
sel = labels == big.tree_id
ax[2].scatter(cloud.x[sel], cloud.attrs["height"][sel], s=0.3, c=LEAF)
ax[2].axhline(2.0, c="k", lw=0.8, ls="--")
ax[2].set(xlabel="x (m)", ylabel="height above ground (m)", title=f"Tree {big.tree_id} from the side", aspect="equal");""",
]

NOTEBOOKS["06_qsm"] = [
    md("""# 6. Quantitative structure models

From the points of one tree to a connected set of cylinders with volumes and
branch orders. This one starts on a synthetic tree whose every cylinder and
leaf is known, scanned as a terrestrial scanner would scan it, so the model can
be checked against the exact answer; a real tree from the Litchfield tile
follows."""),
    SETUP + """
from sylva import filters, leaves, qsm

truth = synthetic.tree_model("broadleaf", dbh=0.35, height=14.0, seed=4)
print(f"truth: DBH {truth.dbh:.3f} m, height {truth.height:.1f} m, wood volume {truth.qsm.total_volume:.3f} m3 "
      f"(stem {truth.qsm.stem_volume:.3f} m3), leaf area {truth.leaf_area:.0f} m2 in {len(truth.leaves['area']):,} leaves")""",
    md("""`synthetic.tree_model` grows a tree from an archetype (here a broadleaf
tree whose stem forks at 80 % of its height) with pipe-model branch radii and
leaves of known size and angle; its wood is a table of cylinders, `truth.qsm`.
The tree is scanned from three positions 8 m away with the model of a RIEGL
VZ-400 (0.35 mrad beam, 7 mm exit diameter) at 0.04 degrees, with 5 mm of
range noise and mixed pixels, so the echoes carry the errors of a real scan.
They keep the true class of the surface they came from. The
[synthetic data guide](../guide/synthetic.md) describes the models."""),
    """\
parts = []
for k, (x, y) in enumerate([(8, 0), (-4, 7), (-4, -7)]):
    shots = synthetic.scan(truth.points, origin=(x, y, 1.5), resolution_deg=0.04, scanner="vz400",
                           range_noise=0.005, seed=k)
    parts.append(shots.to_pointcloud())
tree = filters.voxel_downsample(sylva.PointCloud.concatenate(parts), 0.01)
print(tree.without(*[a for a in tree.attrs if a not in ("classification", "range_spread")]))
print(f"mixed pixels (hits spread over more than 5 cm in range): {np.mean(tree.attrs['range_spread'] > 0.05):.1%} of the echoes")""",
    md("""## Leaf / wood separation

`wood_points` combines local anisotropy with a topological cue: points that many
shortest paths from the base to the crown pass through are wood. The echoes
know which surface they came from, so the filter can be checked both ways:
how many of the points it keeps are wood, and how many of the wood echoes it
keeps."""),
    """\
wood = qsm.wood_points(tree, voxel_size=0.02)
is_wood = tree.attrs["classification"] == 5
true_wood = tree[is_wood]
d, _ = filters.knn(true_wood.xyz, wood.xyz, 1)
back, _ = filters.knn(wood.xyz, true_wood.xyz, 1)
print(f"{len(wood):,} wood points kept: {np.mean(d[:, 0] < 0.02):.1%} of them on wood echoes; "
      f"{np.mean(back[:, 0] < 0.02):.1%} of the wood echoes kept")

on = d[:, 0] < 0.02                          # kept points that lie on wood echoes
fig, ax = plt.subplots(1, 2, figsize=(7.5, 5), sharex=True, sharey=True)
for a, xz, wood_mask in ((ax[0], tree, is_wood), (ax[1], wood, on)):
    slab = np.abs(xz.y - truth.base[1]) < 0.3           # a 60 cm slice through the stem
    a.scatter(xz.x[slab & wood_mask], xz.z[slab & wood_mask], s=0.3, c=WOOD, label="wood")
    a.scatter(xz.x[slab & ~wood_mask], xz.z[slab & ~wood_mask], s=0.3, c=LEAF, label="leaf")
    a.set(aspect="equal", xlabel="x (m)")
ax[0].set(title="All echoes, by true class", ylabel="z (m)")
ax[1].set(title="Points kept by wood_points,\\nby true class")
fig.suptitle("A 60 cm slice through the stem")
ax[1].legend(markerscale=12, loc="upper left", bbox_to_anchor=(1.0, 1.0));""",
    md("""`wood_points` keeps nearly every wood echo, but most of what it keeps
are leaf echoes close to the twigs: this tree's planophile leaves sit along
its terminal branches, and from a few metres away a twig with its leaves
looks like a thicker branch.

## Cylinder model

The model is fitted to the points `wood_points` kept, and, to separate the
two sources of error, also to the true wood echoes; both are compared with
the tree's own cylinders, order by order."""),
    """\
import pandas as pd

model = qsm.build_qsm(wood, base_xy=tuple(truth.base[:2]))


def volume_by_order(m):
    v = np.pi * m.column("radius") ** 2 * m.column("length")
    order = np.minimum(m.column("branch_order").astype(int), 3)
    return np.bincount(order, weights=v, minlength=4)


on_wood = qsm.build_qsm(true_wood, base_xy=tuple(truth.base[:2]))
rows = {"QSM, wood_points": volume_by_order(model), "QSM, true wood echoes": volume_by_order(on_wood),
        "truth": volume_by_order(truth.qsm)}
table = pd.DataFrame(1000 * np.array(list(rows.values())), index=list(rows),
                     columns=["stem", "order 1", "order 2", "order 3 and up"])
table["total"] = table.sum(axis=1)
print(f"DBH: model {model.dbh:.3f} m, truth {truth.dbh:.3f} m; "
      f"fitted to points: {model.metrics()['measured_volume_fraction']:.0%} of the volume")
table.round(0).astype(int).rename_axis("wood volume (L)")""",
    md("""The stem comes within 6 to 10 % of the truth either way, and the DBH
within 4 mm; even on the true wood echoes the model is a little too wide,
since the beam footprint and mixed pixels widen what a scan sees. The
first-order branches show where the larger error comes from: fitted to the
true wood echoes they are 12 % over, fitted to the points `wood_points` kept
60 % over, because the leaf echoes along the twigs make the branches look
thicker. The finer orders come out close to the truth, in part because the
model's taper and pipe-model priors fill in what the scans did not resolve.
Side by side:"""),
    """\
from matplotlib.collections import PolyCollection


ORDER_COLOURS = ["C0", "C1", "C2", "C4"]


def side_view(m, ax):
    # Each cylinder drawn at its true width: the outline of its side view, coloured by order.
    a, b = m.start[:, [0, 2]], m.end[:, [0, 2]]
    d = b - a
    n = np.c_[-d[:, 1], d[:, 0]] / np.maximum(np.hypot(*d.T), 1e-9)[:, None] * m.column("radius")[:, None]
    colours = [ORDER_COLOURS[min(o, 3)] for o in m.column("branch_order").astype(int)]
    ax.add_collection(PolyCollection(np.stack([a + n, b + n, b - n, a - n], axis=1),
                                     facecolor=colours, edgecolor=colours, lw=0.3))
    ax.autoscale_view()


fig, ax = plt.subplots(1, 2, figsize=(7.5, 5.5), sharex=True, sharey=True)
for a, m, title in ((ax[0], truth.qsm, "The tree's own cylinders"), (ax[1], model, "The model, from the scans")):
    side_view(m, a)
    a.set(aspect="equal", xlabel="x (m)", title=f"{title}\\n{len(m):,} cylinders")
ax[0].set_ylabel("z (m)")
for o, name in enumerate(("0 (stem)", "1", "2", "3 and up")):
    ax[1].plot([], [], c=ORDER_COLOURS[o], lw=4, label=name)
ax[1].legend(title="branch order", loc="upper left", bbox_to_anchor=(1.0, 1.0));""",
    md("""## Leaf area and leaf angles

The leaf echoes give the leaf area the scans saw, voxel by voxel
(`leaves.leaf_area_density`), and the leaf angle distribution from their
local normals (`leaves.leaf_angle_distribution`); both have an exact answer
here."""),
    """\
foliage = tree[tree.attrs["classification"] == 4]
area = leaves.leaf_area_density(foliage, voxel_size=0.25)
angles = leaves.leaf_angle_distribution(foliage)
print(f"leaf area: {area.total_area:.0f} m2 seen, {truth.leaf_area:.0f} m2 true ({area.total_area / truth.leaf_area:.0%})")
print(f"mean leaf angle: {angles.mean_deg:.0f} deg measured ({angles.de_wit}), "
      f"{truth.leaf_angles().mean_deg:.0f} deg true (planophile)")""",
    md("""The scans see about four fifths of the leaf area: leaves inside the
crown are hidden behind others from all three positions. The leaf angles
come out much steeper than the truth and close to a uniform distribution:
normals estimated from a few noisy echoes on a 10 cm leaf, some of them
mixed with the leaf behind, scatter in every direction. Both numbers are
the kind of check a real scan never offers."""),
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
nearly two thirds of the volume lies in the trunk and the main limbs, which
were fitted to points, but only 4 % of the length was, so the finer branches
come from the taper and pipe-model priors. The stem volume is supported by
the data here; a third of the total volume, and the branch length, are not. The height cut keeps the
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
fig, ax = plt.subplots(1, 3, figsize=(10, 3.8), sharey=True)
ax[0].barh(hb, np.asarray(counts) / 1000, height=0.5, align="edge", color="C0")
ax[0].set(title="Points", xlabel="points per 0.5 m layer (thousands)", ylabel="height above ground (m)")
ax[1].plot(grid.vertical_profile(), grid.z_levels() - grid.origin[2])
ax[1].set(title="Voxel occupancy", xlabel="fraction of 0.25 m voxels occupied")
ax[2].plot(pad, z)
ax[2].set(title="Plant area density", xlabel="PAD (m² m⁻³)")
fig.suptitle("Three vertical profiles of the same tile");""",
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
synthetic plot instead, where the leaf area is known -- see
[Pulse data](../guide/pulses.md) and the [canopy
benchmark](../benchmarks/canopy.md) for whole plots read from RIEGL `.rxp`,
where Sylva's profiles match pylidar-tls-canopy to within 4 %.

The plot (`synthetic.plot`) is 50 m square, so that pulses at 57.5 degrees
reach the top of the canopy before they leave it: 125 broadleaf and
eucalypt trees with shrubs, grass and dead wood, scanned once from its centre
with the model of a RIEGL VZ-400, which sees from 30 degrees zenith down.
The truth is the trees' leaf area, from the leaves the generator placed."""),
    """\
p = synthetic.plot(size=50, density=500, archetypes={"broadleaf": 1, "eucalypt": 1}, seed=1)
x, y = 25.0, 25.0
shots = synthetic.scan(p.points, origin=(x, y, 1.5 + p.ground_height(x, y)), resolution_deg=0.1,
                       scanner="vz400", range_noise=0.005, seed=0)
print(shots, f"- {np.mean(shots.echo_count == 0):.0%} of pulses returned nothing")
print(f"true leaf area index of the trees: {p.trees['leaf_area'].sum() / 50**2:.2f} (plus bark and understorey)")
echoes = shots.echo_xyz()
echo_height = echoes[:, 2] - p.ground_height(echoes[:, 0], echoes[:, 1])
zen, gap = canopy.gap_fraction_zenith(shots, echo_height, min_height=1.0, zenith_edges=np.arange(30, 95, 5.0))
print(f"effective PAI (hinge, 57.5 deg): {canopy.lai_from_gap_fraction(zen, gap, 'hinge'):.2f}")
fig, ax = plt.subplots()
ax.plot(zen, gap, "o-")
ax.axvline(57.5, c="k", ls="--", lw=0.8)
ax.text(58.5, 0.92, "hinge angle", fontsize=8.5)
ax.axvspan(0, 30, color=CONTEXT, alpha=0.5, lw=0)
ax.text(15, 0.5, "not scanned", ha="center", fontsize=8.5)
ax.set(xlabel="view zenith angle (deg)", ylabel="gap fraction (pulses above 1 m)", ylim=(0, 1.02), xlim=(0, 90),
       title="Gap fraction seen by one synthetic scan");""",
    md("""The effective PAI lands at about half the trees' leaf area index, and
that is the point of the exercise: the leaves are clumped into crowns,
which lets more pulses through than the same leaf area spread evenly, so an
*effective* PAI is a lower bound on the real one. A clumping correction, or
the ray-traced voxels of notebook 9, recover more of it. Miller's integral
is not used here: it needs every zenith ring from 0 to 90 degrees, and the
scanner sees none above 30. The rise beyond 70 degrees is the edge of the
plot: pulses that close to horizontal leave it before they reach the
canopy."""),
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
    md("""Left, the share of the pulses in each direction that end in an echo
inside the tile; right, where the pulses start. The pulses of the scan
position in the corner point into the tile (azimuth 180 to 270 degrees,
clockwise from +y) and all end in it, since those going down reach the
ground. The rest entered through the sides of the tile, and many of them
cross it without a return."""),
    """\
hit = shots.echo_count > 0
az_edges, zen_edges = np.arange(0, 361, 6), np.arange(0, 131, 3)
fired, _, _ = np.histogram2d(az, zen, [az_edges, zen_edges])
returned, _, _ = np.histogram2d(az[hit], zen[hit], [az_edges, zen_edges])
fraction = np.where(fired >= 20, returned / np.maximum(fired, 1), np.nan).T     # at least 20 pulses per cell
fig, ax = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.5, 1]})
im = ax[0].imshow(fraction, origin="upper", extent=(0, 360, 130, 0), aspect="auto", vmin=0, vmax=1)
ax[0].axhline(90, c="w", lw=0.8, ls="--")
ax[0].text(4, 88, "horizon", c="w", fontsize=8, va="bottom")
fig.colorbar(im, ax=ax[0], label="fraction of pulses with a return")
ax[0].set(xlabel="azimuth (deg)", ylabel="zenith (deg)", title="Pulses by direction (blank: fewer than 20)")
ax[1].scatter(shots.origin[::20, 0], shots.origin[::20, 1], s=0.3, c="C0")
ax[1].set(title="Pulse origins, from above", xlabel="x (m)", ylabel="y (m)", aspect="equal")
keys, n = np.unique(np.round(shots.origin[:, :2], 1), axis=0, return_counts=True)
scanner = keys[n.argmax()]                    # the origin most pulses share
ax[1].plot(*scanner, "o", ms=9, mfc="none", mec="C3", mew=1.5)
ax[1].annotate("the one real scan\\nposition in the tile", xy=scanner, xytext=(10, 10), ha="center", fontsize=8.5, bbox=dict(fc="white", ec="none", alpha=0.8),
               arrowprops=dict(arrowstyle="->", lw=0.8));""",
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
density as λ / G. It follows [Vincent et al. 2017](../references.md) and the rayvoxel tool; see the
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
from matplotlib.colors import LogNorm

j = 15                                        # the row of voxels at y = 7.5 to 8 m
ext = (grid.origin[0], grid.origin[0] + grid.voxel_size * state.shape[2],
       grid.origin[2], grid.origin[2] + grid.voxel_size * state.shape[0])
state_colours = ListedColormap(["white", "0.45", "#CFE3F0", LEAF])      # unobserved, occluded, empty, filled
fig, ax = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
im = ax[0].imshow(state[:, j, :], origin="lower", extent=ext, cmap=state_colours, vmin=-0.5, vmax=3.5)
cb = fig.colorbar(im, ax=ax[0], ticks=range(4), shrink=0.9)
cb.ax.set_yticklabels(names)
im = ax[1].imshow(grid.num_beams[:, j, :], origin="lower", extent=ext, norm=LogNorm(1, 1e4))
fig.colorbar(im, ax=ax[1], shrink=0.9, label="pulses through the voxel")
im = ax[2].imshow(grid.num_hits[:, j, :], origin="lower", extent=ext, vmax=50)
fig.colorbar(im, ax=ax[2], shrink=0.9, label="echoes in the voxel", extend="max")
for a, title in zip(ax, ("Voxel state", "Pulses", "Echoes")):
    a.set(title=title, xlabel="x (m)")
ax[0].set_ylabel("z (m)")
fig.suptitle("A vertical slice of 0.5 m voxels at y = 7.5 to 8 m");""",
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
usable = z[beams >= 200].max()
fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
well = (grid.num_beams >= 200) & (grid.num_hits > 0)
ax[0].loglog(grid.attenuation_fpl[well], grid.attenuation_ppl[well], ".", ms=1.5, alpha=0.3, c="C0")
ax[0].plot([1e-3, 40], [1e-3, 40], "k", lw=0.8, label="1:1")
ax[0].set(xlabel="λ, FPL (m⁻¹)", ylabel="λ, PPL (m⁻¹)", title="The two estimators, per voxel\\n(voxels with ≥ 200 pulses)")
ax[0].legend(loc="upper left")
for a in ax[1:]:
    a.axhspan(usable, z.max() + 0.25, color=CONTEXT, alpha=0.5, lw=0)
ax[1].plot(beams, z, "C0")
ax[1].axvline(200, color="k", ls="--", lw=0.8)
ax[1].text(230, 0.5, "200", fontsize=8.5)
ax[1].set(xscale="log", xlabel="median pulses per voxel", ylabel="z (m)", title="Sampling by layer\\n(grey: median below 200)")
for mb, style in ((20, "-"), (200, "--")):
    ax[2].plot(grid.profile("pad_ppl", min_beams=mb), z, style, label=f"min_beams = {mb}")
ax[2].set(xlabel="PAD (m² m⁻³)", ylabel="z (m)", title="The PPL profile, two thresholds"); ax[2].legend(loc="upper right")
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
within 4 %.

The slice below shows the same voxel by voxel: most of the dense-looking
voxels in the upper canopy are ones that fewer than 200 pulses reached."""),
    """\
ok = beams >= 200
print("PAI over the well-sampled layers only:",
      round(float(np.nansum(grid.profile("pad_ppl", min_beams=200)[ok]) * grid.voxel_size), 2),
      " over every layer:",
      round(float(np.nansum(grid.profile("pad_ppl", min_beams=200)) * grid.voxel_size), 2))
pad_slice = grid.pad_ppl[:, j, :]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
ax[0].imshow(pad_slice, origin="lower", extent=ext, vmin=0, vmax=3)
ax[1].imshow(np.where(grid.num_beams[:, j, :] >= 200, np.nan, 1.0), origin="lower", extent=ext,
             cmap=ListedColormap([CONTEXT]))
im = ax[1].imshow(np.where(grid.num_beams[:, j, :] >= 200, pad_slice, np.nan), origin="lower", extent=ext, vmin=0, vmax=3)
fig.colorbar(im, ax=ax, shrink=0.9, label="PAD, PPL (m² m⁻³)", extend="max")
ax[0].set(title="Every voxel", xlabel="x (m)", ylabel="z (m)")
ax[1].set(title="Voxels with ≥ 200 pulses (grey: fewer)", xlabel="x (m)")
fig.suptitle("Plant area density in the slice at y = 7.5 to 8 m");""",
    md("""## Checking the estimators against a known scene

To see whether the numbers are right, the scene has to be known. A synthetic
plot (`synthetic.plot`, 20 m square: 20 broadleaf and eucalypt trees with
shrubs, grass and dead wood on rough ground) is scanned from five positions
with the model of a RIEGL VZ-400, its beam footprint, 5 mm of range noise
and mixed pixels. The echoes keep their true class, so the leaf area density
can be estimated from the leaf echoes alone and compared with the leaf area
the generator placed."""),
    """\
p = synthetic.plot(size=20, density=500, archetypes={"broadleaf": 1, "eucalypt": 1}, seed=1)
positions = [(4, 4), (16, 4), (4, 16), (16, 16), (10, 10)]
sim = Shots.concatenate([
    synthetic.scan(p.points, origin=(x, y, p.ground_height(x, y) + 1.5), resolution_deg=0.1,
                   scanner="vz400", range_noise=0.005, seed=k)
    for k, (x, y) in enumerate(positions)])
box = ((0, 0, float(p.points.z.min()) - 0.5), (20, 20, float(p.points.z.min()) + 29.5))
sim_grid = voxels.ray_voxelize(sim, 0.5, box, ground_class=2, leaf_classes=[4], wood_classes=[5],
                               attenuation=["fpl", "ppl"])
pai = float(np.nansum(sim_grid.profile("pad_ppl", min_beams=20)) * 0.5)
lai = float(np.nansum(sim_grid.profile("lad_ppl", min_beams=20)) * 0.5)
print(sim)
print(f"PAI {pai:.2f}  LAI {lai:.2f}  true LAI of the trees {p.trees['leaf_area'].sum() / 20**2:.2f}")""",
    md("""The leaf area index comes out within about a tenth of the truth: with
five positions and the misses traced, the voxels see most of the crowns.
The PAI is higher, as it should be: it also counts the wood, the shrubs and
the grass, which the truth here leaves out."""),
    md("""## Leaf angles

With `inclination=True`, normals of the echoes give an inclination angle
distribution per tree, and G is integrated over it and over the tree's beam
zeniths instead of assuming a spherical distribution. The synthetic leaves
have known orientations: the broadleaf trees' are planophile (mostly
horizontal blades, a mean inclination of 27 degrees), the eucalypts'
erectophile (hanging leaves, 63 degrees)."""),
    """\
import pandas as pd

inc = voxels.ray_voxelize(sim, 0.5, box, ground_class=2, leaf_classes=[4], wood_classes=[5], inclination=True)
leaf_tree = np.asarray(p.leaves["tree"])
leaf_inc = np.degrees(np.arccos(np.abs(np.asarray(p.leaves["normal"])[:, 2])))
leaf_w = np.asarray(p.leaves["area"])
rows = []
for tid, t in inc.tree_iad.items():
    if tid == 0:
        continue
    k = leaf_tree == tid
    centres = np.degrees(t["bin_centres"])
    rows.append({"tree": tid, "archetype": p.trees["archetype"][tid - 1],
                 "mean, true (deg)": np.average(leaf_inc[k], weights=leaf_w[k]),
                 "mean, estimated (deg)": np.sum(centres * t["liad"]) / np.sum(t["liad"]),
                 "estimated type": t["liad_de_wit"], "G_leaf": t["g_leaf"]})
rows = pd.DataFrame(rows)
summary = rows.groupby("archetype").agg(trees=("tree", "size"), true_mean=("mean, true (deg)", "mean"),
                                        estimated_mean=("mean, estimated (deg)", "mean"),
                                        estimated_range=("mean, estimated (deg)", lambda v: f"{v.min():.0f} to {v.max():.0f}"),
                                        G_leaf=("G_leaf", "mean"))
summary.round(2)""",
    """\
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for a, kind in zip(ax, ("broadleaf", "eucalypt")):
    tid = int(rows.tree[rows.archetype == kind].iloc[0])
    t = inc.tree_iad[tid]
    edges = np.linspace(0, 90, len(t["bin_centres"]) + 1)
    k = leaf_tree == tid
    true_hist = np.histogram(leaf_inc[k], edges, weights=leaf_w[k])[0]
    a.stairs(true_hist / true_hist.sum(), edges, color="k", lw=1.5, label="true leaves")
    a.stairs(t["liad"] / np.sum(t["liad"]), edges, color=LEAF, lw=2, label="estimated, leaf echoes")
    a.stairs(t["wiad"] / np.sum(t["wiad"]), edges, color=WOOD, lw=1.5, ls="--", label="estimated, wood echoes")
    a.set(xlabel="inclination of the surface normal from vertical (deg)", xlim=(0, 90), title=f"Tree {tid}, {kind}")
ax[0].set_ylabel("fraction of the surface")
ax[1].legend(loc="upper left");""",
    md("""The estimates tell the two kinds of tree apart, but both are pulled
towards the middle: the broadleaf leaves come out steeper than they are,
the eucalypt leaves flatter. The normals are fitted to a few noisy echoes on
each leaf, and their scatter flattens any distribution towards a uniform
one. The wood is mostly stem and limbs, near-vertical surfaces whose normals
lie close to 90 degrees from vertical, hence its peak there."""),
    md("## Wood volume, files and streaming\n\nQSM cylinders can be rasterised into the same grid; `write` produces a `.vox` voxel-space file or a text table; and a shots file is voxelised without being loaded. The cylinders here are the largest tree's own, from the generator."),
    """\
tid = int(p.trees["tree_id"][np.argmax(p.trees["dbh"])])
model = p.qsm(tid)
sim_grid.add_wood_volume(model)
reach = np.r_[model.start[:, :2], model.end[:, :2]]
print(f"wood volume in the grid {sim_grid.wood_volume.sum():.3f} m3 of the {model.total_volume:.3f} m3 in tree {tid}'s "
      f"cylinders; the rest lies beyond the grid, as the crown reaches from x = {reach[:, 0].min():.1f} to "
      f"{reach[:, 0].max():.1f} m and y = {reach[:, 1].min():.1f} to {reach[:, 1].max():.1f} m")

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
`sylva.geo.coords` and the `PointCloud` methods `translate`, `rotate` and
`recentre`. The [guide](../guide/coordinates.md) gives the rules behind
them."""),
    SETUP + "\nfrom sylva import filters\nfrom sylva.geo import coords\n\ncloud = sylva.read(DATA / \"litch_tile.laz\")",
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

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(local.x[::20], local.y[::20], s=0.2, c=CONTEXT, label="local frame")
ax.scatter(turned.x[::20], turned.y[::20], s=0.2, c="C0", label="rotated 30 deg about the centre")
ax.plot(*centre[:2], "+", c="k", ms=10, mew=1.5)
ax.set(aspect="equal", xlabel="x, local (m)", ylabel="y, local (m)", title="The tile rotated about its centre (+)")
ax.legend(markerscale=15, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2);""",
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
fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
for i, r in enumerate(raw):
    ax[0].scatter(r.x[::15], r.y[::15], s=0.2, c=f"C{i}", label=files[i].stem)
ax[0].set(aspect="equal", title="Each scan in its own scanner frame", xlabel="x, scanner (m)", ylabel="y, scanner (m)")
fig.legend(markerscale=15, loc="outside lower center", ncol=3)
for i in range(len(files)):
    m = merged.attrs["scan_id"] == i
    ax[1].scatter(merged.x[m][::15], merged.y[m][::15], s=0.2, c=f"C{i}")
ax[1].set(aspect="equal", title="After apply_transforms, in the tile's frame", xlabel="x (m)", ylabel="y (m)");""",
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

Moving values between point clouds and rasters with `sylva.geo.interpolate`:
labels computed on a thinned copy carried back to every point, terrain
models interpolated from the ground returns, and rasters read back onto the
points as attributes. The [guide](../guide/interpolation.md) describes the
methods."""),
    SETUP + "\nimport pandas as pd\nfrom sylva import filters, ground, trees\nfrom sylva.geo import interpolate\n\ncloud = sylva.read(DATA / \"litch_tile.laz\")",
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
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(cloud.x[slab], cloud.attrs["height"][slab], s=0.2, c=CONTEXT, label="same label")
ax.scatter(cloud.x[differ], cloud.attrs["height"][differ], s=1.5, c="C3", label="label differs")
ax.set(title="Where the transferred tree label differs from the\\nfull segmentation (slice y = 6.5 to 8.5 m)",
       xlabel="x (m)", ylabel="height above ground (m)", aspect="equal")
ax.legend(markerscale=6, loc="upper right");""",
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
fig, ax = plt.subplots(2, 2, figsize=(8.5, 7), sharex=True, sharey=True)
lim = np.nanpercentile(np.r_[lowest.data.ravel(), holes.data.ravel()], [1, 99])
im = ax[0, 0].imshow(lowest.data, origin="lower", extent=ext, vmin=lim[0], vmax=lim[1])
fig.colorbar(im, ax=ax[0, 0], shrink=0.9, label="elevation (m)"); ax[0, 0].set_title("Lowest point per cell")
im = ax[0, 1].imshow(holes.data, origin="lower", extent=ext, vmin=lim[0], vmax=lim[1])
fig.colorbar(im, ax=ax[0, 1], shrink=0.9, label="elevation (m)")
ax[0, 1].set_title("TIN, max_distance 0.5 m")
for a, m in zip(ax[1], ("tin", "natural")):
    im = a.imshow(dtms[m].data - lowest.data, origin="lower", extent=ext, cmap=DIVERGING, vmin=-0.2, vmax=0.2)
    fig.colorbar(im, ax=a, shrink=0.9, label="difference (m)", extend="both")
    a.set_title(f"{'TIN' if m == 'tin' else 'Natural neighbour'} minus lowest point")
for a in ax[1]:
    a.set_xlabel("x (m)")
for a in ax[:, 0]:
    a.set_ylabel("y (m)")""",
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
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(cloud.x[slab], h[slab], c=np.clip(relative[slab], 0, 1), s=0.3)
ax.set(title="Height of each point as a fraction of the canopy\\nheight above it (slice y = 9 to 11 m)",
       xlabel="x (m)", ylabel="height above ground (m)", aspect="equal")
fig.colorbar(sc, ax=ax, shrink=0.8, label="height / CHM");""",
]

NOTEBOOKS["13_masking"] = [
    md("""# 13. Masking

Selecting points with `sylva.geo.masks`: by polygons read from a file, by the
raster cell under each point, by an expression over the attributes, and by
the distance to another cloud. A mask is a boolean array, one entry per
point, so masks from different sources combine with `&`, `|` and `~`. The
[guide](../guide/masking.md) has the details."""),
    SETUP + """
import json
from sylva import ground\nfrom sylva.geo import masks

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
fig, ax = plt.subplots(figsize=(7, 4.6))
colours = np.array([CONTEXT, "C0", "C1"])
ax.scatter(cloud.x[::10], cloud.y[::10], s=0.2, c=colours[which[::10] + 1])
for i, f in enumerate(subplots):
    for part in f.parts:
        ax.plot(*part.exterior.T, "k", lw=0.8)
        for h in part.holes:
            ax.plot(*h.T, "k--", lw=0.8)
    ax.scatter([], [], s=20, c=f"C{i}", label=f"{f.properties['name']} ({f.properties['treatment']})")
ax.scatter([], [], s=20, c=CONTEXT, label="outside, or in the hole")
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
ax.set(aspect="equal", xlabel="x (m)", ylabel="y (m)", title="Points by subplot");""",
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

fig, ax = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
im = ax[0].imshow(chm.data, origin="lower", extent=(chm.xmin, chm.xmax, chm.ymin, chm.ymax))
ax[0].contour(chm.data >= 10, levels=[0.5], extent=(chm.xmin, chm.xmax, chm.ymin, chm.ymax), colors="w", linewidths=0.8)
fig.colorbar(im, ax=ax[0], shrink=0.9, label="canopy height (m)")
ax[0].set(title="CHM, with the 10 m contour", xlabel="x (m)", ylabel="y (m)")
m = low_veg & under_tall
ax[1].scatter(cloud.x[low_veg & ~m][::4], cloud.y[low_veg & ~m][::4], s=0.2, c="C1", label="elsewhere")
ax[1].scatter(cloud.x[m][::4], cloud.y[m][::4], s=0.2, c="C0", label="under canopy of 10 m or more")
ax[1].set(aspect="equal", title="Vegetation 0.3 to 2 m above ground", xlabel="x (m)")
ax[1].legend(markerscale=15, loc="upper left", bbox_to_anchor=(1.0, 1.0));""",
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
ax.scatter(cloud.x[side & ~block], cloud.z[side & ~block], s=0.2, c=CONTEXT, label="kept in both epochs")
ax.scatter(lost.x, lost.z, s=0.4, c="C3", label="found by difference (5 cm)")
ax.scatter(cloud.x[missed], cloud.z[missed], s=3, c="C0", label="removed, but within 5 cm of a kept point")
ax.add_patch(plt.Rectangle((14, 0.5), 3, 5.5, fill=False, ls="--", lw=0.8, ec="k"))
ax.set(xlim=(12, 19), ylim=(-0.5, 8), xlabel="x (m)", ylabel="z (m)",
       title="The removed block (dashed), side view of y = 14 to 17 m")
ax.legend(markerscale=5, loc="upper left");""",
]


SYNTH_SETUP = """\
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sylva
from sylva import synthetic

tmp = Path(tempfile.mkdtemp())      # files written here are temporary
""" + STYLE

NOTEBOOKS["14_change"] = [
    md("""# 14. Change detection

`sylva.change` compares two epochs of one plot: which trees died, which were
recruited, how much the survivors grew, and where the points, surfaces, voxels
and cylinder models differ. Every change is reported with an uncertainty or a
level of detection, and what the data cannot support is labelled rather than
reported as change. The [guide](../guide/change.md) has the details.

This notebook runs on `synthetic.forest_epochs`, a 30 m plot scanned twice
from five positions with known changes between the scans, so every result can
be checked against the truth."""),
    SYNTH_SETUP + "\nfrom sylva import change, filters, ground, qsm, trees, voxels",
    md("""## Two epochs with known changes

Between the epochs the survivors grow by a known DBH and height increment,
two trees die, two are recruited (one of them 0.37 m from the stump of a dead
tree, as after felling), one tree loses a limb and part of one crown is
thinned. Epoch 2 is scanned with its own range noise (5 mm against 3 mm), from
scanners displaced by about 0.5 m, and is delivered in a frame rotated by
1.5 degrees and shifted by almost a metre, as an independently registered
revisit would be."""),
    """\
ep = synthetic.forest_epochs(seed=1)
cloud_1, cloud_2 = ep.clouds
print(f"epoch 1: {len(cloud_1):,} echoes, epoch 2: {len(cloud_2):,} echoes")
from collections import Counter
print("known changes:", dict(Counter(c["kind"] for c in ep.changes)))
truth_1 = ep.trees[0]                      # the trees of epoch 1, in the frame of epoch 1
print("trees in epoch 1:", len(truth_1["tree_id"]), " range noise (m):", ep.range_noise)""",
    md("""## Inventory, alignment and tree matching

Each epoch goes through the operational sequence of the trees guide: DTM,
stems, segmentation and heights. `align_epochs` then registers epoch 2 onto
epoch 1 on the stable features only (the stem axes and the terrain), since a
registration on all points would be pulled by the very change it should
reveal. Its uncertainty is part of the result."""),
    """\
def inventory(cloud):
    cloud = ground.normalize_height(cloud, ground.make_dtm(cloud))
    stems = trees.detect_stems(cloud)
    stems, _ = trees.merge_branches(cloud, stems)
    labels = trees.segment_trees(cloud, stems)
    trees.tree_heights(cloud, labels, stems)
    return trees.prune_trees(stems, labels) + (cloud,)


stems_1, labels_1, norm_1 = inventory(cloud_1)
stems_2, labels_2, norm_2 = inventory(cloud_2)
al = change.align_epochs(cloud_1, cloud_2)
print(al.report())

centre = np.array([15.0, 15.0, truth_1["z0"].mean(), 1.0])
error = (al.transform @ centre - ep.transform @ centre)[:3]
print("error at the plot centre (mm):", np.round(1000 * error, 2),
      " one sigma (mm):", np.round(1000 * al.sigma_xyz, 2))""",
    md("""`match_trees` moves the stems of epoch 2 into the frame of epoch 1 and
pairs them by an optimal assignment. Stems do not shrink, so the felled tree
and the recruit beside its stump are a death and a recruit, not one tree that
lost two thirds of its diameter."""),
    """\
m = change.match_trees(stems_1, stems_2, transform=al)
print(m.counts())
for kind, found in (("death", m.deaths), ("recruit", m.recruits)):
    true = ep.of_kind(kind)
    for t in found:
        xy = np.array([t.x, t.y]) if kind == "death" else al.transform[:2, :2] @ [t.x, t.y] + al.transform[:2, 3]
        d = min(np.hypot(c["x"] - xy[0], c["y"] - xy[1]) for c in true)
        print(f"{kind:8s} at ({xy[0]:5.2f}, {xy[1]:5.2f}), DBH {100 * t.dbh:4.1f} cm: "
              f"{100 * d:.1f} cm from a true {kind}")""",
    md("""## Increments and their detection limits

`tree_increments` measures each survivor's DBH increment on its stem profile
(paired circle fits at the same heights in both epochs, which cancels the
stem's own shape) and gives it a minimum detectable increment (MDI) at 95 %
confidence. The true increments are joined by position."""),
    """\
inc = change.tree_increments(m, norm_1, norm_2, labels_1, labels_2, noise_a=0.003, noise_b=0.005)
growth = {c["tree_id"]: c for c in ep.of_kind("growth")}
nearest = [int(truth_1["tree_id"][np.argmin(np.hypot(truth_1["x"] - x, truth_1["y"] - y))])
           for x, y in zip(inc["x"], inc["y"])]
table = inc.to_pandas()[["x", "y", "d_dbh", "d_dbh_mdi", "dbh_change", "d_height", "d_height_mdi", "height_change"]]
table.insert(0, "true_tree", nearest)
table.insert(4, "true_d_dbh", [growth[t]["d_dbh"] for t in nearest])
table.insert(8, "true_d_height", [growth[t]["d_height"] for t in nearest])
within = np.abs(table["d_dbh"] - table["true_d_dbh"]) <= table["d_dbh_mdi"]
print(f"DBH increment within its MDI of the truth: {within.sum()} of {len(table)}")
cols = [c for c in table.columns if c.startswith(("d_", "true_d")) or c.endswith("mdi")]
(table.assign(**{c: 1000 * table[c] for c in cols if "dbh" in c})
      .round({c: 1 for c in cols} | {"x": 1, "y": 1}).rename(columns={c: c + (" (mm)" if "dbh" in c else " (m)") for c in cols}))""",
    md("""Two survivors grew by only 0.5 mm, less than a scan can resolve, and are
reported as `below_detection` rather than as growth. Every height increment
is below its detection level of about a metre: a tree top that no pulse hit
leaves no trace, so a height difference of a few decimetres between two scans
is not evidence of growth, even where it happens to be right."""),
    """\
fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
ax[0].scatter(truth_1["x"], truth_1["y"], s=2000 * truth_1["dbh"] ** 2, facecolors="none", edgecolors="0.4",
              label="epoch 1 stems (size by DBH)")
for kind, marker, colour in (("death", "x", "C3"), ("recruit", "+", "C0")):
    pts = np.array([[c["x"], c["y"]] for c in ep.of_kind(kind)])
    ax[0].scatter(*pts.T, marker=marker, s=90, c=colour, linewidths=2, label=f"true {kind}s")
removed = ep.of_kind("branch_removed")[0]
ax[0].plot([removed["base_x"], removed["tip_x"]], [removed["base_y"], removed["tip_y"]], "C1", lw=2.5, label="limb removed")
ax[0].set(xlim=(0, 30), ylim=(0, 30), aspect="equal", xlabel="x (m)", ylabel="y (m)", title="The plot and its known changes")
ax[0].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), markerscale=0.7)
order = np.argsort(table["true_d_dbh"].to_numpy())
t = table.iloc[order]
ax[1].errorbar(np.arange(len(t)), 1000 * t["d_dbh"], yerr=1000 * t["d_dbh_mdi"], fmt="o", ms=4, c="C0",
               label="measured ± MDI")
ax[1].scatter(np.arange(len(t)), 1000 * t["true_d_dbh"], marker="_", s=200, c="k", label="true", zorder=3)
ax[1].set(xlabel="survivor, in order of true increment", ylabel="DBH increment (mm)", title="DBH increments of the survivors")
ax[1].legend(loc="upper left");""",
    md("""## The plot summary

`plot_summary` gives growth, mortality and recruitment per hectare and year,
with 95 % intervals from Monte Carlo draws of every tree's measurement errors.
The intervals cover measurement and registration only; a plot is still one
sample of its stand. The true basal-area figures come from the known trees."""),
    """\
summary = change.plot_summary(inc, area=30 * 30, years=5)
print(summary.report())

ha_yr = 30 * 30 / 1e4 * 5
ba = lambda d: np.pi / 4 * d ** 2
d1 = dict(zip(truth_1["tree_id"], truth_1["dbh"]))
true_growth = sum(ba(d1[t] + g["d_dbh"]) - ba(d1[t]) for t, g in growth.items()) / ha_yr
true_mortality = sum(ba(c["dbh"]) for c in ep.of_kind("death")) / ha_yr
for name, true in (("basal_area_growth", true_growth), ("basal_area_mortality", true_mortality)):
    e = summary[name]
    print(f"{name:21s} {e.estimate:.3f} [{e.low:.3f}, {e.high:.3f}] m²/ha/yr, true {true:.3f}")""",
    md("""## Change in the points

`distances(a, b, "c2c")` gives, for each point of `b`, the distance to the
nearest point of `a`. Taken from the second epoch to the first
(`distances(epoch_2, epoch_1)`), it marks the points of epoch 1 that have no
counterpart in epoch 2: the dead trees, the removed limb and the thinned
foliage. It is unsigned and biased upwards by point spacing and noise, so it
serves as a first look rather than as a test."""),
    """\
cloud_2a = al.apply(cloud_2)                           # epoch 2 in the frame of epoch 1
gone = change.distances(cloud_2a, cloud_1, "c2c")      # for each point of epoch 1
print(gone, "| median", f"{100 * np.median(gone.distance):.1f} cm")
far = gone.distance > 0.3
print(f"{far.mean():.1%} of the epoch 1 points are more than 30 cm from any point of epoch 2")""",
    md("""M3C2 (Lague et al. 2013) measures a signed distance along a normal with a
95 % level of detection. On a stem, radial normals make the distance the
radius increment. Here the core points are the epoch 1 stem points of one
survivor between 1 and 3 m, thinned to 5 cm; the registration uncertainty of
the alignment enters the level of detection."""),
    """\
tree = 3
k = list(truth_1["tree_id"]).index(tree)
cx, cy, z0 = truth_1["x"][k], truth_1["y"][k], truth_1["z0"][k]


def stem_band(c):
    r, h = np.hypot(c.x - cx, c.y - cy), c.z - z0
    return c[(r < 0.5) & (h > 1) & (h < 3) & (c.attrs["classification"] == 5)]


band_1, band_2 = stem_band(cloud_1), stem_band(cloud_2a)
core = filters.voxel_downsample(band_1, 0.05)
normals = np.column_stack([core.x - cx, core.y - cy, np.zeros(len(core))])
normals /= np.linalg.norm(normals, axis=1)[:, None]
m3 = change.distances(band_1, band_2, "m3c2", core, normals=normals, normal_scale=0.1,
                      projection_scale=0.15, max_depth=0.05, registration_sigma=al.registration_sigma)
print(m3)
print(f"median distance {1000 * np.nanmedian(m3.distance):.1f} mm, median LoD95 {1000 * np.nanmedian(m3.lod):.1f} mm, "
      f"{m3.significant.mean():.0%} significant; true radius increment {500 * growth[tree]['d_dbh']:.1f} mm")""",
    md("""A difference of rasters (`dod`) compares two surfaces on one lattice. With
canopy height models the dead trees show as large, coherent losses. The level
of detection is a fixed 1 m here, since scans from below do not see the upper
surface of a crown the same way twice; isolated cells still change by metres
where one epoch saw through a gap in the crowns and the other did not, which
is why a CHM difference from terrestrial scans needs a look at its spatial
pattern, not only at its significant cells."""),
    """\
bounds = (0, 0, 30, 30)


def chm(cloud):
    cloud = ground.normalize_height(cloud, ground.make_dtm(cloud, 0.5, bounds=bounds))
    return ground.make_chm(cloud, 0.5, bounds=bounds)


dd = change.dod(chm(cloud_1), chm(cloud_2a), min_detectable=1.0)
print(f"significant over {dd.area_changed:.0f} of {dd.area_compared:.0f} m²: "
      f"{dd.volume_gained:.0f} m³ gained, {dd.volume_lost:.0f} m³ lost")

fig, ax = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
s = cloud_1[::5]
ax[0].scatter(s.x, s.y, s=0.1, c=CONTEXT)
ax[0].scatter(cloud_1.x[far], cloud_1.y[far], s=0.2, c="C3")
dead = np.array([[c["x"], c["y"]] for c in ep.of_kind("death")])
ax[0].scatter(*dead.T, marker="x", c="k", s=50, label="dead stems")
ax[0].set(aspect="equal", xlim=(0, 30), ylim=(0, 30), xlabel="x (m)", ylabel="y (m)",
          title="Epoch 1 points > 30 cm from epoch 2 (red)")
ax[0].legend(loc="lower right")
r = dd.thresholded()
im = ax[1].imshow(r.data, origin="lower", extent=(r.xmin, r.xmin + r.data.shape[1] * r.resolution,
                                                  r.ymin, r.ymin + r.data.shape[0] * r.resolution),
                  cmap=DIVERGING, vmin=-15, vmax=15)
fig.colorbar(im, ax=ax[1], shrink=0.9, label="CHM change, epoch 2 minus epoch 1 (m)")
ax[1].set(xlabel="x (m)", title="CHM change beyond ±1 m (white: within)");""",
    md("""## Change in the voxels

A voxel that holds no echoes in the later epoch has only lost its contents if
pulses went through it. `occupancy` compares two ray-traced grids (the pulses
of both epochs in one frame, so the pulses of epoch 2 are moved by the
alignment first) and calls a voxel `lost` or `gained` only when the epoch
that found it empty sent enough pulses to have seen its contents; otherwise
it is `unobserved`."""),
    """\
shots_2a = ep.shots[1].transform(al.transform)
box = ((0, 0, -1), (30, 30, 23))
grids = [voxels.ray_voxelize(s, 0.5, box, ground_class=2, occlusion=True) for s in (ep.shots[0], shots_2a)]
occ = change.occupancy(*grids)
print(occ)

lost = occ.centers("lost")
dead = np.array([[c["x"], c["y"]] for c in ep.of_kind("death")])
near_dead = np.min(np.hypot(lost[:, None, 0] - dead[:, 0], lost[:, None, 1] - dead[:, 1]), axis=1) < 5
print(f"{near_dead.mean():.0%} of the lost voxels lie within 5 m of a dead stem")""",
    md("""The lost voxels that are not near a dead tree belong to the removed limb,
the thinned crown and the edges of crowns that moved as the trees grew,
which is also why voxels are gained. The layer means compare the plant area
density only over the voxels both epochs sampled well, so that what one epoch
did not see does not bias the comparison."""),
    """\
lay = occ.layers
fig, ax = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1.3, 1]})
cols = [np.bincount(np.ravel_multi_index(((occ.centers(n)[:, 1] // 0.5).astype(int),
                                          (occ.centers(n)[:, 0] // 0.5).astype(int)), (60, 60)),
                    minlength=3600).reshape(60, 60) for n in ("lost", "gained")]
im = ax[0].imshow(cols[1] - cols[0], origin="lower", extent=(0, 30, 0, 30), cmap=DIVERGING, vmin=-10, vmax=10)
ax[0].scatter(*dead.T, marker="x", c="k", s=60, label="dead stems")
fig.colorbar(im, ax=ax[0], shrink=0.9, label="gained minus lost voxels per 0.5 m column", extend="both")
ax[0].set(xlabel="x (m)", ylabel="y (m)", title="Voxel occupancy change"); ax[0].legend(loc="lower right")
ax[1].plot(lay["pad_a"], lay["z"], label="epoch 1")
ax[1].plot(lay["pad_b"], lay["z"], "--", label="epoch 2")
ax[1].set(xlabel="mean PAD (m² m⁻³)", ylabel="z (m)", title="PAD by layer, over the voxels\\nboth epochs sampled")
ax[1].legend(loc="upper right");""",
    md("""## Change in a QSM

`compare_qsms` compares two cylinder models of one tree in one frame: the
radius increment along the stem, branches matched, lost and new, and volume
change split into what both models measured (`trusted_change`) and what came
from the priors that fill in a QSM where no points were fitted. Given a
ray-traced grid of the later epoch, a branch missing from the later model is
only called lost when the later pulses saw its space empty.

The tree here is the survivor that lost a limb. Its wood points are taken
from the scans' true labels, to keep this section about the comparison
rather than about segmentation."""),
    """\
tree = ep.of_kind("branch_removed")[0]["tree_id"]
k = list(truth_1["tree_id"]).index(tree)
base = (truth_1["x"][k], truth_1["y"][k])


def wood(c):
    return c[(c.attrs["tree_id"] == tree) & (c.attrs["classification"] == 5)]


qsm_1, qsm_2 = qsm.build_qsm(wood(cloud_1), base_xy=base), qsm.build_qsm(wood(cloud_2a), base_xy=base)
w = wood(cloud_1).xyz
grid_2 = voxels.ray_voxelize(shots_2a, 0.1, bounds=(tuple(w.min(0) - 1), tuple(w.max(0) + 1)), occlusion=True)
qc = change.compare_qsms(qsm_1, qsm_2, grid_b=grid_2)
{k: int(v) if k.startswith("n_") else round(float(v), 4) for k, v in qc.summary().items()}""",
    """\
true = growth[tree]
print(f"taper increment {1000 * qc.taper_increment:.1f} ± {1000 * qc.taper_sigma:.1f} mm "
      f"over {qc.n_taper_bins} bins (true radius increment {500 * true['d_dbh']:.1f} mm)")
removed = ep.of_kind("branch_removed")[0]
print(f"lost branches: {qc.lost['status'].tolist()}, volume {1000 * qc.lost['volume'].sum():.1f} L "
      f"(true {1000 * removed['volume']:.1f} L), measured share {qc.lost['measured'].round(2).tolist()}, "
      f"trusted {qc.lost['trusted'].tolist()}")
print(f"volume change {1000 * qc.change:.1f} L, of which trusted {1000 * qc.trusted_change:.1f} "
      f"± {1000 * qc.trusted_sigma:.1f} L")""",
    md("""The lost limb is found, and the later grid confirms that its space was
seen and is empty, so it is `lost` rather than `unobserved`. Only a small
share of the limb's length was fitted to points in epoch 1, however; the
rest of its radius came from the QSM's priors, so its volume is not counted
as trusted change. The trusted change is the stem growth, which the taper
increment measures to within its stated uncertainty."""),
    """\
fig, ax = plt.subplots(1, 2, figsize=(10, 4.8), gridspec_kw={"width_ratios": [1, 1.2]})
lost_ids = set(qc.lost["id"].tolist())
for model, colour, dx, name in ((qsm_1, "C0", 0.0, "epoch 1"), (qsm_2, "C1", 6.0, "epoch 2")):
    s, e = model.start, model.start + model.cylinders[:, 3:6] * model.column("length")[:, None]
    for i in range(len(s)):
        lost_here = model is qsm_1 and model.column("branch_id")[i] in lost_ids
        ax[0].plot([s[i, 0] + dx, e[i, 0] + dx], [s[i, 2], e[i, 2]], c="k" if lost_here else colour,
                   lw=max(0.5, 60 * model.column("radius")[i]))
    ax[0].text(base[0] + dx, s[:, 2].min() - 1.0, name, ha="center", va="center")
ax[0].plot([], [], c="k", lw=2, label="limb lost")
ax[0].legend(loc="upper left")
ax[0].set(aspect="equal", ylim=(qsm_1.start[:, 2].min() - 1.8, None), xlabel="x (m; epoch 2 shifted by 6 m)", ylabel="z (m)",
          title="The two QSMs")
tp = qc.taper
mid = (tp["z0"] + tp["z1"]) / 2
ok = tp["trusted"]
pooled, sigma = 1000 * qc.taper_increment, 1000 * qc.taper_sigma
ax[1].axvspan(pooled - sigma, pooled + sigma, color="C0", alpha=0.2, lw=0, label="pooled over the trusted bins, ± 1 sigma")
ax[1].axvline(pooled, c="C0", lw=1.5)
ax[1].errorbar(1000 * tp["increment"][ok], mid[ok], xerr=1000 * tp["sigma"][ok], fmt="o", ms=4, c="C0",
               ecolor=CONTEXT, elinewidth=1, label="one trusted bin, ± 1 sigma")
ax[1].plot(1000 * tp["increment"][~ok], mid[~ok], "x", c="0.5", label="bin not trusted")
ax[1].axvline(500 * true["d_dbh"], c="k", ls="--", lw=1, label="true")
ax[1].set(xlabel="stem radius increment (mm)", ylabel="height above base (m)", title="Taper increment, bin by bin and pooled")
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2);""",
]


NOTEBOOKS["15_als"] = [
    md("""# 15. Airborne lidar

`sylva.als` works on airborne (ALS) and UAV lidar delivered as tiles: a
catalogue built from the tile headers, ground classification and terrain and
canopy models over the whole catalogue, area-based metrics, individual tree
detection, and canopy structure from the pulses reconstructed with the flight
trajectory. Every catalogue function processes the area in chunks that each
carry a buffer of points from their neighbours, so the tile edges do not show
in the results. The guides are [Airborne lidar tiles](../guide/als.md),
[Area-based ALS metrics](../guide/als_metrics.md),
[Airborne trees](../guide/als_trees.md) and
[Canopy structure from airborne lidar](../guide/als_canopy.md).

Everything here is flown over synthetic scenes with `synthetic.als_flight`, so
the terrain, the trees and the plant area are known."""),
    SYNTH_SETUP + "\nfrom sylva import als",
    md("""## A simulated flight

`als_flight` flies parallel lines over a scene with an oscillating mirror, a
beam of finite divergence sampled by sub-beams, and up to five returns per
pulse, each with the LAS attributes a real survey carries (`gps_time`,
`return_number`, `scan_angle`, `intensity`, the flight line in
`point_source_id`) and its true `classification`. The scene is 40 trees on a
gently sloping 100 m square; the trajectory is kept as a table."""),
    """\
rng = np.random.default_rng(0)
stems = [(x, y, 0.3, h) for x, y, h in
         zip(rng.uniform(5, 95, 40), rng.uniform(5, 95, 40), rng.uniform(10, 25, 40))]
scene = synthetic.forest(stems, size=100.0, ground_points=100, margin=0.0)
flight = synthetic.als_flight(scene, altitude=80.0, speed=10.0, line_spacing=40.0,
                              pulse_rate=50_000, bounds=(0, 0, 100, 100))
pts = flight.points
print(pts)
print(f"{flight.n_pulses:,} pulses fired; returns per pulse:",
      np.bincount(pts.attrs["number_of_returns"])[1:].tolist())
traj = flight.trajectory
print("trajectory columns:", list(traj), f"({len(traj['time']):,} samples)")""",
    """\
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
s = pts[::20]
order = np.argsort(s.z)
sc = ax[0].scatter(s.x[order], s.y[order], c=s.z[order], s=0.3)
for line in np.unique(traj["line"]):
    k = traj["line"] == line
    ax[0].plot(traj["x"][k], traj["y"][k], "k", lw=1.2)
ax[0].set(aspect="equal", xlim=(-15, 115), ylim=(-15, 115), xlabel="x (m)", ylabel="y (m)",
          title="Returns by height, and the flight lines (black)")
fig.colorbar(sc, ax=ax[0], shrink=0.9, label="z (m)")
slab = (pts.y > 48) & (pts.y < 52)
for c, colour, name in ((2, GROUND, "ground"), (4, LEAF, "leaves"), (5, WOOD, "wood")):
    k = slab & (pts.attrs["classification"] == c)
    ax[1].scatter(pts.x[k], pts.z[k], s=1 if c == 5 else 0.3, c=colour, label=name)
ax[1].set(xlabel="x (m)", ylabel="z (m)", title="Side view of y = 48 to 52 m, by true class")
ax[1].legend(markerscale=10, loc="upper left");""",
    md("""## Tiles and the catalogue

The returns are written as four 50 m LAZ tiles. `als.catalog` reads only the
headers and checks the tiling (overlaps, gaps, mixed coordinate systems,
point formats or scales) before anything is processed."""),
    """\
cat = flight.write_tiles(tmp / "tiles", size=50.0, epsg=32755)
print(cat.report())""",
    md("""## Ground, terrain and canopy height

`classify_ground` runs the cloth simulation filter chunk by chunk and writes
classified tiles; `dtm` and `chm` build rasters on one grid shared by the
whole catalogue. The true terrain is `synthetic.terrain_height`, so the DTM
error is known in every cell. `normalize` writes tiles with heights above
ground in place of z."""),
    """\
ground_cat = als.classify_ground(cat, tmp / "ground", method="csf", cloth_resolution=1.0)
dtm = als.dtm(ground_cat, resolution=1.0)
chm = als.chm(ground_cat, resolution=0.5)
err = dtm.data - synthetic.terrain_height(*dtm.cell_centers())
tin = als.dtm(ground_cat, resolution=1.0, method="tin")
err_tin = tin.data - synthetic.terrain_height(*tin.cell_centers())
for name, e in (("lowest", err), ("tin", err_tin)):
    print(f"DTM ({name:6s}) error: mean {e.mean() * 100:+.1f} cm, mean absolute {np.abs(e).mean() * 100:.1f} cm, "
          f"largest {np.abs(e).max() * 100:.0f} cm")
print(f"CHM: tallest {np.nanmax(chm.data):.1f} m; {np.mean(chm.data > 2):.0%} of the cells above 2 m")
norm = als.normalize(ground_cat, tmp / "normalised", replace_z=True)
h = norm.read((0, 0, 100, 100)).z
print(norm, f"| heights {h.min():.2f} to {h.max():.2f} m")""",
    """\
fig, ax = plt.subplots(1, 3, figsize=(10, 3.4), sharex=True, sharey=True)
ext = lambda r: (r.xmin, r.xmin + r.data.shape[1] * r.resolution, r.ymin, r.ymin + r.data.shape[0] * r.resolution)
for a, e, r, name in ((ax[0], err, dtm, "lowest return per cell"), (ax[1], err_tin, tin, "TIN")):
    im = a.imshow(100 * e, origin="lower", extent=ext(r), cmap=DIVERGING, vmin=-8, vmax=8)
    a.set(xlabel="x (m)", title=f"DTM error, {name}")
fig.colorbar(im, ax=ax[:2], shrink=0.9, label="DTM minus true terrain (cm)", extend="both")
ax[0].set_ylabel("y (m)")
im = ax[2].imshow(chm.data, origin="lower", extent=ext(chm))
fig.colorbar(im, ax=ax[2], shrink=0.9, label="canopy height (m)")
ax[2].set(xlabel="x (m)", title="Canopy height model, 0.5 m");""",
    md("""The default DTM takes the lowest ground return in each cell, which lies
below the terrain at the cell centre: the terrain slopes by 5 %, so the lowest
point of a 1 m cell is about 2.5 cm below its centre, and the lowest of many
returns with 2 cm of range noise lies a few centimetres lower still. A TIN
through the ground returns (`method="tin"`) removes that bias.

## Area-based metrics

`grid_metrics` computes the standard set of metrics ([Roussel et al. 2020](../references.md): height percentiles, cover,
entropy and the rest; `als.metric_names()` lists them) on a grid over the
catalogue, and `plot_metrics` the same for plots, here four circles of 15 m
radius. With `min_height=0` the returns below the DTM are left out."""),
    """\
grid = als.grid_metrics(ground_cat, 20.0, ["zmax", "zq95", "cover", "zentropy"], min_height=0.0)
print("zq95 on a 20 m grid (m), south row first:")
print(np.round(grid["zq95"].data, 1))
plots = als.plot_metrics(ground_cat, [(25, 25), (75, 25), (25, 75), (75, 75)], radius=15.0,
                         metrics=["n", "zmax", "zq95", "zmean", "cover"], min_height=0.0,
                         ids=["SW", "SE", "NW", "NE"])
plots.to_pandas().round(2)""",
    md("""The north-west plot holds no tree centre, only the edge of a crown from
outside it, hence its low `zq95` and cover. The grid is anchored at the
catalogue's south-west corner, so its last row and column (north and east)
lie almost entirely beyond the survey and hold few returns or none.

## Individual trees

For tree detection the scene is `synthetic.crown_forest`: 100 trees on a
hectare with closed, ellipsoidal crowns of known size, flown at about 20
returns per m². `find_trees` finds tree tops with a local maximum filter on
the CHM (a window a fifth of the tree height) and grows crowns from them by
the method of Dalponte and Coomes (2016), and with `out` writes the tiles
again with a `tree_id` per point. `synthetic.forest_trees` gives the truth:
each tree's top, height and crown outline."""),
    """\
stand = synthetic.stand(100, size=100.0, min_spacing=3.0, heights=(10, 25), seed=11)
crowns = synthetic.crown_forest(stand, size=100.0, ground_points=100, margin=0.0, seed=11)
flight_2 = synthetic.als_flight(crowns, pulse_rate=4_000, line_spacing=30.0, bounds=(0, 0, 100, 100), seed=11)
cat_2 = flight_2.write_tiles(tmp / "stand", size=50.0, epsg=32755)

found = als.find_trees(cat_2, out=tmp / "labelled", method="dalponte2016",
                       window=als.LinearWindow(0.0, 0.2, 2.0, 20.0), max_cr=20)
truth = synthetic.forest_trees(crowns)
print(found, f"| {len(truth['tree_id'])} true trees | {len(flight_2.points):,} returns")
print(als.catalog(tmp / "labelled").report().splitlines()[0], "with a tree_id attribute")""",
    md("""Each top is assigned to the true tree of the highest return within
0.75 m of it, as in the guide's validation. A tree is found when at least one
top lies on it; every other top is a commission error (a second top on the
same tree, or a top on no tree)."""),
    """\
p = flight_2.points
tid = p.attrs["tree_id"]
top_tree = np.zeros(len(found), int)
for i, (x, y) in enumerate(zip(found.x, found.y)):
    near = np.flatnonzero(np.hypot(p.x - x, p.y - y) < 0.75)
    top_tree[i] = tid[near[np.argmax(p.z[near])]] if len(near) else 0
hit = np.unique(top_tree[top_tree > 0])
first = np.array([np.flatnonzero(top_tree == t)[0] for t in hit])
row = np.searchsorted(truth["tree_id"], hit)
dh = found.height[first] - truth["height"][row]
print(f"found {len(hit)} of {len(truth['tree_id'])} trees ({len(hit) / len(truth['tree_id']):.0%}); "
      f"commission {1 - len(hit) / len(found):.0%} of the {len(found)} tops")
print(f"height of the trees found: bias {dh.mean():+.2f} m, RMSE {np.sqrt(np.mean(dh ** 2)):.2f} m")
ratio = found.crown_area[first] / truth["crown_area"][row]
print(f"crown area found / true crown area: median {np.median(ratio):.2f}")""",
    """\
chm_2 = als.chm(cat_2, resolution=0.5)
fig, ax = plt.subplots(figsize=(6.5, 7))
im = ax.imshow(chm_2.data, origin="lower", extent=ext(chm_2), cmap="Greys", vmin=0, vmax=40)
fig.colorbar(im, ax=ax, shrink=0.8, label="canopy height (m)")
for poly in found.crowns:
    if len(poly):
        ax.fill(*poly.T, fc="none", ec="k", lw=0.6)
missed = np.setdiff1d(truth["tree_id"], hit)
k = np.searchsorted(truth["tree_id"], missed)
extra = np.setdiff1d(np.arange(len(found)), first)
ax.scatter(found.x[first], found.y[first], s=16, c="C0", label="first top on a true tree")
ax.scatter(found.x[extra], found.y[extra], s=16, c="C1", label="further top (commission)")
ax.scatter(truth["top_x"][k], truth["top_y"][k], marker="x", s=40, c="C3", linewidths=1.8, label="true tree missed")
ax.plot([], [], c="k", lw=0.6, label="crowns found")
ax.set(xlim=(0, 100), ylim=(0, 100), aspect="equal", xlabel="x (m)", ylabel="y (m)", title="Crowns and tree tops found, over the CHM")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2);""",
    md("""The trees missed are, with few exceptions, overtopped: a shorter tree whose
top lies under the crown of a taller neighbour is no local maximum in a
canopy surface, so no method working on the surface can find it. At about
20 returns per m² a crown's surface is ragged and has several local maxima,
which gives the further tops on the smaller crowns; the found crowns are
correspondingly smaller than the true ones. The validation in the guide shows how both change with
stand density and point density.

## Canopy structure from the pulses

Each airborne pulse is a ray from the aircraft down through the canopy.
`als.pulses` groups the returns of each pulse (by `gps_time` and flight line)
and starts it at the sensor position interpolated from the trajectory, which
gives a `Shots` object like those of a terrestrial scan. Its report shows
how well the trajectory fits: the returns of a multiple-return pulse lie on
one line through the sensor."""),
    """\
trajectory = als.Trajectory.from_dict(flight.trajectory)
shots, report = als.pulses(cat.read((0, 0, 50, 50)), trajectory, report=True)
print(shots)
print({k: report[k] for k in ("n_returns", "n_pulses", "n_incomplete", "n_missing_returns")},
      f"| sensor to return line: median {1000 * report['line_offset_median']:.1f} mm")""",
    md("""Pulses are incomplete where the tile was cut: some of their returns fell
in the next tile. The catalogue functions below read each chunk with a buffer
so that they see whole pulses.

A known answer needs a scene whose plant area is known exactly: a layer of
5 cm spheres between 5 and 15 m, a turbid medium in which a thin beam is
stopped with probability `n π r²` per metre in any direction. That is the
extinction of spherical leaf angles (G = 0.5) at a plant area density of
`2 n π r²`, here 0.3 m² m⁻³, and so a plant area index of 3. One sub-beam
per pulse makes every pulse a single ray."""),
    """\
rng = np.random.default_rng(1)
n = 935_000
xyz = np.column_stack([rng.uniform(-15, 55, n), rng.uniform(-15, 55, n), rng.uniform(5, 15, n)])
layer = sylva.PointCloud(xyz, {"classification": np.full(n, 4, np.uint8)})
print(f"true PAD {2 * n / (70 * 70 * 10) * np.pi * 0.05 ** 2:.3f} m² m⁻³")
flight_3 = synthetic.als_flight(layer, altitude=60.0, line_spacing=20.0, footprint_samples=1,
                                target_radius=0.05, terrain_slope=0.0, bounds=(0, 0, 40, 40))
cat_3 = flight_3.write_tiles(tmp / "layer", size=20.0, epsg=32755)
traj_3 = als.Trajectory.from_dict(flight_3.trajectory)

prof = als.gap_profile(cat_3, traj_3, resolution=10.0, min_height=2.0, buffer=15.0)
inner = np.zeros(prof.shape[1:], bool)
inner[1:3, 1:3] = True                      # the central 20 m, away from the clipped edge
height, pad = prof.profile(inner)
print(f"gap profile: PAI {prof.pooled_pai(inner):.3f}, PAD 6-14 m {pad[(height >= 6) & (height < 14)].mean():.3f}")

vox = als.ray_voxelize(cat_3, traj_3, voxel_size=1.0, z_range=(-1, 19), buffer=15.0)
pai = vox.pai(min_beams=5)
cx, cy = pai.cell_centers()
middle = (cx > 10) & (cx < 30) & (cy > 10) & (cy < 30)
print(f"ray-traced voxels: median column PAI {np.nanmedian(pai.data[middle]):.3f}, "
      f"reach {vox.reach:.1f} m (within the 15 m buffer)")""",
    md("""`gap_profile` inverts the Beer-Lambert law layer by layer from the returns
counted in each 10 m cell (the profile of MacArthur and Horn, with each
return's beam angle from the trajectory), and `ray_voxelize` traces every
pulse through a voxel lattice. Both recover the layer. The flight is clipped
to the 40 m square, so pulses whose ground return fell outside it lost their
last return and the cells along that edge read too dense, which is why the
example keeps to the middle."""),
    """\
vh, vpad = vox.profile(min_beams=5, mask=middle)       # heights above the grid floor
fig, ax = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
ax[0].plot([0, 0.3, 0.3, 0], [5, 5, 15, 15], c=CONTEXT, lw=6, solid_joinstyle="miter", label="truth")
ax[0].stairs(np.nan_to_num(pad), np.r_[height, height[-1] + prof.bin_size], orientation="horizontal",
             baseline=None, color="C0", lw=1.5, label="gap profile")
ax[0].plot(vpad, vox.origin[2] + vh + 0.5, "o", ms=3.5, c="C1", label="ray-traced voxels")
ax[0].set(xlabel="PAD (m² m⁻³)", ylabel="height (m)", xlim=(-0.01, 0.4),
          title="Plant area density, central 20 m")
ax[0].legend(loc="upper right")
im = ax[1].imshow(pai.data, origin="lower", extent=ext(pai), vmin=2, vmax=4)
ax[1].plot([10, 30, 30, 10, 10], [10, 10, 30, 30, 10], "w--", lw=1.2)
fig.colorbar(im, ax=ax[1], shrink=0.9, label="PAI (true 3)", extend="both")
ax[1].set(xlabel="x (m)", ylabel="y (m)", title="PAI per 1 m column (dashed: central 20 m)");""",
]


NOTEBOOKS["16_waveform"] = [
    md("""# 16. Full waveforms

A full-waveform scanner digitises the received power of every pulse, a sample
every nanosecond or so, instead of reporting only the returns its own detector
found. `sylva.waveform` reads waveforms from LAS (wave packets) and PulseWaves
files, decomposes them into Gaussian echoes (Hofton et al. 2000; Wagner et
al. 2006), and turns the echoes into `Shots`, so that they feed the ray-traced
voxels like any other pulses. The [guide](../guide/waveform.md) has the
details.

`synthetic.waveforms` makes the waveforms of known targets: the system pulse
(a Gaussian of 1.5 ns standard deviation) convolved with the targets along
each beam, plus a background and Gaussian noise, and digitised. Every echo
found can therefore be checked against the target that made it."""),
    SYNTH_SETUP + "\nfrom sylva import Shots, als, voxels, waveform",
    md("""## Three waveforms

Each echo of the `Shots` given to `synthetic.waveforms` is a target, with
its peak from the echo attribute `amplitude` and its depth along the beam
from `extent`. The first pulse meets two targets 0.6 m apart and an extended
target (0.3 m deep, like a sloping surface in the footprint); the second, two
targets 0.3 m apart; the third, nothing."""),
    """\
origin = np.zeros((3, 3))
direction = np.tile([0.0, 0.0, -1.0], (3, 1))
ranges = np.array([50.0, 50.6, 55.0, 50.0, 50.3])
targets = Shots(origin, direction, np.array([0, 3, 5]), np.array([3, 2, 0]), ranges,
                {"amplitude": np.array([60.0, 40.0, 80.0, 50.0, 50.0]),
                 "extent": np.array([0.0, 0.0, 0.3, 0.0, 0.0])})
wf3, truth3 = synthetic.waveforms(targets, noise=2.0, seed=3)
echoes3 = wf3.decompose()
print(wf3)
print("true ranges (m):     ", truth3.range.round(3), "widths (ns):", truth3.width.round(2))
print("echoes per waveform: ", echoes3.stats["n_echoes"].tolist())
print("decomposed ranges (m):", echoes3.range.round(3), "widths (ns):", echoes3.width.round(2))""",
    md("""The first waveform gives back its three echoes, the extended target as a
wider one. The two targets 0.3 m apart in the second (1.3 pulse standard
deviations) merge into one echo: with the default smoothing two echoes are
separated from about two standard deviations apart, 0.45 m for this pulse.
The third waveform, whose record starts at the scanner since the pulse met
nothing, holds only noise and gives no echo."""),
    """\
fig, ax = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
titles = ["Three targets, three echoes", "Two targets 0.3 m apart, one echo", "No target: noise only, no echo"]
for i, a in enumerate(ax):
    w = wf3[i]
    r = w.positions()[:, 2] * -1                # range below the scanner, along the beam
    a.plot(r, w.samples, ".", ms=3, c="0.5", label="samples")
    k = echoes3.waveform == i
    bg, noise = echoes3.stats["background"][i], echoes3.stats["noise"][i]
    a.axhline(bg + 4 * noise, c="k", ls="--", lw=0.8, label="detection threshold")
    model = np.full_like(r, bg)
    for j, (t, amp, wid) in enumerate(zip(echoes3.time[k], echoes3.amplitude[k], echoes3.width[k])):
        g = amp * np.exp(-0.5 * ((w.times() - t) / wid) ** 2)
        a.plot(r, bg + g, lw=1, c="C1", label="fitted echoes" if j == 0 else None)
        model += g
    if k.any():
        a.plot(r, model, c="C0", lw=1.5, label="fitted sum")
    for j, rt in enumerate(truth3.range[truth3.waveform == i]):
        a.axvline(rt, c="C3", ls=":", lw=1.2, label="true targets" if j == 0 else None)
    a.set(xlim=(48, 58) if i < 2 else (r[0], r[-1]), xlabel="range (m)", title=f"Waveform {i}\\n{titles[i]}")
ax[0].set_ylabel("sample value")
fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower center", ncol=5);""",
    md("""## An airborne survey as waveforms

The targets are now the returns of a simulated airborne survey
(`synthetic.als_flight`) over a 50 m plot with ten trees: each pulse becomes a
shot from the scanner position through its returns, with the returns'
intensities as amplitudes. The noise is 2 sample units, so the weakest
returns are only a few times the noise."""),
    """\
rng = np.random.default_rng(0)
stems = [(x, y, 0.3, h) for x, y, h in
         zip(rng.uniform(5, 45, 10), rng.uniform(5, 45, 10), rng.uniform(10, 25, 10))]
scene = synthetic.forest(stems, size=50.0, ground_points=100, margin=0.0)
flight = synthetic.als_flight(scene, altitude=80.0, line_spacing=40.0, pulse_rate=20_000, bounds=(0, 0, 50, 50))

pts = flight.points                          # the returns of a pulse are consecutive and share gps_time
t = pts.attrs["gps_time"]
first = np.flatnonzero(np.r_[True, t[1:] != t[:-1]])
count = np.diff(np.r_[first, len(t)])
origin = flight.sensor_positions(t[first])
ranges = np.linalg.norm(pts.xyz - np.repeat(origin, count, axis=0), axis=1)
last = first + count - 1
direction = (pts.xyz[last] - origin) / ranges[last, None]
targets = Shots(origin, direction, first, count, ranges, {"amplitude": pts.attrs["intensity"] / 100.0})

wf, truth = synthetic.waveforms(targets, gps_time=t[first], noise=2.0, seed=1)
print(wf, "|", truth, "| amplitudes (5, 50, 95 %):", np.percentile(truth.amplitude, [5, 50, 95]).round(1))""",
    md("""## Writing and reading LAS wave packets

`write_las` writes LAS 1.4 point format 9 with the wave packets in an
extended VLR (or, with `external=True`, in a `.wdp` file beside it).
`waveform.info` describes a file without reading its waveforms, and
`waveform.read` reads them back; `waveform.chunks` reads a bounded number at
a time for files larger than memory."""),
    """\
path = tmp / "flight_wdp.las"
wf.write_las(path)
info = waveform.info(path)
print(f"{path.stat().st_size / 1e6:.0f} MB;", {k: info[k] for k in ("version", "point_format", "n_records", "packets")})
print("descriptor:", info["descriptors"][1])

back = waveform.read(path)
print(back)
print("samples identical:", np.array_equal(back.samples, wf.samples),
      "| gps_time identical:", np.array_equal(back.gps_time, wf.gps_time),
      f"| largest shift of a first sample: {1000 * np.abs(back.sample_positions()[:len(wf.samples)] - wf.sample_positions()).max():.2f} mm")""",
    md("""## Decomposition against the truth

The waveforms are decomposed chunk by chunk, as a large file would be. LAS
stores no scanner position, so the echoes of the file have a position but no
range; they are compared with the true targets by position, within the same
waveform."""),
    """\
parts, found_parts = [], []
for chunk in waveform.chunks(path, size=50_000):
    echoes = chunk.decompose()
    found_parts.append(echoes)
    parts.append(echoes.to_shots(chunk, origin=flight.sensor_positions(chunk.gps_time)))
shots = Shots.concatenate(parts)
print(shots)

echoes = back.decompose()                    # the whole file at once, for the comparison
lo = np.searchsorted(echoes.waveform, truth.waveform, "left")
hi = np.searchsorted(echoes.waveform, truth.waveform, "right")
miss = np.full(len(truth), np.inf)
for i in np.flatnonzero(hi > lo):
    miss[i] = np.linalg.norm(echoes.xyz[lo[i]:hi[i]] - truth.xyz[i], axis=1).min()
found = miss < 0.3
print(f"{len(echoes):,} echoes found for {len(truth):,} targets; {found.mean():.1%} of the targets "
      f"have an echo within 0.3 m, a median {1000 * np.median(miss[found]):.1f} mm away")
last_sample = wf.anchor + wf.direction * (wf.metres_per_ns * (wf.offset + (wf.sample_count - 1) * wf.interval))[:, None]
record_end = np.linalg.norm(last_sample - origin, axis=1)      # range of each waveform's last sample
inside = truth.range <= record_end[truth.waveform]
print(f"{(~inside).sum():,} targets ({(~inside).mean():.1%}) lie beyond the end of their digitised record; "
      f"of the {inside.sum():,} inside it, {found[inside].mean():.1%} are found")
snr = truth.amplitude / echoes.stats["noise"].mean()
for a, b in ((0, 6), (6, 10), (10, np.inf)):
    k = inside & (snr >= a) & (snr < b)
    print(f"  amplitude {a:>2} to {b:<4} times the noise: {k.sum():>7,} targets in the record, {found[k].mean():6.1%} found, "
          f"median distance {1000 * np.median(miss[k & found]):5.1f} mm")""",
    """\
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
bins = np.linspace(0, 60, 31)
counts = np.array([np.histogram(snr[k], bins)[0] for k in (found, ~found & inside, ~inside)])
share = counts / np.maximum(counts.sum(axis=0), 1)
bottom = np.zeros(len(bins) - 1)
for c, colour, name in zip(share, ["C0", "C3", "0.6"], ["found", "missed, in the record", "beyond the end of the record"]):
    ax[0].bar(bins[:-1], c, width=np.diff(bins), bottom=bottom, align="edge", color=colour, label=name)
    bottom += c
ax[0].axvline(4, c="k", ls="--", lw=1, label="detection threshold")
ax[0].set(xlabel="true amplitude / noise standard deviation", ylabel="fraction of the targets", ylim=(0, 1),
          xlim=(0, 60), title="Which targets are found, by amplitude")
fig.legend(loc="outside lower center", ncol=4)
ax[1].hist(1000 * miss[found], np.linspace(0, 40, 41), color="C0")
ax[1].set(xlabel="distance from the true target (mm)", ylabel="echoes", title="Position error of the echoes found");""",
    md("""Within the digitised record, the targets missed are the weak ones, below
or near the detection threshold of four noise standard deviations; those
found lie within millimetres of the truth. Most of the targets missed,
however, were never digitised: `synthetic.waveforms` records 120 samples at
1 ns from 3 m before a pulse's first target, which covers 18 m of range, so a
ground return more than 15 m below the first canopy return falls beyond the
end of the record. Almost all of these (98 %) are ground returns under tall
crowns; a longer record (`n_samples`) keeps them. A real scanner's
record length sets the same limit, and it is worth checking against the
canopy height before a survey's waveforms are used for canopy structure.

## From echoes to ray-traced voxels

`to_shots` makes one shot per pulse with its echoes sorted by range; a pulse
in which nothing was found stays a shot without an echo, which the tracer
counts as free space. With origins from the trajectory, the shots go straight
into `voxels.ray_voxelize`. For comparison, the discrete returns of the same
flight are made into pulses with `als.pulses`."""),
    """\
box = ((0, 0, -2), (50, 50, 30))
grid_wf = voxels.ray_voxelize(shots, 1.0, box)
grid_dr = voxels.ray_voxelize(als.pulses(pts, als.Trajectory.from_dict(flight.trajectory)), 1.0, box)
grid_tr = voxels.ray_voxelize(truth.to_shots(wf, origin=flight.sensor_positions(wf.gps_time)), 1.0, box)
print(grid_wf)
z = grid_wf.z_levels() + 0.5
pad = {name: g.profile("pad_fpl") for name, g in
       (("waveform echoes", grid_wf), ("discrete returns", grid_dr), ("true targets", grid_tr))}
print("true targets and discrete returns give the same profile:",
      np.allclose(pad["true targets"], pad["discrete returns"], equal_nan=True))
k = (z > 5) & (z < 25)
print(f"mean PAD 5-25 m: waveform {np.nanmean(pad['waveform echoes'][k]):.4f}, "
      f"discrete {np.nanmean(pad['discrete returns'][k]):.4f} m² m⁻³")""",
    """\
fig, ax = plt.subplots(figsize=(6, 4.5))
for (name, p), style, width in zip(pad.items(), ("-", "-", ":"), (1.5, 3, 1.5)):
    ax.plot(p[k], z[k], style, lw=width, label=name)
ax.set(xlabel="mean PAD (m² m⁻³)", ylabel="z (m)", title="Plant area density by layer, 1 m voxels")
ax.legend(loc="upper right");""",
    md("""The pulses built from the true targets reproduce the discrete-return pulses
exactly, so the difference between the waveform and discrete profiles comes
entirely from the echoes the decomposition missed. A pulse that loses an
echo gives its remaining echoes a larger share of the pulse and, when the echo
lost was its last, ends earlier; both raise the density read in the canopy. The same
effect is present, and not visible, in a discrete-return survey whose
detector drops weak returns; a waveform survey at least lets the threshold
be chosen and its effect measured."""),
]


NOTEBOOKS["17_synthetic"] = [
    md("""# 17. Synthetic trees, plots and scans

`sylva.synthetic` makes point clouds whose right answer is known, so that a
method can be checked against the truth rather than against another method.
Notebooks 3, 6, 7 and 9 use the realistic generators shown here:
`tree_model` grows a tree from an archetype with its cylinders, leaves and
leaf angles as the truth; `plot` makes a mixed stand on rough ground with
understorey and dead wood and a truth table; and `scan` with a finite beam
scans either one with range noise, a beam footprint, mixed pixels and
multiple returns. The [guide](../guide/synthetic.md) describes the models
and the checks behind them."""),
    SYNTH_SETUP + "\nimport pandas as pd\nfrom sylva import filters",
    md("""## Five archetypes

Each archetype sets the architecture (where the crown starts, how the stem
forks, how branches bend), the crown shape and the leaves: their size and
their angle distribution. Every tree here has a DBH of 30 cm and a target
height of 16 m, except the shrub: 4 m, with 3 cm stems."""),
    """\
names = ["broadleaf", "conifer", "eucalypt", "savanna", "shrub"]
models = {n: synthetic.tree_model(n, dbh=0.03 if n == "shrub" else 0.3, height=4.0 if n == "shrub" else 16.0,
                                  seed=2) for n in names}
pd.DataFrame.from_dict({n: {"DBH (cm)": 100 * t.dbh, "height (m)": t.height, "crown base (m)": t.crown_base,
                  "wood volume (L)": 1000 * t.qsm.total_volume, "stem share of the wood": t.qsm.stem_volume / t.qsm.total_volume,
                  "leaf area (m2)": t.leaf_area, "leaves": len(t.leaves["area"]),
                  "mean leaf angle (deg)": t.leaf_angles().mean_deg}
              for n, t in models.items()}, orient="index").round(2)""",
    """\
fig, ax = plt.subplots(1, 5, figsize=(10, 3.8), sharey=True, gridspec_kw={"width_ratios": [1, 0.8, 1, 1.3, 0.5]})
for a, (name, t) in zip(ax, models.items()):
    c = t.points[::4]
    wood = c.attrs["classification"] == 5
    a.scatter(c.x[wood] - t.base[0], c.z[wood], s=0.1, c=WOOD)
    a.scatter(c.x[~wood] - t.base[0], c.z[~wood], s=0.1, c=LEAF)
    a.set(aspect="equal", title=name, xlabel="x (m)")
ax[0].set_ylabel("z (m)")
fig.suptitle("The archetypes from the side: wood (brown) and leaves (green)");""",
    md("""The truth of a tree is complete: `t.qsm` is its wood as a table of
cylinders (the points lie on them), `t.leaves` lists every leaf with its
centre, normal and area, and `t.leaf_angles()` gives the leaf angle
distribution and its G function. The leaf angles below are each archetype's
default: planophile (mostly horizontal) blades for the broadleaf and the
savanna tree, erectophile (hanging) leaves for the eucalypt, and spherical
for the conifer's shoots and the shrub."""),
    """\
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
edges = np.linspace(0, 90, 19)
for i, (name, t) in enumerate(models.items()):
    inc = np.degrees(np.arccos(np.abs(t.leaves["normal"][:, 2])))
    h = np.histogram(inc, edges, weights=t.leaves["area"])[0]
    ax[0].stairs(h / h.sum(), edges, color=f"C{i}", lw=1.5, label=name)
    order = t.volume_by_order()
    ax[1].bar(np.array(list(order)) + 0.16 * (i - 2), 1000 * np.array(list(order.values())), width=0.16,
              color=f"C{i}", label=name)
ax[0].set(xlabel="leaf inclination (deg)", ylabel="fraction of the leaf area", xlim=(0, 90), title="Leaf angles")
ax[1].set(xlabel="branch order (0: stem)", ylabel="wood volume (L)", yscale="log", xticks=range(4),
          title="Wood volume by branch order")
ax[1].legend(loc="upper right");""",
    md("""## A stand

`plot` draws stem diameters from a distribution (here a Weibull), gives
each tree an archetype by weight and a height from the archetype's
height-diameter curve, places the stems without overlap, and adds terrain
with a slope and micro-relief, shrubs, grass, fallen logs and stumps. Every
point carries a `label`; the trees come with a truth table."""),
    """\
p = synthetic.plot(size=30, density=600, archetypes={"broadleaf": 1, "eucalypt": 1, "conifer": 1},
                   slope=0.15, roughness=0.08, seed=1)
print(f"{len(p.points):,} points; {p.stem_density:.0f} stems/ha, basal area {p.basal_area:.1f} m2/ha, "
      f"leaf area index of the trees {p.trees['leaf_area'].sum() / 30**2:.2f}")
pd.DataFrame(p.trees)[["tree_id", "archetype", "dbh", "height", "crown_base", "wood_volume", "leaf_area"]].head(6).round(2)""",
    """\
LABEL_COLOURS = {1: ("ground", GROUND), 2: ("stem", WOOD), 3: ("branch", "#9C6B3C"), 4: ("leaf", LEAF),
                 5: ("understorey", GRASS), 6: ("dead wood", "k")}
label = p.points.attrs["label"]
fig, ax = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1, 1.4]})
s = p.points[::10]
s = s[s.attrs["label"] != 1]                  # the ground, which the side view shows
order = np.argsort(s.z)
colours = np.array([LABEL_COLOURS[v][1] for v in range(1, 7)], dtype=object)[s.attrs["label"][order] - 1]
ax[0].scatter(s.x[order], s.y[order], c=list(colours), s=0.2)
ax[0].set(aspect="equal", xlabel="x (m)", ylabel="y (m)", title="From above, without the ground", xlim=(0, 30), ylim=(0, 30))
slab = np.abs(p.points.y - 15) < 1.0
for v, (name, colour) in LABEL_COLOURS.items():
    k = slab & (label == v)
    ax[1].scatter(p.points.x[k], p.points.z[k], s=0.3 if v != 6 else 1.0, c=colour, label=name)
ax[1].set(aspect="equal", xlabel="x (m)", ylabel="z (m)", title="A 2 m slice at y = 14 to 16 m")
ax[1].legend(markerscale=10, loc="upper left", bbox_to_anchor=(1.0, 1.0));""",
    md("""## A scan with a finite beam

Any beam option switches `scan` from taking the nearest point in each
angular cell to casting a beam: a cone of the scanner's divergence from its
exit aperture, sampled by sub-beams, with each point standing for a small
patch of surface. Hits closer in range than the receiver's resolution
merge into one echo at their energy-weighted mean range, which is how a
footprint straddling a stem's edge puts a point in the gap behind it: a
mixed pixel. Range noise is added along the beam. Here the stand is scanned
from its centre with the model of a RIEGL VZ-400."""),
    """\
x0, y0 = 15.0, 15.0
origin = (x0, y0, p.ground_height(x0, y0) + 1.5)
shots = synthetic.scan(p.points, origin=origin, resolution_deg=0.06, scanner="vz400", range_noise=0.005, seed=1)
echoes = shots.to_pointcloud()
spread = echoes.attrs["range_spread"]
print(shots)
print(f"echoes per pulse: {dict(enumerate(np.bincount(shots.echo_count).tolist()))}; "
      f"mixed pixels (hits spread over more than 5 cm in range): {np.mean(spread > 0.05):.1%} of the echoes")""",
    """\
mixed = spread > 0.05
label = echoes.attrs["label"]
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
slab = np.abs(echoes.y - y0) < 1.0
hb = ax[0].hexbin(echoes.x[slab], echoes.z[slab], C=mixed[slab], reduce_C_function=np.mean, gridsize=(60, 48),
                  mincnt=5, vmin=0, vmax=0.5, extent=(0, 30, -1, 24))
fig.colorbar(hb, ax=ax[0], shrink=0.9, label="share of mixed pixels", extend="max")
ax[0].plot(*origin[::2], "k^", ms=7)
ax[0].annotate("scanner", origin[::2], xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5)
ax[0].set(aspect="equal", xlabel="x (m)", ylabel="z (m)", xlim=(0, 30), ylim=(-1, 24),
          title="Mixed pixels (spread > 5 cm) in a 2 m slice\\nthrough the scanner")
names = {v: n for v, (n, _) in LABEL_COLOURS.items()}
kinds = [v for v in names if np.any(label == v)]
ax[1].barh([names[v] for v in kinds], [100 * mixed[label == v].mean() for v in kinds],
           color=[LABEL_COLOURS[v][1] for v in kinds])
ax[1].invert_yaxis()
ax[1].set(xlabel="mixed pixels (% of the echoes)", title="By the surface hit");""",
    md("""Mixed pixels gather where surfaces lie within the receiver's range
resolution of each other: in the foliage, where a footprint falls on a leaf
and the leaves just behind it, in the grass and shrubs, and on the ground
far from the scanner, where the footprint grazes it and spreads along it in
range. A stem seen
against the distant background gives two separate returns at its edge
instead, which is why stems and the ground come out cleanest. Every echo
keeps the attributes of its strongest target (`classification`, `label`,
`tree_id`), so each of these errors can be traced back to its cause."""),
]

NOTEBOOKS["18_buttress"] = [
    md("""# 18. Buttresses

A cylinder cannot follow a flanged base: where a tropical tree flares into
buttresses, a circle fitted to a slice of the stem spans the gaps between the
flanges. `trees.detect_buttress` finds such bases and `qsm.buttress_mesh`
rebuilds them as a closed mesh, which `Buttress.fuse` joins to the cylinder
model of the stem above. The [QSM guide](../guide/qsm.md#buttresses) has the
details; this notebook runs them on real trees.

The data are two trees from the destructive-harvest data of Burt et al.
(2021), each cloud holding a single tree: one with a flanged base, kept whole
(46 m tall, thinned to 1 cm up to 6 m, where the buttress functions read the
points, and to 5 cm above), and one with a round stem, kept to its lowest 8 m.
Both are in a frame with the stem at the origin and its foot at z = 0, so that z
is the height above ground the functions take; `make_buttress_subset.py`
cuts them."""),
    """\
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sylva
from sylva import qsm, trees

DATA = Path("data")          # the two trees, cut by make_buttress_subset.py
tmp = Path(tempfile.mkdtemp())      # files written here are temporary
""" + STYLE,
    """\
def read_tree(name):
    tree = sylva.read(DATA / name)
    return tree.with_attrs(height=tree.z)       # z is already the height above ground in this frame


tree = read_tree("buttress_tree.laz")
round_tree = read_tree("round_tree.laz")
lo, hi = tree.bounds
print(tree)
print("extent of the flanged tree:", np.round(hi - lo, 1), "m")
print(round_tree)
near = np.hypot(tree.x, tree.y) < 2.5       # the points around the stem, for the slice plots""",
    """\
def frame(ax, vertices):
    \"\"\"Axes limits and box proportions that match the data, with few ticks.\"\"\"
    lo, hi = vertices.min(0), vertices.max(0)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]),
           xlabel="x (m)", ylabel="y (m)", zlabel="height (m)")
    ax.set_box_aspect(hi - lo)
    ax.locator_params(nbins=4)                       # the boxes can be narrow
    ax.zaxis.labelpad = 8


def draw_mesh(ax, vertices, faces, colour, azim, elev=22):
    \"\"\"A triangle mesh with flat shading, lit from above and from one side.\"\"\"
    tri = vertices[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    sun = np.array([np.cos(np.radians(azim + 40)), np.sin(np.radians(azim + 40)), 1.0])
    shade = 0.35 + 0.65 * np.abs(normal @ sun) / np.linalg.norm(sun)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=shade[:, None] * np.array(to_rgb(colour)), linewidths=0))
    ax.view_init(elev=elev, azim=azim)


rng = np.random.default_rng(0)
whole = tree[rng.choice(len(tree), 80_000, replace=False)]     # a sample, to keep the figure light
foot = tree[tree.z < 2.5]                                      # the base, at full density
fig = plt.figure(figsize=(10, 6))
for k, (cloud, title, view) in enumerate([(whole, "The whole tree (a sample of the points)", 10),
                                          (foot, "The lowest 2.5 m (every point)", 22)]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    ax.scatter(cloud.x, cloud.y, cloud.z, s=0.4, c=cloud.z, cmap="viridis", linewidths=0, depthshade=False)
    frame(ax, cloud.xyz)
    ax.view_init(elev=view, azim=-60)
    ax.set_title(title)""",
    md("""The flanged tree's cloud also holds a sparse scatter of ground returns around
its foot. They are left in: `detect_buttress` reads only bark-like points and
`buttress_mesh` does not close ground into the solid. The slice plots below show
only the points within 2.5 m of the stem.

## Is the base buttressed?

`detect_buttress` looks only at bark-like points (locally planar, with a
near-horizontal normal) and asks two things: how much of the base a circle
explains compared with the stem higher up, and whether protrusions persist at
the same angles through the lowest metre. It finds the stem itself when it is
not told where it is."""),
    """\
b = trees.detect_buttress(tree)
r = trees.detect_buttress(round_tree)
for name, d in (("flanged", b), ("round", r)):
    print(f"{name:8s} buttressed: {str(d['buttressed']):5s} {d['ridges']} ridges; a circle explains "
          f"{d['base_circle_fit']:.0%} of the bark points below 1 m and {d['stem_circle_fit']:.0%} above 2 m")
print(f"the flanges of the first tree end {b['top']:.1f} m up; its round stem has a radius of {b['stem_radius']:.2f} m")""",
    """\
cx, cy = b["centre"]
theta = np.linspace(0, 2 * np.pi, 200)
fig, axes = plt.subplots(1, 4, figsize=(10, 3), sharex=True, sharey=True)
for ax, z in zip(axes, [0.3, 0.8, 1.4, 3.0]):
    s = (np.abs(tree.z - z) < 0.025) & near
    ax.scatter(tree.x[s], tree.y[s], s=1.5, c=WOOD)
    ax.plot(cx + b["stem_radius"] * np.cos(theta), cy + b["stem_radius"] * np.sin(theta), "--", c="0.4", lw=1)
    ax.set(title=f"{z} m", aspect="equal", xlabel="x (m)")
axes[0].set_ylabel("y (m)")
fig.suptitle("The flanged tree in slices 5 cm thick; dashed: the circle of its round stem");""",
    md("""Near the ground the section of the first tree is a star of flanges reaching
1.7 m from the stem axis, and the dashed circle of its round stem explains
little of it. The flanges merge into the trunk as the section rises, and by 3 m
it is round. The second tree has no flanges: a circle explains 72 % of its
base, so `detect_buttress` leaves it to the cylinder model."""),
    md("""## The base as a mesh

`buttress_mesh` rasterises thin slices of the points, closes the gaps that
occlusion leaves in the bark, and builds the sections from the top down, so
that each one contains the one above it. The stacked sections become a
watertight surface."""),
    """\
base = qsm.buttress_mesh(tree, b["centre"], top=b["top"])
print(f"{base.volume:.2f} m3 below {base.top:.1f} m in {len(base.vertices):,} vertices; "
      f"the outline was open (bark unseen) in {base.open.sum()} of {len(base.open)} slices")""",
    """\
def section(mesh, z):
    \"\"\"The segments where a mesh crosses the plane at height z, as an (n, 2, 2) array of xy.\"\"\"
    v, f = mesh.vertices, mesh.faces
    d = v[f, 2] - z                                   # height of each corner above the plane
    found = []
    for i, j in ((0, 1), (1, 2), (2, 0)):             # a triangle that crosses the plane cuts two of its edges
        cut = d[:, i] * d[:, j] < 0
        t = d[cut, i] / (d[cut, i] - d[cut, j])
        a, c = v[f[cut, i], :2], v[f[cut, j], :2]
        found.append((np.flatnonzero(cut), a + t[:, None] * (c - a)))
    face = np.concatenate([i for i, _ in found])
    pts = np.concatenate([p for _, p in found])[np.argsort(face, kind="stable")]
    return pts.reshape(-1, 2, 2)


fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
for a, z in zip(ax[:2], [0.3, 1.0]):
    s = (np.abs(tree.z - z) < 0.025) & near
    a.scatter(tree.x[s], tree.y[s], s=1.5, c=WOOD)
    a.add_collection(LineCollection(section(base, z), colors="C0", linewidths=1.2))
    a.set(title=f"Section at {z} m", aspect="equal", xlabel="x (m)")
    a.autoscale()
ax[0].set_ylabel("y (m)")
ax[0].plot([], [], c=WOOD, marker=".", ls="", label="points")
ax[0].plot([], [], c="C0", label="mesh")
ax[0].legend(loc="upper left")
ax[2].plot(base.areas, base.heights, c="C0")
ax[2].axvline(np.pi * b["stem_radius"] ** 2, ls="--", c="0.4", label="round stem")
ax[2].axhline(base.top, ls=":", c="0.4", label="top")
ax[2].set(xlabel="cross-section area (m$^2$)", ylabel="height (m)", title="Area by height")
ax[2].legend();""",
    md("""The mesh follows the flanges and the notches between them, and the
cross-section area falls from 3.0 m$^2$ at the ground to 1.1 m$^2$ at the top.
That is still above the 0.6 m$^2$ of the round stem, because the section is not
yet round there (see the 1.4 m slice above).

The mesh can only follow what was scanned. Where the bark is not seen to the
end of a notch between two flanges, the outline closes it, so it can run a
little outside the points, as along the lower edge of the 1 m section."""),
    """\
fig = plt.figure(figsize=(10, 4.6))
for k, azim in enumerate([-60, 30]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    draw_mesh(ax, base.vertices, base.faces, "C0", azim)
    frame(ax, base.vertices)
fig.suptitle(f"The buttress mesh from two sides: {base.volume:.2f} m$^3$ below {base.top:.1f} m");""",
    md("""## Against a cylinder model

A cylinder model fitted from the ground up has to cover the base with circles.
The volume below the top of the buttress, and the DBH at 1.3 m (which on this
tree still falls inside the flanges), show what that costs. There is no
measured volume for these trees here: the [QSM guide](../guide/qsm.md#buttresses)
and the [benchmark](../benchmarks/qsm.md) score buttress meshes against felled
trees."""),
    """\
plain = qsm.build_qsm(tree, base_xy=b["centre"])                                    # cylinders from the ground up
stem = qsm.build_qsm(tree[tree.attrs["height"] > base.top], base_xy=b["centre"])   # the stem and crown above the flanges
cyl_below = plain.total_volume - plain.volume_above(base.top_z)
print(f"DBH at 1.3 m: {plain.dbh:.2f} m from cylinders fitted from the ground, {stem.dbh:.2f} m on the round stem")
print(f"wood below {base.top:.1f} m: {cyl_below:.2f} m3 from the cylinders, {base.volume:.2f} m3 from the mesh "
      f"({base.volume / cyl_below - 1:+.0%})")
s = stem.summary()
print(f"the model above: {s['n_cylinders']:,} cylinders, {s['stem_volume_m3']:.1f} m3 of stem and "
      f"{s['branch_volume_m3']:.1f} m3 of branches, up to order {s['max_branch_order']}")""",
    md("""The circles cannot follow the flanges, so the base is mis-sized and the DBH
taken at 1.3 m is inflated, whichever way the volume goes. On the 15 buttressed
trees of the harvest benchmark the cylinder model alone is 10 % low in volume
against the felled trees, and 2 % high with the mesh below the top (see the
guide)."""),
    md("""## One mesh of the whole tree

`Buttress.fuse` cuts the cylinder model at the top of the buttress and puts both
surfaces in one object. The buttress replaces the wood below its top, so nothing
is counted twice and the volume is the one `total_volume` reports. The two parts
stay watertight and labelled (`part` per face). Fewer `sides` per tube keep the
mesh light for a tree with this many branches."""),
    """\
fused = base.fuse(stem, sides=8)
print(f"fused: {fused.volume:.1f} m3 = {fused.buttress_volume:.2f} m3 buttress + {fused.wood_volume:.1f} m3 cylinders above it, "
      f"{len(fused.faces):,} faces")
assert abs(fused.volume - base.total_volume(stem)) < 1e-9
fused.to_obj(tmp / "tree.obj")        # objects "buttress" and "wood"
fused.to_ply(tmp / "tree.ply")        # one mesh, the base in bark brown
print({f.name: f"{f.stat().st_size / 1e6:.0f} MB" for f in sorted(tmp.iterdir())})""",
    """\
fig = plt.figure(figsize=(10, 7))
centroid_z = fused.vertices[fused.faces, 2].mean(axis=1)
for k, (title, below, azim, elev) in enumerate([("The whole tree", np.inf, -60, 10),
                                                ("The lowest 8 m", 8.0, -60, 12)]):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d")
    for part, colour in ((0, "C0"), (1, "C1")):             # the buttress, then the cylinders above it
        faces = fused.faces[(fused.part == part) & (centroid_z < below)]
        draw_mesh(ax, fused.vertices, faces, colour, azim, elev)
    frame(ax, fused.vertices[fused.vertices[:, 2] < below])
    ax.set_title(title)
fig.suptitle("The fused mesh: buttress (blue) and cylinders (orange)");""",
    md("""On a real plot, run `detect_buttress` on each segmented tree and mesh the
ones it flags; `qsm.build_plot(..., buttress=True)` does both for every tree
and `plot.write_meshes` writes one fused surface per tree."""),
]


def build(name: str, cells: list, execute: bool = True) -> None:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = [nbformat.v4.new_markdown_cell(c) if isinstance(c, md) else nbformat.v4.new_code_cell(c.rstrip()) for c in cells]
    if execute:
        import tempfile
        with tempfile.TemporaryDirectory() as work:      # files the notebooks write stay out of the repo
            (Path(work) / "data").symlink_to(HERE / "data")   # ... but data is read from the repo
            (Path(work) / "sylva.mplstyle").symlink_to(HERE / "sylva.mplstyle")
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
