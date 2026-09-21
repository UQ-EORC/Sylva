"""Generate and execute the example notebooks.

    python docs/examples/build_notebooks.py

Each notebook is defined below as a list of cells: a string starting with
``#`` followed by a space, or with plain prose, is Markdown when passed
through ``md()``; everything else is code. The notebooks run on
``sylva.synthetic`` scenes, so they need no data, and are saved with their
outputs so the documentation does not have to execute them.
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
import numpy as np
import matplotlib.pyplot as plt
import sylva
from sylva import synthetic

plt.rcParams.update({"figure.dpi": 90, "figure.figsize": (7, 4), "axes.grid": False})"""

NOTEBOOKS: dict[str, list] = {}

NOTEBOOKS["01_pointclouds_io"] = [
    md("""# 1. Point clouds and I/O

`sylva.PointCloud` is an `(N, 3)` float64 array plus named per-point attributes.
It is what readers return and what almost every function takes."""),
    SETUP,
    md("A synthetic plot to work with: sloped terrain and four trees. `classification` and `tree_id` are the ground truth."),
    """\
cloud = synthetic.forest()
print(cloud)
lo, hi = cloud.bounds
print("extent:", np.round(hi - lo, 2), "m")
{k: (v.dtype, v.min(), v.max()) for k, v in cloud.attrs.items()}""",
    md("Indexing with a mask, indices or a slice keeps the attributes aligned. `with_attrs` returns a copy with extra columns."),
    """\
wood = cloud[cloud.attrs["classification"] == 5]
cloud = cloud.with_attrs(range=np.linalg.norm(cloud.xyz - [10, 10, 1.5], axis=1).astype(np.float32))
print(wood, cloud, sep="\\n")""",
    """\
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
s = ax[0].scatter(cloud.x[::5], cloud.y[::5], c=cloud.z[::5], s=0.3, cmap="viridis")
ax[0].set(title="top view, coloured by z", xlabel="x (m)", ylabel="y (m)", aspect="equal")
fig.colorbar(s, ax=ax[0], label="z (m)")
ax[1].scatter(cloud.x[::5], cloud.z[::5], c=cloud.attrs["classification"][::5], s=0.3, cmap="Set1")
ax[1].set(title="side view, coloured by class", xlabel="x (m)", ylabel="z (m)", aspect="equal");""",
    md("""## Reading and writing

`sylva.read` / `sylva.write` pick the format from the extension: LAS/LAZ (extra
attributes become typed extra-bytes dimensions), PLY, and delimited text.
RIEGL `.rxp` needs RiVLib, see `sylva.io.read_rxp`."""),
    """\
import tempfile, pathlib
tmp = pathlib.Path(tempfile.mkdtemp())
for name in ("plot.laz", "plot.ply", "plot.csv"):
    sylva.write(cloud, tmp / name)
    back = sylva.read(tmp / name)
    print(f"{name:9s} {(tmp / name).stat().st_size / 1e6:6.2f} MB  {len(back):,} points  attrs: {sorted(back.attrs)}")""",
    """\
laz = sylva.read(tmp / "plot.laz")
print("max coordinate error after LAZ (1 mm scale):", np.abs(laz.xyz - cloud.xyz).max())
print("range attribute dtype preserved:", laz.attrs["range"].dtype)""",
    md("Plain arrays come in through `PointCloud.from_array`, naming the columns after x, y, z."),
    """\
arr = np.column_stack([cloud.xyz, cloud.attrs["classification"]])
sylva.PointCloud.from_array(arr, names={3: "classification"})""",
]

NOTEBOOKS["02_filtering"] = [
    md("""# 2. Filtering

Thinning, cropping, outlier removal and local geometry from `sylva.filters`."""),
    SETUP + "\nfrom sylva import filters\n\ncloud = synthetic.forest()",
    md("## Subsampling"),
    """\
thin = {
    "voxel 5 cm": filters.voxel_downsample(cloud, 0.05),
    "voxel 5 cm (centroid)": filters.voxel_downsample(cloud, 0.05, method="centroid"),
    "random 20 %": filters.random_subsample(cloud, fraction=0.2),
    "min distance 5 cm": filters.min_distance_subsample(cloud, 0.05),
}
for k, v in thin.items():
    print(f"{k:24s} {len(v):>8,} of {len(cloud):,}")""",
    md("## Crops"),
    """\
box = filters.crop_box(cloud, (0, 0, -1), (10, 10, 30))
cyl = filters.crop_cylinder(cloud, (8.0, 15.0), radius=3.0)
near = filters.range_filter(cloud, origin=(10, 10, 1.5), max_range=6.0)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True, sharey=True)
for a, (name, c) in zip(ax, {"crop_box": box, "crop_cylinder": cyl, "range_filter": near}.items()):
    a.scatter(cloud.x[::20], cloud.y[::20], s=0.2, c="0.8")
    a.scatter(c.x[::5], c.y[::5], s=0.3, c="C2")
    a.set(title=name, aspect="equal")""",
    md("## Outliers\n\nAdd 500 stray points and remove them again. Statistical removal compares each point's mean neighbour distance with the cloud-wide distribution; radius removal needs a minimum number of neighbours."),
    """\
rng = np.random.default_rng(1)
lo, hi = cloud.bounds
noise = rng.uniform(lo, hi + [0, 0, 5], (500, 3))
noisy = sylva.PointCloud(np.vstack([cloud.xyz, noise]))
is_noise = np.arange(len(noisy)) >= len(cloud)

sor = filters.statistical_outlier_removal(noisy, k=8, std_ratio=2.0, return_mask=True)
ror = filters.radius_outlier_removal(noisy, radius=0.25, min_neighbors=4, return_mask=True)
for name, keep in (("statistical", sor), ("radius", ror)):
    print(f"{name:12s} removed {np.sum(~keep & is_noise)} of 500 stray points and {np.sum(~keep & ~is_noise)} real ones")""",
    md("## Local geometry\n\nPCA over the `k` nearest neighbours gives normals and the planarity / linearity used by the wood filter: stems and branches are planar or linear at this scale, foliage is neither."),
    """\
t = synthetic.tree(seed=3)
planarity, linearity = filters.planarity_linearity(t, k=20)
fig, ax = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
for a, v, name in ((ax[0], planarity, "planarity"), (ax[1], linearity, "linearity")):
    s = a.scatter(t.x, t.z, c=v, s=0.4, cmap="magma", vmin=0, vmax=1)
    a.set(title=name, xlabel="x (m)", aspect="equal")
ax[0].set_ylabel("z (m)")
fig.colorbar(s, ax=ax, shrink=0.8);""",
    md("## Clustering\n\n`euclidean_clusters` labels connected components (points closer than `radius`). Above the ground the four trees separate."),
    """\
above = cloud[cloud.z - synthetic.terrain_height(cloud.x, cloud.y) > 0.5]
above = filters.voxel_downsample(above, 0.1)
labels = filters.euclidean_clusters(above.xyz, radius=0.35, min_points=50)
print("clusters:", labels.max() + 1, " unassigned points:", np.sum(labels < 0))
plt.scatter(above.x, above.y, c=labels, s=0.5, cmap="tab10")
plt.gca().set(aspect="equal", xlabel="x (m)", ylabel="y (m)");""",
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
    SETUP + "\nfrom sylva import ground\n\ncloud = synthetic.forest().without('classification')",
    md("""## Ground classification

Two filters: the cloth simulation filter (CSF, Zhang et al. 2016) drapes a
cloth over the inverted cloud; the progressive morphological filter (PMF,
Zhang et al. 2003) opens a minimum surface with growing windows. CSF suits
multi-scan plots and open terrain, PMF copes better with single scans where
some cells have no ground return at all."""),
    """\
truth = synthetic.forest().attrs["classification"] == 2
for name, f in (("CSF", ground.classify_ground_csf), ("PMF", ground.classify_ground_pmf)):
    g = ground.ground_mask(f(cloud))
    print(f"{name}: {g.sum():,} ground points; recall {np.mean(g[truth]):.3f}, false positives {np.sum(g & ~truth):,}")
cloud = ground.classify_ground_csf(cloud)""",
    md("## DTM and height normalisation"),
    """\
dtm = ground.make_dtm(cloud, resolution=0.5)
X, Y = dtm.cell_centers()
print("DTM", dtm.shape, "cells; RMSE against the true surface:",
      round(float(np.sqrt(np.nanmean((dtm.data - synthetic.terrain_height(X, Y)) ** 2))), 3), "m")
cloud = ground.normalize_height(cloud, dtm)          # adds the 'height' attribute
flat = ground.flatten(cloud, dtm)                    # or replace z itself""",
    """\
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].scatter(cloud.x[::5], cloud.z[::5], s=0.2, c=cloud.attrs["classification"][::5], cmap="coolwarm")
ax[0].set(title="elevation (ground in red)", xlabel="x (m)", ylabel="z (m)")
ax[1].scatter(cloud.x[::5], cloud.attrs["height"][::5], s=0.2, c="C2")
ax[1].set(title="height above ground", xlabel="x (m)", ylabel="height (m)");""",
    md("## Canopy height model"),
    """\
from sylva import canopy
chm = ground.make_chm(cloud, resolution=0.25)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, r, title, cmap in ((ax[0], dtm, "DTM (m)", "terrain"), (ax[1], chm, "CHM (m)", "YlGn")):
    im = a.imshow(r.data, origin="lower", extent=(r.xmin, r.xmax, r.ymin, r.ymax), cmap=cmap)
    a.set_title(title); fig.colorbar(im, ax=a, shrink=0.8)
print("canopy cover above 2 m:", round(canopy.canopy_cover(chm.data, 2.0), 3))
print("tallest point:", round(float(chm.data.max()), 2), "m (the tallest stem is 15 m, with foliage above it)")
chm.to_ascii_grid("chm.asc")   # or .to_geotiff() with rasterio""",
]

NOTEBOOKS["05_trees"] = [
    md("""# 5. Trees

Stem detection and DBH, segmentation of the cloud into trees, heights and
crown metrics. Everything here works on a height-normalised cloud (notebook 4)."""),
    SETUP + """
from sylva import ground, trees

cloud = ground.classify_ground_csf(synthetic.forest().without("classification"))
cloud = ground.normalize_height(cloud, ground.make_dtm(cloud, 0.5))""",
    md("## Stems\n\n`detect_stems` fits RANSAC circles in horizontal slices around breast height, links them vertically into stems, and reports DBH at 1.3 m with quality measures."),
    """\
stems = trees.detect_stems(cloud)
print(f"{'id':>2} {'x':>6} {'y':>6} {'dbh':>6} {'cover':>6} {'slices':>6} {'quality':>7}")
for t in stems:
    print(f"{t.tree_id:>2} {t.x:6.2f} {t.y:6.2f} {t.dbh:6.3f} {t.inlier_fraction:6.2f} {t.n_slices:>6} {t.quality:7.2f}")
print("\\ntruth (x, y, dbh):", [(x, y, d) for x, y, d, _ in synthetic.DEFAULT_TREES])""",
    md("## Segmentation\n\nEvery point is assigned to a stem by shortest path over a neighbourhood graph, then heights are measured and spurious candidates pruned."),
    """\
labels = trees.segment_trees(cloud, stems)
trees.tree_heights(cloud, labels, stems)
stems, labels = trees.prune_trees(stems, labels)
crowns = trees.crown_metrics_all(cloud, labels)
for t in stems:
    c = crowns[t.tree_id]
    print(f"tree {t.tree_id}: DBH {t.dbh:.3f} m, height {t.height:5.2f} m, crown area {c['crown_area']:5.1f} m2, "
          f"crown base {c['crown_base_height']:.1f} m")""",
    """\
truth = synthetic.forest().attrs["tree_id"]
veg = truth > 0
match = {t.tree_id: np.bincount(truth[labels == t.tree_id]).argmax() for t in stems}
correct = np.mean([match.get(l, -1) == g for l, g in zip(labels[veg][::10], truth[veg][::10])])
print(f"vegetation points on the right tree: {correct:.1%}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
m = labels > 0
ax[0].scatter(cloud.x[m][::3], cloud.y[m][::3], c=labels[m][::3], s=0.4, cmap="tab10")
for t in stems:
    ax[0].add_patch(plt.Circle((t.x, t.y), t.dbh / 2, fill=False, color="k"))
    ax[0].annotate(str(t.tree_id), (t.x + 0.4, t.y + 0.4))
ax[0].set(title="tree labels", aspect="equal")
ax[1].scatter(cloud.x[m][::3], cloud.attrs["height"][m][::3], c=labels[m][::3], s=0.4, cmap="tab10")
ax[1].set(title="side view", xlabel="x (m)", ylabel="height (m)");""",
    md("## Taper\n\n`dbh_profile` fits circles up the stem."),
    """\
big = max(stems, key=lambda t: t.dbh)
heights = np.arange(0.5, 8.0, 0.5)
diam = trees.dbh_profile(cloud, (big.x, big.y), heights=heights)
plt.plot(diam[:, 1], diam[:, 0], "o-")
plt.gca().set(xlabel="diameter (m)", ylabel="height (m)", title=f"taper of tree {big.tree_id}");""",
    md("Save the result: `sylva.write(cloud.with_attrs(tree_id=labels), 'plot_trees.laz')`."),
]

NOTEBOOKS["06_qsm"] = [
    md("""# 6. Quantitative structure models

From the points of one tree to a connected set of cylinders with volumes and
branch orders."""),
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
start, end = model.start, model.end
order = model.column("branch_order").astype(int)
fig, ax = plt.subplots(figsize=(5, 6))
for s, e, r, o in zip(start, end, model.column("radius"), order):
    ax.plot([s[0], e[0]], [s[2], e[2]], color=f"C{min(o, 9)}", lw=max(0.6, r * 120), solid_capstyle="round")
ax.set(aspect="equal", xlabel="x (m)", ylabel="z (m)", title="cylinders, coloured by branch order");""",
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
from sylva import canopy, ground

cloud = ground.classify_ground_csf(synthetic.forest().without("classification"))
cloud = ground.normalize_height(cloud, ground.make_dtm(cloud, 0.5))""",
    md("## Occupancy and the contact-frequency profile\n\n`voxelize` counts points per voxel. `pad_profile_voxel` turns the fraction of occupied voxels per layer into plant area density (a simplified Hosoi & Omasa 2006, assuming vertical beams and G = 0.5). It ignores occlusion, which is why pulse data is better."),
    """\
veg = cloud[cloud.attrs["height"] > 0.5]
grid = canopy.voxelize(veg, 0.25)
z, pad = canopy.pad_profile_voxel(veg, voxel_size=0.25)
print("voxel grid (nx, ny, nz):", grid.shape, " PAI:", round(float(np.sum(pad) * 0.25), 2))

hb, counts = canopy.vertical_profile(veg, bin_size=0.5)
fig, ax = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
ax[0].barh(hb, counts, height=0.5, align="edge"); ax[0].set(title="points per 0.5 m", ylabel="height (m)")
ax[1].plot(grid.vertical_profile(), grid.z_levels() - grid.origin[2]); ax[1].set(title="fraction of voxels occupied")
ax[2].plot(pad, z); ax[2].set(title="PAD (m2 m-3)");""",
    md("""## Gap fraction from pulses

A scanner in the middle of the plot fires pulses on an angular grid
(`synthetic.scan`; with RIEGL data use `ScanPosition.read_shots`). A pulse with no
echo above `min_height` is a gap. `lai_from_gap_fraction` inverts P(θ) by the
hinge angle (57.5°) or Miller's integral."""),
    """\
shots = synthetic.scan(synthetic.forest(), origin=(10.0, 10.0, 1.5), resolution_deg=0.25)
print(shots, f"- {np.mean(shots.echo_count == 0):.0%} of pulses returned nothing")
print(f"true leaf area index of the scene: {synthetic.leaf_area(synthetic.forest()) / 20**2:.2f} (plus bark)")
echo_height = shots.echo_xyz()[:, 2] - synthetic.terrain_height(*shots.echo_xyz()[:, :2].T)
zen, gap = canopy.gap_fraction_zenith(shots, echo_height, min_height=1.0, zenith_edges=np.arange(0, 95, 5.0))
for method in ("hinge", "miller"):
    print(f"effective PAI ({method}): {canopy.lai_from_gap_fraction(zen, gap, method):.2f}")
plt.plot(zen, gap, "o-"); plt.gca().set(xlabel="zenith (deg)", ylabel="gap fraction", ylim=(0, 1.02));""",
]

NOTEBOOKS["08_shots"] = [
    md("""# 8. Pulses and shots files

`sylva.Shots` stores pulses rather than points: an origin and direction per
pulse, with a CSR list of echo ranges. Pulses without a return stay in the data,
because they say where there was nothing."""),
    SETUP + "\nfrom sylva import Shots",
    """\
scene = synthetic.forest()
shots = synthetic.scan(scene, origin=(10.0, 10.0, 1.5), resolution_deg=0.2)
print(shots)
print("echoes per pulse:", dict(enumerate(np.bincount(shots.echo_count))))
print("echo attributes:", sorted(shots.echo_attrs))""",
    md("The echo arrays are flat; `echo_start` / `echo_count` say which belong to which pulse."),
    """\
s = int(np.flatnonzero(shots.echo_count == 2)[0])
a = shots.echo_start[s]
print("pulse", s, "direction", np.round(shots.direction[s], 3), "ranges", np.round(shots.echo_range[a:a + 2], 2))
zen, az = shots.zenith_azimuth()
first = shots.echo_rank() == 0
print("first returns:", first.sum(), " later returns:", (~first).sum())""",
    """\
hit = shots.echo_count > 0
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.scatter(az[~hit][::7], zen[~hit][::7], s=0.2, c="lightskyblue", label="no return")
ax.scatter(az[hit][::7], zen[hit][::7], s=0.2, c="darkgreen", label="return")
ax.set(xlabel="azimuth (deg)", ylabel="zenith (deg)", ylim=(130, 0), title="the scan as the scanner sees it")
ax.legend(markerscale=20, loc="lower right");""",
    md("Conversions: `to_pointcloud` gives the echoes as points (with `return_number`, `number_of_returns`, `range`); `from_pointcloud` and `from_ray_cloud` go the other way; `transform`, `subset` and `concatenate` behave as for point clouds."),
    """\
points = shots.to_pointcloud()
upward = shots.subset(zen < 60)
both = Shots.concatenate([shots, synthetic.scan(scene, origin=(4.0, 16.0, 1.5), resolution_deg=0.2)])
print(points, upward, both, sep="\\n")""",
    md("""## Shots files

`save` writes a Parquet file with one row per pulse: a scan index, two beam
angles and list columns for the echoes. A pulse without a return costs its two
angles and nothing else. Compare with storing the same pulses as a ray cloud,
where every miss is a far point carrying all the attributes."""),
    """\
import tempfile, pathlib
tmp = pathlib.Path(tempfile.mkdtemp())
both.save(tmp / "plot.parquet")

# The same pulses as a LAZ ray cloud: echoes, plus a point 100 m out for each miss.
miss = both.echo_count == 0
far = both.origin[miss] + 100 * both.direction[miss]
first_echo = both.echo_start[~miss]
ends = np.vstack([both.echo_xyz(), far])
starts = np.vstack([both.origin[both.shot_of_echo()], both.origin[miss]])
bound = np.r_[np.ones(both.n_echoes, np.uint8), np.zeros(miss.sum(), np.uint8)]
off = (starts - ends).astype(np.float32)
sylva.write(sylva.PointCloud(ends, {"sx": off[:, 0], "sy": off[:, 1], "sz": off[:, 2], "bound": bound}), tmp / "rays.laz")
for f in ("plot.parquet", "rays.laz"):
    print(f"{f:13s} {(tmp / f).stat().st_size / 1e6:6.2f} MB")""",
    """\
info = Shots.file_info(tmp / "plot.parquet")
print({k: v for k, v in info.items() if k != "scans"})
print("scanner positions:\\n", info["scans"])
back = Shots.load(tmp / "plot.parquet")
print("max echo position error after the float32 round trip:", f"{np.abs(back.echo_xyz() - both.echo_xyz()).max() * 1000:.3f} mm")""",
    md("Any Parquet reader opens the file (polars, pyarrow, duckdb, R arrow). Row groups can be read one at a time with `Shots.load(path, groups=[...])`, and `sylva.voxels.ray_voxelize` accepts the path and streams it (notebook 9)."),
]

NOTEBOOKS["09_voxels"] = [
    md("""# 9. Ray-traced voxels

`sylva.voxels` traces every pulse through a voxel grid and estimates the
attenuation coefficient λ of each voxel from how far pulses got, then plant area
density as λ / G. It follows AMAPVox and the rayvoxel tool; see the
[guide](../guide/voxels.md) for the estimators."""),
    SETUP + "\nfrom sylva import Shots, voxels",
    md("Four scan positions around the plot, combined into one set of pulses. The echoes carry `classification` (2 ground, 4 leaf, 5 wood) and `tree_id`."),
    """\
scene = synthetic.forest()
positions = [(3, 3), (17, 3), (3, 17), (17, 17), (10, 10)]
shots = Shots.concatenate([
    synthetic.scan(scene, origin=(x, y, synthetic.terrain_height(x, y) + 1.5), resolution_deg=0.2)
    for x, y in positions])
print(shots)""",
    """\
grid = voxels.ray_voxelize(
    shots, voxel_size=0.5, bounds=((0, 0, -0.5), (20, 20, 16.5)),
    ground_class=2, leaf_classes=[4], wood_classes=[5],
    attenuation=["fpl", "ppl"], laser="VZ-400", occlusion=True,
)
print(grid)
print("raw fields:", ", ".join(grid.fields[:12]), "...")""",
    md("## What the pulses saw\n\n`state` is 0 unobserved, 1 occluded (only reached behind a last echo), 2 empty, 3 filled. Arrays are `(nz, ny, nx)`."),
    """\
state = grid.state
names = ["unobserved", "occluded", "empty", "filled"]
print({n: f"{np.mean(state == i):.1%}" for i, n in enumerate(names)})
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
j = 10                                        # the row of voxels at y = 5 m, through two trees
ax[0].imshow(state[:, j, :], origin="lower", cmap="viridis", vmin=0, vmax=3); ax[0].set_title("state")
ax[1].imshow(np.log10(grid.num_beams[:, j, :] + 1), origin="lower", cmap="magma"); ax[1].set_title("log10 beams")
ax[2].imshow(grid.num_hits[:, j, :], origin="lower", cmap="Greens", vmax=50); ax[2].set_title("echoes");""",
    md("## Attenuation and area density\n\nFPL and PPL agree where voxels are well sampled. Leaf and wood area density split λ by echo class; `transmittance` is the beam-section weighted gap probability. The scene's true leaf area is known. Expect the estimate to be of the right size but low: the pseudo-scanner only sees each leaf disc through its 12 points, so some pulses slip through leaves a real beam would hit."),
    """\
ok = (grid.num_beams >= 20) & (grid.num_hits > 0)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
ax[0].loglog(grid.attenuation_fpl[ok], grid.attenuation_ppl[ok], ".", ms=1.5, alpha=0.4)
ax[0].plot([1e-3, 20], [1e-3, 20], "k", lw=0.6); ax[0].set(xlabel="λ FPL (1/m)", ylabel="λ PPL (1/m)")
z = grid.z_levels() + 0.25
for name in ("pad_ppl", "lad_ppl", "wad_ppl"):
    ax[1].plot(grid.profile(name, min_beams=20), z, label=name)
ax[1].set(xlabel="area density (m2 m-3)", ylabel="z (m)"); ax[1].legend()
ax[2].imshow(grid.transmittance[:, j, :], origin="lower", cmap="bone", vmin=0, vmax=1); ax[2].set_title("transmittance")
print("PAI:", round(float(np.nansum(grid.profile("pad_ppl", min_beams=20)) * grid.voxel_size), 2),
      " LAI:", round(float(np.nansum(grid.profile("lad_ppl", min_beams=20)) * grid.voxel_size), 2),
      " true LAI of the scene:", round(synthetic.leaf_area(scene) / 20**2, 2))""",
    md("## Leaf angles\n\nWith `inclination=True`, normals of the echoes give an inclination angle distribution per tree, and G is integrated over it and over the tree's beam zeniths instead of assuming a spherical distribution. The synthetic leaves are randomly oriented discs, so the result should be close to spherical with G ≈ 0.5."),
    """\
inc = voxels.ray_voxelize(shots, 0.5, ((0, 0, -0.5), (20, 20, 16.5)), ground_class=2,
                          leaf_classes=[4], wood_classes=[5], inclination=True)
for tid, t in inc.tree_iad.items():
    print(f"tree {tid}: leaves {t['liad_de_wit']:12s} G_leaf {t['g_leaf']:.2f}   wood {t['wiad_de_wit']:12s} G_wood {t['g_wood']:.2f}")
t = inc.tree_iad[3]
plt.step(np.degrees(t["bin_centres"]), t["liad"], where="mid", label="leaf")
plt.step(np.degrees(t["bin_centres"]), t["wiad"], where="mid", label="wood")
plt.gca().set(xlabel="inclination of the surface normal (deg)", ylabel="fraction", title="tree 3"); plt.legend();""",
    md("## Wood volume, files and streaming\n\nQSM cylinders can be rasterised into the same grid; `write` produces an AMAPVox `.vox` file or a text table; and a shots file is voxelised without loading it."),
    """\
from sylva import qsm
tree3 = scene[scene.attrs["tree_id"] == 3]
model = qsm.build_qsm(tree3[tree3.attrs["classification"] == 5], base_xy=(8.0, 15.0))
grid.add_wood_volume(model)
print(f"wood volume in the grid {grid.wood_volume.sum():.3f} m3 of {model.total_volume:.3f} m3 in the QSM")

n = grid.write("plot.vox")
print(n, "voxels written;", open("plot.vox").read().split("\\n")[6][:110], "...")

shots.save("plot.parquet")
streamed = voxels.ray_voxelize("plot.parquet", 0.5, ((0, 0, -0.5), (20, 20, 16.5)), ground_class=2)
print("streamed from file:", streamed, "- same hits:", int(streamed.num_hits.sum()) == int(grid.num_hits.sum()))""",
]


def build(name: str, cells: list, execute: bool = True) -> None:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells = [nbformat.v4.new_markdown_cell(c) if isinstance(c, md) else nbformat.v4.new_code_cell(c) for c in cells]
    if execute:
        import tempfile
        with tempfile.TemporaryDirectory() as work:      # files the notebooks write stay out of the repo
            NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": work}}).execute()
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
