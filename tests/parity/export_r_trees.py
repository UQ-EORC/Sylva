"""Write inputs and Python outputs for the R package's tree and quality tests.

    python tests/parity/export_r_trees.py

The R tests (r/sylva/tests/testthat/test-trees-python.R and
test-quality-python.R) read the same points, call the R API and require the
Python results to 1e-9.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parity import cases_quality as cq  # noqa: E402
from parity import cases_trees as ct  # noqa: E402
from parity.export_r_fixtures import OUT, r_value  # noqa: E402
from sylva import PointCloud, quality, trees  # noqa: E402

FIELDS = ["tree_id", "x", "y", "dbh", "height", "n_points", "inlier_fraction", "n_slices", "rmse", "lean_deg",
          "quality"]


def table(ts):
    return {f: np.array([float(getattr(t, f)) for t in ts]) for f in FIELDS}


def save(d, name, a):
    np.savetxt(d / f"{name}.csv", np.atleast_2d(np.asarray(a, dtype=float)).reshape(len(a), -1), delimiter=",",
               fmt="%.17g")


def plot(rng):
    """Three stems with ball crowns over flat ground, heights equal to z."""
    parts = [np.column_stack([rng.uniform(0, 16, (1500, 2)), rng.normal(0, 0.01, 1500)])]
    for x, y, r, h in [(4.0, 4.0, 0.15, 9.0), (11.0, 5.0, 0.1, 7.0), (7.0, 12.0, 0.22, 10.0)]:
        n = int(350 * h)
        t = rng.uniform(0, 2 * np.pi, n)
        rr = r + rng.normal(0, 0.003, n)
        parts.append(np.column_stack([x + rr * np.cos(t), y + rr * np.sin(t), rng.uniform(0, h, n)]))
        v = rng.normal(size=(800, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        v *= rng.uniform(0.6, 1.0, (800, 1)) ** (1 / 3) * 2.0
        parts.append(v + [x, y, h - 2.0])
    return np.vstack(parts)


def trees_fixtures():
    d = OUT / "trees"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(31)
    exp = {}
    # Circles and hulls.
    theta = rng.uniform(0, np.pi, 150)
    xy = np.column_stack([3 + 0.2 * np.cos(theta), -1 + 0.2 * np.sin(theta)]) + rng.normal(0, 0.002, (150, 2))
    xy = np.vstack([xy, rng.uniform(2.5, 3.5, (40, 2))])
    save(d, "circle_xy", xy)
    exp["fit_circle"] = np.array(trees.fit_circle(xy))
    cx, cy, r, inl = trees.fit_circle_ransac(xy, threshold=0.01, seed=3)
    exp["ransac"] = np.array([cx, cy, r])
    exp["ransac_inliers"] = inl.astype(float)
    exp["hull"] = trees.convex_hull_area(xy)
    # The plot chain.
    xyz = plot(rng)
    save(d, "plot_xyz", xyz)
    cloud = PointCloud(xyz, {"height": xyz[:, 2].copy()})
    found = trees.detect_stems(cloud)
    exp["stems"] = table(found)
    kept, merged = trees.merge_branches(cloud, found)
    exp["merged"] = table(kept)
    exp["merged_into"] = np.asarray(merged, dtype=float)
    labels = trees.segment_trees(cloud, kept)
    save(d, "labels", labels)
    trees.tree_heights(cloud, labels, kept)
    exp["heights"] = table(kept)
    exp["dbh_profile"] = trees.dbh_profile(cloud, (kept[0].x, kept[0].y), heights=np.array([1.0, 2.0, 3.5]))
    allm = trees.crown_metrics_all(cloud, labels)
    ids = sorted(allm)
    exp["crowns"] = {"tree_id": np.array(ids, dtype=float),
                     **{k: np.array([allm[i][k] for i in ids]) for k in allm[ids[0]]}}
    exp["crown_1"] = trees.crown_metrics(cloud, labels, 1)
    exp["crown_shape"] = trees.crown_shape(cloud[labels == 1], base_xy=(kept[0].x, kept[0].y), crown_base=5.0)
    # Pruning on the parity candidates, with an extra column carried along.
    cands, lab = ct._candidates(0)
    save(d, "candidates", np.column_stack([table(cands)[f] for f in FIELDS] + [[t.extra["plot"] for t in cands]]))
    save(d, "candidate_labels", lab)
    for k, kw in enumerate([{}, {"merge_radius": 2.0, "max_dbh": 0.9, "min_quality_short": 0.4}]):
        p, pl = trees.prune_trees(cands, lab, **kw)
        exp[f"prune_{k}"] = {**table(p), "plot": np.array([t.extra["plot"] for t in p])}
        exp[f"prune_{k}_labels"] = pl.astype(float)
    exp["basal_area"] = trees.basal_area(cands, 2500.0, min_dbh=0.1)
    # Buttresses.
    flanged = ct._base(1, True, True, n=12_000)
    save(d, "buttress_xyz", flanged.xyz)
    save(d, "buttress_h", flanged.attrs["height"])
    for name, kw in [("buttress", {"base_xy": (3.0, -2.0)}), ("buttress_auto", {}),
                     ("buttress_raw", {"bark_only": False, "voxel": 0.0, "bins": 24})]:
        b = trees.detect_buttress(flanged, **kw)
        exp[name] = {k: (float(v) if k != "centre" else np.asarray(v)) for k, v in b.items()}
    (d / "expected.R").write_text("expected <- " + r_value(exp) + "\n")
    print(f"wrote {d}")


def quality_fixtures():
    d = OUT / "quality"
    d.mkdir(parents=True, exist_ok=True)
    exp = {}
    cloud, ids, stems = cq._stems(41, n_per=500)
    save(d, "stems_xyz", cloud.xyz)
    save(d, "scan_ids", ids)
    save(d, "stems", stems)
    q = quality.stem_noise(cloud, scan_id=ids, stems=stems, thickness=0.15, min_scan_points=5)
    for name in ("slices", "scan_slices", "scans"):
        exp[name] = {k: np.asarray(v, dtype=float) for k, v in getattr(q, name).items()}
    exp["residual"] = q.residual
    exp["summary"] = {k: float(v) for k, v in q.summary().items()}
    exp["summary_4"] = {k: float(v) for k, v in q.summary(min_scan_slices=4).items()}
    rng = np.random.default_rng(42)
    o = rng.uniform(-30, 30, (5, 3))[rng.integers(0, 5, 300)] + rng.normal(0, 0.004, (300, 3))
    save(d, "origins", o)
    exp["scan_ids"] = quality.scan_ids_from_origins(o).astype(float)
    exp["scan_ids_coarse"] = quality.scan_ids_from_origins(o, tolerance=5.0).astype(float)
    (d / "expected.R").write_text("expected <- " + r_value(exp) + "\n")
    print(f"wrote {d}")


if __name__ == "__main__":
    trees_fixtures()
    quality_fixtures()
