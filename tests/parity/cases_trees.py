"""Parity cases for sylva.trees (pruning, basal area, buttress detection)."""

import numpy as np

from sylva import PointCloud, trees


def _candidates(seed, n=40):
    """Candidates on a 20 m square with some near-duplicates, and point labels."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-10, 10, (n, 2))
    dup = rng.choice(n, n // 4, replace=False)
    xy[dup[: len(dup) // 2]] = xy[dup[len(dup) // 2: 2 * (len(dup) // 2)]] + rng.normal(0, 0.15, (len(dup) // 2, 2))
    labels = rng.integers(-1, n + 1, 5000)
    out = []
    for i in range(n):
        height = float(rng.uniform(0.5, 30.0)) if rng.uniform() > 0.1 else float("nan")
        t = trees.Tree(i + 1, float(xy[i, 0]), float(xy[i, 1]), float(rng.uniform(0.05, 1.2)), height,
                       int((labels == i + 1).sum()), float(rng.uniform()), int(rng.integers(2, 12)),
                       float(rng.uniform(0, 0.02)), float(rng.uniform(0, 20)), float(rng.uniform()))
        t.extra["plot"] = float(seed * 100 + i)
        out.append(t)
    return out, labels


def _table(prefix, ts):
    fields = ["tree_id", "x", "y", "dbh", "height", "n_points", "inlier_fraction", "n_slices", "rmse",
              "lean_deg", "quality"]
    out = {f"{prefix}{f}": np.array([getattr(t, f) for t in ts]) for f in fields}
    out[f"{prefix}extra_plot"] = np.array([t.extra.get("plot", np.nan) for t in ts], dtype=float)
    return out


def prune_trees():
    out = {}
    settings = [{}, {"merge_radius": 0.5}, {"min_height": 10.0, "max_dbh": 0.9},
                {"min_quality_short": 0.4, "short_slices": 6}, {"merge_radius": 2.0, "min_height": 0.0}]
    for seed in range(3):
        cands, labels = _candidates(seed)
        for k, kw in enumerate(settings):
            kept, lab = trees.prune_trees(cands, labels, **kw)
            out.update(_table(f"s{seed}_{k}_", kept))
            out[f"s{seed}_{k}_labels"] = lab
    # Ids larger than any label, and gaps in the ids.
    cands, labels = _candidates(5, n=12)
    for i, t in enumerate(cands):
        t.tree_id = 3 * i + 50
    kept, lab = trees.prune_trees(cands, labels % 7, merge_radius=1.0)
    out.update(_table("gaps_", kept))
    out["gaps_labels"] = lab
    return out


def basal_area():
    rng = np.random.default_rng(3)
    dbh = rng.uniform(0.02, 1.5, 300)
    dbh[rng.choice(300, 20, replace=False)] = np.nan
    ts = [trees.Tree(i + 1, 0.0, 0.0, float(d)) for i, d in enumerate(dbh)]
    return {"array": trees.basal_area(dbh, 2500.0), "trees": trees.basal_area(ts, 2500.0),
            "min_dbh": trees.basal_area(ts, np.pi * 30.0 ** 2, min_dbh=0.1),
            "scalar": trees.basal_area(0.4, 100.0), "empty": trees.basal_area([], 100.0)}


def count_ridges():
    rng = np.random.default_rng(4)
    out = {}
    for k in range(40):
        p = np.round(rng.uniform(0, 1, 36) * 10) / 10
        if k % 3 == 0:
            p = np.clip(np.convolve(np.r_[p, p[:4]], np.ones(5) / 5, "valid"), 0, 1)
        out[f"p{k}"] = trees._count_ridges(p)
        out[f"p{k}_level"] = trees._count_ridges(p, level=0.4, dip=0.1)
    out["none"] = trees._count_ridges(np.zeros(36))
    out["all"] = trees._count_ridges(np.ones(36))
    out["wrap"] = trees._count_ridges(np.r_[0.9, 0.9, 0.2, 0.9, 0.3, 0.8, 0.95, 0.7, 0.9])
    return out


def _base(seed, flanges, clutter, n=120_000, lean=0.0):
    """A 0.25 m stem to 6 m, optionally with five flanges fading out by 2 m
    and a grass clump; the centre drifts by ``lean`` per metre."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    h = rng.uniform(0, 6, n)
    r = 0.25 * (1 + (3 * np.clip(1 - h / 2, 0, None) * np.cos(2.5 * t) ** 8 if flanges else 0))
    pts = np.column_stack([r * np.cos(t) + 3.0 + lean * h, r * np.sin(t) - 2.0, h + 100.0])
    pts[:, :2] += rng.normal(0, 0.003, (n, 2))
    if clutter:
        g = rng.normal(0, 1, (n // 6, 3)) * [0.35, 0.35, 0.25] + [3.7, -1.8, 0.5]
        g = g[g[:, 2] > 0.05]
        g[:, 2] += 100.0
        pts = np.vstack([pts, g])
    return PointCloud(pts, {"height": pts[:, 2] - 100.0})


def _buttress(prefix, d):
    out = {f"{prefix}{k}": v for k, v in d.items() if k != "centre"}
    out[f"{prefix}centre"] = np.asarray(d["centre"])
    return out


def detect_buttress():
    out = {}
    flanged = _base(1, True, False)
    out.update(_buttress("flanged_", trees.detect_buttress(flanged, base_xy=(3.0, -2.0))))
    out.update(_buttress("flanged_auto_", trees.detect_buttress(flanged)))
    out.update(_buttress("flanged_raw_", trees.detect_buttress(flanged, base_xy=(3.0, -2.0, 100.0), bark_only=False,
                                                               voxel=0.0, bins=24, low=1.2)))
    clutter = _base(2, False, True)
    out.update(_buttress("round_", trees.detect_buttress(clutter, base_xy=(3.0, -2.0))))
    out.update(_buttress("round_raw_", trees.detect_buttress(clutter, base_xy=(3.0, -2.0), bark_only=False)))
    both = _base(3, True, True, lean=0.02)
    out.update(_buttress("both_", trees.detect_buttress(both, max_radius=2.0, slice_height=0.2, max_height=4.0)))
    # Without a height attribute z is used; too few points for any circle.
    sparse = PointCloud(_base(4, True, False, n=400).xyz - [0.0, 0.0, 100.0])
    out.update(_buttress("sparse_", trees.detect_buttress(sparse, base_xy=(3.0, -2.0))))
    return out


CASES = {"prune_trees": prune_trees, "basal_area": basal_area, "count_ridges": count_ridges,
         "detect_buttress": detect_buttress}
