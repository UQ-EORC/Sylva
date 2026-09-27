"""Parity cases for sylva.quality (scan ids, weighted medians, summaries)."""

import numpy as np

from sylva import PointCloud, quality
from sylva.quality import StemNoise


def _stems(seed, noise=0.003, shift=(0.02, 0.0), n_per=1500):
    """Nine stems on a grid, seen by four scans from the corners; scan 1 shifted."""
    rng = np.random.default_rng(seed)
    scans = np.array([[-15.0, -15.0], [15.0, -15.0], [15.0, 15.0], [-15.0, 15.0]])
    pts, ids, stems = [], [], []
    for i, (x, y) in enumerate(np.array(np.meshgrid([-6.0, 0.0, 6.0], [-6.0, 0.0, 6.0])).reshape(2, -1).T):
        r0 = 0.1 + 0.025 * i
        stems.append((x, y))
        for k, s in enumerate(scans):
            face = np.arctan2(s[1] - y, s[0] - x)
            a = face + rng.uniform(-np.pi / 2, np.pi / 2, n_per)
            r = r0 + rng.normal(0, noise, n_per)
            p = np.c_[x + r * np.cos(a), y + r * np.sin(a), rng.uniform(0.8, 3.2, n_per)]
            if k == 1:
                p[:, :2] += shift
            pts.append(p)
            ids.append(np.full(n_per, k))
    xyz = np.vstack(pts)
    return PointCloud(xyz, {"height": xyz[:, 2].copy()}), np.concatenate(ids), np.array(stems)


def scan_ids_from_origins():
    rng = np.random.default_rng(11)
    pos = rng.uniform(-30, 30, (6, 3))
    which = rng.integers(0, 6, 4000)
    o = pos[which] + rng.normal(0, 0.004, (4000, 3))
    # Exact half-cells: NumPy rounds them to even.
    half = np.array([[0.025, 0.075, -0.025], [0.125, -0.075, 0.175], [0.0, 0.0, 0.0], [-0.125, 0.025, 0.225]])
    return {"default": quality.scan_ids_from_origins(o), "coarse": quality.scan_ids_from_origins(o, tolerance=5.0),
            "half": quality.scan_ids_from_origins(half), "one": quality.scan_ids_from_origins(o[:1]),
            "list": quality.scan_ids_from_origins([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.0, 2.0, 3.0]])}


def wmedian():
    rng = np.random.default_rng(12)
    out = {}
    for k in range(20):
        n = int(rng.integers(1, 60))
        x = np.round(rng.normal(0, 1, n), 1 if k % 2 else 6)
        w = rng.integers(0, 5, n).astype(float) if k % 3 else rng.uniform(0, 2, n)
        if k % 4 == 0:
            x[rng.choice(n, max(1, n // 5), replace=False)] = np.nan
        out[f"m{k}"] = quality._wmedian(x, w)
    out["none"] = quality._wmedian([np.nan, 1.0], [1.0, 0.0])
    out["even"] = quality._wmedian([1.0, 2.0, 3.0, 4.0], [1, 1, 1, 1])
    return out


def _stem_noise_tables(prefix, q):
    out = {}
    for name in ("slices", "scan_slices", "scans"):
        for k, v in getattr(q, name).items():
            out[f"{prefix}{name}_{k}"] = v
    out[f"{prefix}residual"] = q.residual
    return out


def _summary(prefix, s):
    return {f"{prefix}{k}": v for k, v in s.items()}


def stem_noise():
    cloud, ids, stems = _stems(21)
    q = quality.stem_noise(cloud, scan_id=ids, stems=stems)
    out = _stem_noise_tables("multi_", q)
    out.update(_summary("multi_summary_", q.summary()))
    out.update(_summary("multi_summary5_", q.summary(min_scan_slices=5)))
    out.update(_summary("multi_summary99_", q.summary(min_scan_slices=99)))
    single = quality.stem_noise(_stems(22, noise=0.005, shift=(0.0, 0.0))[0], stems=stems, iterations=1)
    out.update(_stem_noise_tables("single_", single))
    out.update(_summary("single_summary_", single.summary()))
    empty = quality.stem_noise(cloud, scan_id=ids, stems=np.array([[50.0, 50.0]]))
    out.update(_summary("empty_summary_", empty.summary()))
    return out


def summary():
    sl = {"stem": np.array([0, 0, 1, 2]), "height": np.array([1.5, 2.0, 1.5, 1.75]),
          "n_points": np.array([100, 80, 120, 30]),
          "sigma": np.array([0.005, 0.004, 0.006, np.nan]), "sigma_first": np.array([0.006, 0.006, 0.007, 0.01]),
          "tail_fraction": np.array([0.0, 0.1, 0.05, 0.2])}
    ss = {"scan": np.array([0, 1, 1]), "slice": np.array([0, 1, 2]), "n_points": np.array([50, 50, 0]),
          "sigma_within": np.array([0.004, 0.003, 0.01]), "sigma_local": np.array([0.003, 0.002, 0.01])}
    sc = {"scan": np.array([0, 1, 2, 3]), "n_points": np.array([500, 400, 5, 300]),
          "n_slices": np.array([4, 3, 0, 4]), "tx": np.array([0.002, -0.002, 0.6, 0.001]),
          "ty": np.array([0.0, 0.001, 0.0, -0.003]), "sigma_within": np.array([0.004, 0.004, np.nan, 0.005]),
          "sigma_local": np.array([0.003, 0.003, np.nan, 0.004])}
    q = StemNoise(sl, ss, sc, np.zeros(0))
    out = _summary("a_", q.summary())
    out.update(_summary("b_", q.summary(min_scan_slices=0)))
    out.update(_summary("c_", q.summary(min_scan_slices=4)))
    return out


CASES = {"scan_ids_from_origins": scan_ids_from_origins, "wmedian": wmedian, "stem_noise": stem_noise,
         "summary": summary}
