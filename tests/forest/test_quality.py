import numpy as np
import pytest

from sylva import PointCloud, quality


def _stems(rng, noise=0.003, shift=(0.02, 0.0), n_per=6000):
    """Nine 0.1-0.3 m stems on a grid, seen by four scans from the plot's
    corners (each sees the half of a stem that faces it); scan 1 shifted."""
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


def test_stem_noise_recovers_noise_and_offset(rng):
    cloud, ids, stems = _stems(rng)
    q = quality.stem_noise(cloud, scan_id=ids, stems=stems)
    s = q.summary()
    assert s["n_stems"] == 9 and s["n_scans"] == 4
    assert s["sigma_local"] == pytest.approx(0.003, abs=0.001)
    assert s["sigma_total"] > s["sigma_corrected"]  # moving scan 1 back tightens the stems
    sc = q.scans
    # Offsets relative to the mean of four: scan 1 +15 mm, the others -5 mm in x.
    k = sc["scan"] == 1
    assert sc["tx"][k][0] == pytest.approx(0.015, abs=0.002) and abs(sc["ty"][k][0]) < 0.002
    assert np.all(np.abs(sc["tx"][~k] + 0.005) < 0.002)
    assert s["worst_scan"] == 1 and s["registration_max"] == pytest.approx(0.015, abs=0.002)
    assert np.isfinite(q.residual).sum() > 0.3 * len(cloud)  # 10 cm slices every 25 cm


def test_stem_noise_grows_with_noise(rng):
    lo = quality.stem_noise(_stems(rng, noise=0.002, shift=(0, 0))[0], stems=_stems(rng)[2]).summary()
    hi = quality.stem_noise(_stems(rng, noise=0.008, shift=(0, 0))[0], stems=_stems(rng)[2]).summary()
    assert lo["sigma_local"] == pytest.approx(0.002, abs=0.001)
    assert hi["sigma_local"] == pytest.approx(0.008, abs=0.0015)
    assert lo["n_scans"] == 1 and "registration_rms" not in lo


def test_scan_ids_from_origins():
    o = np.array([[0, 0, 1.5], [0.01, 0, 1.5], [10, 0, 1.6], [10, 0.02, 1.6], [0, 0, 1.51]])
    ids = quality.scan_ids_from_origins(o)
    assert ids[0] == ids[1] == ids[4] and ids[2] == ids[3] and ids[0] != ids[2]


def test_summary_skips_unsupported_scans():
    from sylva.quality import StemNoise

    sl = {"stem": np.array([0, 0]), "height": np.array([1.5, 2.0]),
          "n_points": np.array([100, 100]),
          "sigma": np.array([0.005, 0.005]), "sigma_first": np.array([0.006, 0.006]),
          "tail_fraction": np.array([0.0, 0.0])}
    ss = {"scan": np.array([0, 1]), "slice": np.array([0, 1]), "n_points": np.array([50, 50]),
          "sigma_within": np.array([0.004, 0.004]), "sigma_local": np.array([0.003, 0.003])}
    # Scan 2 saw five stem points and no slice: its 0.6 m offset is fitted to nothing.
    sc = {"scan": np.array([0, 1, 2]), "n_points": np.array([500, 500, 5]),
          "n_slices": np.array([4, 4, 0]), "tx": np.array([0.002, -0.002, 0.6]),
          "ty": np.array([0.0, 0.0, 0.0]), "sigma_within": np.array([0.004, 0.004, np.nan]),
          "sigma_local": np.array([0.003, 0.003, np.nan])}
    s = StemNoise(sl, ss, sc, np.zeros(0)).summary()
    assert s["n_scans_registered"] == 2 and s["registration_max"] == pytest.approx(0.002)
    loose = StemNoise(sl, ss, sc, np.zeros(0)).summary(min_scan_slices=0)
    assert loose["registration_max"] == pytest.approx(0.6)
