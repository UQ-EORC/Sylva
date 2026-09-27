"""Parity cases for sylva.shots: CSR index arithmetic, angles, the misses
added from a scan pattern, and the functions already on the core."""

import tempfile
from pathlib import Path

import numpy as np

from sylva import PointCloud, Shots

PATTERN = {
    "theta_start": 30.0,
    "theta_delta": 0.5,
    "theta_count": 40,
    "phi_start": 0.0,
    "phi_delta": 3.0,
    "phi_count": 120,
}


def _shots(seed, n=400, origin=(0.0, 0.0, 1.5), max_echoes=4, spread=0.0):
    """Random pulses with zero to ``max_echoes`` echoes and two attributes."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    counts = rng.integers(0, max_echoes + 1, n)
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ranges = np.concatenate([np.sort(rng.uniform(0.5, 60.0, c)) for c in counts])
    o = np.asarray(origin, float) + rng.normal(0, spread, (n, 3))
    m = int(counts.sum())
    attrs = {
        "amplitude": rng.uniform(0, 40, m).astype(np.float32),
        "deviation": rng.integers(0, 60, m).astype(np.uint16),
    }
    return Shots(o, d, start, counts, ranges, attrs)


def _pattern_scan(seed, keep=0.7, origin=(1.0, -2.0, 1.5)):
    """Pulses on PATTERN's grid with jittered azimuths; a fraction ``keep``
    of them is observed (one echo each), the rest is missing."""
    rng = np.random.default_rng(seed)
    theta = PATTERN["theta_start"] + PATTERN["theta_delta"] * np.arange(PATTERN["theta_count"])
    phi = np.arange(PATTERN["phi_count"]) * PATTERN["phi_delta"]
    tt, pp = np.meshgrid(np.radians(theta), np.radians(phi))
    tt = tt.ravel() + rng.normal(0, 1e-4, tt.size)
    pp = pp.ravel() + rng.normal(0, 1e-3, pp.size)
    ok = rng.uniform(size=tt.size) < keep * (0.5 + 0.5 * np.cos(tt) ** 2)
    tt, pp = tt[ok], pp[ok]
    d = np.column_stack([np.sin(tt) * np.sin(pp), np.sin(tt) * np.cos(pp), np.cos(tt)])
    n = len(d)
    return Shots(
        np.tile(origin, (n, 1)),
        d,
        np.arange(n),
        np.ones(n, np.int64),
        rng.uniform(1, 30, n),
        {"reflectance": rng.normal(-5, 3, n)},
    )


def _flat(prefix, s):
    return {
        f"{prefix}origin": s.origin,
        f"{prefix}direction": s.direction,
        f"{prefix}echo_start": s.echo_start,
        f"{prefix}echo_count": s.echo_count,
        f"{prefix}echo_range": s.echo_range,
        **{f"{prefix}attr_{k}": v for k, v in s.echo_attrs.items()},
        f"{prefix}attr_names": np.array(sorted(s.echo_attrs), dtype=str),
    }


def index_arithmetic():
    out = {}
    for seed in range(3):
        s = _shots(seed, max_echoes=seed + 2)
        out[f"s{seed}/shot_of_echo"] = s.shot_of_echo()
        out[f"s{seed}/echo_rank"] = s.echo_rank()
        out[f"s{seed}/echo_xyz"] = s.echo_xyz()
        zen, az = s.zenith_azimuth()
        out[f"s{seed}/zenith"] = zen
        out[f"s{seed}/azimuth"] = az
    empty = _shots(9, n=5, max_echoes=0)
    out["empty/shot_of_echo"] = empty.shot_of_echo()
    out["empty/echo_rank"] = empty.echo_rank()
    out["empty/echo_xyz"] = empty.echo_xyz()
    # Axis-aligned beams: atan2 of signed zeros and exact quadrants.
    axes = Shots(
        np.zeros((6, 3)),
        np.array([[0, 1, 0], [1, 0, 0], [0, -1, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1.0]]),
        np.zeros(6, int),
        np.zeros(6, int),
        np.zeros(0),
    )
    out["axes/zenith"], out["axes/azimuth"] = axes.zenith_azimuth()
    return out


def subset():
    s = _shots(4)
    rng = np.random.default_rng(40)
    out = {}
    for name, mask in {
        "random": rng.uniform(size=s.n_shots) < 0.4,
        "none": np.zeros(s.n_shots, bool),
        "all": np.ones(s.n_shots, bool),
    }.items():
        out.update(_flat(f"{name}/", s.subset(mask)))
    out.update(_flat("ints/", s.subset((np.arange(s.n_shots) % 3 == 0).astype(int))))
    return out


def concatenate():
    a, b, c = _shots(5, n=50), _shots(6, n=70), _shots(7, n=30, max_echoes=0)
    del b.echo_attrs["deviation"]
    out = _flat("abc/", Shots.concatenate([a, b, c]))
    out.update(_flat("a/", Shots.concatenate([a])))
    out.update(_flat("ac/", Shots.concatenate([a, c])))
    # Mixed dtypes are promoted as NumPy promotes them.
    d = _shots(8, n=20)
    d.echo_attrs["amplitude"] = d.echo_attrs["amplitude"].astype(np.float64)
    d.echo_attrs["deviation"] = d.echo_attrs["deviation"].astype(np.int32)
    out.update(_flat("mixed/", Shots.concatenate([a, d])))
    return out


def fill_missing():
    out = {}
    s = _pattern_scan(10)
    for name, kw in {
        "estimated": {},
        "given": {"pulses_per_line": 118, "seed": 3},
        "stride": {"shot_stride": 2, "seed": 11},
        "large": {"pulses_per_line": 130, "seed": 12345678901},
    }.items():
        f = s.fill_missing(PATTERN, **kw)
        out[f"{name}/n_shots"] = np.array(f.n_shots)
        out[f"{name}/origin_added"] = f.origin[s.n_shots :][:3]
        out[f"{name}/direction_added"] = f.direction[s.n_shots :]
        out[f"{name}/echo_start"] = f.echo_start
        out[f"{name}/equal_head"] = np.array(np.array_equal(f.direction[: s.n_shots], s.direction))
    # Nothing missing: the shots come back unchanged.
    full = s.fill_missing(PATTERN, pulses_per_line=1)
    out["none/same"] = np.array(full is s)
    # Several origins: the added misses start at their mean.
    t = _shots(13, n=300, spread=0.01)
    out.update(_flat("spread/", t.fill_missing(PATTERN, pulses_per_line=3, seed=1)))
    return out


def core_functions():
    """Already on the core: points, transforms, the file format and the builders."""
    s = _shots(20)
    out = {}
    pc = s.to_pointcloud()
    out["pc/xyz"] = pc.xyz
    for k, v in pc.attrs.items():
        out[f"pc/{k}"] = v
    rng = np.random.default_rng(21)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    m = np.eye(4)
    m[:3, :3] = q * np.sign(np.linalg.det(q))
    m[:3, 3] = [3.0, -1.0, 2.0]
    out.update(_flat("transform/", s.transform(m)))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "s.parquet"
        s.save(path, double=True, row_group_size=100)
        out.update(_flat("load/", Shots.load(path)))
        out.update(_flat("groups/", Shots.load(path, groups=[1, 3])))
        info = Shots.file_info(path)
        for k in ("n_shots", "n_echoes", "n_groups"):
            out[f"info/{k}"] = np.array(info[k])
        out["info/echo_attrs"] = np.array(sorted(info["echo_attrs"]), dtype=str)
    xyz = rng.uniform(-10, 10, (200, 3))
    t = np.repeat(np.arange(120.0), [1, 2] * 40 + [2] * 40)[:200]
    cloud = PointCloud(xyz, {"gps_time": t, "intensity": rng.uniform(0, 1, 200)})
    out.update(_flat("from_pc/", Shots.from_pointcloud(cloud, origin=(0.5, 0.5, 1.0))))
    ray = PointCloud(
        xyz,
        {
            "nx": rng.normal(size=200),
            "ny": rng.normal(size=200),
            "nz": rng.uniform(1, 2, 200),
            "alpha": (rng.uniform(size=200) < 0.9).astype(np.float64),
        },
    )
    out.update(_flat("from_ray/", Shots.from_ray_cloud(ray)))
    return out


CASES = {
    f.__name__: f for f in (index_arithmetic, subset, concatenate, fill_missing, core_functions)
}
