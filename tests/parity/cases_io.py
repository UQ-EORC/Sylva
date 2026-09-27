"""Parity cases for sylva.io."""

import tempfile
from pathlib import Path

import numpy as np

from sylva import PointCloud, io


def _cloud(seed=31, n=500):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0, 50, (n, 3)) + [500000.0, 7000000.0, 100.0]
    return PointCloud(xyz, {
        "intensity": rng.integers(0, 65535, n).astype(np.uint16),
        "classification": rng.integers(1, 3, n).astype(np.uint8),
        "gps_time": rng.uniform(0, 1e5, n),
        "height": rng.uniform(0, 30, n).astype(np.float32),
        "tree_id": rng.integers(-1, 20, n).astype(np.int32),
        "red": rng.integers(0, 65535, n).astype(np.uint16),
        "green": rng.integers(0, 65535, n).astype(np.uint16),
        "blue": rng.integers(0, 65535, n).astype(np.uint16),
    })


def _flat(prefix, c):
    out = {f"{prefix}_xyz": c.xyz, f"{prefix}_names": np.array(sorted(c.attrs))}
    for k, v in c.attrs.items():
        out[f"{prefix}_{k}"] = v
    return out


def round_trips():
    c = _cloud()
    out = {}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for name, kw in [("las6.las", {}), ("las7.laz", {"point_format": 7, "scale": 0.01}),
                         ("ply_bin.ply", {}), ("ply_ascii.ply", {"binary": False}),
                         ("txt.txt", {}), ("csv.csv", {})]:
            io.write(c, d / name, **kw)
            out.update(_flat(name.split(".")[0], io.read(d / name)))
    return out


def ascii_columns():
    rng = np.random.default_rng(32)
    a = rng.uniform(0, 10, (40, 5))
    out = {}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        np.savetxt(d / "plain.xyz", a, fmt="%.6f")
        np.savetxt(d / "semi.txt", a, fmt="%.6f", delimiter=";", header="x;y;z;a;b", comments="")
        (d / "pts.pts").write_text("40\n" + "\n".join(" ".join(f"{v:.6f}" for v in r) for r in a) + "\n")
        out.update(_flat("plain", io.read_ascii(d / "plain.xyz")))
        out.update(_flat("named", io.read_ascii(d / "plain.xyz", columns=["r", "s"])))
        out.update(_flat("all", io.read_ascii(d / "plain.xyz", columns=["x", "y", "z", "p", "q"])))
        out.update(_flat("header", io.read_ascii(d / "semi.txt")))
        out.update(_flat("pts", io.read(d / "pts.pts")))
        m = rng.normal(size=(4, 4))
        (d / "sop.dat").write_text("\n".join(" ".join(repr(float(v)) for v in r) for r in m) + "\n")
        out["matrix"] = io.read_matrix_file(d / "sop.dat")
    return out


CASES = {"round_trips": round_trips, "ascii_columns": ascii_columns}
