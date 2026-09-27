"""Write inputs and Python outputs for the R package's shots tests.

    python tests/parity/export_r_shots.py

The R tests (r/sylva/tests/testthat/test-shots-python.R) rebuild the same
pulses from these files, call the R API and require the Python results to
1e-9. ``python.parquet`` is a shots file written by the Python package, for
R to read.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r_fixtures_util import write_expected  # noqa: E402

from parity import cases_shots as cs  # noqa: E402
from parity.export_r_fixtures import OUT, save_shots  # noqa: E402
from sylva import Shots  # noqa: E402

PATTERN = {
    "theta_start": 40.0,
    "theta_delta": 1.0,
    "theta_count": 30,
    "phi_start": 0.0,
    "phi_delta": 4.0,
    "phi_count": 90,
}


def save(d, name, s):
    save_shots(d, name, s)
    names = sorted(s.echo_attrs)
    if names:
        np.savetxt(
            d / f"{name}_attrs.csv",
            np.column_stack([np.asarray(s.echo_attrs[k], float) for k in names]),
            delimiter=",",
            fmt="%.17g",
            header=",".join(names),
            comments="",
        )


def flat(s):
    out = {
        "origin": s.origin,
        "direction": s.direction,
        "echo_start": s.echo_start.astype(float),
        "echo_count": s.echo_count.astype(float),
        "echo_range": s.echo_range,
    }
    out["attrs"] = {k: np.asarray(v, float) for k, v in sorted(s.echo_attrs.items())} or {
        "none": np.zeros(0)
    }
    return out


def main():
    d = OUT / "shots"
    d.mkdir(parents=True, exist_ok=True)
    a, b = cs._shots(1, n=60), cs._shots(2, n=40, max_echoes=2)
    del b.echo_attrs["deviation"]
    save(d, "a", a)
    save(d, "b", b)
    e = {
        "shot_of_echo": a.shot_of_echo() + 1.0,
        "echo_rank": a.echo_rank() + 1.0,
        "echo_xyz": a.echo_xyz(),
    }
    e["zenith"], e["azimuth"] = a.zenith_azimuth()
    keep = np.arange(a.n_shots) % 3 != 1
    e["subset"] = flat(a.subset(keep))
    e["concatenate"] = flat(Shots.concatenate([a, b, a]))

    cs.PATTERN = PATTERN
    p = cs._pattern_scan(3, keep=0.6)
    save(d, "pattern", p)
    e["pattern"] = PATTERN
    fills = {}
    for name, kw in {
        "estimated": {},
        "given": {"pulses_per_line": 80, "seed": 4},
        "stride": {"shot_stride": 3, "seed": 9},
    }.items():
        f = p.fill_missing(PATTERN, **kw)
        fills[name] = {
            "n_shots": float(f.n_shots),
            "added": f.direction[p.n_shots :],
            "origin": f.origin[-1],
        }
    e["fill"] = fills

    path = d / "python.parquet"
    a.save(path, double=True, row_group_size=25)
    info = Shots.file_info(path)
    e["info"] = {
        "n_shots": float(info["n_shots"]),
        "n_echoes": float(info["n_echoes"]),
        "n_groups": float(info["n_groups"]),
        "min": np.asarray(info["bounds"][0]),
        "max": np.asarray(info["bounds"][1]),
        "scans": np.asarray(info["scans"]),
    }
    e["groups"] = flat(Shots.load(path, groups=[0, 2]))
    write_expected(d, e)


if __name__ == "__main__":
    main()
