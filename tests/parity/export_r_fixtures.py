"""Write inputs and Python outputs for the R package's cross-language tests.

    python tests/parity/export_r_fixtures.py

The R tests (r/sylva/tests/testthat/test-*-python.R) rebuild the same
objects from these files, call the R API and require the Python results to
1e-9.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parity import cases_canopy as cc  # noqa: E402
from sylva import canopy  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "r" / "sylva" / "tests" / "testthat" / "fixtures"


def num(x):
    x = float(x)
    return "NaN" if np.isnan(x) else ("Inf" if x == np.inf else ("-Inf" if x == -np.inf else repr(x)))


def arr(v):
    """R source for an array, Python's row-major values given R's dimensions."""
    a = np.asarray(v, dtype=float)
    vals = "c(" + ", ".join(num(x) for x in a.ravel()) + ")"
    if a.ndim <= 1:
        return vals
    if a.ndim == 2:
        return f"matrix({vals}, nrow = {a.shape[0]}, byrow = TRUE)"
    return f"aperm(array({vals}, c({', '.join(str(d) for d in reversed(a.shape))})), {a.ndim}:1)"


def r_value(v):
    if isinstance(v, dict):
        return "list(" + ", ".join(f"{k} = {r_value(x)}" for k, x in v.items()) + ")"
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if isinstance(v, np.ndarray):
        return arr(v)
    return num(v)


def save_shots(d, name, s):
    np.savetxt(d / f"{name}_rays.csv", np.column_stack([s.origin, s.direction, s.echo_start, s.echo_count]),
               delimiter=",", fmt="%.17g")
    np.savetxt(d / f"{name}_range.csv", s.echo_range, fmt="%.17g")


def canopy_fixtures():
    d = OUT / "canopy"
    d.mkdir(parents=True, exist_ok=True)
    cc.PATTERN = {"theta_start": 30.0, "theta_delta": 1.5, "theta_count": 70, "phi_start": 0.0,
                  "phi_delta": 6.0, "phi_count": 60}
    expected = {"pattern": cc.PATTERN}
    prof = canopy.GapProfile.empty()
    scans = [cc._scan(100 + k, origin=(k * 4.0, 0.0, 1.5), cover=0.5) for k in range(2)]
    for k, s in enumerate(scans):
        save_shots(d, f"scan{k}", s)
        prof.add_scan(s, cc._heights(s))
    r = prof.report()
    expected["report"] = dict(r)
    expected["pgap"] = np.asarray(prof.pgap())
    expected["pai_weighted"] = np.asarray(prof.pai_profile("weighted"))
    expected["pavd_linear"] = np.asarray(prof.pavd_profile("linear"))
    expected["clumping_47"] = float(prof.clumping(47.5))
    s = scans[0]
    hit = s.subset(s.echo_count > 0)
    edges = prof.zenith_edges
    expected["fired_pattern"] = np.asarray(canopy.fired_pulses_per_ring(hit, cc.PATTERN, edges))
    expected["fired_points"] = np.asarray(canopy.fired_pulses_from_points(hit, edges))
    c, g = canopy.gap_fraction_pattern(hit, cc._heights(hit), cc.PATTERN, min_height=2.0)
    expected["gap_pattern"] = np.asarray(g)
    rng = np.random.default_rng(7)
    pts = np.column_stack([rng.uniform(-10, 10, (3000, 2)), np.zeros(3000)])
    pts[:, 2] = 0.03 * pts[:, 0] + 1.0 + rng.normal(0, 0.01, 3000) + (rng.uniform(size=3000) < 0.2) * 5.0
    np.savetxt(d / "ground_points.csv", pts, delimiter=",", fmt="%.17g")
    expected["ground_plane"] = np.asarray(canopy.fit_ground_plane(pts))
    (d / "expected.R").write_text("expected <- " + r_value(expected) + "\n")
    print(f"wrote {d}")


if __name__ == "__main__":
    canopy_fixtures()
