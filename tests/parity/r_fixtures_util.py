"""Helpers for the R fixture exporters of the io, filters, ground, raster,
registration, pointcloud and limits modules."""

import numpy as np
from export_r_fixtures import OUT, r_value  # noqa: F401


def save_cloud(d, name, cloud):
    """``name.csv`` with x, y, z and every attribute, one column each."""
    cols = {"x": cloud.x, "y": cloud.y, "z": cloud.z, **cloud.attrs}
    header = ",".join(cols)
    rows = np.column_stack([np.asarray(v, dtype=float) for v in cols.values()])
    fmt = ["%.17g" if np.asarray(v).dtype.kind == "f" else "%d" for v in cols.values()]
    np.savetxt(d / f"{name}.csv", rows, delimiter=",", fmt=fmt, header=header, comments="")


def r_string(v):
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, np.ndarray) and v.dtype.kind == "U":
        return "c(" + ", ".join(r_string(str(x)) for x in v.ravel()) + ")"
    if isinstance(v, dict):
        return "list(" + ", ".join(f"{k} = {r_string(x)}" for k, x in v.items()) + ")"
    return r_value(v)


def write_expected(d, expected):
    (d / "expected.R").write_text("expected <- " + r_string(expected) + "\n")
    print(f"wrote {d}")
