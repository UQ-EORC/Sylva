"""Write inputs and Python outputs for the R package's RiSCAN tests.

    python tests/parity/export_r_riscan.py

The synthetic projects of cases_riscan.py go to ``projects.R`` as a manifest
(every file with its bytes, every empty folder), which the R tests
(r/sylva/tests/testthat/test-riscan-python.R) write into a temporary folder
before reading them; ``expected.R`` holds what the Python package reads, with
paths relative to each project.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parity import cases_riscan as cr  # noqa: E402
from parity.export_r_fixtures import OUT, r_value  # noqa: E402
from sylva import riscan  # noqa: E402


def r_string(s):
    out = (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{out}"'


def value(v):
    """R source for a value, strings included (r_value writes numbers)."""
    if isinstance(v, dict):
        return "list(" + ", ".join(f"{k} = {value(x)}" for k, x in v.items()) + ")"
    if isinstance(v, str):
        return r_string(v)
    a = np.asarray(v)
    if a.dtype.kind == "U":
        return (
            "c(" + ", ".join(r_string(str(x)) for x in a.ravel()) + ")"
            if a.size
            else "character(0)"
        )
    if a.dtype.kind == "b":
        return r_value(a.astype(float)) if a.ndim else ("TRUE" if bool(a) else "FALSE")
    if a.size == 0:
        return "numeric(0)"
    return r_value(a if a.ndim else float(a))


def project_manifest(tmp):
    files, dirs = {}, []
    for p in sorted(tmp.rglob("*")):
        rel = p.relative_to(tmp).as_posix()
        if p.is_dir():
            if not any(p.iterdir()):
                dirs.append(rel)
        else:
            files[rel] = p.read_bytes().decode()
    return files, dirs


def riscan_fixtures():
    d = OUT / "riscan"
    d.mkdir(parents=True, exist_ok=True)
    expected = {}
    files, dirs = {}, []
    tmp = Path(tempfile.mkdtemp(prefix="sylva-riscan-fixtures-"))
    try:
        for key, build in [
            ("rsp", cr._rsp_project),
            ("legacy", cr._legacy_project),
            ("proj", cr._proj_project),
            (
                "empty",
                lambda t: (t / "empty.riproject").mkdir(parents=True) or t / "empty.riproject",
            ),
        ]:
            root = build(tmp / key)
            out = cr._describe(riscan.read_riscan_project(root), root)
            out["root"] = root.relative_to(tmp).as_posix()
            # Drop the recording's padding: every per-position array ends with a blank entry.
            n = int(out["len"])
            for k, v in out.items():
                if k not in (
                    "name",
                    "path",
                    "has_pop",
                    "pop",
                    "origins",
                    "gnss_positions",
                    "len",
                    "root",
                    "with_scans",
                    "with_scans_any",
                ):
                    out[k] = np.asarray(v)[:n]
            for k in ("with_scans", "with_scans_any"):
                out[k] = np.asarray(out[k])[:-1]
            expected[key] = out
        files, dirs = project_manifest(tmp)
    finally:
        shutil.rmtree(tmp)
    lines = ["files <- list("]
    lines.append(",\n".join(f"  {r_string(k)} = {r_string(v)}" for k, v in files.items()))
    lines.append(")")
    lines.append("dirs <- c(" + ", ".join(r_string(x) for x in dirs) + ")")
    (d / "projects.R").write_text("\n".join(lines) + "\n")

    rng = np.random.default_rng(60)
    xyz, amplitude = cr._stream(61, step=0.05, n_lines=60, n_shots=60)
    np.savetxt(d / "stream.csv", np.column_stack([xyz, amplitude]), delimiter=",", fmt="%.17g")
    expected["steps"] = np.asarray(riscan.angular_steps(xyz))
    expected["mask_current"] = riscan.riscan_like_mask(xyz, amplitude, "current")
    expected["mask_legacy"] = riscan.riscan_like_mask(xyz, amplitude, "legacy")
    expected["mask_legacy_steps"] = riscan.riscan_like_mask(
        xyz, amplitude, "legacy", steps=(0.05, 0.05), min_neighbours=3, weak_db=20.0
    )
    (d / "settings.txt").write_text(
        "# RiSCAN export\nriegl.Deviation; 0; 12\nrange, 2.5, 30\nREFLECTANCE , -20, 5\n"
    )
    settings = riscan.read_export_settings(d / "settings.txt")
    expected["settings"] = {k: np.asarray(v) for k, v in settings.items()}
    attrs = {
        "deviation": rng.choice([0, 3, 11, 12, 13, 65535], len(xyz)).astype(float),
        "reflectance": rng.uniform(-25, 10, len(xyz)),
    }
    np.savetxt(
        d / "attributes.csv",
        np.column_stack([attrs["deviation"], attrs["reflectance"]]),
        delimiter=",",
        fmt="%.17g",
    )
    expected["settings_mask"] = riscan.export_settings_mask(settings, xyz, attrs)
    fixes = [
        (-27.5 + rng.uniform(0, 1e-3), 153.0 + rng.uniform(0, 1e-3), rng.uniform(0, 50))
        for _ in range(6)
    ]
    fixes[2] = None
    expected["gnss_in"] = np.array([[np.nan] * 3 if f is None else f for f in fixes])
    expected["gnss_local"] = riscan.gnss_to_local(fixes)
    angles = rng.uniform(-360, 360, (5, 3))
    expected["angles"] = angles
    expected["rotations"] = np.array([riscan._rotation_zyx(*a) for a in angles])
    (d / "expected.R").write_text("expected <- " + value(expected) + "\n")
    print(f"wrote {d}")


if __name__ == "__main__":
    riscan_fixtures()
