"""The API reference is generated from docstrings: keep every public name documented,
and keep the promises the operational guide makes."""

import importlib
import inspect

import numpy as np
import pytest

MODULES = ["pointcloud", "raster", "io", "shots", "riscan", "filters", "ground", "trees", "canopy",
           "voxels", "registration", "coreg", "qsm", "leaves", "quality", "fusion",
           "synthetic", "synthetic.als", "synthetic.model", "als", "als.canopy", "als.metrics",
           "als.tiles", "als.trees", "change.als", "geo.coords", "geo.interpolate", "geo.masks",
           "util.limits", "util.progress"]


def _public(mod):
    for name, obj in vars(mod).items():
        if name.startswith("_") or getattr(obj, "__module__", None) != mod.__name__:
            continue
        if inspect.isfunction(obj):
            yield name, obj
        elif inspect.isclass(obj):
            yield name, obj
            for mname, member in vars(obj).items():
                if mname.startswith("_"):
                    continue
                if isinstance(member, property):
                    yield f"{name}.{mname}", member
                elif isinstance(member, (staticmethod, classmethod)) or inspect.isfunction(member):
                    yield f"{name}.{mname}", getattr(obj, mname)


def _cases():
    for m in MODULES:
        mod = importlib.import_module(f"sylva.{m}")
        for name, obj in _public(mod):
            yield pytest.param(obj, id=f"{m}.{name}")


@pytest.mark.parametrize("obj", list(_cases()))
def test_public_api_is_documented(obj):
    doc = inspect.getdoc(obj)
    assert doc, "missing docstring"
    if isinstance(obj, property) or inspect.isclass(obj):
        return
    params = [p for p in inspect.signature(obj).parameters if p not in ("self", "cls")]
    # One-line docstrings are fine for trivial accessors; anything with arguments needs a
    # Parameters section so the reference shows units and defaults.
    if params:
        assert "Parameters" in doc, "arguments without a Parameters section"


def test_pipeline_repeats_exactly():
    from sylva import ground, qsm, synthetic, trees, voxels

    cloud = synthetic.forest()
    cloud = ground.normalize_height(cloud, ground.make_dtm(cloud))

    def run():
        stems = trees.detect_stems(cloud)
        labels = trees.segment_trees(cloud, stems)
        model = qsm.build_qsm(cloud[labels == 1])
        grid = voxels.ray_voxelize(synthetic.scan(cloud), 0.5)
        pad = np.nan_to_num(grid.pad_fpl)
        return [(t.x, t.y, t.dbh) for t in stems], labels, model.cylinders, pad

    a, b = run(), run()
    assert a[0] == b[0]
    for x, y in zip(a[1:], b[1:], strict=True):
        assert np.array_equal(x, y)
