"""The API reference is generated from docstrings: keep every public name documented,
and keep the promises the operational guide makes."""

import importlib
import inspect
import pkgutil
from pathlib import Path

import numpy as np
import pytest

MODULES = ["pointcloud", "raster", "io", "shots", "riscan", "filters", "ground", "trees", "canopy",
           "voxels", "registration", "registration_report", "coreg", "qsm", "leaves", "quality",
           "fusion",
           "synthetic", "synthetic.als", "synthetic.model", "als", "als.canopy", "als.metrics",
           "als.tiles", "als.trees", "als.catalogue", "als.engine", "als.ops", "als.trajectory",
           "change.als", "change", "coreg.pipeline", "geo.coords", "geo.interpolate", "geo.masks",
           "util.image", "util.limits", "util.progress"]


#: Packages that were one module before they were split: each submodule is checked.
SPLIT = ("qsm", "fusion", "change.als", "coreg", "coreg.pipeline", "change")


def _modules():
    """The listed modules, with each split package replaced by its submodules (and itself)."""
    seen = []
    for m in MODULES:
        mod = importlib.import_module(f"sylva.{m}")
        names = [m]
        if m in SPLIT and hasattr(mod, "__path__"):
            names += [f"{m}.{info.name}" for info in pkgutil.iter_modules(mod.__path__)
                      if not info.name.startswith("_") and info.name not in ("cli", "convert")]
        seen += [n for n in names if n not in seen]
    return seen


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
    for m in _modules():
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


# --- The guides name real commands, options and Python objects --------------

DOCS = Path(__file__).resolve().parents[1] / "docs"
README = DOCS.parent / "README.md"


def _doc_pages():
    return sorted(DOCS.rglob("*.md")) + [README]


def _cli_parser():
    """The ``sylva`` argument parser, caught before it parses anything."""
    import argparse

    from sylva import cli

    caught = {}

    class Caught(Exception):
        pass

    def grab(self, *args, **kwargs):
        caught["parser"] = self
        raise Caught

    original = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = grab
    try:
        cli.main([])
    except Caught:
        pass
    finally:
        argparse.ArgumentParser.parse_args = original
    return caught["parser"]


def _options(parser):
    return {opt for action in parser._actions for opt in action.option_strings}


def _shell_lines():
    """Every ``sylva ...`` command line in a fenced block of the docs, continuations joined."""
    for page in _doc_pages():
        fenced, pending = False, []
        for number, raw in enumerate(page.read_text().splitlines(), 1):
            line = raw.strip()
            if line.startswith("```"):
                fenced = not fenced
                continue
            if not fenced:
                continue
            if line.endswith("\\"):
                pending.append(line[:-1])
                continue
            if pending:
                line, pending = " ".join(pending) + " " + line, []
            line = line.lstrip("$ ").strip()
            if line.startswith("sylva "):
                yield f"{page.relative_to(DOCS.parent)}:{number}", line.split(" #")[0]


def test_documented_command_lines_parse():
    """Every command and option the guides show exists in the CLI."""
    import argparse
    import shlex

    parser = _cli_parser()
    commands = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices
    global_options = _options(parser)
    problems = []
    for where, line in _shell_lines():
        tokens = shlex.split(line)[1:]
        while tokens and tokens[0].startswith("-"):
            if tokens.pop(0) not in global_options:
                problems.append(f"{where}: unknown global option in `{line}`")
        if not tokens:
            continue
        command, options = tokens[0], tokens[1:]
        if command not in commands:
            problems.append(f"{where}: unknown command `{command}`")
            continue
        known = _options(commands[command])
        for token in options:
            if token.startswith("--") and token.split("=")[0] not in known:
                problems.append(f"{where}: `{command}` has no option {token.split('=')[0]}")
    assert not problems, "\n".join(problems)


def test_documented_names_resolve():
    """Every ``sylva.x.y`` the docs mention is importable (file names in the docs are allowed)."""
    import re

    files = {p.name for p in DOCS.rglob("*") if p.is_file()}
    missing = set()
    for page in _doc_pages():
        for name in set(re.findall(r"\bsylva(?:\.[A-Za-z_][A-Za-z0-9_]*)+", page.read_text())):
            if name in files:
                continue
            parts = name.split(".")
            obj = importlib.import_module("sylva")
            for depth, part in enumerate(parts[1:], 1):
                try:
                    obj = getattr(obj, part)
                except AttributeError:
                    try:
                        obj = importlib.import_module(".".join(parts[: depth + 1]))
                    except ImportError:
                        missing.add(f"{page.relative_to(DOCS.parent)}: {name}")
                        break
    assert not missing, "\n".join(sorted(missing))
