"""MkDocs hooks.

- The docstrings use a few Sphinx roles (:func:`x`); show them as plain code.
- ``<!-- function-index -->`` in a page is replaced by a table of every public
  class, method and function, read from the source with griffe (no build of
  the Rust core needed), so the index cannot drift from the code.
"""

import re
from pathlib import Path

ROLE = re.compile(r":(?:func|class|meth|attr|data|mod|obj):(<code>.*?</code>)")
MARKER = "<!-- function-index -->"

#: Module -> API page, in the order of the index.
PAGES = {
    "pointcloud": "pointcloud.md", "raster": "pointcloud.md", "io": "io.md", "shots": "shots.md",
    "riscan": "shots.md", "filters": "filters.md", "ground": "ground.md", "registration": "registration.md",
    "coreg": "coreg.md", "trees": "trees.md", "qsm": "qsm.md", "leaves": "leaves.md",
    "canopy": "canopy.md", "voxels": "voxels.md", "voxels.blocks": "voxel_blocks.md",
    "quality": "quality.md", "waveform": "waveform.md",
    "als": "als.md", "als.tiles": "tiles.md", "als.metrics": "als_metrics.md",
    "als.canopy": "als_canopy.md", "als.trajectory": "als_canopy.md", "als.trees": "als_trees.md",
    "change": "change.md", "change.als": "change_als.md", "fusion": "fusion.md",
    "geo.coords": "coords.md", "geo.interpolate": "interpolate.md", "geo.masks": "masks.md",
    "synthetic": "synthetic.md", "synthetic.model": "synthetic.md",
    "util.progress": "progress.md", "util.limits": "limits.md",
}


def _summary(obj) -> str:
    doc = obj.docstring.value if obj.docstring else ""
    first = doc.strip().split("\n\n")[0].replace("\n", " ").strip()
    return first.replace("|", "\\|")


#: Submodules the index leaves out (they have no API page of their own).
SKIP = {"als.tiles_trees"}


def _submodules(name, mod) -> list:
    """The public submodules below ``mod`` that have no page of their own, at any depth."""
    found = []
    for sub, m in mod.modules.items():
        path = f"{name}.{sub}"
        if sub.startswith("_") or path in PAGES or path in SKIP or m.is_alias:
            continue
        found += [(path, m)] + _submodules(path, m)
    return found


def _anchor_module(pkg, where: str, member: str, shown: set) -> str:
    """The module a page documents ``member`` under: the nearest one, from where it is
    defined upwards, that the page renders and that exports it (``sylva.qsm`` for
    ``sylva.qsm.model.QSM``), else where it is defined."""
    parts = where.split(".")
    for end in range(len(parts), 0, -1):
        path = ".".join(parts[:end])
        if f"sylva.{path}" in shown and (end == len(parts) or member in pkg[path].members):
            return path
    return where


def _index(src: Path) -> str:
    import griffe

    pkg = griffe.load("sylva", search_paths=[str(src)], docstring_parser="numpy")
    out = []
    for name, page in PAGES.items():
        mod = pkg[name]
        shown = set(re.findall(r"^::: (\S+)", (src.parent / "docs" / "api" / page).read_text(), re.M))
        rows = []
        # A package (sylva.coreg) is indexed through its own members and any
        # public submodules that do not have a page of their own.
        modules = [(name, mod)] + _submodules(name, mod)
        members = [(where, member) for where, m in modules for member in m.members.values()]
        for where, member in members:
            kind = member.kind.value
            if member.is_alias or member.name.startswith("_") or kind not in ("class", "function"):
                continue
            path = f"sylva.{_anchor_module(pkg, where, member.name, shown)}.{member.name}"
            rows.append(f"| [`{member.name}`]({page}#{path}) | {kind} | {_summary(member)} |")
            if kind == "class":
                for sub in member.members.values():
                    is_property = "property" in sub.labels
                    if sub.name.startswith("_") or not sub.docstring:
                        continue
                    if sub.kind.value != "function" and not is_property:
                        continue
                    label = "property" if is_property else "method"
                    link = f"[`{member.name}.{sub.name}`]({page}#{path}.{sub.name})"
                    rows.append(f"| {link} | {label} | {_summary(sub)} |")
        if rows:
            header = ["| Name | Kind | Summary |", "|---|---|---|"]
            out += [f"## `sylva.{name}`", "", *header, *rows, ""]
    return "\n".join(out)


def on_page_markdown(markdown, config, **kwargs):
    if MARKER in markdown:
        src = Path(config["config_file_path"]).parent / "python"
        markdown = markdown.replace(MARKER, _index(src))
    return markdown


def on_page_content(html, **kwargs):
    return ROLE.sub(r"\1", html)
