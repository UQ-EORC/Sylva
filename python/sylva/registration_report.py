# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""How well a set of scans agree: pair residuals and a self-contained HTML report."""

from __future__ import annotations

import base64
import datetime
import html
import io
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _core, registration
from .filters import voxel_downsample
from .pointcloud import PointCloud

__all__ = ["PairResidual", "pair_residuals", "write_report"]

#: Fewest points a pair needs inside the search radius to be judged.
MIN_POINTS = 100


@dataclass
class PairResidual:
    """How well two registered scans agree where they overlap.

    Attributes
    ----------
    i, j
        Indices of the two scans.
    median
        Median distance (m) from the points of one scan to the nearest point
        of the other, over the points within the search radius, averaged over
        the two directions. NaN if fewer than 100 points are that close. It
        includes the point spacing, so it never reaches zero.
    p90
        90th percentile of the same distances (m): the tail, where a rotation
        or a sloping floor shows before the median moves.
    overlap
        Fraction of the points of one scan that are within the search radius
        of the other, averaged over the two directions. Low values mean
        little overlap, or a misregistration larger than the radius.
    n
        Points inside the search radius, both directions together.
    """

    i: int
    j: int
    median: float
    p90: float
    overlap: float
    n: int


def _registered(clouds, transforms, voxel):
    """The clouds moved into one frame and thinned, and the scanner position of each."""
    if transforms is not None and len(transforms) != len(clouds):
        raise ValueError("transforms needs one matrix per cloud")
    frames, origins = [], []
    for k, cloud in enumerate(clouds):
        if transforms is not None:
            matrix = np.asarray(transforms[k], dtype=np.float64)
            if matrix.shape != (4, 4):
                raise ValueError("each transform must be 4x4")
            cloud = cloud.transform(matrix)
            origins.append(matrix[:3, 3])
        else:
            origins.append(cloud.xyz.mean(axis=0))
        frames.append(voxel_downsample(cloud, voxel) if voxel else cloud)
    return frames, np.array(origins)


def _neighbour_pairs(origins, neighbours):
    """Each scan with its nearest ``neighbours`` scans, as sorted ``(i, j)`` with ``i < j``."""
    n = len(origins)
    pairs = set()
    for i in range(n):
        dist = np.linalg.norm(origins - origins[i], axis=1)
        dist[i] = np.inf
        for j in np.argsort(dist)[:neighbours]:
            pairs.add((min(i, int(j)), max(i, int(j))))
    return sorted(pairs)


def _one_way(a, b, search, max_points, rng):
    """Distances from a sample of ``a`` to ``b`` within ``search``, and the sample size."""
    if len(a) > max_points:
        a = a[rng.choice(len(a), max_points, replace=False)]
    d = _core.knn(np.ascontiguousarray(b), np.ascontiguousarray(a), 1)[0][:, 0]
    return d[d < search], len(a)


def _residuals(frames, pairs, search, max_points):
    rng = np.random.default_rng(0)
    out = []
    for i, j in pairs:
        a, b = frames[i].xyz, frames[j].xyz
        dab, nab = _one_way(a, b, search, max_points, rng)
        dba, nba = _one_way(b, a, search, max_points, rng)
        both = np.concatenate([dab, dba])
        medians = [np.median(d) for d in (dab, dba) if len(d) >= MIN_POINTS // 2]
        if len(both) < MIN_POINTS or not medians:
            out.append(PairResidual(i, j, np.nan, np.nan, len(both) / (nab + nba), len(both)))
            continue
        out.append(PairResidual(i, j, float(np.mean(medians)), float(np.percentile(both, 90)),
                                len(both) / (nab + nba), len(both)))
    return out


def pair_residuals(clouds: list[PointCloud], transforms: list[np.ndarray] | None = None,
                   pairs: list[tuple[int, int]] | None = None, *, neighbours: int = 4,
                   search: float = 1.0, voxel: float = 0.05,
                   max_points: int = 50_000) -> list[PairResidual]:
    """Measure how well each pair of neighbouring scans agrees.

    No ground truth is needed: where two scans see the same surface, the
    distance from each point to the nearest point of the other scan is the
    registration error plus the point spacing. Compare pairs with each other
    and one registration with another, not against zero.

    Parameters
    ----------
    clouds
        One cloud per scan.
    transforms
        One ``(4, 4)`` matrix per cloud (RiSCAN SOPs, or ``transform_for`` of a
        :mod:`sylva.coreg` result); None if the clouds are already in one
        frame. The translation of each matrix is taken as the scanner position.
    pairs
        ``(i, j)`` pairs to measure. Default each scan with its ``neighbours``
        nearest scanner positions.
    neighbours
        Neighbours per scan when ``pairs`` is None.
    search
        Points further than this (m) from the other scan are not counted: they
        are outside the overlap. Larger than the biggest misfit you expect.
    voxel
        Voxel size (m) the clouds are thinned to first, which also evens out
        the density near each scanner. 0 keeps every point.
    max_points
        Most points sampled from each scan of a pair, at random (seeded).

    Returns
    -------
    list of PairResidual
        In the order of ``pairs``.

    Raises
    ------
    ValueError
        If ``transforms`` is the wrong length or a matrix is not 4x4.
    """
    frames, origins = _registered(clouds, transforms, voxel)
    if pairs is None:
        pairs = _neighbour_pairs(origins, neighbours)
    return _residuals(frames, pairs, search, max_points)


def _png(fig) -> str:
    """The figure as a base64 PNG for an ``<img>``, and the figure closed."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _short(names):
    """Scan names without the prefix they all share."""
    prefix = ""
    if len(names) > 1:
        prefix = names[0]
        for name in names[1:]:
            while not name.startswith(prefix):
                prefix = prefix[:-1]
    return [n[len(prefix):] or n for n in names]


def _map_figure(origins, names, results, tolerance, max_distance):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import PowerNorm

    fig, ax = plt.subplots(figsize=(7.5, 6.5), layout="constrained")
    norm = PowerNorm(0.5, 0.0, max_distance)
    ok = [r for r in results if np.isfinite(r.median)]
    lost = [r for r in results if not np.isfinite(r.median)]
    if lost:
        ax.add_collection(LineCollection([origins[[r.i, r.j], :2] for r in lost], colors="#999999",
                                         linestyles="dashed", linewidths=1.2))
    if ok:
        lines = LineCollection([origins[[r.i, r.j], :2] for r in ok], cmap="viridis", norm=norm,
                               linewidths=3)
        lines.set_array(np.array([r.median for r in ok]))
        ax.add_collection(lines)
        bar = fig.colorbar(lines, ax=ax, label="median distance between the pair (m)", extend="max")
        ticks = [t for t in (0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5) if t <= max_distance]
        bar.set_ticks(ticks)
        for r in sorted(ok, key=lambda r: -r.median)[:12]:
            if r.median > tolerance:
                mid = origins[[r.i, r.j], :2].mean(axis=0)
                ax.annotate(f"{100 * r.median:.0f} cm", mid, fontsize=7, ha="center",
                            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none",
                                  "alpha": 0.8})
    ax.scatter(origins[:, 0], origins[:, 1], s=36, c="black", zorder=3)
    for (x, y, *_), label in zip(origins, _short(names), strict=True):
        ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Scan positions and the pairs measured; the 12 worst are labelled\n"
                 "(dashed: no usable overlap)", fontsize=10)
    return fig


def _matrix_figure(names, results, max_distance):
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    n = len(names)
    grid = np.full((n, n), np.nan)
    for r in results:
        grid[r.i, r.j] = grid[r.j, r.i] = r.median
    size = min(9.0, 3.0 + 0.22 * n)
    fig, ax = plt.subplots(figsize=(size + 1.2, size), layout="constrained")
    shown = ax.imshow(grid, cmap="viridis", norm=PowerNorm(0.5, 0.0, max_distance))
    fig.colorbar(shown, ax=ax, label="median distance (m)", extend="max", shrink=0.8)
    labels = _short(names)
    step = max(1, n // 40)
    ticks = range(0, n, step)
    ax.set_xticks(ticks, [labels[k] for k in ticks], rotation=90, fontsize=7)
    ax.set_yticks(ticks, [labels[k] for k in ticks], fontsize=7)
    ax.set_title("Median distance for every pair measured (blank: not measured)")
    return fig


def _pair_figures(source, target, centre, radius, max_distance, label, section_radius,
                  thickness):
    """The overlay and the cross sections of one pair, as HTML ``<img>`` or a note."""
    drawers = (("Overlay", lambda: registration.plot_overlay(
                   source, target, None, centre=(float(centre[0]), float(centre[1])),
                   radius=radius, max_distance=max_distance)),
               ("Cross sections", lambda: registration.plot_sections(
                   source, target, None, n=3, radius=section_radius, thickness=thickness)))
    out = []
    for what, draw in drawers:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                image = _png(draw())
            except ValueError as exc:
                out.append(f'<p class="note">{what} not drawn: {html.escape(str(exc))}.</p>')
                continue
        out.append(f'<img alt="{what} of {html.escape(label)}" '
                   f'src="data:image/png;base64,{image}">')
        if caught:
            out.append(f'<p class="note">{what}: {html.escape(str(caught[0].message))}.</p>')
    return out


_CSS = """
:root { --bg:#ffffff; --fg:#1b1f23; --muted:#5b6670; --line:#d8dee4; --card:#f5f7f9;
        --ok:#1a7f4b; --check:#b54708; --none:#6b7280; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14181c; --fg:#e6e9ec; --muted:#9aa5b1; --line:#2c343c; --card:#1c2228;
          --ok:#4cc38a; --check:#f0a05a; --none:#9aa5b1; } }
body { background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0; }
main { max-width:1000px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:1.6rem; margin:0 0 4px; } h2 { font-size:1.2rem; margin:32px 0 8px; }
h3 { font-size:1rem; margin:20px 0 6px; }
p.sub, p.note { color:var(--muted); margin:4px 0; }
.cards { display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:10px;
         margin:16px 0; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px;
        padding:10px 12px; }
.card b { display:block; font-size:1.4rem; }
.card span { color:var(--muted); font-size:.85rem; }
img { max-width:100%; height:auto; background:#fff; border-radius:6px; }
table { border-collapse:collapse; width:100%; font-size:.9rem; }
th, td { text-align:left; padding:5px 8px; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:600; }
td.n, th.n { text-align:right; font-variant-numeric:tabular-nums; }
.ok { color:var(--ok); } .check { color:var(--check); font-weight:600; }
.none { color:var(--none); }
.scroll { overflow-x:auto; }
"""


def _cm(value) -> str:
    return "n/a" if not np.isfinite(value) else f"{100 * value:.1f}"


def _status(r, tolerance):
    if not np.isfinite(r.median):
        return "none", "no overlap"
    return ("ok", "ok") if r.median <= tolerance else ("check", "check")


def write_report(path: str | Path, clouds: list[PointCloud],
                 transforms: list[np.ndarray] | None = None, names: list[str] | None = None, *,
                 pairs: list[tuple[int, int]] | None = None, neighbours: int = 4,
                 search: float = 1.0, tolerance: float = 0.05, max_distance: float | None = None,
                 worst: int = 3, voxel: float = 0.05, title: str = "Registration report") -> Path:
    """Write a self-contained HTML report on how well a set of scans agree.

    The page has a map of the scan positions with every measured pair coloured
    by its residual, the same as a matrix, tables of the pairs (worst first)
    and of the scans, and for the worst pairs an overlay and cross sections
    (:func:`sylva.registration.plot_overlay`, :func:`sylva.registration.plot_sections`).
    The cross sections are thin slabs, so they need clouds dense enough to fill them:
    a coarse ``voxel`` gives sparse sections.
    Figures are embedded, so the file can be moved or emailed on its own. It
    describes the registration as given: to compare two, write a report for
    each. Needs matplotlib.

    Parameters
    ----------
    path
        Output ``.html`` file.
    clouds
        One cloud per scan, thinned (every 5th to 10th pulse is plenty).
    transforms
        One ``(4, 4)`` matrix per cloud (RiSCAN SOPs, or ``transform_for`` of a
        :mod:`sylva.coreg` result); None if the clouds share a frame already.
    names
        One label per scan. Default ``scan 1``, ``scan 2``, ...
    pairs, neighbours, search, voxel
        As for :func:`pair_residuals`.
    tolerance
        Median distance (m) above which a pair is marked "check". The distance
        includes the point spacing, so choose it from a registration you trust.
    max_distance
        Upper end (m) of the colour scale, which goes with the square root of
        the distance. Default the 95th percentile of the pair medians, at
        least 0.1 m, so that the pairs are told apart whatever the scale of
        the misfit.
    worst
        How many of the worst pairs get an overlay and cross sections.
    title
        Page title.

    Returns
    -------
    pathlib.Path
        The file written.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If there are fewer than two clouds, or ``names`` or ``transforms`` is
        the wrong length.
    """
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise ImportError("write_report needs matplotlib: pip install matplotlib") from exc
    if len(clouds) < 2:
        raise ValueError("a report compares at least two scans")
    if names is None:
        names = [f"scan {k + 1}" for k in range(len(clouds))]
    if len(names) != len(clouds):
        raise ValueError("names needs one label per cloud")
    frames, origins = _registered(clouds, transforms, voxel)
    if pairs is None:
        pairs = _neighbour_pairs(origins, neighbours)
    results = _residuals(frames, pairs, search, 50_000)
    esc = html.escape
    if max_distance is None:
        medians = [r.median for r in results if np.isfinite(r.median)]
        max_distance = max(0.1, float(np.percentile(medians, 95))) if medians else 0.1

    judged = [r for r in results if np.isfinite(r.median)]
    ranked = sorted(results, key=lambda r: -np.inf if not np.isfinite(r.median) else -r.median)
    within = sum(r.median <= tolerance for r in judged)
    cards = [(str(len(clouds)), "scans"), (str(len(results)), "pairs measured"),
             (f"{within} of {len(results)}", f"pairs within {100 * tolerance:g} cm"),
             (_cm(float(np.median([r.median for r in judged]))) + " cm" if judged else "n/a",
              "median of the pair medians")]
    worst_pair = next((r for r in ranked if np.isfinite(r.median)), None)
    if worst_pair is not None:
        cards.append((f"{_cm(worst_pair.median)} cm",
                      f"worst pair, {names[worst_pair.i]} and {names[worst_pair.j]}"))

    body = [f"<h1>{esc(title)}</h1>",
            f'<p class="sub">{datetime.date.today().isoformat()} · {len(clouds)} scans, '
            f"thinned to {voxel:g} m voxels · search radius {search:g} m</p>",
            '<div class="cards">'
            + "".join(f'<div class="card"><b>{esc(v)}</b><span>{esc(k)}</span></div>'
                      for v, k in cards) + "</div>",
            "<h2>Where the scans disagree</h2>",
            f'<img alt="Map of scan positions with pair residuals" src="data:image/png;base64,'
            f'{_png(_map_figure(origins, names, results, tolerance, max_distance))}">']
    if len(clouds) <= 80:
        body += [f'<img alt="Residual of every pair" src="data:image/png;base64,'
                 f'{_png(_matrix_figure(names, results, max_distance))}">']

    rows = []
    for r in ranked:
        cls, label = _status(r, tolerance)
        rows.append(f"<tr><td>{esc(names[r.i])} – {esc(names[r.j])}</td>"
                    f'<td class="n">{_cm(r.median)}</td><td class="n">{_cm(r.p90)}</td>'
                    f'<td class="n">{100 * r.overlap:.0f}</td><td class="{cls}">{label}</td></tr>')
    body += ["<h2>Pairs, worst first</h2>", '<div class="scroll"><table><thead><tr><th>Pair</th>'
             '<th class="n">Median (cm)</th><th class="n">90th percentile (cm)</th>'
             '<th class="n">Overlap (%)</th><th>Status</th></tr></thead><tbody>'
             + "".join(rows) + "</tbody></table></div>",
             f'<p class="note">Status is "check" above {100 * tolerance:g} cm, "no overlap" when '
             f"fewer than {MIN_POINTS} points lie within {search:g} m of the other scan.</p>"]

    per_scan = []
    for k, name in enumerate(names):
        mine = [r for r in results if k in (r.i, r.j)]
        meds = [r.median for r in mine if np.isfinite(r.median)]
        bad = max((r for r in mine if np.isfinite(r.median)), key=lambda r: r.median, default=None)
        other = "" if bad is None else names[bad.j if bad.i == k else bad.i]
        per_scan.append((np.median(meds) if meds else np.inf, name, len(mine), other,
                         bad.median if bad else np.nan))
    rows = [f"<tr><td>{esc(name)}</td><td class=\"n\">{count}</td>"
            f'<td class="n">{_cm(med if np.isfinite(med) else np.nan)}</td>'
            f"<td>{esc(other) or '–'}</td><td class=\"n\">{_cm(worst_med)}</td></tr>"
            for med, name, count, other, worst_med in sorted(per_scan, key=lambda s: -s[0])]
    body += ["<h2>Scans, worst first</h2>", '<div class="scroll"><table><thead><tr><th>Scan</th>'
             '<th class="n">Pairs</th><th class="n">Median of its pairs (cm)</th>'
             '<th>Worst partner</th><th class="n">Worst pair (cm)</th></tr></thead><tbody>'
             + "".join(rows) + "</tbody></table></div>"]

    shown = [r for r in ranked if np.isfinite(r.median)][:worst]
    if shown:
        body.append("<h2>The worst pairs up close</h2>")
    for r in shown:
        label = f"{names[r.i]} and {names[r.j]}: median {_cm(r.median)} cm"
        gap = np.linalg.norm(origins[r.i, :2] - origins[r.j, :2])
        body.append(f"<h3>{esc(label)}</h3>")
        body += _pair_figures(frames[r.j], frames[r.i], 0.5 * (origins[r.i, :2] + origins[r.j, :2]),
                              float(np.clip(0.5 * gap, 5.0, 12.0)), max_distance, label,
                              float(np.clip(2 * r.median, 1.0, 3.0)), max(0.1, 2 * voxel))

    body.append('<p class="note">Distances are from each point of one scan to the nearest point '
                "of the other, in both directions, over points within the search radius. They "
                "include the point spacing, so no pair reaches zero: compare pairs, and one "
                "registration with another.</p>")
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{esc(title)}</title><style>{_CSS}</style></head><body><main>"
            + "\n".join(body) + "</main></body></html>")
    path = Path(path)
    path.write_text(page, encoding="utf-8")
    return path
