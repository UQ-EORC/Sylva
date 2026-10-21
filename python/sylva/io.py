# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Reading and writing point clouds.

``read`` and ``write`` dispatch on file extension:

| Format | read | write |
|---|---|---|
| ``.las`` / ``.laz`` | yes | yes |
| ``.ply`` (ascii / binary, incl. raycloudtools ray clouds) | yes | yes |
| ``.xyz`` ``.txt`` ``.csv`` ``.pts`` | yes | yes |
| ``.rxp`` (RIEGL, needs RiVLib) | yes | no |
| ``.rdbx`` (RIEGL RDB 2, needs librdb) | yes | no |

Pulse data has its own format, see :meth:`sylva.Shots.save`.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _core
from .pointcloud import PointCloud
from .shots import Shots

__all__ = [
    "read", "write", "read_ascii", "read_rxp", "read_rxp_shots", "find_rivlib", "find_rivlib_shim",
    "read_rdbx", "rdbx_info", "find_rdblib", "RdbxInfo", "RdbxAttribute",
    "read_matrix_file",
]


def read(path: str | Path) -> PointCloud:
    """Read a point cloud, choosing the reader from the file extension.

    Parameters
    ----------
    path
        ``.las``, ``.laz``, ``.ply``, ``.xyz``, ``.txt``, ``.csv``, ``.asc``,
        ``.pts``, ``.rxp`` or ``.rdbx``. The extension is matched
        case-insensitively.

    Returns
    -------
    PointCloud
        Coordinates in the file's frame (``.rxp`` and a scan position's
        ``.rdbx``: scanner frame, see :func:`read_rxp` and
        :func:`read_rdbx`). Attributes depend on the format:

        - LAS/LAZ: ``intensity``, ``return_number``, ``number_of_returns``,
          ``classification``, ``scan_angle``, ``user_data``,
          ``point_source_id``, plus ``gps_time`` and ``red``/``green``/``blue``
          when the point format has them, and every extra-bytes dimension
          under its own name and type. ``crs`` is the header's OGC WKT, or
          ``"EPSG:n"`` (``"EPSG:h+v"`` with a vertical CRS) from its GeoTIFF
          keys; None if the header declares neither.
        - PLY: every ``vertex`` property except x, y, z, with its type.
          raycloudtools ray clouds give ``nx``, ``ny``, ``nz`` (point to
          sensor) and ``alpha``; see :meth:`sylva.Shots.from_ray_cloud`.
        - Text: see :func:`read_ascii`.
        - RXP: see :func:`read_rxp` (default options).
        - RDBX: every attribute of the database, see :func:`read_rdbx`.

    Raises
    ------
    OSError
        If the file cannot be opened or is malformed.
    ValueError
        For an unsupported extension or invalid option.
    """
    xyz, attrs = _core.read(str(path))
    crs = _core.read_las_crs(str(path)) if Path(path).suffix.lower() in (".las", ".laz") else None
    return PointCloud(xyz, attrs, crs)


def write(cloud: PointCloud, path: str | Path, point_format: int = 6, scale: float | None = None,
          binary: bool = True) -> None:
    """Write a point cloud, choosing the writer from the file extension.

    Parameters
    ----------
    cloud
        Points to write, with all their attributes.
    path
        ``.las``, ``.laz``, ``.ply``, ``.xyz``, ``.txt``, ``.asc``, ``.pts``
        (space separated) or ``.csv`` (comma separated). Overwritten if it
        exists.
    point_format
        LAS point data record format (LAS/LAZ only). 6 (LAS 1.4, with GPS
        time) is the default; use 7 or 8 to keep RGB.
    scale
        LAS coordinate quantisation in the units of the coordinates
        (LAS/LAZ only); the offset is the floor of the minimum coordinate.
        By default 0.001 (millimetres), or 1e-7 degrees (about 1 cm) when
        the cloud's ``crs`` is geographic (longitude and latitude).
    binary
        Binary little-endian PLY if True, ASCII otherwise (PLY only).

    Notes
    -----
    LAS/LAZ: attributes that are not standard dimensions of ``point_format``
    are written as typed extra bytes, so ``height``, ``tree_id`` and the like
    survive a round trip. The cloud's ``crs`` is stored as an OGC WKT VLR
    (an EPSG code as the registry's WKT); a CRS given as a PROJ string has
    no WKT form and is left out with a warning. Text files get a header
    line naming the columns and coordinates to 0.1 mm; integer attributes
    are written as integers. PLY and text files have no place for a CRS.

    Raises
    ------
    OSError
        If the file cannot be opened or is malformed.
    ValueError
        For an unsupported extension or invalid option.
    """
    crs_wkt = None
    geographic = False
    if getattr(cloud, "crs", None) is not None and Path(path).suffix.lower() in (".las", ".laz"):
        try:
            info = _core.crs_info(cloud.crs)
            crs_wkt, geographic = info["wkt"], bool(info["geographic"])
        except ValueError:                       # WKT Sylva cannot interpret: store it as given
            text = cloud.crs.strip()
            crs_wkt = text if text[:1].isalpha() and "[" in text else None
        if crs_wkt is None:
            warnings.warn(f"the CRS {cloud.crs!r} has no WKT form and is not stored in {path}; "
                          "give it as an EPSG code or WKT to keep it", UserWarning, stacklevel=2)
    if scale is None:
        scale = 1e-7 if geographic else 0.001
    _core.write(str(path), cloud.xyz, cloud.attrs, point_format, scale, binary, crs_wkt)


def read_ascii(path: str | Path, columns: list[str] | None = None) -> PointCloud:
    """Read a delimited text point cloud.

    The delimiter (comma, semicolon, tab or whitespace) is detected from the
    first data line. Lines starting with ``#`` are skipped, a leading line
    holding a single integer (PTS point count) is skipped, and a non-numeric
    first line is taken as a header.

    Parameters
    ----------
    path
        Text file whose first three columns are x, y, z.
    columns
        Names for the columns. Either all columns (the first three are then
        ignored) or only the ones after x, y, z. Overrides a header line.
        Without names or header, extra columns are called ``col3``,
        ``col4``, ...

    Returns
    -------
    PointCloud
        Extra columns as float64 attributes.

    Raises
    ------
    OSError
        On a non-numeric value, a row with a different column count, fewer
        than three columns, or a file that cannot be opened.
    """
    xyz, attrs = _core.read_ascii(str(path), columns)
    return PointCloud(xyz, attrs)


def read_rxp(path: str | Path, **options) -> PointCloud:
    """Read a RIEGL ``.rxp`` scan as points, converted like RiSCAN PRO.

    Needs RiVLib, which RIEGL does not allow to be redistributed; see
    :func:`find_rivlib` for where it is looked for. With the small
    ``libsylva_rivlib`` shim built from the SDK (see
    :func:`find_rivlib_shim`) the scan is converted the way RiSCAN PRO's
    import does it: multiple-time-around (MTA) ambiguity is resolved, echoes
    are placed on the pulse they belong to, the window and weak-echo filters
    are applied and MTA-uncertain echoes are dropped. Without the shim the
    plain ``libscanifc`` reading is used (every echo in zone 1, no
    resolution), which is what earlier versions did.

    .. warning::

       MTA resolution is Sylva's own reimplementation and is a work in
       progress: it is close to RiSCAN PRO's import (about 99% of the same
       points, 99.98% of the same zones on the reference scan) but not
       identical to it, and the disagreements fall in far canopy returns and
       echoes RIEGL's ``rimta5`` sets aside as uncertain. Where the exact
       RiSCAN point set matters, import in RiSCAN PRO and read the ``.rdbx``
       with :func:`read_rdbx`. See the IO guide for the measured figures.

    Parameters
    ----------
    path
        The ``.rxp`` file of one scan position.
    **options
        Keyword options:

        - ``mta`` (``"auto"``): ``"auto"`` resolves MTA when the shim is
          found, ``"on"`` requires it, ``"off"`` reads plainly.
        - ``max_zone`` (6): MTA zones considered (1..8).
        - ``keep_uncertain`` (False): keep MTA-uncertain echoes, flagged in
          ``mta_uncertain_point``, instead of dropping them as RiSCAN does.
        - ``mta_model`` (None): how the zone and keep/drop are decided from
          the evidence. ``None`` or ``"default"`` uses the learned model
          shipped with Sylva (boosted trees trained on rimta5's output of a
          VZ-600i scan), ``"rule"`` the hand-written thresholds, a path a
          model JSON written by ``scripts/train_mta.py``.
        - ``shim`` (path): ``libsylva_rivlib`` or a directory containing it.
        - ``max_lines`` (None): stop after this many scan lines (quick looks).
        - ``library`` (path): ``libscanifc`` or a directory containing it.
        - ``drop_pseudo_echoes`` (True): skip echoes RIEGL flags as pseudo.
        - ``min_range`` (0.5): drop echoes closer than this (m); removes the
          tripod and operator.
        - ``max_range`` (inf): drop echoes farther than this (m).
        - ``stride`` (1): keep every n-th echo. Splits multi-echo pulses, so
          do not use it for pulse data.
        - ``shot_stride`` (1): keep every n-th pulse with all its echoes;
          the right way to thin data for gap fraction and ray tracing.
        - ``max_points`` (None): stop after this many points.
        - ``echoes`` (``"all"``): ``"all"``, ``"first"``, ``"last"`` or
          ``"single"``.

    Returns
    -------
    PointCloud
        Points in the scanner's own coordinate system (SOCS), scanner at the
        origin. Apply the SOP (:func:`read_matrix_file` or
        :class:`sylva.riscan.RiscanProject`) to put them in project
        coordinates. Attributes: ``amplitude`` and ``reflectance`` (dB),
        ``deviation`` (pulse-shape deviation, higher = more distorted),
        ``echo_type`` (0 single, 1 first, 2 interior, 3 last) and
        ``gps_time`` (s, scanner clock). With MTA resolution also
        ``mta_zone``, ``return_number`` and ``number_of_returns`` (recounted
        over the kept echoes), ``scan_line_index`` (RiSCAN's convention:
        negative, -1 is the first line), ``shot_index_line`` and
        ``mirror_facet``.

    Raises
    ------
    ValueError
        If RiVLib (or, with ``mta="on"``, the shim) cannot be found or an
        error is reported reading the file.
    """
    xyz, attrs = _core.read_rxp(str(path), **options)
    return PointCloud(xyz, attrs)


def read_rxp_shots(path: str | Path, **options) -> Shots:
    """Read a RIEGL ``.rxp`` scan as pulses.

    Echoes sharing a timestamp are grouped into one pulse. RIEGL records only
    pulses that returned something, so misses must be added afterwards with
    :meth:`sylva.Shots.fill_missing` for gap fraction and ray tracing.

    Parameters
    ----------
    path
        The ``.rxp`` file of one scan position.
    **options
        As for :func:`read_rxp`. Thin with ``shot_stride``, not ``stride``.

    Returns
    -------
    Shots
        Pulses in the scanner frame (origin at 0, 0, 0), with the echo
        attributes of :func:`read_rxp`. Transform with the SOP before
        combining scans.

    Raises
    ------
    ValueError
        If RiVLib cannot be found or reports an error reading the file.
    """
    return Shots._from_core(_core.read_rxp_shots(str(path), **options))


def find_rivlib_shim(hint: str | Path | None = None) -> Path:
    """Locate ``libsylva_rivlib``, Sylva's shim over RiVLib's C++ class.

    The shim is what makes :func:`read_rxp` resolve MTA ambiguity like
    RiSCAN PRO. It is built once from the RiVLib SDK with
    ``crates/sylva-rs/rivlib-shim/build.sh`` (RiVLib is linked statically;
    the result stays on your machine).

    Parameters
    ----------
    hint
        A file, or a directory to search (three levels deep) first.

    Returns
    -------
    Path
        The library found. After ``hint`` the search order is
        ``$SYLVA_RIVLIB_SHIM``, then the same roots as :func:`find_rivlib`
        (``$RIVLIB_PATH``, ``~/.local/lib``, ``/opt``, ...).

    Raises
    ------
    ValueError
        If it is not found, with the build instructions.
    """
    return Path(_core.find_rivlib_shim(None if hint is None else str(hint)))


def find_rivlib(hint: str | Path | None = None) -> Path:
    """Locate RiVLib's ``libscanifc`` shared library.

    Parameters
    ----------
    hint
        A file, or a directory to search (three levels deep) first.

    Returns
    -------
    Path
        The library found. After ``hint``, the search order is
        ``$RIVLIB_PATH``, ``$RIVLIB_HOME``, ``~/.local/lib``, ``~/.local``,
        ``~/lib``, ``~/opt``, ``~``, ``/opt``, ``/usr/local/lib``,
        ``/usr/local`` and ``/usr/lib``, looking in subdirectories whose
        name contains "rivlib".

    Raises
    ------
    ValueError
        If no ``libscanifc`` is found. Set ``RIVLIB_PATH`` to the extracted
        RiVLib directory.
    """
    return Path(_core.find_rivlib(str(hint) if hint else None))


def read_rdbx(path: str | Path, **options) -> PointCloud:
    """Read a RIEGL ``.rdbx`` database (RDB 2) as points.

    RiSCAN PRO writes one when it imports a scan: the echoes of the ``.rxp``
    after MTA resolution and the import filters, with the processing
    settings stored beside them as metadata (see :func:`rdbx_info`). Needs
    RIEGL's ``librdb`` from the RDB SDK, which RIEGL does not allow to be
    redistributed; see :func:`find_rdblib` for where it is looked for.

    Parameters
    ----------
    path
        The ``.rdbx`` file, e.g.
        ``project.rdb/SCANS/ScanPos001/SINGLESCANS/<scan>/<scan>.rdbx``.
    **options
        Keyword options:

        - ``library`` (path): ``librdb`` or a directory containing it.
        - ``attributes`` (None): names of the attributes to read besides
          the coordinates, by RDB name (``"riegl.reflectance"``), without
          the namespace (``"reflectance"``) or by the name used here
          (``"return_number"``). Everything by default.
        - ``filter`` (None): a filter in RDB's own syntax, with RDB names,
          for example ``"riegl.reflectance > -10 && riegl.deviation < 20"``;
          the library evaluates it while reading.
        - ``stride`` (1): keep every n-th point.
        - ``max_points`` (None): stop after this many points.

    Returns
    -------
    PointCloud
        Points of ``riegl.xyz`` in the frame the database stores (a scan
        position's own database: the scanner's coordinate system, like
        :func:`read_rxp`). Every other attribute is a column, in the type
        the database suggests for it, except that ``int16`` is widened to
        ``int32`` and ``uint64`` to ``int64``. Names: ``riegl.target_index``
        is ``return_number``, ``riegl.target_count`` ``number_of_returns``,
        ``riegl.timestamp`` ``gps_time`` (s), ``riegl.class``
        ``classification`` and ``riegl.id`` ``id``, so a LAS/LAZ written
        from the cloud carries them as standard dimensions; ``riegl.rgba``
        gives ``red``, ``green``, ``blue``, ``alpha``; any other attribute
        keeps its name without the ``riegl.`` namespace, dots replaced by
        underscores, and a vector attribute of length n gives ``name_0`` ..
        ``name_{n-1}``. The usual scan attributes are therefore
        ``amplitude``, ``reflectance`` (dB), ``deviation``, ``mta_zone``,
        ``mta_uncertain_point``, ``dyntrig_uncertain_point`` and so on.

    Raises
    ------
    ValueError
        If ``librdb`` cannot be found, the database cannot be opened, an
        attribute is unknown or the filter does not parse (the message
        carries the library's reason).
    """
    xyz, attrs = _core.read_rdbx(str(path), **options)
    return PointCloud(xyz, attrs)


@dataclass(frozen=True)
class RdbxAttribute:
    """One point attribute of an ``.rdbx`` database, as the database describes it.

    ``dtype`` is the type the database suggests for a buffer (numpy style);
    :func:`read_rdbx` widens ``int16`` to ``int32`` and ``uint64`` to
    ``int64``. ``named_values`` maps the values of an enumerated attribute
    (``riegl.class``, ``riegl.fwa``, ...) to their names.
    """

    name: str
    title: str
    unit: str
    length: int
    dtype: str
    resolution: float
    minimum: float
    maximum: float
    invalid: float | None
    named_values: dict[str, str]


@dataclass(frozen=True)
class RdbxInfo:
    """What an ``.rdbx`` database holds besides its points, see :func:`rdbx_info`."""

    point_count: int
    primary_attribute: str
    attributes: dict[str, RdbxAttribute]
    metadata: dict[str, str]
    bounds: tuple[np.ndarray, np.ndarray] | None
    library_version: str

    def json(self, name: str):
        """A metadata entry parsed as JSON (RIEGL's entries all are).

        Parameters
        ----------
        name
            The entry, a key of ``metadata``. For example
            ``"riegl.mta_settings"`` holds the MTA zone width, modulation
            depth and zone count the scan was resolved with,
            ``"riegl.notch_filter"`` the near-range notch and
            ``"riegl.device"`` the scanner type and serial number.

        Returns
        -------
        object
            The parsed value: usually a dict, as ``json.loads`` returns it.

        Raises
        ------
        KeyError
            If the database has no entry of that name.
        ValueError
            If the entry is not JSON (user-written entries may be plain text).
        """
        return json.loads(self.metadata[name])


def rdbx_info(path: str | Path, library: str | Path | None = None) -> RdbxInfo:
    """Describe an ``.rdbx`` database without reading its points.

    Parameters
    ----------
    path
        The ``.rdbx`` file.
    library
        ``librdb`` or a directory containing it, if not found on its own.

    Returns
    -------
    RdbxInfo
        The point count, the attribute the index is built on (usually
        ``riegl.xyz``), every attribute by RDB name with its type, length,
        unit and declared range, every metadata entry as text (``json()``
        parses one), the extent of the coordinates from the index, and the
        version of the ``librdb`` used.

    Raises
    ------
    ValueError
        If ``librdb`` cannot be found or the database cannot be opened.
    """
    d = _core.rdbx_info(str(path), str(library) if library else None)
    attributes = {a["name"]: RdbxAttribute(**{**a, "named_values": dict(a["named_values"])})
                  for a in d["attributes"]}
    bounds = d["bounds"]
    if bounds is not None:
        bounds = (np.asarray(bounds[0]), np.asarray(bounds[1]))
    return RdbxInfo(d["point_count"], d["primary_attribute"], attributes, d["metadata"], bounds,
                    d["library_version"])


def find_rdblib(hint: str | Path | None = None) -> Path:
    """Locate RIEGL's ``librdb`` shared library (the RDB SDK).

    Parameters
    ----------
    hint
        A file, or a directory to search (three levels deep) first.

    Returns
    -------
    Path
        The library found. After ``hint``, the search order is
        ``$RDBLIB_PATH``, ``$RDBLIB_HOME``, ``~/.local/lib``, ``~/.local``,
        ``~/lib``, ``~/opt``, ``~``, ``/opt``, ``/usr/local/lib``,
        ``/usr/local`` and ``/usr/lib``, looking in subdirectories whose
        name contains "rdb" (the SDK unpacks to
        ``rdblib-<version>-<platform>/library/librdb.so``).

    Raises
    ------
    ValueError
        If no ``librdb`` is found. Set ``RDBLIB_PATH`` to the extracted
        RDB SDK directory.
    """
    return Path(_core.find_rdblib(str(hint) if hint else None))


def read_matrix_file(path: str | Path) -> np.ndarray:
    """Read a 4x4 transform stored as 16 whitespace-separated numbers.

    Parameters
    ----------
    path
        Text file in row-major order, as RiSCAN exports SOP/POP matrices
        (``.dat``, ``.txt``).

    Returns
    -------
    numpy.ndarray
        ``(4, 4)`` float64 matrix for :meth:`PointCloud.transform`.

    Raises
    ------
    OSError
        If the file cannot be read or does not hold exactly 16 numbers.
    """
    return _core.read_matrix_file(str(path))
