"""Reading and writing point clouds.

``read`` and ``write`` dispatch on file extension:

| Format | read | write |
|---|---|---|
| ``.las`` / ``.laz`` | yes | yes |
| ``.ply`` (ascii / binary, incl. raycloudtools ray clouds) | yes | yes |
| ``.xyz`` ``.txt`` ``.csv`` ``.pts`` | yes | yes |
| ``.rxp`` (RIEGL, needs RiVLib) | yes | no |

Pulse data has its own format, see :meth:`sylva.Shots.save`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import _core
from .pointcloud import PointCloud
from .shots import Shots

__all__ = [
    "read", "write", "read_ascii", "read_rxp", "read_rxp_shots", "find_rivlib",
    "read_matrix_file",
]


def read(path: str | Path) -> PointCloud:
    """Read a point cloud, choosing the reader from the file extension.

    Parameters
    ----------
    path
        ``.las``, ``.laz``, ``.ply``, ``.xyz``, ``.txt``, ``.csv``, ``.asc``,
        ``.pts`` or ``.rxp``. The extension is matched case-insensitively.

    Returns
    -------
    PointCloud
        Coordinates in the file's frame (``.rxp``: scanner frame, see
        :func:`read_rxp`). Attributes depend on the format:

        - LAS/LAZ: ``intensity``, ``return_number``, ``number_of_returns``,
          ``classification``, ``scan_angle``, ``user_data``,
          ``point_source_id``, plus ``gps_time`` and ``red``/``green``/``blue``
          when the point format has them, and every extra-bytes dimension
          under its own name and type.
        - PLY: every ``vertex`` property except x, y, z, with its type.
          raycloudtools ray clouds give ``nx``, ``ny``, ``nz`` (point to
          sensor) and ``alpha``; see :meth:`sylva.Shots.from_ray_cloud`.
        - Text: see :func:`read_ascii`.
        - RXP: see :func:`read_rxp` (default options).

    Raises
    ------
    OSError
        If the file cannot be opened or is malformed.
    ValueError
        For an unsupported extension or invalid option.
    """
    xyz, attrs = _core.read(str(path))
    return PointCloud(xyz, attrs)


def write(cloud: PointCloud, path: str | Path, point_format: int = 6, scale: float = 0.001,
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
        LAS coordinate quantisation in metres (LAS/LAZ only). 0.001 keeps
        millimetres; the offset is the floor of the minimum coordinate.
    binary
        Binary little-endian PLY if True, ASCII otherwise (PLY only).

    Notes
    -----
    LAS/LAZ: attributes that are not standard dimensions of ``point_format``
    are written as typed extra bytes, so ``height``, ``tree_id`` and the like
    survive a round trip. Text files get a header line naming the columns
    and coordinates to 0.1 mm; integer attributes are written as integers.

    Raises
    ------
    OSError
        If the file cannot be opened or is malformed.
    ValueError
        For an unsupported extension or invalid option.
    """
    _core.write(str(path), cloud.xyz, cloud.attrs, point_format, scale, binary)


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
    """Read a RIEGL ``.rxp`` scan as points.

    Needs RiVLib's ``libscanifc``, which RIEGL does not allow to be
    redistributed; see :func:`find_rivlib` for where it is looked for.

    Parameters
    ----------
    path
        The ``.rxp`` file of one scan position.
    **options
        Keyword options:

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
        ``gps_time`` (s).

    Raises
    ------
    ValueError
        If RiVLib cannot be found or reports an error reading the file.
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
