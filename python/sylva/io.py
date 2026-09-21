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
    """Read a point cloud, choosing the reader from the file extension."""
    xyz, attrs = _core.read(str(path))
    return PointCloud(xyz, attrs)


def write(cloud: PointCloud, path: str | Path, point_format: int = 6, scale: float = 0.001,
          binary: bool = True) -> None:
    """Write a point cloud, choosing the writer from the file extension.

    ``point_format``/``scale`` apply to LAS/LAZ; ``binary`` to PLY. Attributes
    that are not standard LAS dimensions are stored as extra bytes.
    """
    _core.write(str(path), cloud.xyz, cloud.attrs, point_format, scale, binary)


def read_ascii(path: str | Path, columns: list[str] | None = None) -> PointCloud:
    """Read delimited text; ``columns`` names the attribute columns (after x, y, z)."""
    xyz, attrs = _core.read_ascii(str(path), columns)
    return PointCloud(xyz, attrs)


def read_rxp(path: str | Path, **options) -> PointCloud:
    """Read a RIEGL ``.rxp`` (scanner coordinates, scanner at the origin).

    Needs RiVLib's ``libscanifc`` (set ``RIVLIB_PATH`` or pass ``library=``).
    Options: ``drop_pseudo_echoes=True``, ``min_range=0.5``, ``max_range``,
    ``stride=1``, ``max_points``, ``echoes="all"|"first"|"last"|"single"``.
    """
    xyz, attrs = _core.read_rxp(str(path), **options)
    return PointCloud(xyz, attrs)


def read_rxp_shots(path: str | Path, **options) -> Shots:
    """Read a RIEGL ``.rxp`` as pulses (see :func:`read_rxp` for options)."""
    return Shots._from_core(_core.read_rxp_shots(str(path), **options))


def find_rivlib(hint: str | Path | None = None) -> Path:
    """Locate RiVLib's ``libscanifc`` shared library."""
    return Path(_core.find_rivlib(str(hint) if hint else None))


def read_matrix_file(path: str | Path) -> np.ndarray:
    """Read a 4x4 whitespace-delimited matrix (RIEGL SOP/POP ``.dat``)."""
    return _core.read_matrix_file(str(path))
