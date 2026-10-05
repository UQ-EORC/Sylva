# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Triangle-mesh helpers and OBJ/PLY writers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import _core


def _rgb(color):
    """One RGB triple as three ints (0-255), or None."""
    return None if color is None else [int(c) for c in np.broadcast_to(np.asarray(color, dtype=np.uint8), (3,))]


def _faces(faces) -> np.ndarray:
    """``(m, 3)`` vertex indices as uint32; an error for negative ones."""
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if (f < 0).any():
        raise ValueError("face indices must not be negative")
    return np.ascontiguousarray(f, dtype=np.uint32)


def write_obj(path: str | Path, meshes: list[tuple[np.ndarray, np.ndarray]],
              names: list[str] | None = None) -> None:
    """Write several meshes to one OBJ file, one named object each.

    Parameters
    ----------
    path
        Output file.
    meshes
        ``(vertices, faces)`` pairs with 0-based face indices, e.g. from
        :meth:`QSM.mesh` for every tree of a plot.
    names
        Object names; ``tree_1``, ``tree_2``, ... if None.
    """
    meshes = list(meshes)
    names = [str(names[i]) if names else f"tree_{i + 1}" for i in range(len(meshes))]
    _core.write_obj(str(path), [(np.ascontiguousarray(v, dtype=float).reshape(-1, 3), _faces(f)) for v, f in meshes],
                    names)


def write_ply_mesh(path: str | Path, vertices: np.ndarray, faces: np.ndarray,
                   face_rgb: np.ndarray | None = None) -> None:
    """Write a binary little-endian PLY triangle mesh.

    Parameters
    ----------
    path
        Output file.
    vertices
        ``(n, 3)`` coordinates, stored as float32.
    faces
        ``(m, 3)`` 0-based vertex indices.
    face_rgb
        Optional ``(m, 3)`` uint8 colour per face.
    """
    faces = np.ascontiguousarray(np.asarray(faces).astype(np.int32, copy=False).reshape(-1, 3))
    rgb = None
    if face_rgb is not None:
        rgb = np.ascontiguousarray(np.broadcast_to(np.asarray(face_rgb, dtype=np.uint8), (len(faces), 3)))
    _core.write_ply_mesh(str(path), np.ascontiguousarray(vertices, dtype=float).reshape(-1, 3), faces, rgb)
