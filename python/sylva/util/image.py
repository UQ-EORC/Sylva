# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Images for the eye: colour maps and PNG files, without an image library.

Used by :meth:`sylva.Shots.scan_image` and
:func:`sylva.canopy.hemispherical_photo` to turn a field into colours and
write it out. Four colour maps are built in; with matplotlib installed any
of its maps can be named as well.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

__all__ = ["COLORMAPS", "colormap", "to_rgb", "write_png"]

#: Built-in colour maps as nine evenly spaced hex anchors (the 9-class
#: versions of matplotlib's perceptually uniform maps) plus a grey ramp.
#: Any name may take an ``_r`` suffix to reverse it.
COLORMAPS: dict[str, list[str]] = {
    "viridis": ["#440154", "#472d7b", "#3b528b", "#2c728e", "#21908c",
              "#27ad81", "#5dc863", "#aadc32", "#fde725"],
    "inferno": ["#000004", "#1b0c42", "#4b0c6b", "#781c6d", "#a52c60",
              "#cf4446", "#ed6925", "#fb9a06", "#fcffa4"],
    "magma": ["#000004", "#180f3e", "#451077", "#721f81", "#9f2f7f",
            "#cd4071", "#f1605d", "#fd9567", "#fcfdbf"],
    "gray": ["#000000", "#ffffff"],
}


def _hex(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def colormap(name: str = "viridis", n: int = 256) -> np.ndarray:
    """A colour map as a lookup table.

    Parameters
    ----------
    name
        One of :data:`COLORMAPS`, optionally with ``_r`` to reverse it, or
        the name of a matplotlib colour map when matplotlib is installed.
    n
        Number of entries.

    Returns
    -------
    numpy.ndarray
        ``(n, 3)`` RGB in 0-1.

    Raises
    ------
    ValueError
        For a name that is neither built in nor known to matplotlib.
    """
    base, reverse = (name[:-2], True) if name.endswith("_r") else (name, False)
    if base in COLORMAPS:
        anchors = np.array([_hex(h) for h in COLORMAPS[base]])
        x = np.linspace(0.0, 1.0, len(anchors))
        t = np.linspace(0.0, 1.0, n)
        lut = np.column_stack([np.interp(t, x, anchors[:, c]) for c in range(3)])
        return lut[::-1] if reverse else lut
    try:
        import matplotlib
        return np.asarray(matplotlib.colormaps[name](np.linspace(0.0, 1.0, n)))[:, :3]
    except (ImportError, KeyError):
        known = ", ".join(COLORMAPS)
        raise ValueError(f"unknown colour map {name!r}; built in: {known} (add _r to reverse); "
                         "install matplotlib for its maps") from None


def to_rgb(data: np.ndarray, cmap: str = "viridis", vmin: float | None = None,
           vmax: float | None = None, log: bool = False, nan_color=(0, 0, 0)) -> np.ndarray:
    """Colour an array of values.

    Parameters
    ----------
    data
        Values; NaN is drawn in ``nan_color``.
    cmap
        Colour map name (see :func:`colormap`).
    vmin, vmax
        Values mapped to the ends of the colour map. Defaults are the 2nd
        and 98th percentiles of the finite values, so a few outliers do not
        flatten the rest.
    log
        Map ``log10`` of the values instead (values at or below zero are
        drawn as NaN).
    nan_color
        RGB in 0-255 for NaN.

    Returns
    -------
    numpy.ndarray
        ``uint8`` of shape ``data.shape + (3,)``.
    """
    v = np.asarray(data, dtype=float)
    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(v > 0, np.log10(v), np.nan)
    finite = np.isfinite(v)
    if vmin is None:
        vmin = float(np.percentile(v[finite], 2)) if finite.any() else 0.0
    elif log:
        vmin = np.log10(vmin)
    if vmax is None:
        vmax = float(np.percentile(v[finite], 98)) if finite.any() else 1.0
    elif log:
        vmax = np.log10(vmax)
    if not vmax > vmin:
        vmax = vmin + 1.0
    lut = colormap(cmap)
    t = np.clip((np.nan_to_num(v, nan=vmin) - vmin) / (vmax - vmin), 0.0, 1.0)
    idx = np.rint(t * (len(lut) - 1)).astype(int)
    rgb = (lut[idx] * 255.0 + 0.5).astype(np.uint8)
    rgb[~finite] = np.asarray(nan_color, dtype=np.uint8)
    return rgb


def write_png(path: str | Path, image: np.ndarray) -> Path:
    """Write an 8-bit greyscale or RGB image as a PNG.

    Parameters
    ----------
    path
        Output file.
    image
        ``(rows, cols)`` greyscale or ``(rows, cols, 3)`` RGB, ``uint8``
        (other dtypes are clipped to 0-255 and converted).

    Returns
    -------
    pathlib.Path
        The file written.

    Raises
    ------
    ValueError
        For another shape.
    """
    a = np.asarray(image)
    if a.dtype != np.uint8:
        a = np.clip(np.nan_to_num(a), 0, 255).astype(np.uint8)
    if a.ndim == 2:
        colour_type = 0
    elif a.ndim == 3 and a.shape[2] == 3:
        colour_type = 2
    else:
        raise ValueError(f"image must be (rows, cols) or (rows, cols, 3), got {a.shape}")
    rows, cols = a.shape[:2]
    raw = np.concatenate([np.zeros((rows, 1), np.uint8), a.reshape(rows, -1)], axis=1).tobytes()

    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
        return struct.pack(">I", len(body)) + kind + body + crc

    header = struct.pack(">IIBBBBB", cols, rows, 8, colour_type, 0, 0, 0)
    out = Path(path)
    out.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                    + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    return out
