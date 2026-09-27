# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Coordinate systems: shifts, rotations, reprojection and transform files.

Shifting and rotating are methods of :class:`~sylva.PointCloud`
(:meth:`~sylva.PointCloud.translate`, :meth:`~sylva.PointCloud.rotate`,
:meth:`~sylva.PointCloud.recentre`); this module builds their matrices,
reprojects between coordinate reference systems and applies registration
results to whole surveys.

A CRS is given as an EPSG code (``"EPSG:7855"``, ``7855``, a compound
``"EPSG:7855+5711"``), a PROJ string (``"+proj=utm +zone=55 +south
+ellps=GRS80"``) or WKT (as found in LAS files, ESRI ``.prj`` files or
exported by GIS). EPSG definitions are built in (the EPSG registry's
horizontal CRSs with codes up to 65535) and reprojection runs in the Rust
core without PROJ or GDAL.

How exact a reprojection is depends on the datums involved:

| Transformation | Example | Result |
|---|---|---|
| Same datum, other projection | MGA zone 55 to GDA2020 latitude/longitude; UTM to WGS 84 | exact (float rounding) |
| Datums tied to WGS 84 by zero parameters | GDA94, NAD83, ETRS89, NZGD2000 to WGS 84 | exact under that convention |
| Helmert (``towgs84``) datum change | OSGB36 to WGS 84 | exact to the published parameters; heights change |
| Datums without parameters between them | GDA2020 to WGS 84 or GDA94 | approximate: not applied (null transformation) |
| Grid-based datum change | NAD27; NTv2 grids | refused (grids are not available) |
| Vertical datum change | AHD to ellipsoidal heights | approximate: heights not changed |

Approximate transformations issue an :class:`ApproximateTransformationWarning`
naming what was not applied; :func:`transformation` reports it beforehand.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _core
from .pointcloud import PointCloud

__all__ = [
    "ApproximateTransformationWarning",
    "CRS",
    "Transformation",
    "apply_transforms",
    "crs_info",
    "reproject",
    "rotation_matrix",
    "same_crs",
    "transformation",
    "translation_matrix",
]

_CLOUD_SUFFIXES = (".las", ".laz", ".ply", ".xyz", ".txt", ".csv", ".asc", ".pts", ".rxp")
_WRITABLE = (".las", ".laz", ".ply", ".xyz", ".txt", ".csv", ".asc", ".pts")


class ApproximateTransformationWarning(UserWarning):
    """A reprojection could not apply part of the transformation (a datum
    shift without known parameters, or a change of vertical datum)."""


# --------------------------------------------------------------------------- #
# Matrices
# --------------------------------------------------------------------------- #


def _finite3(value, what: str) -> np.ndarray:
    v = np.asarray(value, dtype=np.float64).reshape(-1)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ValueError(f"{what} must be three finite numbers, got {value!r}")
    return v


def translation_matrix(dx: float, dy: float, dz: float = 0.0) -> np.ndarray:
    """A 4x4 translation.

    Parameters
    ----------
    dx, dy, dz
        Offset (m).

    Returns
    -------
    numpy.ndarray
        ``(4, 4)`` matrix for :meth:`PointCloud.transform`.

    Raises
    ------
    ValueError
        If an offset is not finite.
    """
    d = _finite3([dx, dy, dz], "the offset")
    return _core.coords_translation_matrix(*map(float, d))


def rotation_matrix(angle_deg: float, axis: str | Iterable[float] = "z",
                    about: Iterable[float] | None = None) -> np.ndarray:
    """A 4x4 rotation about an axis through a point.

    Parameters
    ----------
    angle_deg
        Angle in degrees, right-handed (counter-clockwise looking down the
        axis towards the origin). Multiples of 90 degrees are exact.
    axis
        ``"x"``, ``"y"``, ``"z"`` or a 3-vector of any non-zero length.
    about
        A point on the axis; the origin if None.

    Returns
    -------
    numpy.ndarray
        ``(4, 4)`` matrix ``[R | c - R c]`` for centre ``c``.

    Raises
    ------
    ValueError
        If the angle, axis or centre is not finite, or the axis is zero or
        an unknown name.
    """
    angle = float(angle_deg)
    if not np.isfinite(angle):
        raise ValueError(f"angle_deg must be finite, got {angle_deg!r}")
    if isinstance(axis, str):
        names = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
        key = axis.strip().lower()
        if key not in names:
            raise ValueError(f'axis must be "x", "y", "z" or a 3-vector, got {axis!r}')
        a = names[key]
    else:
        a = tuple(map(float, _finite3(axis, "axis")))
        if not any(a):
            raise ValueError("axis must not be the zero vector")
    c = None if about is None else tuple(map(float, _finite3(about, "about")))
    return _core.coords_rotation_matrix(a, angle, c)


# --------------------------------------------------------------------------- #
# Coordinate reference systems
# --------------------------------------------------------------------------- #


def _crs_text(crs, what: str = "crs") -> str:
    """A CRS definition as the core takes it."""
    if isinstance(crs, bool) or crs is None:
        raise ValueError(f"{what} must be an EPSG code, PROJ string or WKT, got {crs!r}")
    if isinstance(crs, (int, np.integer)):
        return f"EPSG:{int(crs)}"
    if isinstance(crs, str):
        if not crs.strip():
            raise ValueError(f"{what} is empty")
        return crs.strip()
    if hasattr(crs, "to_wkt"):                      # pyproj / rasterio CRS objects
        return str(crs.to_wkt())
    raise ValueError(f"{what} must be an EPSG code, PROJ string or WKT, got {type(crs).__name__}")


@dataclass(frozen=True)
class CRS:
    """What Sylva understands of a coordinate reference system.

    Attributes
    ----------
    name
        Name from the definition (``"GDA2020 / MGA zone 55"``).
    label
        ``"EPSG:7855"`` (``"EPSG:7855+5711"`` for a compound CRS) when the
        code is known, otherwise the name.
    epsg
        EPSG code of the horizontal part, or None.
    vertical_epsg
        EPSG code of the vertical part, or None.
    proj4
        PROJ string the coordinates are transformed with.
    wkt
        WKT written to LAS files, or None (PROJ strings have none).
    datum
        Datum identity used to decide whether two CRSs share a datum.
    geographic
        True when x and y are longitude and latitude in degrees.
    """

    name: str
    label: str
    epsg: int | None
    vertical_epsg: int | None
    proj4: str
    wkt: str | None
    datum: str | None
    geographic: bool | None


def crs_info(crs) -> CRS:
    """Parse a CRS.

    Parameters
    ----------
    crs
        EPSG code (``"EPSG:7855"``, ``7855``, ``"EPSG:7855+5711"``,
        ``"urn:ogc:def:crs:EPSG::7855"``), PROJ string, WKT (1, ESRI or 2),
        or an object with a ``to_wkt()`` method (pyproj, rasterio).

    Returns
    -------
    CRS

    Raises
    ------
    ValueError
        If the definition is not recognised, the EPSG code is not in the
        built-in tables, or WKT without an EPSG code uses a projection Sylva
        cannot convert.
    """
    d = _core.crs_info(_crs_text(crs))
    return CRS(d["name"], d["label"], d["epsg"], d["vertical_epsg"], d["proj4"], d["wkt"],
               d["datum"], d["geographic"])


def same_crs(a, b) -> bool:
    """Whether two CRS definitions describe the same system.

    Parameters
    ----------
    a, b
        CRS definitions, as for :func:`crs_info`.

    Returns
    -------
    bool
        True for equal definitions, for the same EPSG code(s) written
        differently (``7855``, ``"EPSG:7855"``, its WKT), or for the same
        PROJ definition. False if either cannot be parsed.
    """
    if a is None or b is None:
        return a is b
    ta, tb = _crs_text(a), _crs_text(b)
    if ta == tb:
        return True
    try:
        ia, ib = crs_info(ta), crs_info(tb)
    except ValueError:
        return False
    if ia.epsg is not None and ib.epsg is not None:
        return (ia.epsg, ia.vertical_epsg) == (ib.epsg, ib.vertical_epsg)
    return ia.proj4 == ib.proj4 and ia.vertical_epsg == ib.vertical_epsg


@dataclass(frozen=True)
class Transformation:
    """How :func:`reproject` gets from one CRS to another.

    Attributes
    ----------
    kind
        ``"identity"`` (same CRS), ``"conversion"`` (same datum, other
        projection), ``"helmert"`` (datum change by published ``towgs84``
        parameters) or ``"null datum"`` (datums without parameters between
        them; not applied).
    exact
        True when the result is exact to float rounding (and to the
        published parameters of a Helmert change).
    changes_z
        True when heights change (Helmert datum changes, which treat z as
        ellipsoidal height).
    note
        Explanation; says what is not applied when ``exact`` is False.
    """

    kind: str
    exact: bool
    changes_z: bool
    note: str


def transformation(src_crs, dst_crs) -> Transformation:
    """Describe the transformation between two CRSs without applying it.

    Parameters
    ----------
    src_crs, dst_crs
        CRS definitions, as for :func:`crs_info`.

    Returns
    -------
    Transformation

    Raises
    ------
    ValueError
        If a CRS cannot be parsed or the transformation needs a datum grid.
    """
    d = _core.crs_plan(_crs_text(src_crs, "src_crs"), _crs_text(dst_crs, "dst_crs"))
    return Transformation(d["kind"], d["exact"], d["changes_z"], d["note"])


def reproject(data, dst_crs, src_crs=None):
    """Transform coordinates into another coordinate reference system.

    Parameters
    ----------
    data
        A :class:`~sylva.PointCloud`, or an ``(N, 2)`` or ``(N, 3)`` array of
        x, y (, z). Geographic coordinates are x = longitude and
        y = latitude in degrees.
    dst_crs
        Target CRS: EPSG code, PROJ string or WKT (see :func:`crs_info`).
    src_crs
        CRS of ``data``; defaults to the cloud's ``crs``.

    Returns
    -------
    PointCloud or numpy.ndarray
        Same type as ``data``. A cloud keeps its attributes and gets
        ``crs = dst_crs`` (an integer code becomes ``"EPSG:n"``). Points
        outside a projection's domain, or with NaN coordinates, become NaN.
        z changes only when the datum does by Helmert parameters (see
        :func:`transformation`).

    Raises
    ------
    ValueError
        If there is no source CRS, a CRS cannot be parsed, the
        transformation needs a datum grid, or an array has the wrong shape.
    TypeError
        For a :class:`~sylva.Raster` or another unsupported type. Rasters
        are not reprojected: a regular grid does not stay regular under a
        change of projection, so it must be resampled (e.g. with GDAL's
        ``gdalwarp``) rather than have its extent moved.

    Warns
    -----
    ApproximateTransformationWarning
        When part of the transformation cannot be applied (a datum change
        without known parameters, or a vertical datum change).

    Notes
    -----
    Uses proj4rs (a Rust port of PROJ.4) with the EPSG registry's PROJ.4
    definitions. Transformations that PROJ carries out with grids (NTv2,
    NADCON, geoid models) are not available; the GDA94 to GDA2020 Helmert
    can be applied by giving GDA2020 as a PROJ string with its ``towgs84``
    parameters (see the coordinates guide).

    Examples
    --------
    >>> cloud = sylva.read("plot.laz")                 # crs from the LAS header
    >>> lonlat = sylva.coords.reproject(cloud, "EPSG:7844")
    """
    from .raster import Raster

    if isinstance(data, Raster):
        raise TypeError("rasters are not reprojected: a regular grid must be resampled into "
                        "the new CRS (e.g. with gdalwarp), not have its extent moved; reproject "
                        "the point cloud and rebuild the raster instead")
    if isinstance(data, PointCloud):
        src = src_crs if src_crs is not None else data.crs
        if src is None:
            raise ValueError("the cloud has no crs; give src_crs")
        dst = _crs_text(dst_crs, "dst_crs")
        xyz = _reproject_xyz(data.xyz, _crs_text(src, "src_crs"), dst)
        return PointCloud(xyz, dict(data.attrs), dst)
    if isinstance(data, (np.ndarray, list, tuple)):
        a = np.asarray(data, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] not in (2, 3):
            raise ValueError(f"coordinates must have shape (N, 2) or (N, 3), got {a.shape}")
        if src_crs is None:
            raise ValueError("give src_crs for an array of coordinates")
        xyz = a if a.shape[1] == 3 else np.column_stack([a, np.zeros(len(a))])
        out = _reproject_xyz(np.ascontiguousarray(xyz), _crs_text(src_crs, "src_crs"),
                             _crs_text(dst_crs, "dst_crs"))
        return out if a.shape[1] == 3 else out[:, :2]
    raise TypeError(f"cannot reproject a {type(data).__name__}; give a PointCloud or an array")


def _reproject_xyz(xyz: np.ndarray, src: str, dst: str) -> np.ndarray:
    out, plan = _core.crs_reproject(xyz, src, dst)
    if not plan["exact"]:
        warnings.warn(f"approximate reprojection: {plan['note']}",
                      ApproximateTransformationWarning, stacklevel=3)
    return out


# --------------------------------------------------------------------------- #
# Transform files
# --------------------------------------------------------------------------- #


def _is_riscan_project(d: Path) -> bool:
    return (d.suffix.lower() in (".riscan", ".proj") or (d / "project.rsp").is_file()
            or (d / "all_sop.csv").is_file())


def _load_transforms(transforms) -> dict[str, np.ndarray] | list[np.ndarray]:
    """Transforms as ``{name: matrix}`` or a positional list."""
    if isinstance(transforms, (str, Path)):
        path = Path(transforms)
        if path.is_dir():
            if _is_riscan_project(path):
                from .riscan import read_riscan_project

                project = read_riscan_project(path)
                found = {p.name: p.sop for p in project if p.sop is not None}
                if not found:
                    raise ValueError(f"no SOP matrices in the RiSCAN project {path}")
                return found
            files = sorted(f for f in path.iterdir()
                           if f.is_file() and f.suffix.lower() == ".dat")
            if not files:
                raise ValueError(f"no .DAT matrix files in {path}")
            from .io import read_matrix_file

            return {f.stem: read_matrix_file(f) for f in files}
        if path.suffix.lower() == ".json":
            from .coreg import load_transforms

            return load_transforms(path)
        from .io import read_matrix_file

        return [read_matrix_file(path)]                 # one matrix, for every scan
    if isinstance(transforms, Mapping):
        return {str(k): np.asarray(v, dtype=np.float64) for k, v in transforms.items()}
    a = np.asarray(transforms, dtype=np.float64)
    if a.shape == (4, 4):
        return [a]
    if a.ndim == 3 and a.shape[1:] == (4, 4):
        return list(a)
    raise ValueError("transforms must be a {name: 4x4} mapping, a list of 4x4 matrices, a "
                     "transforms.json, a .DAT file or a directory of .DAT files or a RiSCAN project")


def _scan_list(scans) -> list[tuple[str | None, PointCloud | Path]]:
    if isinstance(scans, PointCloud):
        return [(None, scans)]
    if isinstance(scans, (str, Path)):
        p = Path(scans)
        if p.is_dir():
            files = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in _CLOUD_SUFFIXES)
            if not files:
                raise ValueError(f"no point cloud files in {p}")
            return [(f.stem, f) for f in files]
        return [(p.stem, p)]
    if isinstance(scans, Mapping):
        return [(str(k), v if isinstance(v, PointCloud) else Path(v)) for k, v in scans.items()]
    out = []
    for s in scans:
        if isinstance(s, PointCloud):
            out.append((None, s))
        elif isinstance(s, (str, Path)):
            out.append((Path(s).stem, Path(s)))
        else:
            raise ValueError(f"scans must be paths or PointClouds, got {type(s).__name__}")
    return out


def _boundary_match(token: str, name: str) -> bool:
    t, n = token.lower(), name.lower()
    return t == n or (t.startswith(n) and not t[len(n)].isalnum())


def _match(name: str | None, source, transforms: dict[str, np.ndarray]) -> str:
    """The transform name for a scan: its own name, else a path component
    (file stem or folder) equal to, or starting with, a transform name
    followed by a separator (``ScanPos001_xyz.laz`` gets ``ScanPos001``)."""
    tokens = [name] if name else []
    if isinstance(source, Path):
        tokens += [source.stem] + [q.name for q in source.parents if q.name]
    for t in tokens:
        exact = [k for k in transforms if k.lower() == t.lower()]
        if len(exact) == 1:
            return exact[0]
    for t in tokens:
        hits = [k for k in transforms if _boundary_match(t, k)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise ValueError(f"scan {t!r} matches several transforms: {sorted(hits)}")
    label = name or (str(source) if isinstance(source, Path) else "cloud")
    raise ValueError(f"no transform for scan {label!r}; transforms are named "
                     f"{sorted(transforms)[:10]}")


def apply_transforms(scans, transforms, out: str | Path | None = None,
                     merge: bool = False) -> list[PointCloud] | PointCloud:
    """Put scans into one frame with their registration matrices.

    Parameters
    ----------
    scans
        Point cloud paths or :class:`~sylva.PointCloud` objects: a list,
        a ``{name: path or cloud}`` mapping, one path, or a directory (every
        point cloud file in it). Paths are read with :func:`sylva.read`.
    transforms
        One of

        - ``{name: (4, 4)}``, matched to scans by name (see Notes);
        - a list of ``(4, 4)`` matrices, one per scan in order, or a single
          matrix applied to every scan;
        - a ``transforms.json`` written by
          :meth:`sylva.coreg.SurveyResult.save`
          (read with :func:`sylva.coreg.load_transforms`);
        - a ``.DAT`` file holding one matrix, applied to every scan;
        - a directory of RiSCAN ``.DAT`` matrices (``ScanPos001.DAT``, ...),
          named by file stem, as RiSCAN exports and ``sylva coreg`` writes;
        - a RiSCAN project directory (``.RiSCAN`` or ``.PROJ``); each
          position's SOP is used (the POP is not applied).
    out
        Where to write: a directory for one file per scan (named after the
        scan, keeping its format, ``.laz`` for clouds without a path and
        for ``.rxp``), or a file when ``merge`` is True. Nothing is written
        if None.
    merge
        Return (and write) one cloud with a ``scan_id`` attribute (int32,
        the scan's index), keeping the attributes all scans share.

    Returns
    -------
    list of PointCloud or PointCloud
        The transformed scans in input order, or the merged cloud.

    Raises
    ------
    ValueError
        If a scan has no matching transform, a name matches several, the
        numbers of scans and positional transforms differ, or a matrix is
        not 4x4.
    OSError
        If a file cannot be read or written.

    Notes
    -----
    A scan is matched to a named transform by its mapping key, or by its
    path: the file stem or any parent folder that equals a transform name or
    starts with it followed by a non-alphanumeric character. So
    ``ScanPos001.rxp``, ``ScanPos001_2cm.laz`` and
    ``ScanPos001/scans/240101_1200.rxp`` all get ``ScanPos001``, and
    ``ScanPos0011`` does not. Unnamed clouds need positional transforms.

    Examples
    --------
    >>> clouds = sylva.coords.apply_transforms("scans/", "plot_coreg/transforms.json",
    ...                                        out="plot.laz", merge=True)
    """
    items = _scan_list(scans)
    if not items:
        raise ValueError("no scans given")
    tf = _load_transforms(transforms)
    if isinstance(tf, list):
        if len(tf) == 1:
            mats = tf * len(items)
        elif len(tf) != len(items):
            raise ValueError(f"{len(items)} scans but {len(tf)} transforms")
        else:
            mats = tf
    else:
        if any(name is None and not isinstance(src, Path) for name, src in items):
            raise ValueError("clouds without names cannot be matched to named transforms; "
                             "give scans as a {name: cloud} mapping or transforms as a list")
        mats = [tf[_match(name, src, tf)] for name, src in items]
    for m in mats:
        if np.asarray(m).shape != (4, 4):
            raise ValueError(f"transforms must be 4x4, got shape {np.asarray(m).shape}")

    from .io import read, write

    clouds = []
    for (name, src), m in zip(items, mats, strict=True):
        cloud = src if isinstance(src, PointCloud) else read(src)
        clouds.append(PointCloud(_core.coords_apply(cloud.xyz, np.asarray(m, dtype=np.float64)),
                                 dict(cloud.attrs), cloud.crs))
    if merge:
        from .registration import merge_scans

        merged = merge_scans(clouds)
        if out is not None:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            write(merged, out)
        return merged
    if out is not None:
        outdir = Path(out)
        outdir.mkdir(parents=True, exist_ok=True)
        used: set[str] = set()
        for k, ((name, src), cloud) in enumerate(zip(items, clouds, strict=True)):
            stem = name or f"scan{k:03d}"
            ext = src.suffix.lower() if isinstance(src, Path) and src.suffix.lower() in _WRITABLE else ".laz"
            target = stem + ext
            if target in used:
                target = f"{stem}_{k}{ext}"
            used.add(target)
            write(cloud, outdir / target)
    return clouds
