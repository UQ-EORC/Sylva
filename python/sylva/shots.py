# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Pulse-centric data for ray-based canopy metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _core
from .pointcloud import PointCloud


@dataclass
class Shots:
    """Laser pulses, each with zero or more echoes.

    Point clouds lose the pulses that hit nothing and the order of echoes
    along each beam; gap fraction, clumping and ray-traced voxels need both.
    ``Shots`` keeps one row per pulse and the echoes in compressed sparse
    row (CSR) form: the echoes of shot ``i`` are
    ``echo_range[echo_start[i] : echo_start[i] + echo_count[i]]``.
    A shot with ``echo_count == 0`` is a genuine miss and still carries
    information about free space.

    Build one with :func:`sylva.io.read_rxp_shots`,
    :meth:`sylva.riscan.ScanPosition.read_shots`, :meth:`from_ray_cloud`,
    :meth:`from_pointcloud` or :meth:`load`.

    Attributes
    ----------
    origin, direction
        ``(n_shots, 3)`` beam origin and unit direction.
    echo_start, echo_count
        CSR offsets into the per-echo arrays.
    echo_range
        Range from the origin to each echo, ascending within a shot.
    echo_attrs
        Per-echo attributes (``amplitude``, ``reflectance``, ...).
    """

    origin: np.ndarray
    direction: np.ndarray
    echo_start: np.ndarray
    echo_count: np.ndarray
    echo_range: np.ndarray
    echo_attrs: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.origin = np.ascontiguousarray(self.origin, dtype=np.float64).reshape(-1, 3)
        self.direction = np.ascontiguousarray(self.direction, dtype=np.float64).reshape(-1, 3)
        self.echo_start = np.ascontiguousarray(self.echo_start, dtype=np.int64)
        self.echo_count = np.ascontiguousarray(self.echo_count, dtype=np.int64)
        self.echo_range = np.ascontiguousarray(self.echo_range, dtype=np.float64)
        self.echo_attrs = {k: np.ascontiguousarray(v) for k, v in self.echo_attrs.items()}

    @property
    def n_shots(self) -> int:
        """Number of pulses, including misses."""
        return len(self.origin)

    @property
    def n_echoes(self) -> int:
        """Number of echoes over all pulses."""
        return len(self.echo_range)

    def __repr__(self) -> str:
        return f"Shots(n_shots={self.n_shots:,}, n_echoes={self.n_echoes:,})"

    def _to_core(self) -> dict:
        return {
            "origin": self.origin, "direction": self.direction, "echo_start": self.echo_start,
            "echo_count": self.echo_count, "echo_range": self.echo_range,
            "echo_attrs": self.echo_attrs,
        }

    @classmethod
    def _from_core(cls, d: dict) -> Shots:
        return cls(d["origin"], d["direction"], d["echo_start"], d["echo_count"],
                   d["echo_range"], d["echo_attrs"])

    def shot_of_echo(self) -> np.ndarray:
        """Index of the owning shot for each echo.

        Returns
        -------
        numpy.ndarray
            Length ``n_echoes``.
        """
        return np.repeat(np.arange(self.n_shots), self.echo_count)

    def echo_rank(self) -> np.ndarray:
        """Position of each echo within its shot.

        Returns
        -------
        numpy.ndarray
            Length ``n_echoes``; 0 is the first (nearest) return.
        """
        return np.arange(self.n_echoes) - np.repeat(self.echo_start, self.echo_count)

    def echo_xyz(self) -> np.ndarray:
        """Echo positions in the shots' frame.

        Returns
        -------
        numpy.ndarray
            ``(n_echoes, 3)``, ``origin + direction * range``.
        """
        s = self.shot_of_echo()
        return self.origin[s] + self.direction[s] * self.echo_range[:, None]

    def to_pointcloud(self) -> PointCloud:
        """The echoes as a point cloud; misses are dropped.

        Returns
        -------
        PointCloud
            One point per echo with ``return_number`` (1 = first),
            ``number_of_returns``, ``range`` (m) and every echo attribute.
        """
        xyz, attrs = _core.shots_to_pointcloud(self._to_core())
        return PointCloud(xyz, attrs)

    def transform(self, matrix: np.ndarray) -> Shots:
        """Move the pulses into another frame (e.g. scanner to project with the SOP).

        Parameters
        ----------
        matrix
            ``(4, 4)`` rigid transform. Origins are transformed as points,
            directions are rotated; ranges are unchanged, so the matrix must
            not scale.

        Returns
        -------
        Shots
        """
        m = np.ascontiguousarray(matrix, dtype=np.float64)
        return Shots._from_core(_core.shots_transform(self._to_core(), m))

    def subset(self, mask: np.ndarray) -> Shots:
        """Select pulses.

        Parameters
        ----------
        mask
            Boolean array of length ``n_shots``.

        Returns
        -------
        Shots
            The selected pulses with all their echoes (never part of a pulse).
        """
        mask = np.asarray(mask, dtype=bool)
        keep_shots = np.flatnonzero(mask)
        echo_mask = mask[self.shot_of_echo()]
        count = self.echo_count[keep_shots]
        start = np.concatenate([[0], np.cumsum(count)[:-1]]) if len(count) else np.zeros(0, int)
        return Shots(
            self.origin[keep_shots], self.direction[keep_shots], start, count,
            self.echo_range[echo_mask], {k: v[echo_mask] for k, v in self.echo_attrs.items()},
        )

    def zenith_azimuth(self) -> tuple[np.ndarray, np.ndarray]:
        """Beam angles of each pulse.

        Returns
        -------
        zenith : numpy.ndarray
            Degrees from +z (0 = straight up, 90 = horizontal).
        azimuth : numpy.ndarray
            Degrees clockwise from +y, ``atan2(x, y)`` wrapped to
            ``[0, 360)`` (RIEGL convention).
        """
        d = self.direction
        zen = np.degrees(np.arccos(np.clip(d[:, 2], -1, 1)))
        az = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360.0
        return zen, az

    @classmethod
    def concatenate(cls, parts: list[Shots]) -> Shots:
        """Stack pulse sets, e.g. several scan positions.

        Parameters
        ----------
        parts
            Shots in a common frame.

        Returns
        -------
        Shots
            All pulses in order; only echo attributes present in every part
            are kept.

        Raises
        ------
        ValueError
            If ``parts`` is empty.
        """
        if not parts:
            raise ValueError("no shots to concatenate")
        counts = np.concatenate([p.echo_count for p in parts])
        start = np.concatenate([[0], np.cumsum(counts)[:-1]]) if len(counts) else np.zeros(0, int)
        common = set.intersection(*(set(p.echo_attrs) for p in parts))
        return cls(
            np.vstack([p.origin for p in parts]), np.vstack([p.direction for p in parts]),
            start, counts, np.concatenate([p.echo_range for p in parts]),
            {k: np.concatenate([p.echo_attrs[k] for p in parts]) for k in sorted(common)},
        )

    @staticmethod
    def _zenith_lines(pattern: dict) -> tuple[np.ndarray, np.ndarray]:
        theta = pattern["theta_start"] + pattern["theta_delta"] * np.arange(pattern["theta_count"])
        half = pattern["theta_delta"] / 2
        return theta, np.concatenate([theta - half, [theta[-1] + half]])

    def pulses_per_line(self, pattern: dict, quantile: float = 0.98, shot_stride: int = 1) -> int:
        """Effective number of pulses fired along each zenith line.

        Nominally ``pattern["phi_count"]``, but RIEGL scanners fire ~1 % more
        pulses than the nominal grid (a VZ-2000i at "600 kHz" steps at ~631
        kHz), so the count observed on saturated lines is a better estimate.
        Call in the scanner frame.

        Parameters
        ----------
        pattern
            Scan pattern from :attr:`sylva.riscan.ScanPosition.pattern`.
        quantile
            Quantile of the per-line shot counts taken as "fully sampled".
        shot_stride
            The ``shot_stride`` the shots were read with; the nominal count
            is divided by it.

        Returns
        -------
        int
            The larger of the nominal count and the observed ``quantile``.
        """
        _, edges = self._zenith_lines(pattern)
        zen, _ = self.zenith_azimuth()
        observed, _ = np.histogram(zen, bins=edges)
        return int(max(int(pattern["phi_count"]) / max(int(shot_stride), 1), np.quantile(observed, quantile)))

    def expected_per_zenith(self, pattern: dict, zenith_edges: np.ndarray,
                            pulses_per_line: int | None = None) -> np.ndarray:
        """Number of pulses the scan ``pattern`` fired into each zenith ring.

        ``pattern`` holds ``theta_start``, ``theta_delta``, ``theta_count``
        (zenith lines, degrees) and ``phi_count`` (azimuth steps per line), as
        parsed from a RiSCAN ``project.rsp``.

        Parameters
        ----------
        pattern
            Scan pattern from :attr:`sylva.riscan.ScanPosition.pattern`.
        zenith_edges
            Ring edges in degrees from up.
        pulses_per_line
            Overrides ``phi_count`` (see :meth:`pulses_per_line`).

        Returns
        -------
        numpy.ndarray
            Pulses fired per ring (length ``len(zenith_edges) - 1``): zenith
            lines whose angle falls in the ring times pulses per line.
        """
        theta, _ = self._zenith_lines(pattern)
        lines, _ = np.histogram(theta, bins=np.asarray(zenith_edges, dtype=float))
        n = int(pattern["phi_count"]) if pulses_per_line is None else int(pulses_per_line)
        return lines * n

    def fill_missing(self, pattern: dict, pulses_per_line: int | None = None,
                     seed: int = 0, shot_stride: int = 1) -> Shots:
        """Add the pulses that returned nothing.

        RiVLib's point stream only contains echoes, so a pulse that went to the
        sky is simply absent. Given the angular scan ``pattern`` this compares,
        per zenith line, the pulses fired against the shots observed and adds
        the difference as shots with ``echo_count == 0`` spread uniformly in
        azimuth (the exact azimuths cannot be recovered because the mirror
        angles are not on the nominal grid). Ray-traced metrics then see the
        free space these pulses sampled.

        ``pulses_per_line`` defaults to :meth:`pulses_per_line`, which corrects
        for the scanner firing slightly more pulses than the nominal grid but
        can overshoot by a few per cent. Where the scan looks at the ground,
        the median count on the downward lines is exact, since every pulse
        there returns (see :func:`sylva.canopy.fired_pulses_per_ring`).

        Call this on shots in scanner coordinates (before applying a SOP), as
        the pattern's zenith lines are defined in the scanner frame; all shots
        must share one origin. ``ScanPosition.read_shots(fill_missing=True)``
        does this in the right order.

        Parameters
        ----------
        pattern
            Scan pattern from :attr:`sylva.riscan.ScanPosition.pattern`.
        pulses_per_line
            Pulses fired per zenith line; estimated if None.
        seed
            Seed for the random azimuths, so the output is reproducible.
        shot_stride
            The ``shot_stride`` the shots were read with. The estimate of
            pulses per line is divided by it; without it, thinned shots get
            ``shot_stride`` times too many misses.

        Returns
        -------
        Shots
            The input followed by the added misses (the input itself if none
            are missing).
        """
        rng = np.random.default_rng(seed)
        theta, edges = self._zenith_lines(pattern)
        zen, _ = self.zenith_azimuth()
        observed, _ = np.histogram(zen, bins=edges)
        if pulses_per_line is None:
            pulses_per_line = self.pulses_per_line(pattern, shot_stride=shot_stride)
        missing = np.maximum(int(pulses_per_line) - observed, 0)
        n = int(missing.sum())
        if n == 0:
            return self
        zen_new = np.radians(np.repeat(theta, missing))
        az_new = np.radians(rng.uniform(0, 360, n))
        direction = np.column_stack([
            np.sin(zen_new) * np.sin(az_new), np.sin(zen_new) * np.cos(az_new), np.cos(zen_new)
        ])
        origin = np.tile(self.origin.mean(axis=0), (n, 1))
        empty = Shots(origin, direction, np.zeros(n, np.int64), np.zeros(n, np.int64),
                      np.zeros(0), {k: v[:0] for k, v in self.echo_attrs.items()})
        return Shots.concatenate([self, empty])

    def save(self, path: str | Path, double: bool = False, row_group_size: int = 1 << 20,
             zstd_level: int = 3, origin_tolerance: float = 1e-3) -> None:
        """Write the sylva shots format: a Parquet file with one row per pulse.

        Columns are ``scan`` (index into the scanner positions kept in the
        file metadata), ``zenith`` and ``azimuth`` (rad), ``range`` (a list of
        echo ranges, empty for a pulse without a return) and one list column
        per echo attribute, zstd compressed in row groups of
        ``row_group_size`` pulses that can be read or voxelised one at a time
        (:func:`sylva.voxels.ray_voxelize` accepts the path). Any Parquet
        reader (polars, pyarrow, duckdb, R arrow) opens it.

        Angles and ranges are float32 unless ``double``: echo positions then
        come back to about 0.01 mm per 100 m of range. Origins within
        ``origin_tolerance`` (m) of each other become one scanner position
        (echoes stay put; ray clouds carry per-ray rounding noise on the
        origin); ``0`` keeps them exact. With more than 65 536 positions
        (mobile or airborne scanning) origins are stored per pulse at 0.1 mm,
        delta encoded.

        Parameters
        ----------
        path
            Output ``.parquet`` file; overwritten if it exists.
        double
            Store angles and ranges as float64.
        row_group_size
            Pulses per row group; the unit of partial reads and of streaming
            in :func:`sylva.voxels.ray_voxelize`. Smaller groups use less
            memory when streaming.
        zstd_level
            zstd compression level (1 fast, 22 smallest).
        origin_tolerance
            Distance (m) within which origins are merged into one position.
        """
        _core.write_shots(self._to_core(), str(path), double, int(row_group_size),
                          int(zstd_level), float(origin_tolerance))

    @classmethod
    def load(cls, path: str | Path, groups: list[int] | None = None) -> Shots:
        """Read a shots file written by :meth:`save`.

        Parameters
        ----------
        path
            ``.parquet`` shots file.
        groups
            Row-group indices to read (see :meth:`file_info`); all if None.

        Returns
        -------
        Shots

        Raises
        ------
        OSError
            If the file is missing or is not a Sylva shots file.
        """
        return cls._from_core(_core.read_shots(str(path), groups))

    @staticmethod
    def file_info(path: str | Path) -> dict:
        """Describe a shots file without reading its pulses.

        Parameters
        ----------
        path
            ``.parquet`` shots file.

        Returns
        -------
        dict
            ``n_shots``, ``n_echoes``, ``n_groups``, echo ``bounds``,
            ``scans`` (scanner positions; empty when origins are stored per
            pulse) and ``echo_attrs`` (names).
        """
        return _core.shots_info(str(path))

    @classmethod
    def from_pointcloud(cls, cloud: PointCloud, origin=(0.0, 0.0, 0.0)) -> Shots:
        """Pulses from a single-position point cloud.

        Use when the only record of a scan is its points (no ``.rxp``). Misses
        are not recovered; add them with :meth:`fill_missing` if the scan
        pattern is known.

        Parameters
        ----------
        cloud
            Points of one scan position. Adjacent points with equal
            ``gps_time`` form one multi-echo shot; otherwise each point is a
            single-return pulse. Other attributes become echo attributes.
        origin
            Scanner position in the cloud's frame.

        Returns
        -------
        Shots
        """
        return cls._from_core(_core.shots_from_pointcloud(cloud.xyz, cloud.attrs, tuple(origin)))

    @classmethod
    def from_ray_cloud(cls, cloud: PointCloud) -> Shots:
        """From a raycloudtools ray cloud. ``nx, ny, nz`` (PLY) or ``sx, sy, sz``
        (LAS/LAZ) point from each end point back to the sensor; ``bound == 0``
        (else ``alpha == 0``) marks unbounded rays, which become echo-less
        shots. Returns sharing a ``beam_id`` / ``gps_time`` are joined into one
        multi-echo shot when ``number_of_returns`` is present.

        Parameters
        ----------
        cloud
            A ray cloud read with :func:`sylva.read`.

        Returns
        -------
        Shots
            Origins are the per-ray sensor positions.

        Raises
        ------
        ValueError
            If the cloud has no sensor-vector attributes.

        Warns
        -----
        UserWarning
            If every ray starts at (0, 0, 0): the file was written without
            sensor positions, and ray-traced products would be wrong.
        """
        shots = cls._from_core(_core.shots_from_ray_cloud(cloud.xyz, cloud.attrs))
        if shots.n_shots and np.all(np.abs(shots.origin[:: max(1, shots.n_shots // 10000)]) < 1e-6):
            import warnings

            warnings.warn("every ray of this ray cloud starts at (0, 0, 0): it was written without sensor "
                          "positions, so ray-traced products (voxels, occlusion, sampling) will describe "
                          "pulses from the origin, not the scan", stacklevel=2)
        return shots
