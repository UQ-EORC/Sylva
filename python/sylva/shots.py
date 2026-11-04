# Sylva: LiDAR processing for forest ecology and remote sensing research.
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
from .util import image as _image


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

    def _to_core(self, attrs: bool = True) -> dict:
        d = {
            "origin": self.origin, "direction": self.direction, "echo_start": self.echo_start,
            "echo_count": self.echo_count, "echo_range": self.echo_range,
        }
        if attrs:
            d["echo_attrs"] = self.echo_attrs
        return d

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
        return _core.shots_shot_of_echo(self._to_core(attrs=False))

    def echo_rank(self) -> np.ndarray:
        """Position of each echo within its shot.

        Returns
        -------
        numpy.ndarray
            Length ``n_echoes``; 0 is the first (nearest) return.
        """
        return _core.shots_echo_rank(self._to_core(attrs=False))

    def echo_xyz(self) -> np.ndarray:
        """Echo positions in the shots' frame.

        Returns
        -------
        numpy.ndarray
            ``(n_echoes, 3)``, ``origin + direction * range``.
        """
        return _core.shots_echo_xyz(self._to_core(attrs=False))

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
        mask = np.ascontiguousarray(mask, dtype=bool)
        return Shots._from_core(_core.shots_subset(self._to_core(), mask))

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
        return _core.shots_zenith_azimuth(self.direction)

    def scan_lines(self) -> tuple[np.ndarray, float, str]:
        """Where the scan lines begin, for pulses in acquisition order.

        A scanner sweeps one angle along each line, the zenith (RIEGL: the
        mirror sweeps a vertical line, then the head steps in azimuth) or the
        azimuth (a head turning a full circle per elevation step), and turns
        back between lines; so a line ends where the swept angle jumps back
        against the sweep by more than ten degrees. The swept angle is the
        one that moves more between consecutive pulses. When every shot has
        an echo with a ``gps_time``, the jump must also span the dead time of
        a turn, which keeps the stray out-of-sequence pulses of a fast
        scanner from starting lines of their own.

        Returns
        -------
        starts : numpy.ndarray
            Shot index at which each line begins, then ``n_shots``: line
            ``i`` is ``starts[i]:starts[i + 1]``.
        sweep : float
            ``1.0`` when the swept angle grows along a line (zenith: sky to
            ground), ``-1.0`` when it falls.
        along : {"zenith", "azimuth"}
            The angle swept along a line.
        """
        zen, az = self.zenith_azimuth()
        if self.n_shots < 2:
            return np.array([0, self.n_shots], dtype=np.int64), 1.0, "zenith"
        daz = (np.diff(az) + 180.0) % 360.0 - 180.0
        zenith_sweeps = np.nanmedian(np.abs(np.diff(zen))) >= np.nanmedian(np.abs(daz))
        along = "zenith" if zenith_sweeps else "azimuth"
        time = None
        if "gps_time" in self.echo_attrs and (self.echo_count > 0).all():
            time = np.ascontiguousarray(self.echo_attrs["gps_time"], dtype=float)[self.echo_start]
        angle = zen if along == "zenith" else az
        starts, sweep = _core.shots_scan_lines(np.ascontiguousarray(angle, dtype=float), time)
        return starts.astype(np.int64), float(sweep), along

    def angular_steps(self) -> tuple[float, float]:
        """The scan pattern's angular increments, from the pulses.

        The step along a line is the median change of the swept angle between
        consecutive pulses of a line; the step between lines is the median
        change of the other angle from one line's middle to the next's.

        Returns
        -------
        zenith_step, azimuth_step : float
            Degrees; 0.03 for either that cannot be told.
        """
        zen, az = self.zenith_azimuth()
        starts, _, along = self.scan_lines()
        n_lines = len(starts) - 1
        line = np.repeat(np.arange(n_lines), np.diff(starts))
        within = line[1:] == line[:-1]
        daz = (np.diff(az) + 180.0) % 360.0 - 180.0
        swept, other = (zen, az) if along == "zenith" else (az, zen)
        dswept = (daz if along == "azimuth" else np.diff(zen))[within]
        dswept = np.abs(dswept[np.isfinite(dswept)])
        along_step = float(np.median(dswept[dswept > 1e-6])) if (dswept > 1e-6).any() else 0.03
        mids = np.array([np.nanmedian(other[starts[i]:starts[i + 1]]) for i in range(n_lines)])
        dmid = np.diff(mids)
        if along == "zenith":
            dmid = (dmid + 180.0) % 360.0 - 180.0
        dmid = np.abs(dmid[np.isfinite(dmid)])
        between = float(np.median(dmid[dmid > 1e-6])) if (dmid > 1e-6).any() else 0.03
        return (along_step, between) if along == "zenith" else (between, along_step)

    def _line_columns(self, zen: np.ndarray, az: np.ndarray) -> np.ndarray | None:
        """The scan line of every pulse, for a scanner sweeping the zenith along
        its lines: the line index for the recorded pulses, and for a recovered
        miss the line whose azimuth is nearest once the swing of azimuth along
        a line (a smooth curve of zenith set by the mirror geometry, about a
        degree on a VZ) is taken out. None when the lines cannot be told."""
        rec = self.echo_count > 0
        if rec.sum() < 1000:
            return None
        sub = self if rec.all() else self.subset(rec)
        zr, ar = (zen, az) if rec.all() else sub.zenith_azimuth()
        starts, _, along = sub.scan_lines()
        n_lines = len(starts) - 1
        if along != "zenith" or n_lines < 3:
            return None
        line = np.repeat(np.arange(n_lines), np.diff(starts))
        # Each line's azimuth: the median over its pulses (sort within lines, take the middle).
        order = np.lexsort((ar, line))
        az0 = ar[order][starts[:-1] + np.diff(starts) // 2]
        col = np.empty(len(zen), dtype=np.int64)
        col[rec] = line
        if rec.all():
            return col
        # The swing: the median offset from the line's azimuth per degree of zenith.
        res = (ar - az0[line] + 180.0) % 360.0 - 180.0
        edges = np.arange(np.floor(np.nanmin(zr)), np.ceil(np.nanmax(zr)) + 1.0)
        b = np.clip(np.digitize(zr, edges) - 1, 0, len(edges) - 2)
        order = np.lexsort((res, b))
        counts = np.bincount(b, minlength=len(edges) - 1)
        first = np.concatenate([[0], np.cumsum(counts)[:-1]])
        have = counts > 0
        swing = np.interp(zen[~rec], (0.5 * (edges[:-1] + edges[1:]))[have],
                          res[order][first[have] + counts[have] // 2]) if have.sum() >= 2 else 0.0
        target = (az[~rec] - swing) % 360.0
        # Nearest line by azimuth, the circle closed by a copy either side.
        srt = np.argsort(az0)
        ring = np.concatenate([az0[srt] - 360.0, az0[srt], az0[srt] + 360.0])
        who = np.concatenate([srt, srt, srt])
        k = np.searchsorted(ring, target)
        k = np.clip(k, 1, len(ring) - 1)
        left_closer = (target - ring[k - 1]) < (ring[k] - target)
        col[~rec] = who[np.where(left_closer, k - 1, k)]
        return col

    def scan_image(self, field="range", echo: str = "first", cmap: str = "viridis",
                   grid: str = "angles", steps: tuple[float, float] | None = None,
                   sky_up: bool = True) -> ScanImage:
        """The scanner's own view of the scan: the pulses on their grid,
        zenith down the image and azimuth across it, coloured by a field.

        A pulse that returned nothing is NaN, so the gaps show.

        Parameters
        ----------
        field
            What to show: ``"range"`` (m, default), ``"n_echoes"``,
            ``"zenith"`` or ``"azimuth"`` (degrees), the name of an echo
            attribute (``"amplitude"``, ``"reflectance"``, ``"deviation"``,
            ...), or an array with one value per echo or per shot.
        echo : {"first", "last"}
            Which echo of a multi-echo pulse supplies a per-echo field.
        cmap
            Default colour map for :meth:`ScanImage.save`; see
            :func:`sylva.util.image.colormap`.
        grid : {"angles", "lines"}
            ``"angles"``: each pulse at its zenith and azimuth step of the
            scan pattern, one row per zenith step and one column per azimuth
            step, whatever order the pulses come in; recovered misses land
            where they were fired, and steps the scanner skipped stay NaN.
            For a scanner sweeping the zenith along its lines the columns
            are the lines themselves, since a line's azimuth is not
            constant: on a VZ it swings by about a degree over the sweep, a
            smooth curve of the zenith set by the mirror geometry, which
            would slide pulses into the next column part way along every
            line. A recovered miss goes to the line nearest its azimuth
            with that swing taken out.
            ``"lines"``: scan line by index along the line, as recorded
            (:meth:`scan_lines`); lines shorter than the longest leave NaN
            at their end, and misses appended after the recorded pulses
            form lines of their own.
        steps
            ``(zenith_step, azimuth_step)`` in degrees for the angle grid,
            e.g. ``theta_delta`` and ``phi_delta`` of a RiSCAN scan pattern;
            estimated by :meth:`angular_steps` if None.
        sky_up
            Put the sky (lowest zenith) at the top of the image, whichever
            way the scanner sweeps.

        Returns
        -------
        ScanImage
            On the angle grid ``(zenith steps, azimuth steps)``. On the line
            grid, for a scanner that sweeps the zenith along its lines
            (RIEGL) one column per line and one row per pulse along it; for
            one that sweeps the azimuth, one row per line and one column per
            pulse.

        Raises
        ------
        ValueError
            For an unknown field, echo or grid, or an array of the wrong
            length.

        Examples
        --------
        >>> s = io.read_rxp_shots(pos.rxp, fill_missing=True)
        >>> s.scan_image().save("range.png")
        >>> s.scan_image("reflectance", cmap="gray").save("refl.png", vmin=-20, vmax=0)
        """
        if echo not in ("first", "last"):
            raise ValueError(f"echo must be 'first' or 'last', got {echo!r}")
        if grid not in ("angles", "lines"):
            raise ValueError(f"grid must be 'angles' or 'lines', got {grid!r}")
        has = self.echo_count > 0
        last = self.echo_start + self.echo_count - 1
        pick = np.where(has, self.echo_start if echo == "first" else last, 0)

        def per_shot(v: np.ndarray) -> np.ndarray:
            out = np.full(self.n_shots, np.nan)
            out[has] = v[pick[has]]
            return out

        zen, az = self.zenith_azimuth()
        if isinstance(field, str):
            name = field
            if field == "range":
                values = per_shot(self.echo_range)
            elif field == "n_echoes":
                values = self.echo_count.astype(float)
            elif field in ("zenith", "azimuth"):
                values = zen if field == "zenith" else az
            elif field in self.echo_attrs:
                values = per_shot(np.asarray(self.echo_attrs[field], dtype=float))
            else:
                have = ", ".join(sorted(self.echo_attrs)) or "none"
                raise ValueError(f"unknown field {field!r}; use range, n_echoes, zenith, azimuth "
                                 f"or an echo attribute ({have})")
        else:
            arr = np.asarray(field, dtype=float).ravel()
            name = "value"
            if len(arr) == self.n_shots:
                values = arr
            elif len(arr) == self.n_echoes:
                values = per_shot(arr)
            else:
                raise ValueError(f"field has {len(arr)} values for {self.n_shots} shots / "
                                 f"{self.n_echoes} echoes")
        starts, sweep, along = self.scan_lines()
        n_lines = len(starts) - 1
        if grid == "angles":
            if steps is None:
                steps = self.angular_steps() if self.n_shots > 1 else (0.03, 0.03)
            dz, da = float(steps[0]), float(steps[1])
            if not (dz > 0 and da > 0):
                raise ValueError(f"steps must be positive, got {steps}")
            ok = np.isfinite(zen) & np.isfinite(az)
            line_cols = self._line_columns(zen, az)
            if ok.any():
                rows = np.rint((zen - np.nanmin(zen)) / dz).astype(np.int64)
                if line_cols is not None:
                    cols = line_cols
                else:
                    cols = np.rint((az - np.nanmin(az)) / da).astype(np.int64)
                shape = (int(rows[ok].max()) + 1, int(cols[ok].max()) + 1)
            else:
                rows = cols = np.zeros(0, np.int64)
                shape = (0, 0)
            data = np.full(shape, np.nan)
            # A recorded pulse and a miss on the same step: the value wins.
            order = np.argsort(np.isfinite(values[ok]), kind="stable")
            data[rows[ok][order], cols[ok][order]] = values[ok][order]
            if not sky_up:
                data = data[::-1]
            return ScanImage(data, name, starts, 1.0, "zenith", cmap, grid, (dz, da))
        lengths = np.diff(starts)
        longest = int(lengths.max()) if n_lines else 0
        data = np.full((longest, n_lines), np.nan)   # (index along the line, line)
        if self.n_shots:
            line = np.repeat(np.arange(n_lines), lengths)
            data[np.arange(self.n_shots) - starts[line], line] = values
        if along == "zenith":
            flip = sweep < 0
        else:
            data = data.T   # (line, index): lines step through the zenith
            first, last = zen[starts[0]:starts[1]], zen[starts[-2]:starts[-1]]
            flip = n_lines > 1 and np.nanmean(first) > np.nanmean(last)
        if sky_up and flip:
            data = data[::-1]
        return ScanImage(data, name, starts, sweep, along, cmap, grid)

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
        return cls._from_core(_core.shots_concatenate([p._to_core() for p in parts]))

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
        return int(_core.canopy_pulses_per_line(self.direction, pattern, float(quantile),
                                                int(shot_stride)))

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
        edges = [float(e) for e in np.asarray(zenith_edges, dtype=float)]
        n = None if pulses_per_line is None else int(pulses_per_line)
        return np.rint(_core.canopy_expected_per_zenith(pattern, edges, n)).astype(np.int64)

    def fill_missing_from_timing(self, zenith_range: tuple[float, float] | None = None,
                                 shot_stride: int = 1, return_report: bool = False):
        """Add the pulses that returned nothing, each where it was fired, from
        the pulse timing: the reconstruction of
        :func:`sylva.io.read_rxp_shots` for shots already in memory, e.g.
        built by :meth:`from_pointcloud` from points that kept their
        ``gps_time``. Every shot needs an echo with a ``gps_time``, and the
        shots must be in acquisition order in the scanner's frame.

        Parameters
        ----------
        zenith_range
            The sweep's first and last zenith (degrees) from the scan
            pattern; None takes the extremes recorded, which under-fills
            the sky end of every line when nothing returned from there.
        shot_stride
            Keep every ``shot_stride``-th fired pulse, recorded or not.
        return_report
            Also return the report of :func:`sylva.io.read_rxp_shots`.

        Returns
        -------
        Shots
            The recorded pulses (those on kept stride slots) followed by the
            recovered misses, whose origin is this scan's.
        report : dict
            With ``return_report``.

        Raises
        ------
        ValueError
            If a shot has no echo, or there is no ``gps_time``.
        """
        if "gps_time" not in self.echo_attrs:
            raise ValueError("the shots carry no gps_time echo attribute")
        if not (self.echo_count > 0).all():
            raise ValueError("every shot needs an echo to time it; the shots already hold misses")
        time = np.ascontiguousarray(self.echo_attrs["gps_time"], dtype=float)[self.echo_start]
        misses, keep, report = _core.shots_reconstruct_misses(time, self.direction, zenith_range,
                                                              int(shot_stride))
        kept = self.subset(keep) if shot_stride > 1 else self
        origin = np.median(self.origin, axis=0) if self.n_shots else np.zeros(3)
        zeros = np.zeros(len(misses), dtype=np.int64)
        extra = Shots(np.tile(origin, (len(misses), 1)), misses, zeros, zeros, np.zeros(0),
                      {k: np.zeros(0, dtype=v.dtype) for k, v in kept.echo_attrs.items()})
        out = Shots.concatenate([kept, extra])
        return (out, report) if return_report else out

    def fill_missing(self, pattern: dict, pulses_per_line: int | None = None,
                     seed: int = 0, shot_stride: int = 1) -> Shots:
        """Add the pulses that returned nothing, by count per zenith line.

        Prefer ``read_rxp_shots(..., fill_missing=...)`` or
        ``ScanPosition.read_shots(fill_missing=True)``, which recover each
        miss at the place it was fired from the pulse timing. This method
        only knows the shots, not their timing: given the angular scan
        ``pattern`` it compares, per zenith line, the pulses fired against the
        shots observed and adds the difference as shots with
        ``echo_count == 0`` at *random* azimuths along the line. The count
        per ring is right, so ring-averaged gap fraction is served, but the
        misses land anywhere on the ring, including through foliage that
        returned every pulse. Ray-traced voxels then see free space where
        there was none, and plant area density comes out low in dense canopy
        (28 % low over a Banksia woodland plot against the same pulses with
        their misses in place). Do not use it for voxels.

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
        n = None if pulses_per_line is None else int(pulses_per_line)
        filled = _core.shots_fill_missing(self._to_core(), pattern, n, int(seed), int(shot_stride))
        return self if filled is None else Shots._from_core(filled)

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


@dataclass
class ScanImage:
    """A scan laid out as the scanner acquired it: one column per scan line,
    one row per pulse along the line. Build with :meth:`Shots.scan_image`.

    Attributes
    ----------
    data
        Values of the field, zenith down the image and azimuth across it.
        On the angle grid ``(zenith steps, azimuth steps)``; on the line
        grid ``(pulses per line, lines)`` for a scanner sweeping the zenith
        along its lines, ``(lines, pulses per line)`` for one sweeping the
        azimuth. NaN for pulses that returned nothing, steps the scanner
        skipped and the unfilled end of short lines.
    field
        The field shown.
    line_start
        Shot index at which each line begins, then ``n_shots``.
    sweep
        ``1.0`` when the swept angle grows along a line, else ``-1.0``.
    along
        ``"zenith"`` or ``"azimuth"``: the angle swept along a line.
    cmap
        Default colour map.
    grid
        ``"angles"`` or ``"lines"``: how the pulses were laid out.
    steps
        ``(zenith_step, azimuth_step)`` in degrees of the angle grid; None
        on the line grid.
    """

    data: np.ndarray
    field: str
    line_start: np.ndarray
    sweep: float
    along: str = "zenith"
    cmap: str = "viridis"
    grid: str = "lines"
    steps: tuple[float, float] | None = None

    @property
    def n_lines(self) -> int:
        """Scan lines in the image."""
        return len(self.line_start) - 1

    def to_rgb(self, cmap: str | None = None, vmin: float | None = None, vmax: float | None = None,
               log: bool = False) -> np.ndarray:
        """Colour the image.

        Parameters
        ----------
        cmap
            Colour map; the image's default if None (see
            :func:`sylva.util.image.colormap`).
        vmin, vmax
            Colour limits; the 2nd and 98th percentiles of the field by
            default.
        log
            Colour ``log10`` of the field, which suits range and amplitude.

        Returns
        -------
        numpy.ndarray
            ``uint8`` of shape ``(rows, lines, 3)``; NaN is black.
        """
        return _image.to_rgb(self.data, cmap or self.cmap, vmin, vmax, log)

    def save(self, path: str | Path, cmap: str | None = None, vmin: float | None = None,
             vmax: float | None = None, log: bool = False) -> Path:
        """Write the image as a PNG (see :meth:`to_rgb`).

        Parameters
        ----------
        path
            Output file.
        cmap, vmin, vmax, log
            As for :meth:`to_rgb`.

        Returns
        -------
        pathlib.Path
            The file written.
        """
        return _image.write_png(path, self.to_rgb(cmap, vmin, vmax, log))
