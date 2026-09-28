# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Full-waveform lidar: reading, Gaussian decomposition and conversion to pulses.

A full-waveform scanner records the received power of each pulse as a run
of samples instead of (or besides) a few discrete returns. This module
reads such records from LAS 1.3 / 1.4 files with waveform data packets
(point formats 4, 5, 9 and 10, packets in the file or in a ``.wdp`` file
next to it) and from PulseWaves (``.pls`` with its ``.wvs``), decomposes
them into Gaussian echoes (Hofton et al. 2000; Wagner et al. 2006), and
turns the echoes into :class:`sylva.shots.Shots` for the ray-traced voxel
and canopy tools. RIEGL SDF files need RIEGL's proprietary library and are
not read; export them to LAS with waveform data packets first.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import _core
from .pointcloud import PointCloud
from .shots import Shots

__all__ = ["Waveform", "Waveforms", "Echoes", "info", "read", "chunks", "write_las",
           "write_pulsewaves", "estimate_noise", "smooth", "decompose", "to_shots",
           "backscatter_cross_section", "calibration_constant", "C_HALF"]

#: Range per nanosecond of round-trip time in vacuum, ``c / 2`` (m/ns).
C_HALF = 0.299792458 / 2.0


def _check_path(path) -> str:
    p = Path(path)
    if not p.exists():
        raise OSError(f"{p}: no such file")
    return str(p)


@dataclass
class Waveform:
    """One digitised waveform: a view of a row of :class:`Waveforms`.

    Attributes
    ----------
    samples
        Received power (digitiser units, or physical units after a gain,
        offset or lookup table from the file).
    interval
        Sampling interval (ns).
    anchor, direction
        A point on the beam and the unit beam direction (away from the
        scanner).
    offset
        Time from the anchor to the first sample (ns).
    metres_per_ns
        Range per nanosecond of round-trip time (m/ns).
    origin
        Scanner position, NaN when the file does not give it.
    gps_time, pulse
        Time of the pulse and its identifier.
    """

    samples: np.ndarray
    interval: float
    anchor: np.ndarray
    direction: np.ndarray
    offset: float
    metres_per_ns: float
    origin: np.ndarray
    gps_time: float
    pulse: int

    def times(self) -> np.ndarray:
        """Time of each sample after the first (ns).

        Returns
        -------
        numpy.ndarray
            ``arange(n) * interval``.
        """
        return np.arange(len(self.samples)) * self.interval

    def positions(self) -> np.ndarray:
        """Position of each sample.

        Returns
        -------
        numpy.ndarray
            ``(n, 3)``: ``anchor + direction * metres_per_ns * (offset + t)``.
        """
        t = self.offset + self.times()
        return self.anchor + np.outer(t * self.metres_per_ns, self.direction)


@dataclass
class Waveforms:
    """A set of waveforms, their geometry and their samples.

    The samples of waveform ``i`` are
    ``samples[sample_start[i] : sample_start[i] + sample_count[i]]``;
    sample ``k`` of it lies at
    ``anchor[i] + direction[i] * metres_per_ns[i] * (offset[i] + k * interval[i])``.
    Rows that belong to one pulse (segments of a PulseWaves sampling, or
    its outgoing and returning waveforms) are consecutive and share
    ``pulse``.

    Build one with :func:`read`, :func:`chunks` or
    :func:`sylva.synthetic.waveforms`.

    Attributes
    ----------
    pulse
        Pulse identifier: the index of the pulse's first point record (LAS)
        or of the pulse (PulseWaves).
    gps_time
        Time of the pulse (s, as stored in the file).
    origin
        ``(n, 3)`` scanner positions; NaN where unknown (LAS files never
        give them).
    anchor, direction
        ``(n, 3)`` point on the beam and unit direction away from the scanner.
    offset
        Time from the anchor to the first sample (ns).
    interval
        Sampling interval (ns).
    metres_per_ns
        Range per nanosecond of round-trip time (m/ns), about ``C_HALF``.
    sample_start, sample_count
        CSR offsets into ``samples``.
    samples
        All samples, float32.
    attrs
        Per-waveform attributes: from LAS ``intensity``, ``return_number``,
        ``number_of_returns``, ``classification``, ``point_source_id``,
        ``n_records`` (point records sharing the packet) and ``descriptor``;
        from PulseWaves ``kind`` (1 outgoing, 2 returning), ``channel``,
        ``segment``, ``sampling``, ``intensity``, ``classification`` and
        ``descriptor``.
    """

    pulse: np.ndarray
    gps_time: np.ndarray
    origin: np.ndarray
    anchor: np.ndarray
    direction: np.ndarray
    offset: np.ndarray
    interval: np.ndarray
    metres_per_ns: np.ndarray
    sample_start: np.ndarray
    sample_count: np.ndarray
    samples: np.ndarray
    attrs: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        f64 = lambda a: np.ascontiguousarray(a, dtype=np.float64)  # noqa: E731
        i64 = lambda a: np.ascontiguousarray(a, dtype=np.int64)  # noqa: E731
        self.pulse = i64(self.pulse)
        self.gps_time = f64(self.gps_time)
        self.origin = f64(self.origin).reshape(-1, 3)
        self.anchor = f64(self.anchor).reshape(-1, 3)
        self.direction = f64(self.direction).reshape(-1, 3)
        self.offset = f64(self.offset)
        self.interval = f64(self.interval)
        self.metres_per_ns = f64(self.metres_per_ns)
        self.sample_start = i64(self.sample_start)
        self.sample_count = i64(self.sample_count)
        self.samples = np.ascontiguousarray(self.samples, dtype=np.float32)
        self.attrs = {k: np.ascontiguousarray(v) for k, v in self.attrs.items()}

    def __len__(self) -> int:
        return len(self.anchor)

    @property
    def n_waveforms(self) -> int:
        """Number of waveforms (rows)."""
        return len(self.anchor)

    @property
    def n_samples(self) -> int:
        """Number of samples over all waveforms."""
        return len(self.samples)

    def __repr__(self) -> str:
        return f"Waveforms(n_waveforms={self.n_waveforms:,}, n_samples={self.n_samples:,})"

    def __getitem__(self, i: int) -> Waveform:
        i = int(i)
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(f"waveform {i} out of range for {len(self)} waveforms")
        a = self.sample_start[i]
        return Waveform(self.samples[a:a + self.sample_count[i]].copy(), float(self.interval[i]),
                        self.anchor[i].copy(), self.direction[i].copy(), float(self.offset[i]),
                        float(self.metres_per_ns[i]), self.origin[i].copy(),
                        float(self.gps_time[i]), int(self.pulse[i]))

    def _to_core(self) -> dict:
        return {"pulse": self.pulse, "gps_time": self.gps_time, "origin": self.origin,
                "anchor": self.anchor, "direction": self.direction, "offset": self.offset,
                "interval": self.interval, "metres_per_ns": self.metres_per_ns,
                "sample_start": self.sample_start, "sample_count": self.sample_count,
                "samples": self.samples, "attrs": self.attrs}

    @classmethod
    def _from_core(cls, d: dict) -> Waveforms:
        return cls(d["pulse"], d["gps_time"], d["origin"], d["anchor"], d["direction"],
                   d["offset"], d["interval"], d["metres_per_ns"], d["sample_start"],
                   d["sample_count"], d["samples"], d["attrs"])

    def sample_positions(self) -> np.ndarray:
        """Position of every sample, in the order of ``samples``.

        Returns
        -------
        numpy.ndarray
            ``(n_samples, 3)``.
        """
        return _core.waveform_sample_positions(self._to_core())

    def sample_times(self) -> np.ndarray:
        """Time of every sample after its waveform's first sample (ns).

        Returns
        -------
        numpy.ndarray
            Length ``n_samples``.
        """
        row = np.repeat(np.arange(len(self)), self.sample_count)
        k = np.arange(self.n_samples) - np.repeat(self.sample_start, self.sample_count)
        return k * self.interval[row]

    def subset(self, mask) -> Waveforms:
        """Select waveforms.

        Parameters
        ----------
        mask
            Boolean array of length ``n_waveforms``, or row indices.

        Returns
        -------
        Waveforms
            The selected rows with their samples, repacked.

        Raises
        ------
        ValueError
            If a boolean mask has the wrong length.
        """
        m = np.asarray(mask)
        if m.dtype == bool:
            if len(m) != len(self):
                raise ValueError(f"mask has {len(m)} values for {len(self)} waveforms")
            idx = np.flatnonzero(m)
        else:
            idx = m.astype(np.int64).ravel()
        count = self.sample_count[idx]
        start = np.cumsum(count) - count
        take = np.arange(int(count.sum()), dtype=np.int64) + np.repeat(self.sample_start[idx] - start, count)
        return Waveforms(self.pulse[idx], self.gps_time[idx], self.origin[idx], self.anchor[idx],
                         self.direction[idx], self.offset[idx], self.interval[idx],
                         self.metres_per_ns[idx], start, count, self.samples[take],
                         {k: v[idx] for k, v in self.attrs.items()})

    @classmethod
    def concatenate(cls, parts: list[Waveforms]) -> Waveforms:
        """Stack waveform sets, e.g. the chunks of :func:`chunks`.

        Parameters
        ----------
        parts
            Waveform sets.

        Returns
        -------
        Waveforms
            All rows in order; only attributes present in every part are kept.

        Raises
        ------
        ValueError
            If ``parts`` is empty.
        """
        if not parts:
            raise ValueError("no waveforms to concatenate")
        offs = np.cumsum([0] + [p.n_samples for p in parts[:-1]])
        keys = [k for k in parts[0].attrs if all(k in p.attrs for p in parts)]
        cat = np.concatenate
        return cls(cat([p.pulse for p in parts]), cat([p.gps_time for p in parts]),
                   cat([p.origin for p in parts]), cat([p.anchor for p in parts]),
                   cat([p.direction for p in parts]), cat([p.offset for p in parts]),
                   cat([p.interval for p in parts]), cat([p.metres_per_ns for p in parts]),
                   cat([p.sample_start + o for p, o in zip(parts, offs)]),
                   cat([p.sample_count for p in parts]), cat([p.samples for p in parts]),
                   {k: cat([p.attrs[k] for p in parts]) for k in keys})

    def write_las(self, path: str | Path, scale: float = 0.001, bits: int = 16,
                  external: bool = False, crs_wkt: str | None = None) -> None:
        """Write as LAS 1.4 with waveform data packets; see :func:`write_las`.

        Parameters
        ----------
        path, scale, bits, external, crs_wkt
            As for :func:`write_las`.
        """
        write_las(self, path, scale=scale, bits=bits, external=external, crs_wkt=crs_wkt)

    def write_pulsewaves(self, path: str | Path, scale: float = 1e-4) -> None:
        """Write as PulseWaves; see :func:`write_pulsewaves`.

        Parameters
        ----------
        path, scale
            As for :func:`write_pulsewaves`.
        """
        write_pulsewaves(self, path, scale=scale)

    def decompose(self, **kwargs) -> Echoes:
        """Gaussian decomposition; see :func:`decompose`.

        Parameters
        ----------
        **kwargs
            Parameters of :func:`decompose`.

        Returns
        -------
        Echoes
        """
        return decompose(self, **kwargs)


@dataclass
class Echoes:
    """Echoes found by :func:`decompose` (or the truth of
    :func:`sylva.synthetic.waveforms`), one row per echo, grouped by
    waveform in ascending order.

    Attributes
    ----------
    waveform
        Row of the waveform each echo belongs to.
    time
        Time of the echo's peak after the waveform's first sample (ns).
    amplitude
        Peak of the fitted Gaussian above the background (sample units).
    width
        Standard deviation of the fitted Gaussian (ns). The full width at
        half maximum is ``2.3548 * width``.
    xyz
        ``(n, 3)`` echo positions.
    range
        Range from the scanner along the beam (m); NaN where the origin is
        unknown.
    stats
        Per-waveform results: ``background``, ``noise`` (standard
        deviation), ``rmse`` of the fit, ``n_echoes`` and ``iterations``.
    """

    waveform: np.ndarray
    time: np.ndarray
    amplitude: np.ndarray
    width: np.ndarray
    xyz: np.ndarray
    range: np.ndarray
    stats: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.waveform = np.ascontiguousarray(self.waveform, dtype=np.int64)
        for k in ("time", "amplitude", "width", "range"):
            setattr(self, k, np.ascontiguousarray(getattr(self, k), dtype=np.float64))
        self.xyz = np.ascontiguousarray(self.xyz, dtype=np.float64).reshape(-1, 3)

    def __len__(self) -> int:
        return len(self.time)

    def __repr__(self) -> str:
        return f"Echoes(n={len(self):,})"

    def _to_core(self) -> dict:
        return {"waveform": self.waveform, "time": self.time, "amplitude": self.amplitude,
                "width": self.width, "xyz": self.xyz, "range": self.range}

    @property
    def energy(self) -> np.ndarray:
        """Area under each echo, ``amplitude * width * sqrt(2 pi)`` (sample units times ns)."""
        return self.amplitude * self.width * np.sqrt(2 * np.pi)

    def cross_section(self, calibration: float = 1.0, range=None) -> np.ndarray:
        """Backscatter cross-section of each echo; see :func:`backscatter_cross_section`.

        Parameters
        ----------
        calibration
            Calibration constant ``C_cal`` (see :func:`calibration_constant`);
            1 gives relative values.
        range
            Ranges (m) to use instead of :attr:`range`, e.g. from a
            trajectory when the file has no scanner positions.

        Returns
        -------
        numpy.ndarray
            ``C_cal R^4 amplitude width``; NaN where the range is unknown.
        """
        r = self.range if range is None else range
        return backscatter_cross_section(r, self.amplitude, self.width, calibration)

    def to_pointcloud(self) -> PointCloud:
        """The echoes as points.

        Returns
        -------
        PointCloud
            One point per echo with ``amplitude``, ``width``, ``time``,
            ``range`` and ``waveform`` attributes.
        """
        return PointCloud(self.xyz, {"amplitude": self.amplitude, "width": self.width,
                                     "time": self.time, "range": self.range,
                                     "waveform": self.waveform})

    def to_shots(self, waveforms: Waveforms, origin=None) -> Shots:
        """Pulses with these echoes; see :func:`to_shots`.

        Parameters
        ----------
        waveforms
            The waveforms the echoes were found in.
        origin
            Scanner positions per waveform, as for :func:`to_shots`.

        Returns
        -------
        Shots
        """
        return to_shots(waveforms, self, origin=origin)


def info(path: str | Path) -> dict:
    """Describe a waveform file without reading its waveforms.

    Parameters
    ----------
    path
        LAS/LAZ with waveform data packets, or a PulseWaves ``.pls``.

    Returns
    -------
    dict
        ``format`` (``"las"`` or ``"pulsewaves"``), ``version``,
        ``bounds``, ``compressed``, the wave packet or pulse ``descriptors``;
        for LAS also ``point_format``, ``n_records`` and ``packets``
        (``"internal"``, ``"external"`` with the ``wdp`` path, or
        ``"missing"``); for PulseWaves ``n_pulses``, ``system``, ``software``,
        ``waves`` (the ``.wvs`` path) and ``lookup_tables``.

    Raises
    ------
    OSError
        If the file is missing, is not a waveform file, or is a RIEGL SDF
        file (which only RIEGL's proprietary library reads).
    """
    return _core.waveform_info(_check_path(path))


def read(path: str | Path, start: int = 0, count: int | None = None, kind: str = "returning",
         dedupe: bool = True, lookup: bool = True) -> Waveforms:
    """Read waveforms from a LAS/LAZ or PulseWaves file.

    Parameters
    ----------
    path
        LAS 1.3 / 1.4 (or LAZ) with point format 4, 5, 9 or 10, with its
        waveform packets inside or in a ``.wdp`` file of the same name; or
        a PulseWaves ``.pls`` with its ``.wvs``.
    start, count
        Point records (LAS) or pulses (PulseWaves) to read, from ``start``;
        all by default. Use :func:`chunks` to stream a large file.
    kind
        PulseWaves only: ``"returning"`` (default), ``"outgoing"`` or
        ``"all"`` samplings.
    dedupe
        LAS only: consecutive point records that point to the same packet
        (the returns of one pulse) give one waveform; False gives one per
        record.
    lookup
        PulseWaves only: turn sample values into physical units with the
        file's lookup tables.

    Returns
    -------
    Waveforms

    Raises
    ------
    OSError
        If the file is missing or unreadable, its packets cannot be found,
        or it uses an unsupported variant (compressed packets, PulseWaves
        compression).
    ValueError
        For a bad ``kind``, ``start`` or ``count``.
    """
    wf, _ = _read(path, start, count, kind, dedupe, lookup)
    return wf


def _read(path, start, count, kind, dedupe, lookup):
    if start < 0 or (count is not None and count < 0):
        raise ValueError("start and count must not be negative")
    d, nxt = _core.waveform_read(_check_path(path), int(start), None if count is None else int(count),
                                 bool(dedupe), str(kind), bool(lookup))
    return Waveforms._from_core(d), nxt


def chunks(path: str | Path, size: int = 100_000, kind: str = "returning", dedupe: bool = True,
           lookup: bool = True) -> Iterator[Waveforms]:
    """Stream a waveform file in chunks.

    Each chunk holds about ``size`` point records (LAS) or pulses
    (PulseWaves) and the samples they refer to, so memory stays bounded
    whatever the file size. A pulse belongs to the chunk it starts in: in a
    LAS file the records that continue a chunk's last packet are read with
    it, so no waveform is split or repeated.

    Parameters
    ----------
    path
        As for :func:`read`.
    size
        Records or pulses per chunk.
    kind, dedupe, lookup
        As for :func:`read`.

    Yields
    ------
    Waveforms

    Raises
    ------
    ValueError
        If ``size`` is not positive.
    """
    if size < 1:
        raise ValueError(f"size must be positive, got {size}")
    meta = info(path)
    total = meta["n_records"] if meta["format"] == "las" else meta["n_pulses"]
    start = 0
    while start < total:
        wf, start = _read(path, start, size, kind, dedupe, lookup)
        yield wf


def write_las(waveforms: Waveforms, path: str | Path, scale: float = 0.001, bits: int = 16,
              external: bool = False, crs_wkt: str | None = None) -> None:
    """Write waveforms as LAS 1.4 point format 9 with waveform data packets.

    Each waveform becomes one point at its anchor, with the LAS waveform
    fields that put its samples where they were: the packet offset and size,
    the return point location ``-offset`` (ps) and the vector
    ``-direction * metres_per_ns`` (m/ps). ``gps_time`` and the attributes
    ``intensity``, ``return_number``, ``number_of_returns``,
    ``classification`` and ``point_source_id`` are written where present.
    One wave packet descriptor is written per distinct (number of samples,
    interval) pair, at most 255.

    Parameters
    ----------
    waveforms
        The waveforms. The origin and ``pulse`` are not stored (LAS has no
        field for them).
    path
        Output ``.las`` file.
    scale
        Coordinate resolution (m).
    bits
        8 or 16 bits per sample. Integer samples that fit are stored as
        they are; others with a digitiser gain and offset spanning their
        range (the rounding error is at most half the gain).
    external
        Put the packets in a ``.wdp`` file next to ``path`` instead of an
        extended VLR.
    crs_wkt
        Coordinate reference system as OGC WKT, stored in a VLR.

    Raises
    ------
    ValueError
        For bad arguments, more than 255 descriptors, or an interval that
        is not a whole number of picoseconds.
    """
    if bits not in (8, 16):
        raise ValueError(f"bits must be 8 or 16, got {bits}")
    _core.waveform_write_las(waveforms._to_core(), str(path), float(scale), int(bits),
                             bool(external), crs_wkt)


def write_pulsewaves(waveforms: Waveforms, path: str | Path, scale: float = 1e-4) -> None:
    """Write waveforms as PulseWaves 0.3 (``path`` and a ``.wvs`` next to it).

    Consecutive rows of one pulse on the same beam become one pulse whose
    outgoing (``kind == 1``) and returning rows are the segments of an
    outgoing and a returning sampling. The scanner position becomes the
    anchor where it is known (the file then says the optical centre and the
    anchor coincide). Samples are stored as 16-bit integers, segment
    starts to a thousandth of a sample.

    Parameters
    ----------
    waveforms
        The waveforms; samples must round to integers from 0 to 65535.
    path
        Output ``.pls`` file.
    scale
        Resolution of the anchor and target coordinates (m).

    Raises
    ------
    ValueError
        If a sample does not fit 16 bits or a segment has more than 65535
        samples.
    """
    _core.waveform_write_pulsewaves(waveforms._to_core(), str(path), float(scale))


def estimate_noise(waveforms: Waveforms) -> tuple[np.ndarray, np.ndarray]:
    """Background level and noise of each waveform.

    First robustly: the median of the samples and 1.4826 times their median
    absolute deviation (the standard deviation, for Gaussian noise),
    recomputed without the samples more than three standard deviations
    above the median until these settle, so that echoes do not inflate
    them. Then the mean and standard deviation of the samples within three
    standard deviations of that background (divided by 0.9866 for the
    clipped tails), iterated; this keeps the noise right for digitised
    samples, whose median absolute deviation moves in whole counts.

    Parameters
    ----------
    waveforms
        The waveforms.

    Returns
    -------
    background, noise : numpy.ndarray
        Per waveform, in sample units.
    """
    return _core.waveform_noise(waveforms._to_core())


def smooth(waveforms: Waveforms, sigma: float) -> Waveforms:
    """Smooth the samples with a Gaussian kernel.

    Parameters
    ----------
    waveforms
        The waveforms.
    sigma
        Standard deviation of the kernel (ns), truncated at four; near the
        ends of a waveform the kernel is renormalised over the samples it
        covers.

    Returns
    -------
    Waveforms
        A copy with smoothed samples.

    Raises
    ------
    ValueError
        If ``sigma`` is negative.
    """
    out = Waveforms._from_core(waveforms._to_core())
    out.samples = _core.waveform_smooth(waveforms._to_core(), float(sigma))
    return out


def decompose(waveforms: Waveforms, smooth: float = 1.0, threshold: float = 4.0,
              min_amplitude: float = 0.0, peaks: str = "inflection", max_echoes: int = 10,
              min_width: float = 0.3, max_width: float = 20.0, noise: float | None = None,
              background: float | None = None, max_iter: int = 100,
              refine: bool = True) -> Echoes:
    """Gaussian decomposition of waveforms into echoes (Wagner et al. 2006).

    Each waveform is modelled as a constant background plus one Gaussian per
    echo. The background and noise come from :func:`estimate_noise` (or the
    values given). The waveform is smoothed with a Gaussian of ``smooth``
    ns, and candidate echoes are taken from the smoothed signal where it
    exceeds the threshold: with ``peaks="derivative"`` at the zero
    crossings of its first derivative (the local maxima, as Wagner et al.
    2006), with ``peaks="inflection"`` at the local minima of its second
    derivative, the width from the inflection points around them (Hofton
    et al. 2000), which also separates echoes that overlap too much to make
    a maximum of their own. The amplitude, position and width of every
    candidate are then fitted together to the raw samples by
    Levenberg-Marquardt least squares. Fitted echoes weaker than the
    threshold are dropped, coincident ones merged, and with ``refine`` an
    echo is added where the residual still shows one, until the fit
    settles. Waveforms are processed in parallel; the result does not
    depend on the number of threads.

    Parameters
    ----------
    waveforms
        The waveforms (outgoing ones too, if they are in the set).
    smooth
        Standard deviation of the smoothing kernel (ns) used to find the
        candidates; 0 for none. The fit always uses the raw samples.
    threshold
        Detection threshold in noise standard deviations.
    min_amplitude
        Smallest echo amplitude (sample units), whatever the noise.
    peaks
        ``"inflection"`` or ``"derivative"``.
    max_echoes
        Most echoes per waveform.
    min_width, max_width
        Bounds on each echo's standard deviation (ns).
    noise, background
        Values to use instead of the estimates (sample units).
    max_iter
        Levenberg-Marquardt iterations per fit.
    refine
        Look for echoes in the residual of the fit.

    Returns
    -------
    Echoes
        With per-waveform ``stats``.

    Raises
    ------
    ValueError
        For bad parameters.
    """
    d, stats = _core.waveform_decompose(waveforms._to_core(), float(smooth), float(threshold),
                                        float(min_amplitude), str(peaks), int(max_echoes),
                                        float(min_width), float(max_width),
                                        None if noise is None else float(noise),
                                        None if background is None else float(background),
                                        int(max_iter), bool(refine))
    return Echoes(d["waveform"], d["time"], d["amplitude"], d["width"], d["xyz"], d["range"], stats)


def to_shots(waveforms: Waveforms, echoes: Echoes, origin=None) -> Shots:
    """Pulses from decomposed waveforms, for ray tracing and canopy metrics.

    One shot per pulse (the consecutive rows sharing ``pulse``; outgoing
    rows are ignored), with the echoes of all its returning rows sorted by
    range. A pulse in which no echo was found stays as a shot without an
    echo, which ray-traced products count as free space.

    Parameters
    ----------
    waveforms
        The waveforms.
    echoes
        Echoes found in them by :func:`decompose`.
    origin
        ``(n_waveforms, 3)`` scanner positions (e.g. from a trajectory),
        used where finite. Otherwise the waveform's own origin is used, and
        where that is unknown too the shot starts at the pulse's first
        sample: its free space is then only what the record covers.

    Returns
    -------
    Shots
        Echo attributes ``amplitude``, ``width`` (ns), ``time`` (ns after
        the first sample of its waveform) and ``waveform`` (row).

    Raises
    ------
    ValueError
        If ``origin`` does not have one row per waveform, or the echoes
        refer to rows that are not there.
    """
    o = None if origin is None else np.ascontiguousarray(origin, dtype=np.float64).reshape(-1, 3)
    return Shots._from_core(_core.waveform_to_shots(waveforms._to_core(), echoes._to_core(), o))


def backscatter_cross_section(range, amplitude, width, calibration: float = 1.0) -> np.ndarray:
    """Backscatter cross-section of echoes (Wagner et al. 2006).

    By the radar equation for a Gaussian system pulse, the cross-section of
    an echo is ``sigma = C_cal R^4 P s``: the fourth power of the range
    times the echo's amplitude ``P`` and width ``s``, scaled by a
    calibration constant that folds in the transmitted power, the receiver
    and the atmosphere.

    Parameters
    ----------
    range
        Ranges (m).
    amplitude
        Echo amplitudes.
    width
        Echo widths (ns, standard deviation).
    calibration
        ``C_cal``; 1 gives relative values, comparable within a survey.

    Returns
    -------
    numpy.ndarray
        Cross-sections (m² once calibrated).

    Raises
    ------
    ValueError
        If the arrays differ in length.
    """
    return _core.waveform_cross_section(np.ascontiguousarray(range, dtype=np.float64),
                                        np.ascontiguousarray(amplitude, dtype=np.float64),
                                        np.ascontiguousarray(width, dtype=np.float64),
                                        float(calibration))


def calibration_constant(range, amplitude, width, reflectance, beam_divergence: float,
                         incidence=0.0) -> float:
    """Calibration constant ``C_cal`` from echoes of reference targets.

    For an extended Lambertian target of reflectance ``rho`` hit at
    incidence angle ``alpha`` by a beam of divergence ``beta`` (full angle,
    rad), ``sigma = pi rho R^2 beta^2 cos(alpha)`` (Wagner 2010), so each
    echo gives ``C_cal = pi rho beta^2 cos(alpha) / (R^2 P s)``; the median
    over the echoes is returned.

    Parameters
    ----------
    range, amplitude, width
        Range (m), amplitude and width (ns) of the reference echoes.
    reflectance
        Reflectance of each target (0 to 1), or one value for all.
    beam_divergence
        Full beam divergence (rad).
    incidence
        Incidence angle (rad), per echo or one value.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If no echo gives a finite, positive value.
    """
    r = np.ascontiguousarray(range, dtype=np.float64).ravel()
    n = len(r)
    full = lambda v: np.ascontiguousarray(np.broadcast_to(np.asarray(v, dtype=np.float64), (n,)))  # noqa: E731
    return float(_core.waveform_calibration(r, full(amplitude), full(width), full(reflectance),
                                            float(beam_divergence), full(incidence)))
