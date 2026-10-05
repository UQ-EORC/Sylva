# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Two epochs in one frame, and a record of how each was processed."""

from __future__ import annotations

import dataclasses
import datetime
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import _core
from ..coreg.pipeline import CoregConfig, PairResult, ScanFeatures, prepare_scan, register_pair
from ..coreg.transforms import transform_points
from ..pointcloud import PointCloud

__all__ = ["EpochAlignment", "align_epochs", "Provenance", "ProvenanceWarning", "provenance"]

_STABLE = {"stems+ground": (True, True), "stems": (True, False), "ground": (False, True)}


@dataclass
class EpochAlignment:
    """Result of :func:`align_epochs`: the later epoch in the frame of the
    earlier one, and how well that is known.

    Attributes
    ----------
    transform
        ``(4, 4)`` transform taking the new epoch onto the reference.
    registration_sigma
        One-sigma uncertainty (m) of where a point of the new epoch lands,
        as the RMS over the reference stems (or ``evaluate``) of the 3-D
        displacement uncertainty that the covariance implies; NaN if a
        direction is not constrained (stems alone leave z free).
    sigma_horizontal, sigma_vertical
        Its horizontal and vertical parts. The vertical part is what moves
        stem slices along the stem when increments are measured.
    sigma_xyz
        One-sigma uncertainty of x, y, z at ``centre``.
    covariance
        ``(6, 6)`` covariance of a small rotation (rad, about x, y, z through
        ``centre``) and a translation (m).
    centre
        Point the parameters are expressed about.
    stem_pairs
        ``(n, 2)`` indices of the matched stems in the two stem maps
        (``reference.stem_map``, ``new.stem_map``).
    stem_residuals
        ``(n, 2)`` horizontal residual (m) of each matched stem after
        alignment.
    ground_residuals
        Height (m) of each terrain sample of the new epoch above the
        reference terrain after alignment.
    stem_sigma, ground_sigma
        Robust spread (m) of one stem residual component and of one terrain
        residual: the precision of a single stable feature.
    coarse
        The coregistration pair that gave the starting transform (``None``
        when ``initial`` was given).
    reference, new
        The prepared epochs (:class:`sylva.coreg.ScanFeatures`).
    stable
        The features used.
    settings
        The settings of the call, for :func:`provenance`.
    """

    transform: np.ndarray
    registration_sigma: float
    sigma_horizontal: float
    sigma_vertical: float
    sigma_xyz: np.ndarray
    covariance: np.ndarray
    centre: np.ndarray
    stem_pairs: np.ndarray
    stem_residuals: np.ndarray
    ground_residuals: np.ndarray
    stem_sigma: float
    ground_sigma: float
    coarse: PairResult | None = None
    reference: ScanFeatures | None = None
    new: ScanFeatures | None = None
    stable: str = "stems+ground"
    settings: dict = field(default_factory=dict)

    @property
    def n_stems(self) -> int:
        """Stems matched between the epochs."""
        return len(self.stem_pairs)

    def apply(self, cloud: PointCloud) -> PointCloud:
        """The new epoch's cloud moved into the reference frame.

        Parameters
        ----------
        cloud
            A cloud in the new epoch's frame.

        Returns
        -------
        PointCloud
        """
        return cloud.transform(self.transform)

    def report(self) -> str:
        """Plain-text summary for a log or a QC record."""
        lines = [
            f"Epoch alignment on {self.stable}",
            f"  stems matched     {self.n_stems}  (residual sigma {1000 * self.stem_sigma:.1f} mm per axis)",
            f"  terrain samples   {len(self.ground_residuals)}  (residual sigma {1000 * self.ground_sigma:.1f} mm)",
            f"  registration sigma {1000 * self.registration_sigma:.1f} mm "
            f"(horizontal {1000 * self.sigma_horizontal:.1f}, vertical {1000 * self.sigma_vertical:.1f})",
        ]
        if self.coarse is not None:
            lines.append(f"  coarse: {self.coarse.summary()}")
        return "\n".join(lines)


def _features(epoch, config: CoregConfig, name: str) -> ScanFeatures:
    if isinstance(epoch, ScanFeatures):
        return epoch
    return prepare_scan(epoch, config, name=name)


def _ground(scan: ScanFeatures):
    g = scan.ground
    if g is None:
        return None
    return (np.ascontiguousarray(g.elevation, dtype=float), float(g.origin[0]), float(g.origin[1]),
            float(g.cell_size), np.ascontiguousarray(g.observed, dtype=bool))


def _stems(scan: ScanFeatures) -> np.ndarray:
    return np.array([[s.x, s.y, s.z, s.dbh] for s in scan.stem_map], dtype=float).reshape(-1, 4)


def align_epochs(ref, new, stable: str = "stems+ground", config: CoregConfig | None = None,
                 initial: np.ndarray | None = None, stem_tolerance: float = 0.3,
                 dbh_tolerance: float = 0.35, ground_spacing: float = 1.0, ground_block: float = 5.0,
                 stem_floor: float = 0.002, ground_floor: float = 0.005, huber: float = 2.0,
                 iterations: int = 30, evaluate=None) -> EpochAlignment:
    """Align a later epoch onto an earlier one on features that do not change.

    Between surveys crowns grow, lose limbs and move in the wind, so a
    registration over all points is pulled by the change it should reveal.
    Here a coarse transform comes from :func:`sylva.coreg.register_pair`
    (a global stem-map match refined by ICP), and is then refined on the
    stable features only: the horizontal position of every stem matched in
    both epochs (a stem thickens, its axis stays put) and the height of the
    new terrain above the reference terrain. Robust Gauss-Newton with Huber
    (1964) weights, each feature type weighted by its own spread, gives the
    transform and the covariance of its six parameters, from which
    ``registration_sigma`` follows.

    Parameters
    ----------
    ref, new
        The two epochs: :class:`~sylva.PointCloud`, ``(n, 3)`` arrays, file
        paths, or scans already prepared with :func:`sylva.coreg.prepare_scan`.
        For a multi-scan project, pass its registered, merged cloud
        (:func:`sylva.coreg.merge_clouds`).
    stable
        ``"stems+ground"`` (default), ``"stems"`` (x, y and rotation about
        z only; z and the tilts stay at the coarse transform's) or
        ``"ground"`` (z and the tilts only).
    config
        Settings of the coarse coregistration and of the stem maps and
        terrain models.
    initial
        ``(4, 4)`` starting transform (new onto reference); skips the coarse
        coregistration.
    stem_tolerance
        Farthest (m) a stem may lie from its match under the current
        transform to count as stable.
    dbh_tolerance
        Largest relative DBH difference of a stable stem pair.
    ground_spacing
        Spacing (m) of the terrain samples.
    ground_block
        Terrain errors are taken as correlated within blocks of this side
        (m): all terrain samples together weigh as much as one independent
        sample per block, so the smooth, spatially coherent errors of a
        terrain model do not pass for precision.
    stem_floor, ground_floor
        Smallest error (m) assumed for one stem position (per axis) and one
        terrain sample. No bark surface or terrain model is known better,
        whatever the residuals of an unusually clean plot suggest; on
        synthetic plots without these floors the alignment error exceeded
        its sigma.
    huber
        Residuals beyond this many robust standard deviations are
        down-weighted.
    iterations
        Most Gauss-Newton iterations.
    evaluate
        ``(n, 3)`` points (reference frame) over which
        ``registration_sigma`` is summarised; the reference stems if None.

    Returns
    -------
    EpochAlignment

    Raises
    ------
    ValueError
        If ``stable`` is unknown, or the features do not constrain the
        alignment (too few stems matched and no shared terrain).

    Notes
    -----
    ``registration_sigma`` is the uncertainty of the transform, not the
    residual of one feature: a stem-map residual of 5 mm over 15 stems
    constrains the translation to about 5 / sqrt(15) mm. Stem errors are
    taken as independent, terrain errors as shared within ``ground_block``.
    A bias common to all stems or to the whole terrain of one epoch (a
    different terrain filter, say) is not visible in the residuals and not
    included.

    Examples
    --------
    >>> ep = synthetic.forest_epochs(seed=1)
    >>> al = change.align_epochs(*ep.clouds)
    >>> new_in_ref = al.apply(ep.clouds[1])
    >>> al.registration_sigma
    """
    if stable not in _STABLE:
        raise ValueError(f"stable must be one of {sorted(_STABLE)}, got {stable!r}")
    use_stems, use_ground = _STABLE[stable]
    cfg = config or CoregConfig(verbose=False)
    fa = _features(ref, cfg, "reference")
    fb = _features(new, cfg, "new")
    if not fa.usable or not fb.usable:
        raise ValueError(f"an epoch is unusable: {fa.error or fb.error}")
    coarse = None
    if initial is None:
        coarse = register_pair(fb, fa, cfg, i=1, j=0)
        if not coarse.success:
            warnings.warn(f"the coarse coregistration was not accepted ({coarse.reason}); "
                          "refining its transform anyway", stacklevel=2)
        start = coarse.transform
    else:
        start = np.asarray(initial, dtype=float)
        if start.shape != (4, 4):
            raise ValueError(f"initial must be a (4, 4) matrix, got shape {start.shape}")
    ev = np.zeros((0, 3)) if evaluate is None else np.asarray(evaluate, dtype=float).reshape(-1, 3)
    d = _core.change_align_on_stable(_stems(fa), _stems(fb), _ground(fa), _ground(fb),
                                     np.ascontiguousarray(start, dtype=float), np.ascontiguousarray(ev),
                                     use_stems, use_ground, float(stem_tolerance), float(dbh_tolerance),
                                     float(ground_spacing), float(ground_block), float(stem_floor),
                                     float(ground_floor), float(huber), int(iterations))
    settings = {"stable": stable, "stem_tolerance": stem_tolerance, "dbh_tolerance": dbh_tolerance,
                "ground_spacing": ground_spacing, "ground_block": ground_block, "stem_floor": stem_floor, "ground_floor": ground_floor,
                "huber": huber, "iterations": iterations,
                "coreg": dataclasses.asdict(cfg)}
    return EpochAlignment(
        transform=d["transform"], registration_sigma=d["registration_sigma"],
        sigma_horizontal=d["sigma_horizontal"], sigma_vertical=d["sigma_vertical"],
        sigma_xyz=np.asarray(d["sigma_xyz"]), covariance=d["covariance"], centre=np.asarray(d["centre"]),
        stem_pairs=d["stem_pairs"], stem_residuals=d["stem_residuals"],
        ground_residuals=d["ground_residuals"], stem_sigma=d["stem_sigma"], ground_sigma=d["ground_sigma"],
        coarse=coarse, reference=fa, new=fb, stable=stable, settings=settings)


def _transform_xy(transform, xy) -> np.ndarray:
    """``(n, 2)`` horizontal positions moved by a rigid transform (z = 0)."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(xy) == 0:
        return xy.copy()
    pts = np.column_stack([xy, np.zeros(len(xy))])
    return transform_points(np.asarray(transform, dtype=float), pts)[:, :2]


# ------------------------------------------------------------------ provenance


class ProvenanceWarning(UserWarning):
    """Two epochs were processed with different software or settings."""


def _plain(value):
    """A JSON-ready copy of ``value`` (dataclasses, NumPy, paths)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


@dataclass
class Provenance:
    """Result of :func:`provenance`.

    Attributes
    ----------
    records
        One per epoch: ``sylva_version``, ``settings`` and whatever else was
        recorded (``created``, ``info``).
    differences
        ``(setting, value in the first epoch, value in the other)`` for every
        setting that differs, with dotted keys (``settings.detect_stems.min_slices``)
        and values as text (None where an epoch lacks the setting). With more
        than two epochs, each is compared with the first and the key is
        prefixed ``epoch <k>:``.
    """

    records: list[dict]
    differences: list[tuple]

    @property
    def consistent(self) -> bool:
        """Were all epochs processed with the same version and settings?"""
        return not self.differences

    def to_json(self) -> str:
        """The records as JSON text, to store beside each epoch's products."""
        return json.dumps(self.records, indent=2)


def _record(epoch) -> dict:
    if isinstance(epoch, dict) and "sylva_version" in epoch and "settings" in epoch:
        return _plain(epoch)
    from .. import __version__

    return {"sylva_version": __version__,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "settings": _plain(epoch if epoch is not None else {})}


def provenance(*epochs, warn: bool = True, rtol: float = 1e-9, ignore=("created", "info")) -> Provenance:
    """Record how each epoch was processed, and compare the epochs.

    A difference between two epochs is change only if both were measured the
    same way: another stem-detection setting, voxel size or Sylva version can
    move a DBH or a profile by more than a year's growth. Record each epoch
    when it is processed (``provenance(settings).records[0]``, stored with
    its products), then compare the records of the epochs to be compared.

    Parameters
    ----------
    *epochs
        Per epoch, the settings it was processed with (a dict, possibly
        nested, or a dataclass such as :class:`sylva.coreg.CoregConfig`; the
        ``settings`` of this module's results can be included), or a record
        made earlier by this function.
    warn
        Emit a :class:`ProvenanceWarning` when the epochs differ.
    rtol
        Numbers within this relative tolerance count as equal.
    ignore
        Top-level record keys left out of the comparison.

    Returns
    -------
    Provenance

    Examples
    --------
    >>> rec_a = change.provenance({"detect_stems": {"min_slices": 3}}).records[0]
    >>> rec_b = change.provenance({"detect_stems": {"min_slices": 4}}).records[0]
    >>> change.provenance(rec_a, rec_b).differences   # warns
    [('settings.detect_stems.min_slices', '3', '4')]
    """
    if not epochs:
        raise ValueError("provenance needs at least one epoch")
    records = [_record(e) for e in epochs]

    def compared(r):
        return json.dumps({k: v for k, v in r.items() if k not in ignore})

    diffs: list[tuple] = []
    for k, r in enumerate(records[1:], start=1):
        found = _core.change_provenance_differences(compared(records[0]), compared(r), float(rtol))
        prefix = "" if len(records) == 2 else f"epoch {k}: "
        diffs.extend((prefix + key, a, b) for key, a, b in found)
    if warn and diffs:
        shown = "; ".join(f"{k}: {a} vs {b}" for k, a, b in diffs[:5])
        more = f" (and {len(diffs) - 5} more)" if len(diffs) > 5 else ""
        warnings.warn(f"the epochs were processed differently: {shown}{more}", ProvenanceWarning, stacklevel=2)
    return Provenance(records, diffs)
