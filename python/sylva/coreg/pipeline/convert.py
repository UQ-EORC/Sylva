# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Conversion between the pipeline's Python types and the core's dicts and arrays."""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from ..ground import GroundModel
from ..icp import ICPResult, _plane_information
from ..matching import MatchResult
from ..posegraph import OptimisationResult
from ..reflectors import Reflector, ReflectorMatch
from ..stems import Stem, StemMap, _detector_kwargs
from ..transforms import _mat4
from .results import CoregConfig, PairResult, ScanFeatures, SurveyResult


def _config_core(cfg: CoregConfig) -> dict:
    """The settings as the core takes them: stem detection in the detector's
    coregistration mode, the rest field for field."""
    d = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    d["stems"] = _detector_kwargs(cfg.stems)
    d["matching"] = asdict(cfg.matching)
    icp = asdict(cfg.icp)
    icp["voxel_sizes"] = [float(v) for v in cfg.icp.voxel_sizes]
    if cfg.icp.max_distances is not None:
        icp["max_distances"] = [float(v) for v in cfg.icp.max_distances]
    d["icp"] = icp
    d["refinement_voxel_sizes"] = [float(v) for v in cfg.refinement_voxel_sizes]
    d["refinement_max_distances"] = [float(v) for v in cfg.refinement_max_distances]
    d["riegl_options"] = dict(cfg.riegl_options)
    return d


def _stem_rows(stem_map: StemMap) -> np.ndarray:
    return np.array(
        [
            [
                s.x,
                s.y,
                s.z,
                s.dbh,
                *np.asarray(s.axis, dtype=float).reshape(3),
                s.reference_height,
                s.n_slices,
                s.n_points,
                s.rmse,
                s.coverage,
                s.lean_deg,
            ]
            for s in stem_map
        ],
        dtype=np.float64,
    ).reshape(-1, 13)


def _stems_from_rows(rows: np.ndarray) -> list[Stem]:
    return [
        Stem(
            float(r[0]),
            float(r[1]),
            float(r[2]),
            float(r[3]),
            np.array(r[4:7]),
            float(r[7]),
            int(r[8]),
            int(r[9]),
            float(r[10]),
            float(r[11]),
            float(r[12]),
        )
        for r in rows
    ]


def _reflector_tuples(reflectors) -> list[tuple]:
    return [
        (
            float(r.x),
            float(r.y),
            float(r.z),
            float(r.reflectance),
            float(r.diameter),
            int(r.n_points),
            str(r.name),
        )
        for r in reflectors or []
    ]


def _scan_core(scan: ScanFeatures) -> tuple:
    g = scan.ground
    return (
        scan.name,
        int(scan.n_points),
        None
        if g is None
        else (
            np.ascontiguousarray(g.elevation, dtype=np.float64),
            (float(g.origin[0]), float(g.origin[1])),
            float(g.cell_size),
            np.ascontiguousarray(g.observed, dtype=bool),
        ),
        _stem_rows(scan.stem_map),
        scan.stem_map.name,
        np.ascontiguousarray(np.asarray(scan.icp_points, dtype=np.float64).reshape(-1, 3)),
        _reflector_tuples(scan.reflectors),
        np.ascontiguousarray(np.asarray(scan.icp_heights, dtype=np.float32).reshape(-1)),
        _mat4(scan.levelling),
        [float(v) for v in np.asarray(scan.origin, dtype=float).reshape(3)],
        None if scan.source is None else str(scan.source),
        float(scan.seconds),
        scan.error,
    )


def _scan_from_core(d: dict) -> ScanFeatures:
    ground = None if d["ground"] is None else GroundModel(*d["ground"])
    return ScanFeatures(
        name=d["name"],
        n_points=d["n_points"],
        ground=ground,
        stem_map=StemMap(_stems_from_rows(d["stems"]), name=d["stem_map_name"], ground=ground),
        icp_points=d["icp_points"],
        reflectors=[Reflector(*r) for r in d["reflectors"]],
        icp_heights=d["icp_heights"],
        levelling=d["levelling"],
        origin=d["origin"],
        source=None if d["source"] is None else Path(d["source"]),
        seconds=d["seconds"],
        error=d["error"],
    )


def _match_core(m: MatchResult | None) -> dict | None:
    if m is None:
        return None
    return {
        "transform": _mat4(m.transform),
        "n_inliers": int(m.n_inliers),
        "inlier_rmse": float(m.inlier_rmse),
        "score": float(m.score),
        "correspondences": np.ascontiguousarray(
            np.asarray(m.correspondences, dtype=np.int64).reshape(-1, 2)
        ),
        "n_source": int(m.n_source),
        "n_target": int(m.n_target),
        "success": bool(m.success),
        "ambiguity": float(m.ambiguity),
        "rival": _match_core(m.rival),
    }


def _icp_core(r: ICPResult | None) -> dict | None:
    if r is None:
        return None
    info = r.information
    return {
        "transform": _mat4(r.transform),
        "fitness": float(r.fitness),
        "inlier_rmse": float(r.inlier_rmse),
        "n_correspondences": int(r.n_correspondences),
        "iterations": int(r.iterations),
        "converged": bool(r.converged),
        "history": [float(v) for v in r.history],
        "hessian": None
        if info is None
        else np.ascontiguousarray(np.asarray(info.hessian, dtype=float)),
        "plane_sigma": None if info is None else float(info.sigma),
        "plane_n": None if info is None else int(info.n),
    }


def _icp_from_core(d: dict | None) -> ICPResult | None:
    if d is None:
        return None
    return ICPResult(
        np.asarray(d["transform"]),
        float(d["fitness"]),
        float(d["inlier_rmse"]),
        int(d["n_correspondences"]),
        int(d["iterations"]),
        bool(d["converged"]),
        list(d["history"]),
        _plane_information(d),
    )


def _pair_core(p: PairResult) -> dict:
    r = p.reflector_match
    return {
        "i": int(p.i),
        "j": int(p.j),
        "name_i": p.name_i,
        "name_j": p.name_j,
        "transform": _mat4(p.transform),
        "coarse_transform": _mat4(p.coarse_transform),
        "match": _match_core(p.match),
        "reflector_match": None
        if r is None
        else {
            "transform": _mat4(r.transform),
            "n_inliers": int(r.n_inliers),
            "rmse": float(r.rmse),
            "correspondences": np.ascontiguousarray(
                np.asarray(r.correspondences, dtype=np.int64).reshape(-1, 2)
            ),
            "success": bool(r.success),
        },
        "icp": _icp_core(p.icp),
        "success": bool(p.success),
        "reason": p.reason,
        "seconds": float(p.seconds),
        "matched_source": np.ascontiguousarray(
            np.asarray(p.matched_source, dtype=np.float64).reshape(-1, 3)
        ),
        "matched_target": np.ascontiguousarray(
            np.asarray(p.matched_target, dtype=np.float64).reshape(-1, 3)
        ),
        "coarse_stem_rmse": float(p.coarse_stem_rmse),
        "fine_stem_rmse": float(p.fine_stem_rmse),
        "fitness_above": float(p.fitness_above),
        "rival": _match_core(p.rival),
        "used_icp": bool(p.used_icp),
        "ground_offset": float(p.ground_offset),
        "trusted": bool(p.trusted),
    }


def _pair_from_core(d: dict) -> PairResult:
    r = d["reflector_match"]
    return PairResult(
        i=d["i"],
        j=d["j"],
        name_i=d["name_i"],
        name_j=d["name_j"],
        transform=d["transform"],
        coarse_transform=d["coarse_transform"],
        match=None if d["match"] is None else MatchResult._from_core(d["match"]),
        reflector_match=None
        if r is None
        else ReflectorMatch(
            r["transform"], r["n_inliers"], r["rmse"], r["correspondences"], r["success"]
        ),
        icp=_icp_from_core(d["icp"]),
        success=d["success"],
        reason=d["reason"],
        seconds=d["seconds"],
        matched_source=d["matched_source"],
        matched_target=d["matched_target"],
        coarse_stem_rmse=d["coarse_stem_rmse"],
        fine_stem_rmse=d["fine_stem_rmse"],
        fitness_above=d["fitness_above"],
        rival=None if d["rival"] is None else MatchResult._from_core(d["rival"]),
        used_icp=d["used_icp"],
        ground_offset=d["ground_offset"],
        trusted=d["trusted"],
    )


def _survey_from_core(scans: list[ScanFeatures], d: dict) -> SurveyResult:
    poses = [np.asarray(p) for p in d["poses"]]
    o = d["optimisation"]
    optimisation = None
    if o is not None:
        optimisation = OptimisationResult(
            list(poses),
            int(o["iterations"]),
            bool(o["converged"]),
            float(o["initial_error"]),
            float(o["final_error"]),
            [int(k) for k in o["rejected_edges"]],
            np.asarray(o["edge_errors"]),
        )
    return SurveyResult(
        scans=scans,
        pairs=[_pair_from_core(p) for p in d["pairs"]],
        poses=poses,
        reference=int(d["reference"]),
        optimisation=optimisation,
        registered=[bool(r) for r in d["registered"]],
        seconds=float(d["seconds"]),
        edge_to_pair=[int(k) for k in d["edge_to_pair"]],
    )


def _poses(poses) -> list[np.ndarray]:
    return [_mat4(p) for p in poses]


def _targets(targets) -> tuple[list[tuple], list[np.ndarray]]:
    targets = list(targets)
    return [_scan_core(s) for s, _ in targets], [_mat4(p) for _, p in targets]


def _points(cloud) -> np.ndarray:
    from ...pointcloud import PointCloud

    if isinstance(cloud, PointCloud):
        return np.ascontiguousarray(cloud.xyz, dtype=np.float64)
    return np.ascontiguousarray(np.asarray(cloud, dtype=np.float64).reshape(-1, 3))


def _input_core(cloud):
    """A scan as the core reads it: a path (str) or ``(n, 3)`` points."""
    if isinstance(cloud, (str, Path)):
        return str(Path(cloud))
    return _points(cloud)


def _positions(positions) -> list[list[float]] | None:
    """Approximate positions for the core; None where they would be ignored."""
    if positions is None:
        return None
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or len(positions) == 0:
        return None
    return positions.tolist()
