# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Marker-free coregistration of scan positions in forests.

Registers scans with no targets and no initial alignment, from the trees
themselves.

1. **Per scan** (:func:`prepare_scan`): a raster terrain model gives height
   above ground; horizontal slices through the stem band are fitted with
   circles and linked into a stem map; locally planar points (stems, ground,
   logs) are kept for ICP and foliage is dropped. Tilted scans are levelled
   first, with the scanner's own attitude.
2. **Per pair** (:func:`register_pair`): reflective targets where both scans
   saw three or more, otherwise a global stem-map match (stem distances do
   not depend on the unknown transform, so a sorted pair table makes the
   search near-exhaustive) whose height comes from the two terrain models,
   refined by a robust point-to-plane ICP. A pair is
   accepted only if it fits over all points and above the ground, so the
   ground plane alone cannot confirm a wrong pair.
3. **Whole survey** (:func:`coregister`): accepted pairs are edges of a pose
   graph, each weighted by the directions its surfaces constrain, solved
   with a Huber kernel and outlier rejection; scans left over
   are retried against the combined registered survey; optionally all poses
   are refined jointly (:func:`refine_joint`). The report lists, per pair,
   how far apart the same tree lands from the two scans, a quality measure
   that needs no ground truth.

Scans can be held fixed at trusted poses, so new or badly registered
positions join an existing project, and approximate poses (a RiSCAN SOP, GNSS)
can serve as priors.

Examples
--------
>>> from sylva import coreg, read_riscan_project, riscan
>>> project = read_riscan_project("survey.PROJ")
>>> scans = project.with_scans(require_sop=False)
>>> result = coreg.coregister([p.rxp for p in scans], names=[p.name for p in scans],
...                           levelling=[p.levelling for p in scans],
...                           reflectors=[p.reflectors() for p in scans],
...                           approximate_positions=riscan.gnss_to_local([p.gnss for p in scans]))
>>> print(result.report())
>>> result.save("transforms.json")
"""

from .geometry import KdTree, estimate_normals, planar_filter, voxel_downsample
from .ground import GroundModel, fit_ground
from .icp import (
    ICPConfig,
    ICPResult,
    ICPTarget,
    PlaneInformation,
    evaluate_registration,
    icp,
    plane_information,
)
from .matching import MatchConfig, MatchResult, match_stem_maps
from .pipeline import (
    CoregConfig,
    PairResult,
    ScanFeatures,
    SurveyResult,
    coregister,
    coregister_prepared,
    load_transforms,
    merge_clouds,
    place_from_prior,
    prepare_scan,
    reading_options,
    register_pair,
)
from .posegraph import (
    OptimisationResult,
    PoseGraph,
    PoseGraphEdge,
    default_information,
    plane_edge_information,
)
from .refine import JointRefinement, refine_joint
from .reflectors import (
    Reflector,
    ReflectorMatch,
    detect_reflectors,
    match_reflectors,
    read_reflector_list,
    read_tiepoint_list,
)
from .simulate import Plot, Survey, scan_plot, simulate_plot, simulate_survey
from .stems import Stem, StemDetectionConfig, StemMap, detect_stems, stem_map_from_arrays
from .transforms import (
    invert,
    kabsch,
    kabsch_2d_yaw,
    se3_exp,
    se3_log,
    transform_difference,
    transform_points,
    yaw_transform,
)

__all__ = [
    "CoregConfig",
    "GroundModel",
    "ICPConfig",
    "ICPResult",
    "ICPTarget",
    "JointRefinement",
    "KdTree",
    "MatchConfig",
    "MatchResult",
    "OptimisationResult",
    "PairResult",
    "PlaneInformation",
    "Plot",
    "PoseGraph",
    "PoseGraphEdge",
    "Reflector",
    "ReflectorMatch",
    "ScanFeatures",
    "Stem",
    "StemDetectionConfig",
    "StemMap",
    "Survey",
    "SurveyResult",
    "coregister",
    "coregister_prepared",
    "default_information",
    "detect_reflectors",
    "detect_stems",
    "estimate_normals",
    "evaluate_registration",
    "fit_ground",
    "icp",
    "invert",
    "kabsch",
    "kabsch_2d_yaw",
    "load_transforms",
    "match_reflectors",
    "match_stem_maps",
    "merge_clouds",
    "place_from_prior",
    "planar_filter",
    "plane_edge_information",
    "plane_information",
    "prepare_scan",
    "reading_options",
    "read_reflector_list",
    "read_tiepoint_list",
    "refine_joint",
    "register_pair",
    "scan_plot",
    "se3_exp",
    "se3_log",
    "simulate_plot",
    "simulate_survey",
    "stem_map_from_arrays",
    "transform_difference",
    "transform_points",
    "voxel_downsample",
    "yaw_transform",
]
