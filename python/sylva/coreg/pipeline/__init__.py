# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""End-to-end coregistration of a survey.

The pipeline is staged, and each stage can run on its own:

1. **Per scan** (:func:`prepare_scan`): a terrain model, the stem map, and a
   planarity-filtered subsample for ICP. The expensive part, done once per
   scan however many pairs are tried.
2. **Per pair** (:func:`register_pair`): reflective targets where both scans
   saw them, otherwise a global stem-map match with its height taken from the
   two terrain models, gives a coarse transform that ICP refines; the pair is
   accepted only if it fits over all points and above the ground, and its
   terrain agrees.
3. **Whole survey** (:func:`coregister_prepared`): accepted pairs become the
   edges of a pose graph, each weighted by the directions its surfaces
   constrain, solved with outlier rejection; scans left over are retried
   against the combined registered survey, and optionally every pose is
   refined jointly.

Every stage records its own quality, because the useful question is not "did
it run" but "which scans can I trust". Stem matches take
their height from the shared ground rather than from stem bases (vertical
error dominates stem-based registration, as Tremblay & Béland 2018 and
GlobalMatch, Wang et al. 2023, both found); edges carry the anisotropic
information of their point-to-plane correspondences instead of an isotropic
weight; outlier edges are rejected even while some scans are unregistered;
and recovered scans are tied in by pairwise measurements. Scans can be held
fixed at trusted poses, so that new scans are registered into an existing
project; and approximate poses (a RiSCAN SOP, GNSS and compass) can serve as
priors, refusing results that move a scanner implausibly far and placing
scans that see too few stems.
"""

from .pair import (  # noqa: F401
    register_pair,
)
from .results import (  # noqa: F401
    CoregConfig,
    PairResult,
    ScanFeatures,
    SurveyResult,
    load_transforms,
)
from .scan import (  # noqa: F401
    prepare_scan,
    reading_options,
)
from .survey import (  # noqa: F401
    coregister,
    coregister_prepared,
    merge_clouds,
    place_from_prior,
)

__all__ = [
    "CoregConfig",
    "PairResult",
    "ScanFeatures",
    "SurveyResult",
    "coregister",
    "coregister_prepared",
    "load_transforms",
    "merge_clouds",
    "place_from_prior",
    "prepare_scan",
    "reading_options",
    "register_pair",
]
