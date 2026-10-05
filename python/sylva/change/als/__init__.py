# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Change between two airborne lidar surveys of one area.

Every function works on catalogues of tiles (:func:`sylva.als.catalog`)
through the chunk engine of :mod:`sylva.als`: each chunk reads the same
buffered box from both surveys, and the results do not depend on the chunk
size or the number of workers. As elsewhere in :mod:`sylva.change`, every
change carries its uncertainty or level of detection, and what the data
cannot support is labelled (below detection, no data, uncertain,
undetected, unobserved) rather than reported as change.

::

    from sylva import als, change

    a, b = als.catalog("2019/"), als.catalog("2024/")          # ground classified
    al = change.align_surveys(a, b, stable_classes=(2, 6))      # ground and roofs
    chm = change.chm_change(a, b, resolution=1.0, alignment=al, harmonise=True)
    gaps = change.gap_change(chm, height=2.0, min_area=10.0, years=5)
    trees = change.tree_change(als.find_trees(a), als.find_trees(b), chm, alignment=al)
    metrics = change.metric_change(a, b, resolution=20.0, metrics=["zq95", "cover"], alignment=al)
"""

from .align import (  # noqa: F401
    ALSAlignment,
    align_surveys,
)
from .cli import (  # noqa: F401
    _add_commands,
)
from .gaps import (  # noqa: F401
    GAP_CELLS,
    GapChange,
    Gaps,
    canopy_gaps,
    gap_change,
)
from .metrics import (  # noqa: F401
    MetricChange,
    PAIChange,
    metric_change,
    pai_change,
    profile_change,
)
from .surface import (  # noqa: F401
    SURFACE_CLASSES,
    SurfaceChange,
    chm_change,
    dtm_change,
    harmonise,
    surface_change,
)
from .trees import (  # noqa: F401
    ALSTreeChange,
    tree_change,
)

__all__ = [
    "ALSAlignment", "align_surveys",
    "SurfaceChange", "surface_change", "chm_change", "dtm_change", "harmonise",
    "Gaps", "canopy_gaps", "GapChange", "gap_change",
    "ALSTreeChange", "tree_change",
    "MetricChange", "metric_change", "PAIChange", "pai_change", "profile_change",
    "SURFACE_CLASSES", "GAP_CELLS",
]
