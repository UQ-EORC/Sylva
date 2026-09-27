# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Change between two epochs of one plot.

The principle throughout is to separate real change from noise and from not
having seen something. Every change carries its uncertainty or level of
detection, and whatever the data cannot support is labelled (below
detection, unmeasured, ambiguous) rather than reported as change.

Trees and plots::

    from sylva import change

    al = change.align_epochs(cloud_2019, cloud_2024)          # on stems and ground
    m = change.match_trees(stems_2019, stems_2024, transform=al)
    inc = change.tree_increments(m, cloud_2019, cloud_2024, labels_2019, labels_2024,
                                 noise_a=0.003, noise_b=0.004)
    s = change.plot_summary(inc, area=np.pi * 20**2, years=5)
    change.provenance(settings_2019, settings_2024)           # warns if they differ

Points, surfaces and voxels: :func:`distances` (C2C and M3C2), :func:`dod`
and :func:`occupancy`. QSMs: :func:`compare_qsms` and
:func:`compare_plot_qsms`.

:func:`sylva.synthetic.forest_epochs` makes two scanned epochs of a plot with
known changes, against which all of this is validated.
"""

from .epochs import EpochAlignment, Provenance, ProvenanceWarning, align_epochs, provenance
from .points import DoD, PointDistances, distances, dod
from .qsm import PlotQSMChange, QSMChange, compare_plot_qsms, compare_qsms
from .summary import Estimate, PlotSummary, plot_summary
from .trees import TreeIncrements, TreeMatch, match_trees, tree_increments
from .voxels import OCCUPANCY_CLASSES, Occupancy, occupancy

__all__ = [
    # epochs, trees and plot summary
    "EpochAlignment", "align_epochs", "Provenance", "ProvenanceWarning", "provenance",
    "TreeMatch", "match_trees", "TreeIncrements", "tree_increments",
    "Estimate", "PlotSummary", "plot_summary",
    # points, surfaces and voxels
    "PointDistances", "distances", "DoD", "dod",
    "OCCUPANCY_CLASSES", "Occupancy", "occupancy",
    # QSMs
    "QSMChange", "compare_qsms", "PlotQSMChange", "compare_plot_qsms",
]
