# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Change detection between two epochs of one plot.

The principle throughout is to separate real change from noise and from not
having seen something: every change carries an uncertainty or a level of
detection, and what the data cannot support is labelled (unobserved) rather
than reported as change.
"""

from .points import DoD, PointDistances, distances, dod
from .voxels import OCCUPANCY_CLASSES, Occupancy, occupancy

__all__ = [
    "PointDistances", "distances", "DoD", "dod",
    "OCCUPANCY_CLASSES", "Occupancy", "occupancy",
]
