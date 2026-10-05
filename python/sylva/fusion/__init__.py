# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Terrestrial and airborne lidar together.

A TLS plot measures stems, diameters and the lower canopy in detail; an
airborne survey measures the upper canopy and the terrain over the whole
landscape. This module puts the two in one frame and combines them:

- :func:`register` places a TLS plot (in a local frame, or georeferenced
  with a GNSS error of a few metres) on the ALS survey, and reports the
  residuals and their uncertainty;
- :func:`link_trees` links TLS stems to ALS trees, reports the stems
  under another tree's crown, and gives one table of TLS diameters and ALS
  heights;
- :func:`merge_clouds` and :func:`fuse_profiles` give one point cloud and
  one plant area density profile from both instruments;
- :func:`plot_values`, :func:`fit_model` and :func:`upscale` carry plot
  values from the TLS over the ALS survey with a regression on area-based
  metrics;
- :func:`synthetic_scan` is a terrestrial scanner for synthetic scenes that
  sees them as :func:`sylva.synthetic.als_flight` does, so both instruments
  can be checked against one known forest.

The computations are in the Rust core (``sylva_rs::fusion``).
"""

from .cli import (  # noqa: F401
    _add_commands,
)
from .link import (  # noqa: F401
    TreeLinks,
    link_trees,
)
from .merge import (  # noqa: F401
    FusedProfile,
    fuse_profiles,
    merge_clouds,
)
from .registration import (  # noqa: F401
    Registration,
    register,
)
from .synthetic import (  # noqa: F401
    synthetic_scan,
)
from .upscaling import (  # noqa: F401
    Model,
    Upscaling,
    fit_model,
    plot_values,
    upscale,
)

__all__ = ["Registration", "register", "TreeLinks", "link_trees", "merge_clouds", "FusedProfile",
           "fuse_profiles", "plot_values", "Model", "fit_model", "Upscaling", "upscale",
           "synthetic_scan"]
