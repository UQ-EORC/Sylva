"""Where the benchmark data lives. Override with environment variables.

``SYLVA_DATA``   plots, one folder per site (default ``~/data``)
``SYLVA_HARVEST`` the destructive-harvest benchmark: its ``code/harvest``, ``data/harvest``
                 and ``results/harvest`` folders (default ``~/data/harvest_benchmark``)
"""

from __future__ import annotations

import os
from pathlib import Path

DATA = Path(os.environ.get("SYLVA_DATA", "~/data")).expanduser()
HARVEST = Path(os.environ.get("SYLVA_HARVEST", "~/data/harvest_benchmark")).expanduser()

HARVEST_CODE = HARVEST / "code" / "harvest"
HARVEST_DATA = HARVEST / "data" / "harvest"
HARVEST_RESULTS = HARVEST / "results" / "harvest"

#: Folder of Sylva's own outputs inside a site folder. The name predates the
#: rename of the project and is kept so existing results stay readable.
OUT = "pytls"
