# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""What will not fit in memory, refused before it is asked for.

A voxel grid is ``nx * ny * nz`` cells whatever the cloud holds, so a hectare
asked for at 1 cm is tens of billions of them and the process dies with
nothing worth reading in the log — or takes the machine with it. The
allocations that scale that way are sized first and checked against a budget,
and the error says what it would have needed and what would make it fit::

    >>> voxels.ray_voxelize(shots, 0.01, bounds=plot)     # doctest: +SKIP
    ValueError: a 10000 x 10000 x 4000 voxel grid at 0.01 m needs 51.2 TB,
    and only 41.8 GB is available: try a larger voxel, a smaller area, or
    splitting the plot into tiles. Set SYLVA_MEM_BUDGET (GB) to raise the limit.

The budget is 80 % of what the system reports as free, or ``SYLVA_MEM_BUDGET``
in gigabytes, or whatever :func:`set_budget` was given. It is a guard against
the obvious mistake, not a guarantee: nothing here tracks what is already
held, and a machine can still be pushed over by many smaller pieces.

Checked so far: ray-traced voxel grids, neighbour graphs (tree segmentation
and QSM skeletons) and the buttress raster.
"""

from __future__ import annotations

from .. import _core

__all__ = ["available", "budget", "set_budget", "check", "human"]


def available() -> int | None:
    """Memory the system reports as free (bytes).

    Returns
    -------
    int or None
        None where the system will not say (anything but Linux).
    """
    return _core.memory_available()


def budget() -> int | None:
    """The most one allocation may ask for (bytes).

    Returns
    -------
    int or None
        ``SYLVA_MEM_BUDGET`` (GB) if set, else anything given to
        :func:`set_budget`, else 80 % of :func:`available`. None when nothing
        is known, in which case nothing is refused.
    """
    return _core.memory_budget()


def set_budget(gigabytes: float | None) -> None:
    """Set the budget for this process.

    Parameters
    ----------
    gigabytes
        The limit; None goes back to reading the system.
    """
    _core.set_memory_budget(0 if gigabytes is None else int(gigabytes * 1e9))


def human(bytes_: float) -> str:
    """Bytes as a person writes them ("3.2 TB").

    Parameters
    ----------
    bytes_
        A size.

    Returns
    -------
    str
    """
    return _core.memory_human(float(bytes_))


def check(cells: int, per_cell: int, what: str, hint: str) -> None:
    """Refuse an allocation of ``cells * per_cell`` bytes that will not fit.

    Parameters
    ----------
    cells
        How many of the thing.
    per_cell
        Bytes each.
    what
        What is being made, with its dimensions ("a 400 x 400 x 90 grid").
    hint
        What would make it fit ("a larger voxel, or a smaller area").

    Raises
    ------
    ValueError
        If it is over budget.
    """
    _core.memory_check(int(cells), int(per_cell), what, hint)
