# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Progress reporting for the work that takes a while.

Segmenting a plot, tracing a few million pulses or fitting a QSM can run for
minutes with nothing on the screen. The core counts what it has done in
atomics; nothing is printed and no callback is called on that path, so the
counting costs nothing when no one is watching. Ask to watch and a thread
here reads those counts a few times a second and draws them::

    from sylva import trees
    from sylva.util import progress

    with progress.bar():
        labels = trees.segment_trees(cloud, stems)

A script that wants the numbers rather than a bar passes a callback, which is
called with the running stages as ``(label, done, total)``, outermost first::

    with progress.bar(lambda stages: log(stages[-1])):
        ...

Python loops report too, so a stage that runs over trees or scans shows up
the same way::

    with progress.task("fitting QSMs", len(trees)) as t:
        for tree in trees:
            ...
            t.update()

Nothing is drawn unless the stream is a terminal, so redirected output and
notebooks stay clean; pass ``force=True`` to draw anyway.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager

from .. import _core

__all__ = ["bar", "task", "state"]

#: The tree grows as the work goes: seed, sprout, potted, tree.
_GROWTH = ["\N{SEEDLING}", "\N{HERB}", "\N{POTTED PLANT}", "\N{DECIDUOUS TREE}", "\N{EVERGREEN TREE}"]
#: A sapling swaying in the wind, for a stage whose length is unknown.
_SWAY = ["\N{SEEDLING} ", " \N{SEEDLING}", "\N{HERB} ", " \N{HERB}"]
_ASCII_GROWTH = [".", ",", "i", "Y", "T"]
_ASCII_SWAY = [". ", " .", ", ", " ,"]
_BLOCKS = " ▏▎▍▌▋▊▉█"


def state() -> list[tuple[str, int, int]]:
    """The stages running now, outermost first.

    Returns
    -------
    list of (str, int, int)
        Label, steps done and steps expected (0 when not known). Safe to call
        from another thread while the work runs.
    """
    return _core.progress_state()


def _unicode_ok(stream) -> bool:
    if os.environ.get("SYLVA_PROGRESS_ASCII"):
        return False
    try:
        "".join(_GROWTH).encode(stream.encoding or "ascii")
        return True
    except (UnicodeEncodeError, LookupError, AttributeError):
        return False


def _bar(fraction: float, width: int, blocks: bool) -> str:
    """A bar of ``width`` cells, an eighth of a cell at a time."""
    if not blocks:
        n = int(round(fraction * width))
        return "#" * n + "-" * (width - n)
    eighths = int(round(fraction * width * 8))
    full, rest = divmod(eighths, 8)
    return (_BLOCKS[8] * full + (_BLOCKS[rest] if rest else "")).ljust(width)


def _line(stages, tick: int, elapsed: float, width: int, fancy: bool) -> str:
    label, done, total = stages[-1]
    if len(stages) > 1:
        label = f"{stages[0][0]} · {label}"
    growth = _GROWTH if fancy else _ASCII_GROWTH
    sway = _SWAY if fancy else _ASCII_SWAY
    if total:
        f = min(max(done / total, 0.0), 1.0)
        tree = growth[min(int(f * (len(growth) - 1)), len(growth) - 1)]
        bar = _bar(f, max(10, min(28, width - len(label) - 34)), fancy)
        counts = f"{done:,}/{total:,}"
        body = f"{tree} {label}  {'▕' if fancy else '['}{bar}{'▏' if fancy else ']'} {f:4.0%}  {counts}"
    else:
        body = f"{sway[tick % len(sway)]} {label}  working"
    return f"{body}  {elapsed:.0f}s"


class _Handle:
    """A stage opened from Python; ``update`` counts steps off."""

    def __init__(self, label: str, total: int = 0):
        self._task = _core.ProgressTask(label, int(total))

    def update(self, n: int = 1) -> None:
        """Count ``n`` more steps done."""
        self._task.inc(int(n))

    def set(self, done: int) -> None:
        """Set how many steps are done."""
        self._task.set(int(done))

    def set_total(self, total: int) -> None:
        """Set how many steps there are, once that is known."""
        self._task.set_total(int(total))

    def close(self) -> None:
        """End the stage."""
        self._task.close()


@contextmanager
def task(label: str, total: int = 0):
    """Report a stage of a Python loop.

    Parameters
    ----------
    label
        What the stage is doing, in a few words ("fitting QSMs").
    total
        Steps expected; 0 if not known, which draws a sapling in the wind
        rather than a bar.

    Yields
    ------
    _Handle
        Call ``update(n=1)`` per step, or ``set(done)``.
    """
    h = _Handle(label, total)
    try:
        yield h
    finally:
        h.close()


@contextmanager
def bar(callback=None, stream=None, interval: float = 0.15, force: bool = False):
    """Watch the work while it runs.

    Parameters
    ----------
    callback
        Called with the running stages as ``(label, done, total)``, outermost
        first, every ``interval`` seconds. Draws a bar when None.
    stream
        Where to draw; stderr by default, so piped output stays clean.
    interval
        Seconds between reads.
    force
        Draw even when the stream is not a terminal.

    Notes
    -----
    The bar is drawn from a thread of its own while the core works with the
    GIL released, so it costs the work nothing. Nested calls are harmless:
    only the outermost draws.
    """
    stream = sys.stderr if stream is None else stream
    drawing = callback is None
    if drawing and not force and not (hasattr(stream, "isatty") and stream.isatty()):
        yield  # nothing to draw on
        return
    if drawing and getattr(bar, "_drawing", False):
        yield  # an outer bar is already running
        return

    stop = threading.Event()
    fancy = drawing and _unicode_ok(stream)
    width = 0
    if drawing:
        try:
            width = os.get_terminal_size(stream.fileno()).columns
        except Exception:
            width = 0
        if width < 40:          # no size to be had (a pipe, a bare pty)
            width = 80
        bar._drawing = True
    t0 = time.monotonic()

    def run():
        tick, last = 0, 0
        while not stop.is_set():
            stages = state()
            if stages:
                if callback is not None:
                    callback(stages)
                else:
                    line = _line(stages, tick, time.monotonic() - t0, width, fancy)[: max(width - 1, 20)]
                    stream.write("\r" + line.ljust(last) + "\r" + line)
                    stream.flush()
                    last = len(line)
            elif last:
                stream.write("\r" + " " * last + "\r")
                stream.flush()
                last = 0
            tick += 1
            stop.wait(interval)
        if last:
            stream.write("\r" + " " * last + "\r")
            stream.flush()

    thread = threading.Thread(target=run, name="sylva-progress", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2 * interval + 1)
        if drawing:
            bar._drawing = False
