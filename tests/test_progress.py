import io
import threading
import time

import numpy as np

from sylva import PointCloud, progress, qsm


def test_task_reports_while_it_runs():
    assert progress.state() == []
    with progress.task("counting", 4) as t:
        assert progress.state() == [("counting", 0, 4)]
        t.update()
        t.update(2)
        assert progress.state() == [("counting", 3, 4)]
        with progress.task("inner") as inner:          # nested, length unknown
            inner.update()
            assert [s[0] for s in progress.state()] == ["counting", "inner"]
        t.set(4)
        t.set_total(5)
        assert progress.state() == [("counting", 4, 5)]
    assert progress.state() == []


def test_the_core_reports_its_own_work(single_tree):
    seen = []
    with progress.bar(callback=seen.append, interval=0.005):
        qsm.build_qsm(single_tree, bin_length=0.5)
        time.sleep(0.02)
    labels = {s[0] for stages in seen for s in stages}
    assert "building a QSM" in labels, labels


class _Screen(io.StringIO):
    """A stream that says it is a terminal, so the bar draws into it."""

    encoding = "utf-8"

    def isatty(self):
        return True


def test_the_bar_draws_and_cleans_up_after_itself():
    out = _Screen()
    with progress.bar(stream=out, interval=0.005, force=True):
        with progress.task("growing", 2) as t:
            time.sleep(0.05)
            t.update()
            time.sleep(0.05)
    text = out.getvalue()
    assert "growing" in text and "%" in text
    assert text.endswith("\r") or text.rstrip("\r").endswith(" ")  # the line is wiped at the end
    # The line itself: a tree, a bar, the counts.
    line = progress._line([("growing", 1, 2)], 0, 3.0, 90, True)
    assert line.startswith("🌿") or line.startswith("🪴")
    assert "1/2" in line and "50%" in line
    plain = progress._line([("growing", 1, 2)], 0, 3.0, 90, False)
    assert "#" in plain and "🌱" not in plain


def test_no_bar_when_the_stream_is_not_a_terminal():
    out = io.StringIO()
    with progress.bar(stream=out, interval=0.005):
        with progress.task("quiet", 1) as t:
            t.update()
            time.sleep(0.02)
    assert out.getvalue() == ""


def test_counting_is_safe_from_several_threads():
    with progress.task("threads", 400) as t:
        ts = [threading.Thread(target=lambda: [t.update() for _ in range(100)]) for _ in range(4)]
        [x.start() for x in ts]
        [x.join() for x in ts]
        assert progress.state()[0][1] == 400
