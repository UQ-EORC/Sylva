"""Full-waveform lidar: readers and writers round-trip, the decomposition
recovers known echoes, and the echoes feed the pulse tools."""

import os
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from sylva import PointCloud, Shots, io, synthetic, voxels, waveform

PW = 1.5  # system pulse, ns (standard deviation)
SIG_M = PW * waveform.C_HALF


def make_shots(ranges, amps, extent=None, origin=(0.0, 0.0, 100.0), direction=(0.0, 0.0, -1.0)):
    n = len(ranges)
    count = np.array([len(r) for r in ranges], dtype=np.int64)
    start = np.concatenate([[0], np.cumsum(count)[:-1]]).astype(np.int64)
    flat = np.concatenate([np.asarray(r, float) for r in ranges]) if n else np.zeros(0)
    attrs = {"amplitude": np.concatenate([np.asarray(a, float) for a in amps]) if n else np.zeros(0)}
    if extent is not None:
        attrs["extent"] = np.concatenate([np.asarray(e, float) for e in extent])
    return Shots(np.tile(origin, (n, 1)), np.tile(direction, (n, 1)), start, count, flat, attrs)


def nearest(est, truth, tol):
    out = np.full(len(truth), -1)
    for i in range(len(truth)):
        cand = np.flatnonzero(est.waveform == truth.waveform[i])
        if len(cand):
            d = np.abs(est.range[cand] - truth.range[i])
            if d.min() <= tol:
                out[i] = cand[np.argmin(d)]
    return out


@pytest.fixture(scope="module")
def mixed():
    """Waveforms with 0 to 3 echoes, a slanted beam and a known truth."""
    rng = np.random.default_rng(7)
    n = 300
    ranges, amps = [], []
    for i in range(n):
        k = i % 4
        r0 = rng.uniform(30, 60)
        ranges.append([r0 + 1.5 * j for j in range(k)])
        amps.append(rng.uniform(30, 150, k))
    d = np.array([0.2, -0.1, -1.0]) / np.linalg.norm([0.2, -0.1, -1.0])
    s = make_shots(ranges, amps, origin=(500.0, 800.0, 120.0), direction=d)
    return synthetic.waveforms(s, pulse_width=PW, noise=1.0, background=12.0, seed=3,
                               gps_time=np.arange(n) * 1e-5 + 1000.0)


# ---------------------------------------------------------------- container


def test_sample_positions_follow_the_beam(mixed):
    wf, truth = mixed
    p = wf.sample_positions()
    assert p.shape == (wf.n_samples, 3)
    w = wf[5]
    assert np.allclose(p[wf.sample_start[5]:wf.sample_start[5] + wf.sample_count[5]], w.positions())
    # Sample k lies metres_per_ns * interval * k further along the beam than sample 0.
    step = np.linalg.norm(w.positions()[1] - w.positions()[0])
    assert step == pytest.approx(waveform.C_HALF * w.interval)
    assert np.allclose((w.positions()[-1] - w.positions()[0]) / np.linalg.norm(w.positions()[-1] - w.positions()[0]),
                       w.direction)
    # The truth echo of a one-echo pulse is at its range from the origin.
    i = np.flatnonzero(np.bincount(truth.waveform, minlength=len(wf)) == 1)[0]
    e = np.flatnonzero(truth.waveform == i)[0]
    assert np.allclose(truth.xyz[e], wf.origin[i] + truth.range[e] * wf.direction[i])
    assert np.allclose(wf.sample_times()[:3], [0.0, 1.0, 2.0])


def test_subset_concatenate_and_index(mixed):
    wf, _ = mixed
    sub = wf.subset(np.arange(len(wf)) % 3 == 0)
    assert len(sub) == 100
    assert np.array_equal(sub[1].samples, wf[3].samples)
    back = waveform.Waveforms.concatenate([wf.subset(np.arange(0, 10)), wf.subset(np.arange(10, 20))])
    assert np.array_equal(back.samples, wf.subset(np.arange(20)).samples)
    assert wf[-1].pulse == wf.pulse[-1]
    with pytest.raises(IndexError):
        wf[len(wf)]
    with pytest.raises(ValueError, match="mask"):
        wf.subset(np.ones(3, bool))
    with pytest.raises(ValueError):
        waveform.Waveforms.concatenate([])
    assert "n_waveforms=300" in repr(wf)


def test_malformed_waveforms_raise(mixed):
    wf, _ = mixed
    bad = wf.subset(np.arange(5))
    bad.sample_count[2] = 10_000
    with pytest.raises(ValueError, match="past the end"):
        waveform.decompose(bad)
    bad = wf.subset(np.arange(5))
    bad.interval[0] = 0.0
    with pytest.raises(ValueError, match="interval"):
        bad.sample_positions()


# ---------------------------------------------------------------- files


@pytest.mark.parametrize("external", [False, True])
def test_las_round_trip(tmp_path, mixed, external):
    wf, _ = mixed
    path = tmp_path / "w.las"
    wf.write_las(path, external=external)
    meta = waveform.info(path)
    assert meta["format"] == "las" and meta["version"] == "1.4" and meta["point_format"] == 9
    assert meta["packets"] == ("external" if external else "internal")
    assert (tmp_path / "w.wdp").exists() == external
    back = waveform.read(path)
    assert len(back) == len(wf)
    assert np.array_equal(back.samples, wf.samples)  # integer samples are stored exactly
    assert np.abs(back.sample_positions() - wf.sample_positions()).max() < 2e-3
    assert np.allclose(back.gps_time, wf.gps_time)
    assert np.all(np.isnan(back.origin))  # LAS has no field for it
    # The points themselves are ordinary LAS 1.4 format 9 points.
    cloud = io.read(path)
    assert len(cloud) == len(wf)
    assert np.abs(cloud.xyz - wf.anchor).max() <= 0.0005 + 1e-9


def test_las_float_samples_and_8_bits(tmp_path, mixed):
    wf, _ = mixed
    f = waveform.Waveforms._from_core(wf._to_core())
    f.samples = f.samples * 0.37 - 2.0
    f.write_las(tmp_path / "f.las")
    back = waveform.read(tmp_path / "f.las")
    step = (f.samples.max() - f.samples.min()) / 65535
    assert np.abs(back.samples - f.samples).max() <= 0.5 * step + 1e-4
    f.write_las(tmp_path / "b.las", bits=8)
    back8 = waveform.read(tmp_path / "b.las")
    assert np.abs(back8.samples - f.samples).max() <= 0.5 * (f.samples.max() - f.samples.min()) / 255 + 1e-3
    with pytest.raises(ValueError, match="bits"):
        f.write_las(tmp_path / "x.las", bits=12)


def test_chunks_cover_every_waveform_once(tmp_path, mixed):
    wf, _ = mixed
    path = tmp_path / "c.las"
    wf.write_las(path)
    parts = list(waveform.chunks(path, size=37))
    assert len(parts) == int(np.ceil(len(wf) / 37))
    joined = waveform.Waveforms.concatenate(parts)
    assert np.array_equal(joined.samples, wf.samples)
    assert np.array_equal(joined.pulse, np.arange(len(wf)))
    with pytest.raises(ValueError):
        next(waveform.chunks(path, size=0))


def test_pulsewaves_round_trip(tmp_path, mixed):
    wf, _ = mixed
    path = tmp_path / "w.pls"
    wf.write_pulsewaves(path)
    meta = waveform.info(path)
    assert meta["format"] == "pulsewaves" and meta["n_pulses"] == len(wf)
    back = waveform.read(path)
    assert np.array_equal(back.samples, wf.samples)
    assert np.abs(back.sample_positions() - wf.sample_positions()).max() < 2e-3
    assert np.abs(back.origin - wf.origin).max() < 1e-3  # the origin is the anchor
    assert np.abs(back.gps_time - wf.gps_time).max() < 1e-8
    assert list(waveform.chunks(path, size=1000))[0].n_samples == wf.n_samples
    neg = waveform.Waveforms._from_core(wf._to_core())
    neg.samples = neg.samples - 1000
    with pytest.raises(ValueError, match="16-bit"):
        neg.write_pulsewaves(tmp_path / "n.pls")


def test_pulsewaves_segments_and_outgoing(tmp_path):
    # One pulse with an outgoing waveform and two returning segments.
    d = np.array([0.0, 0.0, -1.0])
    wf = waveform.Waveforms(
        pulse=[4, 4, 4], gps_time=[1.0, 1.0, 1.0], origin=np.tile([0, 0, 50.0], (3, 1)),
        anchor=np.tile([0, 0, 50.0], (3, 1)), direction=np.tile(d, (3, 1)),
        offset=[-3.0, 200.0, 260.0], interval=[1.0, 1.0, 1.0],
        metres_per_ns=np.full(3, waveform.C_HALF), sample_start=[0, 6, 26],
        sample_count=[6, 20, 10], samples=np.arange(36) % 17,
        attrs={"kind": np.array([1, 2, 2], np.uint8)})
    wf.write_pulsewaves(tmp_path / "s.pls")
    every = waveform.read(tmp_path / "s.pls", kind="all")
    assert list(every.attrs["kind"]) == [1, 2, 2]
    assert list(every.attrs["segment"]) == [0, 0, 1]
    assert np.all(every.pulse == 0)
    assert np.allclose(every.offset, wf.offset)
    assert len(waveform.read(tmp_path / "s.pls")) == 2
    assert len(waveform.read(tmp_path / "s.pls", kind="outgoing")) == 1
    with pytest.raises(ValueError, match="kind"):
        waveform.read(tmp_path / "s.pls", kind="both")
    # Outgoing rows are not pulses of their own in the shots.
    est = waveform.decompose(every, noise=1.0, background=0.0)
    shots = waveform.to_shots(every, est)
    assert shots.n_shots == 1


def test_bad_files(tmp_path):
    with pytest.raises(OSError):
        waveform.read(tmp_path / "missing.las")
    (tmp_path / "scan.sdf").write_bytes(b"\x00" * 64)
    with pytest.raises(OSError, match="RIEGL"):
        waveform.info(tmp_path / "scan.sdf")
    io.write(PointCloud(np.zeros((3, 3))), tmp_path / "plain.las")
    with pytest.raises(OSError, match="no waveform"):
        waveform.read(tmp_path / "plain.las")
    with pytest.raises(ValueError):
        waveform.read(tmp_path / "plain.las", start=-1)


def test_missing_wdp_is_reported(tmp_path, mixed):
    wf, _ = mixed
    wf.subset(np.arange(3)).write_las(tmp_path / "e.las", external=True)
    (tmp_path / "e.wdp").unlink()
    assert waveform.info(tmp_path / "e.las")["packets"] == "missing"
    with pytest.raises(OSError, match="wdp"):
        waveform.read(tmp_path / "e.las")


# ---------------------------------------------------------------- processing


def test_noise_estimate(mixed):
    wf, _ = mixed
    bg, sd = waveform.estimate_noise(wf)
    assert np.median(bg) == pytest.approx(12.0, abs=0.5)
    assert np.median(sd) == pytest.approx(1.0, abs=0.25)


def test_smooth_keeps_area_and_lowers_noise(mixed):
    wf, _ = mixed
    sm = waveform.smooth(wf, 1.0)
    assert sm.samples.shape == wf.samples.shape
    quiet = wf.subset(wf.pulse % 4 == 0)  # no echo: background and noise only
    assert waveform.smooth(quiet, 1.0).samples.std() < 0.6 * quiet.samples.std()
    assert np.array_equal(waveform.smooth(wf, 0.0).samples, wf.samples)
    with pytest.raises(ValueError):
        waveform.smooth(wf, -1.0)


def test_decomposition_recovers_known_echoes(mixed):
    wf, truth = mixed
    est = waveform.decompose(wf)
    m = nearest(est, truth, 3 * SIG_M)
    assert (m >= 0).mean() > 0.99
    assert len(est) <= len(truth) * 1.01
    ok = m >= 0
    dr = est.range[m[ok]] - truth.range[ok]
    assert np.abs(dr).max() < 0.03
    assert np.sqrt(np.mean(dr ** 2)) < 0.01
    assert np.median(np.abs(est.amplitude[m[ok]] / truth.amplitude[ok] - 1)) < 0.03
    assert np.median(np.abs(est.width[m[ok]] / truth.width[ok] - 1)) < 0.05
    assert np.allclose(est.xyz[m[ok]], truth.xyz[ok], atol=0.03)
    assert set(est.stats) == {"background", "noise", "rmse", "n_echoes", "iterations"}
    # Waveforms without an echo give none.
    assert np.all(est.stats["n_echoes"][wf.pulse % 4 == 0] == 0)


@pytest.mark.parametrize("peaks", ["inflection", "derivative"])
def test_two_echoes_are_separated(peaks):
    rng = np.random.default_rng(11)
    r0 = rng.uniform(20, 60, 200)
    sep = 3.0 * SIG_M  # three pulse widths apart: a clear dip between them
    s = make_shots([[a, a + sep] for a in r0], [[100.0, 100.0]] * 200)
    wf, truth = synthetic.waveforms(s, pulse_width=PW, noise=1.0, seed=1)
    est = waveform.decompose(wf, peaks=peaks)
    assert (np.bincount(est.waveform, minlength=200) == 2).mean() > 0.98
    m = nearest(est, truth, 0.5 * sep)
    assert (m >= 0).all()
    assert np.abs(est.range[m] - truth.range).max() < 0.05


def test_inflections_resolve_closer_echoes_than_maxima():
    rng = np.random.default_rng(12)
    r0 = rng.uniform(20, 60, 300)
    # Two equal echoes 2 sigma apart: after the 1 ns smoothing their sum has a single
    # maximum, but its second derivative still has two minima.
    sep = 2.0 * SIG_M
    s = make_shots([[a, a + sep] for a in r0], [[100.0, 100.0]] * 300)
    wf, truth = synthetic.waveforms(s, pulse_width=PW, noise=1.0, seed=2)
    two = {p: (np.bincount(waveform.decompose(wf, peaks=p).waveform, minlength=300) == 2).mean()
           for p in ("inflection", "derivative")}
    assert two["inflection"] > 0.95
    assert two["derivative"] < 0.5
    # Less smoothing resolves closer echoes still.
    s = make_shots([[a, a + 1.75 * SIG_M] for a in r0], [[100.0, 100.0]] * 300)
    wf, truth = synthetic.waveforms(s, pulse_width=PW, noise=1.0, seed=3)
    assert (np.bincount(waveform.decompose(wf, smooth=0.5).waveform, minlength=300) == 2).mean() > 0.7


def test_extended_targets_widen_the_echo():
    s = make_shots([[40.0], [40.0]], [[100.0], [100.0]], extent=[[0.0], [0.5]])
    wf, truth = synthetic.waveforms(s, pulse_width=PW, noise=0.5, seed=4)
    assert truth.width[1] == pytest.approx(np.hypot(PW, 0.5 / waveform.C_HALF))
    assert truth.amplitude[1] == pytest.approx(100.0 * PW / truth.width[1])
    est = waveform.decompose(wf)
    assert len(est) == 2
    assert est.width == pytest.approx(truth.width, rel=0.03)
    # The energy is kept.
    assert est.energy == pytest.approx(truth.amplitude * truth.width * np.sqrt(2 * np.pi), rel=0.03)


def test_decompose_edge_cases(mixed):
    wf, _ = mixed
    empty = wf.subset(np.zeros(len(wf), bool))
    est = waveform.decompose(empty)
    assert len(est) == 0 and len(est.stats["noise"]) == 0
    assert waveform.to_shots(empty, est).n_shots == 0
    nan = wf.subset(np.arange(4))
    nan.samples[nan.sample_start[1] + 3] = np.nan
    est = waveform.decompose(nan)
    assert np.all(np.isfinite(est.range))
    flat = wf.subset(np.arange(1))
    flat.samples[:] = 5.0
    assert len(waveform.decompose(flat, min_amplitude=1.0)) == 0
    for kw in [{"peaks": "maxima"}, {"smooth": -1.0}, {"min_width": 2.0, "max_width": 1.0},
               {"max_echoes": 0}, {"noise": -1.0}, {"threshold": -1.0}]:
        with pytest.raises(ValueError):
            waveform.decompose(wf, **kw)


def test_synthetic_arguments_are_checked():
    s = make_shots([[10.0]], [[50.0]])
    for kw in [{"pulse_width": 0.0}, {"interval": -1.0}, {"n_samples": 0}, {"noise": -1.0},
               {"gps_time": [1.0, 2.0]}]:
        with pytest.raises(ValueError):
            synthetic.waveforms(s, **kw)
    a, _ = synthetic.waveforms(s, seed=9)
    b, _ = synthetic.waveforms(s, seed=9)
    assert np.array_equal(a.samples, b.samples)
    miss, truth = synthetic.waveforms(make_shots([[]], [[]]))
    assert len(truth) == 0 and len(miss) == 1


def test_cross_section_and_calibration():
    r = np.array([100.0, 200.0, 300.0])
    width = np.full(3, 1.5)
    rho, beta = 0.4, 0.5e-3
    # Echo amplitudes of Lambertian targets fall with R^2 under the radar equation.
    amp = 1e4 / r ** 2
    c = waveform.calibration_constant(r, amp, width, rho, beta)
    sigma = waveform.backscatter_cross_section(r, amp, width, c)
    assert sigma == pytest.approx(np.pi * rho * r ** 2 * beta ** 2)
    with pytest.raises(ValueError):
        waveform.calibration_constant(r, np.zeros(3), width, rho, beta)
    with pytest.raises(ValueError):
        waveform.backscatter_cross_section(r, amp[:2], width)


# ---------------------------------------------------------------- into pulses


def test_echoes_become_shots_for_ray_tracing(mixed):
    wf, truth = mixed
    est = waveform.decompose(wf)
    shots = est.to_shots(wf)
    assert shots.n_shots == len(wf)  # one per pulse, misses kept
    assert (shots.echo_count == 0).sum() == (wf.pulse % 4 == 0).sum()
    assert np.allclose(shots.origin, wf.origin)
    assert np.all(np.diff(shots.echo_range)[np.diff(shots.shot_of_echo()) == 0] > 0)
    assert np.allclose(np.sort(shots.echo_xyz(), axis=0), np.sort(est.xyz, axis=0), atol=1e-6)
    for k in ("amplitude", "width", "time", "waveform"):
        assert k in shots.echo_attrs
    grid = voxels.ray_voxelize(shots, 2.0)
    assert grid.shape[0] > 0
    cloud = est.to_pointcloud()
    assert len(cloud) == len(est) and "amplitude" in cloud.attrs


def test_shots_without_origin_start_at_the_first_sample(tmp_path, mixed):
    wf, _ = mixed
    wf.write_las(tmp_path / "o.las")
    back = waveform.read(tmp_path / "o.las")
    est = waveform.decompose(back)
    shots = waveform.to_shots(back, est)
    first = np.array([back[i].positions()[0] for i in range(len(back))])
    assert np.allclose(shots.origin, first)
    # A trajectory gives the true origin back.
    shots = waveform.to_shots(back, est, origin=wf.origin)
    ref = waveform.to_shots(wf, waveform.decompose(wf))
    assert np.allclose(shots.echo_range, ref.echo_range, atol=2e-3)
    with pytest.raises(ValueError, match="origin"):
        waveform.to_shots(back, est, origin=np.zeros((2, 3)))


def test_public_api_is_documented():
    import inspect

    for name in waveform.__all__:
        obj = getattr(waveform, name)
        if isinstance(obj, float):
            continue
        assert inspect.getdoc(obj), name
        members = [obj] + ([m for k, m in vars(obj).items() if not k.startswith("_") and callable(m)]
                           if inspect.isclass(obj) else [])
        for m in members:
            if inspect.isclass(m):
                continue
            params = [p for p in inspect.signature(m).parameters if p not in ("self", "cls")]
            if params:
                assert "Parameters" in inspect.getdoc(m), f"{name}.{m.__name__}"
    assert "Parameters" in inspect.getdoc(synthetic.waveforms)


# ---------------------------------------------------------------- a real file

SAMPLE_URL = "https://raw.githubusercontent.com/PulseWaves/PulseWaves/master/data/RIEGL/"
SAMPLE = "100429_152240_2535pt_UTM"


@pytest.fixture(scope="module")
def riegl_sample():
    """The RIEGL LMS-Q680i sample of the PulseWaves repository (LAS 1.4 with a .wdp,
    and the same flight as PulseWaves), about 0.8 MB, cached outside the repository."""
    cache = Path(os.environ.get("SYLVA_TEST_DATA", Path.home() / ".cache" / "sylva-test-data")) / "pulsewaves"
    cache.mkdir(parents=True, exist_ok=True)
    for ext in ("las", "wdp", "pls", "wvs"):
        f = cache / f"{SAMPLE}.{ext}"
        if not f.exists():
            try:
                urllib.request.urlretrieve(SAMPLE_URL + f.name, f)
            except Exception as e:  # pragma: no cover - offline
                pytest.skip(f"sample waveform file not available: {e}")
    return cache / f"{SAMPLE}.las", cache / f"{SAMPLE}.pls"


def test_riegl_sample_matches_riegl_returns(riegl_sample):
    las, _ = riegl_sample
    meta = waveform.info(las)
    assert meta["packets"] == "external" and meta["point_format"] == 9
    wf = waveform.read(las)
    assert len(wf) == 2376 and wf.attrs["n_records"].sum() == 2535
    # The echoes found agree with RIEGL's own returns in the file.
    est = waveform.decompose(wf)
    pts = io.read(las)
    t = pts.attrs["gps_time"]
    d = []
    for i in range(len(wf)):
        mine = est.xyz[est.waveform == i]
        for k in np.flatnonzero(t == wf.gps_time[i]):
            if len(mine):
                d.append(np.linalg.norm(mine - pts.xyz[k], axis=1).min())
    d = np.array(d)
    assert len(d) > 2400
    assert np.median(d) < 0.05
    assert np.percentile(d, 90) < 0.1


def test_riegl_sample_formats_agree(riegl_sample):
    las, pls = riegl_sample
    a = waveform.read(las)
    raw = waveform.read(pls, kind="all", lookup=False)
    assert set(np.unique(raw.attrs["kind"])) == {1, 2}
    # PulseWaves keeps time in microseconds: match the returning segments to the LAS waveforms.
    ret = np.flatnonzero(raw.attrs["kind"] == 2)
    same = 0
    for i in range(0, len(a), 5):
        j = ret[np.abs(raw.gps_time[ret] - a.gps_time[i]) < 1e-6]
        for jj in j:
            b = raw[jj]
            w = a[i]
            along = (b.positions()[0] - w.anchor) @ w.direction
            assert np.linalg.norm(b.positions()[0] - w.anchor - along * w.direction) < 0.01
            k = int(round((along / w.metres_per_ns - w.offset) / w.interval))
            if 0 <= k and k + len(b.samples) <= len(w.samples):
                same += np.array_equal(w.samples[k:k + len(b.samples)], b.samples)
    assert same > 50
