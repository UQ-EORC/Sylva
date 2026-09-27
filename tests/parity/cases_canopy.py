"""Parity cases for sylva.canopy."""

import numpy as np

from sylva import PointCloud, Raster, Shots, canopy

PATTERN = {"theta_start": 30.0, "theta_delta": 0.5, "theta_count": 200, "phi_start": 0.0,
           "phi_delta": 1.0, "phi_count": 360}


def _scan(seed, origin=(0.0, 0.0, 1.5), canopy_top=20.0, cover=0.6):
    """A scan on PATTERN's grid over flat ground and a canopy layer: upward
    pulses hit foliage with probability ``cover`` (one to three echoes),
    downward ones hit the ground; about 1 % of downward pulses miss."""
    rng = np.random.default_rng(seed)
    theta = PATTERN["theta_start"] + PATTERN["theta_delta"] * np.arange(PATTERN["theta_count"])
    phi = np.arange(PATTERN["phi_count"]) * PATTERN["phi_delta"]
    tt, pp = np.meshgrid(np.radians(theta), np.radians(phi))
    tt, pp = tt.ravel(), pp.ravel()
    d = np.column_stack([np.sin(tt) * np.sin(pp), np.sin(tt) * np.cos(pp), np.cos(tt)])
    counts, ranges = [], []
    for k in range(len(d)):
        up = d[k, 2] > 0
        if up:
            if rng.uniform() < cover * (1.0 - 0.3 * abs(np.cos(tt[k]))):
                n = int(rng.integers(1, 4))
                h = np.sort(rng.uniform(3.0, canopy_top, n))
                ranges.extend((h - origin[2]) / d[k, 2])
                counts.append(n)
            else:
                counts.append(0)
        else:
            if rng.uniform() < 0.01:
                counts.append(0)
            else:
                ranges.append(origin[2] / -d[k, 2] + rng.normal(0, 0.01))
                counts.append(1)
    counts = np.array(counts, np.int64)
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    origins = np.tile(np.asarray(origin, float), (len(d), 1))
    return Shots(origins, d, start, counts, np.array(ranges, float))


def _heights(shots):
    return shots.echo_xyz()[:, 2]


def gap_profile():
    prof = canopy.GapProfile.empty()
    for seed in range(3):
        s = _scan(seed, origin=(seed * 5.0, 0.0, 1.5), cover=0.4 + 0.15 * seed)
        prof.add_scan(s, _heights(s))
    r = prof.report()
    out = {k: v for k, v in r.items()}
    out["pgap"] = prof.pgap()
    for m in ("hinge", "linear", "weighted"):
        out[f"pai_{m}_profile_direct"] = prof.pai_profile(m)
        out[f"pavd_{m}"] = prof.pavd_profile(m)
    out["clumping_45"] = prof.clumping(47.5)
    return out


def gap_profile_fired():
    """Streams without misses, fired pulses from the pattern and from the points."""
    edges = canopy.GapProfile.empty().zenith_edges
    prof_pattern, prof_points = canopy.GapProfile.empty(), canopy.GapProfile.empty()
    out = {}
    for seed in range(2):
        s = _scan(10 + seed)
        hit = s.subset(s.echo_count > 0)
        f1 = canopy.fired_pulses_per_ring(hit, PATTERN, edges)
        f2 = canopy.fired_pulses_from_points(hit, edges)
        out[f"fired_pattern_{seed}"] = f1
        out[f"fired_points_{seed}"] = f2
        prof_pattern.add_scan(hit, _heights(hit), fired_per_ring=f1)
        prof_points.add_scan(hit, _heights(hit), fired_per_ring=f2)
    out["pai_pattern"] = prof_pattern.report()["pai_hinge"]
    out["pai_points"] = prof_points.report()["pai_hinge"]
    return out


def gap_fraction_pattern():
    s = _scan(20)
    hit = s.subset(s.echo_count > 0)
    c, g = canopy.gap_fraction_pattern(hit, _heights(hit), PATTERN, min_height=2.0)
    c2, g2 = canopy.gap_fraction_zenith(s, _heights(s), min_height=2.0)
    return {"centres": c, "gap": g, "centres_zenith": c2, "gap_zenith": g2}


def ground_plane():
    rng = np.random.default_rng(30)
    xy = rng.uniform(-20, 20, (20000, 2))
    z = 0.05 * xy[:, 0] - 0.02 * xy[:, 1] + 3.0 + rng.normal(0, 0.02, len(xy))
    trees = rng.uniform(size=len(xy)) < 0.2
    z[trees] += rng.uniform(0.5, 15, trees.sum())
    pts = np.column_stack([xy, z])
    return {"plane": canopy.fit_ground_plane(pts),
            "plane_local": canopy.fit_ground_plane(pts, cell=2.0, centre=(5.0, -3.0), radius=10.0)}


def vertical_profile():
    rng = np.random.default_rng(40)
    h = rng.gamma(2.0, 4.0, 5000)
    cloud = PointCloud(np.column_stack([rng.uniform(size=(5000, 2)), h]), {"height": h})
    b, c = canopy.vertical_profile(cloud, bin_size=1.0)
    b2, c2 = canopy.vertical_profile(cloud, bin_size=0.5, max_height=20.0)
    return {"bins": b, "counts": c, "bins_capped": b2, "counts_capped": c2}


def density_profiles():
    s = _scan(50, cover=0.7)
    grid = canopy.density_grid(s, 2.0, origin=(-30.0, -30.0, -2.0), shape=(30, 30, 12))
    dtm = Raster(np.full((31, 31), 0.0) + 0.01 * np.arange(31)[None, :], -31.0, -31.0, 2.0)
    masked = grid.mask_ground(dtm)
    b, pooled = grid.profile_above_ground(dtm)
    b2, mean = grid.profile_above_ground(dtm, bin_size=4.0, pooled=False)
    return {"profile": grid.profile, "pai": grid.pai, "masked_profile": masked.profile,
            "masked_density": masked.density, "bins": b, "pooled": pooled, "bins_mean": b2, "mean": mean}


CASES = {f.__name__: f for f in (gap_profile, gap_profile_fired, gap_fraction_pattern, ground_plane,
                                  vertical_profile, density_profiles)}
