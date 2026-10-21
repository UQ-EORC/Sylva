"""Raster methods that have no parity recording to check against."""

import numpy as np

from sylva import Raster


def esri_hillshade(z, cell, azimuth, altitude, z_factor=1.0):
    """The ESRI/GDAL hillshade written out longhand, for comparison.

    Takes ``z`` in image order (row 0 north, the opposite of a Raster) and
    shades the interior cells only, leaving NaN around the edge. It reaches
    the answer by way of slope and aspect angles, where ``Raster.hillshade``
    uses the surface normal directly: agreeing to machine precision means
    both the formula and the row order are right.
    """
    a, b, c = z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:]
    d, f = z[1:-1, :-2], z[1:-1, 2:]
    g, h, i = z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cell)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cell)
    zenith = np.radians(90 - altitude)
    az = np.radians((360 - azimuth + 90) % 360)
    slope = np.arctan(z_factor * np.hypot(dzdx, dzdy))
    aspect = np.where(dzdx != 0, np.arctan2(dzdy, -dzdx),
                      np.where(dzdy > 0, np.pi / 2, np.where(dzdy < 0, 3 * np.pi / 2, 0.0)))
    aspect = np.where(aspect < 0, aspect + 2 * np.pi, aspect)
    shade = 255.0 * (np.cos(zenith) * np.cos(slope)
                     + np.sin(zenith) * np.sin(slope) * np.cos(az - aspect))
    out = np.full(z.shape, np.nan)
    out[1:-1, 1:-1] = np.clip(shade, 0.0, None)
    return out


def relief(rows=40, cols=50):
    y, x = np.mgrid[0:rows, 0:cols]
    rng = np.random.default_rng(3)
    return 3 * np.sin(x / 6) + 2 * np.cos(y / 5) + 0.04 * x + rng.normal(0, 0.05, x.shape)


def test_hillshade_matches_gdal():
    z = relief()
    r = Raster(z, 100.0, 200.0, 2.0)
    lights = [(315, 45, 1.0), (90, 30, 1.0), (0, 10, 2.5), (200, 75, 0.3)]
    for azimuth, altitude, z_factor in lights:
        mine = r.hillshade(azimuth, altitude, z_factor).data
        # The reference works in image order, so flip going in and coming back.
        ref = esri_hillshade(z[::-1], r.resolution, azimuth, altitude, z_factor)[::-1]
        interior = np.isfinite(ref)
        np.testing.assert_allclose(mine[interior], ref[interior], atol=1e-9)
        assert mine.min() >= 0.0 and mine.max() <= 255.0


def test_hillshade_lights_the_face_it_should():
    # A plane rising one metre per metre towards the east turns its face
    # westwards, so it is brightest lit from the west and black from the east.
    _, x = np.mgrid[0:7, 0:7]
    r = Raster(x.astype(float), 0.0, 0.0, 1.0)
    np.testing.assert_allclose(r.hillshade(270.0, 45.0).data[1, 1], 255.0)
    assert r.hillshade(90.0, 45.0).data[1, 1] == 0.0
    # Flat ground takes the light at its own altitude, whatever the bearing.
    flat = Raster(np.full((4, 4), 7.0), 0.0, 0.0, 1.0).hillshade(315.0, 30.0)
    np.testing.assert_allclose(flat.data, 255.0 * np.sin(np.radians(30.0)))


def test_hillshade_keeps_holes_and_georeferencing():
    z = relief()
    z[10, 12] = np.nan
    r = Raster(z, 100.0, 200.0, 2.0, crs="EPSG:28356")
    hs = r.hillshade()
    # Only the hole is NaN: around it the missing neighbour is taken as the
    # centre cell rather than spreading the gap.
    assert np.isnan(hs.data[10, 12])
    assert np.isnan(hs.data).sum() == 1
    assert (hs.xmin, hs.ymin, hs.resolution, hs.crs) == (r.xmin, r.ymin, r.resolution, r.crs)
    assert hs.shape == r.shape
    # The input is untouched.
    np.testing.assert_array_equal(np.isnan(r.data), np.isnan(z))
