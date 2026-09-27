# Sylva: terrestrial laser scanning processing for forest ecology.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""The als-* commands give what the functions they call give."""

import numpy as np
import pytest

from sylva import Raster, als, cli, io, synthetic


@pytest.fixture(scope="module")
def tiles(tmp_path_factory):
    rng = np.random.default_rng(3)
    trees = [(float(x), float(y), 0.3, float(h)) for x, y, h in
             zip(rng.uniform(6, 34, 4), rng.uniform(6, 34, 4), rng.uniform(10, 18, 4), strict=True)]
    scene = synthetic.forest(trees, size=40.0, ground_points=100, margin=0.0, seed=3)
    flight = synthetic.als_flight(scene, pulse_rate=8_000, line_spacing=30.0,
                                  bounds=(0.0, 0.0, 40.0, 40.0), seed=1)
    return flight.write_tiles(tmp_path_factory.mktemp("tiles"), size=20.0, epsg=32755)


@pytest.fixture(scope="module")
def merged(tiles):
    return tiles.read()

def test_command_line(tiles, merged, tmp_path, capsys):
    d = str(tiles.paths[0]).rsplit("/", 1)[0]
    cli.main(["--no-progress", "als-catalog", d])
    out = capsys.readouterr().out
    assert "4 tiles" in out and "no problems found" in out
    cli.main(["--no-progress", "als-dtm", d, str(tmp_path / "dtm.asc"), "--resolution", "2"])
    dtm = Raster.from_ascii_grid(tmp_path / "dtm.asc")
    want = als.dtm(tiles, 2.0)
    np.testing.assert_allclose(dtm.data, want.data, atol=1e-4)
    cli.main(["--no-progress", "als-chm", d, str(tmp_path / "chm.asc"), "--resolution", "1",
              "--workers", "2"])
    np.testing.assert_allclose(Raster.from_ascii_grid(tmp_path / "chm.asc").data,
                               als.chm(tiles, 1.0).data, atol=1e-4)
    cli.main(["--no-progress", "als-normalize", d, str(tmp_path / "norm"), "--replace-z"])
    norm = als.catalog(tmp_path / "norm")
    assert len(norm) == 4 and norm.n_points == len(merged)
    cli.main(["--no-progress", "als-chm", str(tmp_path / "norm"), str(tmp_path / "chm2.asc"),
              "--resolution", "1", "--normalized"])
    np.testing.assert_allclose(Raster.from_ascii_grid(tmp_path / "chm2.asc").data,
                               als.chm(tiles, 1.0).data, atol=2e-3)
    cli.main(["--no-progress", "als-ground", d, str(tmp_path / "g"), "--method", "pmf",
              "--resolution", "1"])
    assert len(als.catalog(tmp_path / "g")) == 4
    # Problems are reported, and --strict makes them an error.
    io.write(io.read(tiles.paths[0]), tmp_path / "norm" / "extra.laz")
    with pytest.raises(SystemExit):
        cli.main(["--no-progress", "als-catalog", str(tmp_path / "norm"), "--strict"])
    assert "overlap" in capsys.readouterr().out
