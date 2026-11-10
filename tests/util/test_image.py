import struct
import zlib

import numpy as np
import pytest

from sylva.util import image


def test_colormaps_are_tables_and_reverse():
    for name in image.COLORMAPS:
        lut = image.colormap(name)
        assert lut.shape == (256, 3) and lut.min() >= 0.0 and lut.max() <= 1.0
    assert np.allclose(image.colormap("viridis_r"), image.colormap("viridis")[::-1])
    assert image.colormap("gray", 3).tolist() == [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]]
    with pytest.raises(ValueError, match="unknown colour map"):
        image.colormap("not-a-map")


def test_to_rgb_limits_and_nan():
    data = np.array([[0.0, 5.0, 10.0], [np.nan, 100.0, -3.0]])
    rgb = image.to_rgb(data, "gray", vmin=0, vmax=10, nan_color=(255, 0, 0))
    assert rgb.dtype == np.uint8 and rgb.shape == (2, 3, 3)
    assert rgb[0, 0].tolist() == [0, 0, 0] and rgb[0, 2].tolist() == [255, 255, 255]
    assert rgb[0, 1].tolist() == [128, 128, 128]
    assert rgb[1, 1].tolist() == [255, 255, 255] and rgb[1, 2].tolist() == [0, 0, 0]   # clipped
    assert rgb[1, 0].tolist() == [255, 0, 0]
    # Default limits are robust percentiles; log drops non-positive values.
    auto = image.to_rgb(np.linspace(0, 1, 100), "gray")
    assert auto[0].tolist() == [0, 0, 0] and auto[-1].tolist() == [255, 255, 255]
    assert (image.to_rgb(np.array([0.0, 1.0, 10.0]), "gray", log=True)[0] == 0).all()
    assert image.to_rgb(np.array([3.0, 3.0]), "gray")[0].tolist() == [0, 0, 0]


def test_write_png_round_trip(tmp_path):
    grey = (np.arange(12, dtype=np.uint8) * 20).reshape(3, 4)
    rgb = np.stack([grey, 255 - grey, grey // 2], axis=2)
    for img, colour_type in ((grey, 0), (rgb, 2)):
        path = image.write_png(tmp_path / f"t{colour_type}.png", img)
        raw = path.read_bytes()
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"
        width, height, depth, ctype = struct.unpack(">IIBB", raw[16:26])
        assert (width, height, depth, ctype) == (4, 3, 8, colour_type)
        # The data chunk inflates to the rows, each behind a filter byte of zero.
        i = raw.index(b"IDAT")
        n = struct.unpack(">I", raw[i - 4:i])[0]
        rows = zlib.decompress(raw[i + 4:i + 4 + n])
        expect = np.concatenate([np.zeros((3, 1), np.uint8), img.reshape(3, -1)], axis=1).tobytes()
        assert rows == expect
    # Floats are clipped to bytes; a bad shape is refused.
    image.write_png(tmp_path / "f.png", np.array([[300.0, -1.0]]))
    with pytest.raises(ValueError, match="image must be"):
        image.write_png(tmp_path / "bad.png", np.zeros((2, 2, 4)))


def test_log_colouring_refuses_limits_that_have_no_logarithm():
    import pytest

    from sylva.util import image

    data = np.array([[1.0, 10.0], [100.0, 0.0]])
    for kwargs in ({"vmin": 0.0, "vmax": 100.0}, {"vmin": -1.0}, {"vmax": 0.0}):
        with pytest.raises(ValueError, match="positive with log"):
            image.to_rgb(data, log=True, **kwargs)
    rgb = image.to_rgb(data, log=True, vmin=1.0, vmax=100.0)
    assert rgb.shape == (2, 2, 3) and tuple(rgb[1, 1]) == (0, 0, 0)     # log of 0 is drawn as NaN
