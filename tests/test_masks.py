"""sylva.masks: polygon, raster, expression and distance masks."""

import inspect
import json
import struct

import numpy as np
import pytest

from sylva import PointCloud, Raster, masks

SQUARE = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
HOLE = np.array([[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]])


def cloud_xy(xy, **attrs):
    xy = np.asarray(xy, dtype=float)
    return PointCloud(np.column_stack([xy, np.zeros(len(xy))]), attrs)


def random_cloud(n, seed=0, scale=10.0):
    rng = np.random.default_rng(seed)
    return PointCloud(rng.uniform(0, scale, (n, 3)), {
        "classification": rng.integers(0, 8, n).astype(np.uint8),
        "height": rng.uniform(-1, 30, n),
        "intensity": rng.integers(0, 65535, n).astype(np.uint16),
    })


# ------------------------------------------------------------------ polygons

def test_polygon_interior_edges_and_holes():
    pts = [(0.5, 0.5), (1.5, 1.5), (1.0, 1.5), (0.0, 2.0), (4.0, 4.0), (4.0001, 2.0), (2.0, 0.0),
           (np.nan, 1.0)]
    c = cloud_xy(pts)
    got = masks.inside_polygons(c, masks.Polygon(SQUARE, [HOLE]))
    # Interior, hole interior, hole edge, outer edge, vertex, outside, bottom edge, NaN.
    assert got.tolist() == [True, False, True, True, True, False, True, False]
    assert got.dtype == bool


def test_polygon_matches_matplotlib_away_from_edges():
    from matplotlib.path import Path

    rng = np.random.default_rng(3)
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    r = 5 + 2 * np.sin(5 * t)                               # star-like, concave
    ring = np.column_stack([r * np.cos(t), r * np.sin(t)])
    c = cloud_xy(rng.uniform(-8, 8, (20_000, 2)))
    got = masks.inside_polygons(c, ring)
    ref = Path(ring).contains_points(c.xyz[:, :2])
    # matplotlib is loose on the boundary; compare away from it.
    a, b = ring, np.roll(ring, -1, axis=0)
    p = c.xyz[:, None, :2]
    t = np.clip(np.sum((p - a) * (b - a), -1) / np.sum((b - a) ** 2, -1), 0, 1)
    dist = np.linalg.norm(p - (a + t[..., None] * (b - a)), axis=-1).min(axis=1)
    far = dist > 1e-9
    assert far.sum() > 19_990
    np.testing.assert_array_equal(got[far], ref[far])


def test_polygon_index_and_inputs():
    a = SQUARE
    b = SQUARE + [10, 0]
    c = cloud_xy([(1, 1), (11, 1), (4, 2), (20, 20)])
    # Shared nothing; list of arrays is one feature each.
    assert masks.polygon_index(c, [a, b]).tolist() == [0, 1, 0, -1]
    # A list of vertex pairs is one ring.
    assert masks.polygon_index(c, [tuple(v) for v in a]).tolist() == [0, -1, 0, -1]
    # A 3-D array is a stack of rings.
    assert masks.polygon_index(c, np.stack([a, b])).tolist() == [0, 1, 0, -1]
    mp = masks.MultiPolygon([masks.Polygon(a), masks.Polygon(b)], {"id": 5})
    assert masks.polygon_index(c, mp).tolist() == [0, 0, 0, -1]
    # Adjacent squares share the edge x = 4; the first wins in the index.
    right = SQUARE + [4, 0]
    assert masks.polygon_index(c, [right, a]).tolist() == [1, -1, 0, -1]
    assert masks.inside_polygons(c, []).tolist() == [False] * 4
    # Closed rings with z are accepted.
    closed = np.column_stack([np.vstack([a, a[:1]]), np.ones(5)])
    assert masks.inside_polygons(c, closed).tolist() == [True, False, True, False]


def test_crop_polygons_invert_keeps_attributes():
    c = random_cloud(1000)
    kept = masks.crop_polygons(c, SQUARE)
    rest = masks.crop_polygons(c, SQUARE, invert=True)
    assert len(kept) + len(rest) == len(c)
    assert set(kept.attrs) == set(c.attrs)
    assert np.all(kept.x <= 4) and np.all(kept.y <= 4)
    assert np.all((rest.x > 4) | (rest.y > 4))


def test_polygon_errors():
    c = cloud_xy([(0, 0)])
    with pytest.raises(ValueError, match="distinct vertices"):
        masks.inside_polygons(c, [(0, 0), (1, 1), (0, 0)])
    with pytest.raises(ValueError, match="non-finite"):
        masks.inside_polygons(c, [(0, 0), (1, np.nan), (0, 1)])
    with pytest.raises(ValueError, match=r"\(K, 2\)"):
        masks.Polygon(np.zeros(5))


def test_polygons_empty_cloud():
    c = PointCloud(np.zeros((0, 3)))
    assert masks.inside_polygons(c, SQUARE).shape == (0,)
    assert len(masks.crop_polygons(c, SQUARE)) == 0


def test_many_points_many_polygons_match_bruteforce():
    rng = np.random.default_rng(1)
    # A 20 x 20 grid of small squares with gaps, 400 features.
    polys = [SQUARE * 0.2 + [i, j] for i in range(20) for j in range(20)]
    c = cloud_xy(rng.uniform(0, 20, (200_000, 2)))
    idx = masks.polygon_index(c, polys)
    fx, fy = np.floor(c.x), np.floor(c.y)
    inside = (c.x - fx <= 0.8) & (c.y - fy <= 0.8)
    expect = np.where(inside, fx * 20 + fy, -1).astype(np.int64)
    np.testing.assert_array_equal(idx, expect)


# ------------------------------------------------------------ polygon files

def test_read_geojson(tmp_path):
    fc = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": "EPSG:28355"}},
          "features": [
              {"type": "Feature", "properties": {"plot": 1, "name": "a"},
               "geometry": {"type": "Polygon",
                            "coordinates": [SQUARE.tolist() + [SQUARE[0].tolist()],
                                            HOLE.tolist()]}},
              {"type": "Feature", "properties": {"plot": 2},
               "geometry": {"type": "MultiPolygon",
                            "coordinates": [[(SQUARE + [10, 0]).tolist()],
                                            [(SQUARE + [20, 0]).tolist()]]}},
              {"type": "Feature", "properties": {"plot": 3}, "geometry": None},
          ]}
    path = tmp_path / "plots.geojson"
    path.write_text(json.dumps(fc))
    layer = masks.read_polygons(path)
    assert len(layer) == 3 and layer.crs == "EPSG:28355"
    assert [f.properties["plot"] for f in layer] == [1, 2, 3]
    assert len(layer[0].parts[0].holes) == 1 and len(layer[1].parts) == 2
    assert layer[2].parts == []
    c = cloud_xy([(0.5, 0.5), (1.5, 1.5), (12, 2), (22, 2), (30, 0)])
    assert masks.polygon_index(c, layer).tolist() == [0, -1, 1, 1, -1]
    sub = layer[[f.properties["plot"] == 2 for f in layer]]
    assert len(sub) == 1 and sub.crs == layer.crs
    assert masks.inside_polygons(c, sub).tolist() == [False, False, True, True, False]
    # A directory with a layer name.
    assert len(masks.read_polygons(tmp_path, layer="plots")) == 3


def test_read_polygons_errors(tmp_path):
    with pytest.raises(OSError):
        masks.read_polygons(tmp_path / "missing.shp")
    (tmp_path / "line.geojson").write_text(json.dumps(
        {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}))
    with pytest.raises(OSError, match="LineString, not a polygon"):
        masks.read_polygons(tmp_path / "line.geojson")
    (tmp_path / "other.geojson").write_text("{}")
    with pytest.raises(OSError, match="several layers"):
        masks.read_polygons(tmp_path)
    with pytest.raises(OSError, match="no layer"):
        masks.read_polygons(tmp_path, layer="plots")
    with pytest.raises(OSError, match="invalid GeoJSON"):
        masks.read_polygons(tmp_path / "other.geojson")


def _write_shapefile(base, records, fields):
    """Minimal polygon shapefile writer (ESRI whitepaper, 1998) for tests.

    ``records`` is a list of ring lists; exterior rings clockwise, holes
    counter-clockwise, as the format requires.
    """
    contents = []
    for rings in records:
        pts = np.vstack([np.vstack([r, r[:1]]) for r in rings])
        parts = np.cumsum([0] + [len(r) + 1 for r in rings[:-1]])
        box = [*pts.min(0), *pts.max(0)]
        body = struct.pack("<i4d2i", 5, *box, len(rings), len(pts))
        body += struct.pack(f"<{len(parts)}i", *parts) + pts.astype("<f8").tobytes()
        contents.append(body)
    allpts = np.vstack([np.vstack(r) for rec in records for r in rec])
    box = [*allpts.min(0), *allpts.max(0)]

    def header(length_bytes):
        return (struct.pack(">7i", 9994, 0, 0, 0, 0, 0, length_bytes // 2)
                + struct.pack("<2i4d4d", 1000, 5, *box, 0, 0, 0, 0))

    shp, shx, offset = b"", b"", 100
    for i, body in enumerate(contents):
        shx += struct.pack(">2i", offset // 2, len(body) // 2)
        rec = struct.pack(">2i", i + 1, len(body) // 2) + body
        shp += rec
        offset += len(rec)
    base.with_suffix(".shp").write_bytes(header(100 + len(shp)) + shp)
    base.with_suffix(".shx").write_bytes(header(100 + len(shx)) + shx)
    # dBase III table.
    names = list(fields)
    widths = [10 if isinstance(fields[n][0], (int, np.integer)) else 12 for n in names]
    rec_len = 1 + sum(widths)
    dbf = struct.pack("<B3BIHH20x", 3, 126, 1, 1, len(records), 32 + 32 * len(names) + 1, rec_len)
    for n, w in zip(names, widths, strict=True):
        kind = b"N" if isinstance(fields[n][0], (int, np.integer)) else b"C"
        dbf += struct.pack("<11sc4xBB14x", n.encode(), kind, w, 0)
    dbf += b"\r"
    for i in range(len(records)):
        dbf += b" " + b"".join(
            (str(fields[n][i]).rjust(w) if isinstance(fields[n][i], (int, np.integer))
             else str(fields[n][i]).ljust(w)).encode() for n, w in zip(names, widths, strict=True))
    dbf += b"\x1a"
    base.with_suffix(".dbf").write_bytes(dbf)
    base.with_suffix(".prj").write_text('PROJCS["GDA94 / MGA zone 55"]')


def test_read_shapefile(tmp_path):
    cw = lambda r: r[::-1]  # noqa: E731  (SQUARE is counter-clockwise)
    records = [[cw(SQUARE), HOLE], [cw(SQUARE + [10, 0]), cw(SQUARE + [20, 0])]]
    _write_shapefile(tmp_path / "stands", records, {"ID": [7, 8], "NAME": ["north", "south"]})
    layer = masks.read_polygons(tmp_path / "stands.shp")
    assert layer.crs == 'PROJCS["GDA94 / MGA zone 55"]'
    assert layer.properties == [{"ID": 7, "NAME": "north"}, {"ID": 8, "NAME": "south"}]
    assert [len(f.parts) for f in layer] == [1, 2]
    assert len(layer[0].parts[0].holes) == 1
    c = cloud_xy([(0.5, 0.5), (1.5, 1.5), (12, 2), (22, 2), (30, 0)])
    assert masks.polygon_index(c, layer).tolist() == [0, -1, 1, 1, -1]
    assert len(masks.read_polygons(tmp_path, "stands")) == 2


# ------------------------------------------------------------------- rasters

def _raster():
    # Rows from the south: row 0 = [1, 2, nan], row 1 = [4, 5, 6]; 1 m cells from (10, 20).
    return Raster(np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]]), 10.0, 20.0, 1.0)


def test_raster_mask_range_values_and_edges():
    c = cloud_xy([(10.5, 20.5), (12.5, 20.5), (11.5, 21.5), (9.9, 20.5), (13.0, 20.5),
                  (10.0, 20.0), (12.99, 21.99)])
    r = _raster()
    assert masks.raster_mask(c, r).tolist() == [True, False, True, False, False, True, True]
    assert masks.raster_mask(c, r, min=2, max=5).tolist() == [False] * 2 + [True] + [False] * 4
    assert masks.raster_mask(c, r, max=1).tolist() == [True] + [False] * 4 + [True, False]
    assert masks.raster_mask(c, r, values=[6, 1]).tolist() == [True] + [False] * 4 + [True, True]
    assert masks.raster_mask(c, r, values=5).tolist() == [False, False, True] + [False] * 4
    kept = masks.crop_raster(c, r, min=4)
    assert len(kept) == 2
    assert len(masks.crop_raster(c, r, min=4, invert=True)) == 5


def test_raster_mask_matches_cell_index():
    c = random_cloud(20_000, scale=12)
    rng = np.random.default_rng(5)
    r = Raster(rng.uniform(0, 10, (10, 10)), 1.0, 1.0, 1.0)
    r.data[3, 4] = np.nan
    row, col = r.cell_index(c.x, c.y)
    ok = (row >= 0) & (row < 10) & (col >= 0) & (col < 10)
    val = np.full(len(c), np.nan)
    val[ok] = r.data[row[ok], col[ok]]
    np.testing.assert_array_equal(masks.raster_mask(c, r, min=2.5, max=7.5),
                                  (val >= 2.5) & (val <= 7.5))


def test_raster_mask_errors():
    c = cloud_xy([(0, 0)])
    r = _raster()
    with pytest.raises(ValueError, match="not both"):
        masks.raster_mask(c, r, min=1, values=[1])
    with pytest.raises(ValueError, match="greater than max"):
        masks.raster_mask(c, r, min=3, max=1)
    with pytest.raises(ValueError, match="NaN"):
        masks.raster_mask(c, r, min=np.nan)
    with pytest.raises(ValueError, match="resolution"):
        masks.raster_mask(c, Raster(r.data, 0, 0, 0.0))


# -------------------------------------------------------------- expressions

def test_expression_matches_numpy():
    c = random_cloud(100_000, seed=2)
    h, cls, z = c.attrs["height"], c.attrs["classification"], c.z
    cases = {
        "height > 2 & classification != 2": (h > 2) & (cls != 2),
        "height > 2 and classification != 2": (h > 2) & (cls != 2),
        "1.3 <= z < 5 | classification in (3, 4, 5)":
            ((1.3 <= z) & (z < 5)) | np.isin(cls, [3, 4, 5]),
        "not (classification not in (1, 2)) & -x < -3": np.isin(cls, [1, 2]) & (-c.x < -3),
        "intensity / 65535 * 2 - 1 > 0.5": c.attrs["intensity"] / 65535 * 2 - 1 > 0.5,
        "!(height >= 10) || x + y * 2 == 0": ~(h >= 10) | (c.x + c.y * 2 == 0),
    }
    for expr, want in cases.items():
        np.testing.assert_array_equal(masks.expression(c, expr), want, err_msg=expr)


def test_expression_nan_bool_attrs_and_where():
    c = PointCloud(np.array([[0, 0, 0.5], [1, 2, 3], [2, 4, np.nan]]),
                   {"withheld": np.array([False, True, False]),
                    "height": np.array([0.1, 2.5, 3.0])})
    assert masks.expression(c, "z > 0").tolist() == [True, True, False]
    assert masks.expression(c, "z != z").tolist() == [False, False, True]
    assert masks.expression(c, "withheld").tolist() == [False, True, False]
    assert masks.expression(c, "withheld + 1 == 2 & height > 1").tolist() == [False, True, False]
    kept = c.where("height > 1 & not withheld")
    assert len(kept) == 1 and kept.attrs["height"][0] == 3.0
    assert len(masks.crop_expression(c, "height > 1", invert=True)) == 1
    assert masks.expression(PointCloud(np.zeros((0, 3))), "z > 0").shape == (0,)


def test_expression_errors():
    c = random_cloud(10)
    with pytest.raises(ValueError, match=r"unknown attribute 'hieght' at position 0 "
                                         r"\(did you mean 'height'\?\); the cloud has: "
                                         r"x, y, z, classification, height, intensity"):
        masks.expression(c, "hieght > 2")
    with pytest.raises(ValueError, match=r"syntax error at position 11: expected '\)'"):
        c.where("height > (2")
    with pytest.raises(ValueError, match="position 7: use '=='"):
        masks.expression(c, "height = 2")
    with pytest.raises(ValueError, match="not a condition"):
        masks.expression(c, "height & z > 1")
    with pytest.raises(ValueError, match="gives numbers"):
        masks.expression(c, "height * 2")
    with pytest.raises(ValueError, match="empty"):
        masks.expression(c, "  ")
    with pytest.raises(TypeError):
        masks.expression(c, lambda p: p)
    # Python code is not an expression.
    with pytest.raises(ValueError, match="syntax error"):
        masks.expression(c, "__import__('os').system('true')")


# ------------------------------------------------------------ between clouds

def test_near_matches_kdtree():
    from scipy.spatial import cKDTree

    a = random_cloud(50_000, seed=3)
    b = random_cloud(5_000, seed=4)
    d3, _ = cKDTree(b.xyz).query(a.xyz)
    d2, _ = cKDTree(b.xyz[:, :2]).query(a.xyz[:, :2])
    np.testing.assert_array_equal(masks.near(a, b, 0.3), d3 <= 0.3)
    np.testing.assert_array_equal(masks.near(a, b.xyz, 0.1, horizontal=True), d2 <= 0.1)
    kept = masks.crop_near(a, b, 0.3)
    gone = masks.difference(a, b, 0.3)
    assert len(kept) + len(gone) == len(a)
    assert len(masks.crop_near(a, b, 0.3, invert=True)) == len(gone)
    np.testing.assert_array_equal(gone.xyz, a.xyz[d3 > 0.3])


def test_difference_change_detection():
    before = random_cloud(20_000, seed=6)
    new = PointCloud(np.array([[50.0, 50.0, 50.0], [60.0, 60.0, 60.0]]))
    after = PointCloud.concatenate([before.without(*before.attrs), new])
    appeared = masks.difference(after, before, 0.01)
    np.testing.assert_array_equal(appeared.xyz, new.xyz)
    assert len(masks.difference(before, after, 0.01)) == 0


def test_near_edge_cases():
    a = PointCloud(np.array([[0.0, 0, 0], [1, 0, 0], [np.nan, 0, 0]]))
    assert masks.near(a, np.zeros((0, 3)), 1.0).tolist() == [False] * 3
    assert masks.near(a, a, 0.0).tolist() == [True, True, False]
    assert len(masks.difference(a, np.zeros((0, 3)), 1.0)) == 3
    assert masks.near(PointCloud(np.zeros((0, 3))), a, 1.0).shape == (0,)
    with pytest.raises(ValueError, match="non-negative"):
        masks.near(a, a, -1.0)
    with pytest.raises(ValueError, match="finite"):
        masks.near(a, a, np.inf)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        masks.near(a, np.zeros((3, 2)), 1.0)


def test_masks_combine():
    c = random_cloud(10_000, seed=8)
    m = masks.inside_polygons(c, SQUARE) & ~masks.expression(c, "classification == 2")
    np.testing.assert_array_equal(
        m, (c.x <= 4) & (c.y <= 4) & (c.attrs["classification"] != 2))
    assert len(c[m]) == m.sum()


def test_public_api_documented():
    for name in masks.__all__:
        obj = getattr(masks, name)
        doc = inspect.getdoc(obj)
        assert doc, name
        if inspect.isfunction(obj) and inspect.signature(obj).parameters:
            assert "Parameters" in doc, name
