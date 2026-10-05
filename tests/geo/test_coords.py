# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""Shifts, rotations, CRSs, reprojection and applying transform files."""

import json
import pickle
import warnings

import numpy as np
import pytest

import sylva
from sylva import PointCloud, Raster, cli, filters, io
from sylva.geo import coords

# Flinders Peak, the worked example of the GDA technical manual: geographic
# coordinates and their published MGA zone 55 grid coordinates (the same on
# GDA94 and GDA2020, which share the ellipsoid and projection).
FLINDERS_LONLAT = (144 + 25 / 60 + 29.52440 / 3600, -(37 + 57 / 60 + 3.72030 / 3600))
FLINDERS_MGA55 = (273741.297, 5796489.777)

# The GDA94 -> GDA2020 Helmert (EPSG:8048) written as GDA2020 towgs84 relative
# to GDA94 (which the EPSG definitions tie to WGS 84 by zero parameters).
GDA2020_HELMERT = ("+proj=longlat +ellps=GRS80 +towgs84=-0.06155,0.01087,0.04019,"
                   "-0.0394924,-0.0327221,-0.0328979,0.009994 +no_defs")


def cloud_at(xyz, **kw):
    xyz = np.atleast_2d(np.asarray(xyz, dtype=float))
    return PointCloud(xyz, {"intensity": np.arange(len(xyz), dtype=np.uint16)}, **kw)


@pytest.fixture
def projected():
    rng = np.random.default_rng(3)
    xyz = np.column_stack([rng.uniform(512_000, 512_050, 5000),
                           rng.uniform(5_412_000, 5_412_050, 5000),
                           rng.uniform(95, 125, 5000)])
    return cloud_at(np.round(xyz, 3), crs="EPSG:7855")


# --------------------------------------------------------------------------- #
# translate / rotate / recentre
# --------------------------------------------------------------------------- #


def test_translate_is_exact_and_keeps_everything(projected):
    moved = projected.translate(-512_000, -5_412_000, -100)
    np.testing.assert_array_equal(moved.xyz, projected.xyz - [512_000, 5_412_000, 100])
    assert moved.crs == "EPSG:7855" and set(moved.attrs) == set(projected.attrs)
    back = moved.translate(512_000, 5_412_000, 100)
    np.testing.assert_array_equal(back.xyz, projected.xyz)       # whole metres: exact
    assert moved.translate(0.5, 0.25).xyz[:, 2].tolist() == moved.z.tolist()  # dz defaults to 0


def test_rotate_quarter_turns_exact():
    c = cloud_at([[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(c.rotate(90).xyz, [[-2.0, 1.0, 3.0]])
    np.testing.assert_array_equal(c.rotate(-90, axis="x").xyz, [[1.0, 3.0, -2.0]])
    np.testing.assert_array_equal(c.rotate(180, axis="Y", about=(1, 0, 1)).xyz, [[1.0, 2.0, -1.0]])
    np.testing.assert_array_equal(c.rotate(360).xyz, c.xyz)
    four = c
    for _ in range(4):
        four = four.rotate(90, about=(10, -5, 0))
    np.testing.assert_array_equal(four.xyz, c.xyz)


def test_rotate_matches_rotation_z_and_icp_convention(projected):
    a = projected.rotate(23.5)
    b = projected.transform(sylva.registration.rotation_z(23.5))
    np.testing.assert_allclose(a.xyz, b.xyz, rtol=0, atol=1e-8)


def test_rotate_about_point_and_arbitrary_axis(projected):
    centre = projected.xyz.mean(axis=0)
    r = projected.rotate(37, axis=(1, -2, 0.5), about=centre)
    # Distances to the centre and the axis coordinate are preserved.
    np.testing.assert_allclose(np.linalg.norm(r.xyz - centre, axis=1),
                               np.linalg.norm(projected.xyz - centre, axis=1), atol=1e-8)
    k = np.array([1, -2, 0.5]) / np.linalg.norm([1, -2, 0.5])
    np.testing.assert_allclose((r.xyz - centre) @ k, (projected.xyz - centre) @ k, atol=1e-8)
    back = r.rotate(-37, axis=(1, -2, 0.5), about=centre)
    np.testing.assert_allclose(back.xyz, projected.xyz, rtol=0, atol=1e-8)


def test_repeated_round_trips_do_not_drift(projected):
    c = projected
    centre = (512_025.0, 5_412_025.0, 0.0)
    for _ in range(50):
        c = c.rotate(13.7, about=centre).rotate(-13.7, about=centre)
        c = c.translate(-512_000.25, -5_412_000.5, 3).translate(512_000.25, 5_412_000.5, -3)
    np.testing.assert_allclose(c.xyz, projected.xyz, rtol=0, atol=1e-7)


def test_recentre(projected):
    local, offset = projected.recentre()
    np.testing.assert_array_equal(offset, np.floor(projected.xyz.min(axis=0)))
    assert np.all(local.xyz >= 0) and np.all(local.xyz.max(axis=0) < 60)
    np.testing.assert_array_equal(local.translate(*offset).xyz, projected.xyz)
    assert local.crs == projected.crs
    local, offset = projected.recentre(origin=(512_010, 5_412_010, 100))
    np.testing.assert_array_equal(offset, [512_010, 5_412_010, 100])
    np.testing.assert_array_equal(local.translate(*offset).xyz, projected.xyz)


def test_recentre_ignores_nan_and_handles_empty():
    c = cloud_at([[np.nan, 5.5, -0.5], [500_000.7, 6.2, 3.0]])
    local, offset = c.recentre()
    np.testing.assert_array_equal(offset, [500_000.0, 5.0, -1.0])
    assert np.isnan(local.x[0])
    empty = PointCloud(np.zeros((0, 3)))
    local, offset = empty.recentre()
    assert len(local) == 0 and offset.tolist() == [0, 0, 0]
    assert len(empty.translate(1, 2, 3)) == 0 and len(empty.rotate(10)) == 0


@pytest.mark.parametrize("call, message", [
    (lambda c: c.translate(np.nan, 0), "finite"),
    (lambda c: c.translate(1, np.inf, 0), "finite"),
    (lambda c: c.rotate(np.nan), "finite"),
    (lambda c: c.rotate(10, axis="w"), "axis"),
    (lambda c: c.rotate(10, axis=(0, 0, 0)), "zero"),
    (lambda c: c.rotate(10, axis=(0, 1)), "axis"),
    (lambda c: c.rotate(10, about=(0, np.nan, 0)), "about"),
    (lambda c: c.recentre(origin=(1, 2)), "origin"),
])
def test_invalid_shift_and_rotation(call, message):
    with pytest.raises(ValueError, match=message):
        call(cloud_at([[1.0, 2.0, 3.0]]))


def test_matrices_compose_with_transform(projected):
    m = coords.rotation_matrix(30, "z", about=(512_000, 5_412_000, 0)) @ coords.translation_matrix(1, 2, 3)
    a = projected.transform(m)
    b = projected.translate(1, 2, 3).rotate(30, about=(512_000, 5_412_000, 0))
    np.testing.assert_allclose(a.xyz, b.xyz, rtol=0, atol=1e-8)


# --------------------------------------------------------------------------- #
# crs on PointCloud
# --------------------------------------------------------------------------- #


def test_crs_field_defaults_and_normalises():
    assert PointCloud(np.zeros((2, 3))).crs is None
    assert PointCloud(np.zeros((2, 3)), crs=7855).crs == "EPSG:7855"
    assert PointCloud(np.zeros((2, 3)), crs=np.int64(4326)).crs == "EPSG:4326"
    with pytest.raises(ValueError, match="crs"):
        PointCloud(np.zeros((2, 3)), crs=3.5)
    assert "EPSG:7855" in repr(PointCloud(np.zeros((2, 3)), crs=7855))
    assert "crs" not in repr(PointCloud(np.zeros((2, 3))))


def test_crs_carried_through(projected):
    c = projected
    for derived in (c[c.z > 100], c[:10], c.copy(), c.with_attrs(h=c.z), c.without("intensity"),
                    c.transform(np.eye(4)), filters.voxel_downsample(c, 1.0),
                    filters.voxel_downsample(c, 1.0, method="centroid"),
                    PointCloud.concatenate([c, c]), PointCloud.concatenate([c, cloud_at(c.xyz)])):
        assert derived.crs == "EPSG:7855"


def test_concatenate_crs_rules(projected):
    other = projected.copy()
    other.crs = coords.crs_info(7855).wkt                        # same CRS, spelled as WKT
    assert PointCloud.concatenate([projected, other]).crs == "EPSG:7855"
    assert PointCloud.concatenate([cloud_at([[0, 0, 0]])]).crs is None
    with pytest.raises(ValueError, match="different CRSs"):
        PointCloud.concatenate([projected, cloud_at([[0, 0, 0]], crs=32755)])


def test_old_pickles_still_load(projected):
    blob = pickle.dumps(projected)
    assert pickle.loads(blob).crs == "EPSG:7855"
    old = PointCloud.__new__(PointCloud)
    old.__dict__.update({"xyz": projected.xyz, "attrs": projected.attrs})  # pickled before crs existed
    restored = pickle.loads(pickle.dumps(old))
    assert restored.crs is None and len(restored) == len(projected)


# --------------------------------------------------------------------------- #
# LAS/LAZ CRS
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("ext", [".las", ".laz"])
def test_las_crs_round_trip(tmp_path, projected, ext):
    path = tmp_path / f"a{ext}"
    io.write(projected, path)
    back = io.read(path)
    assert back.crs.startswith('PROJCS["GDA2020 / MGA zone 55"')
    assert coords.same_crs(back.crs, 7855)
    assert coords.crs_info(back.crs).epsg == 7855
    # Written again, the WKT stays as it is.
    io.write(back, tmp_path / f"b{ext}")
    assert io.read(tmp_path / f"b{ext}").crs == back.crs


def test_las_crs_is_standard_wkt_vlr(tmp_path, projected):
    laspy = pytest.importorskip("laspy")
    io.write(projected, tmp_path / "a.las")
    f = laspy.read(tmp_path / "a.las")
    wkt = [v for v in f.header.vlrs if v.user_id == "LASF_Projection" and v.record_id == 2112]
    assert len(wkt) == 1 and f.header.global_encoding.wkt


def test_las_compound_crs(tmp_path, projected):
    c = projected.copy()
    c.crs = "EPSG:7855+5711"
    io.write(c, tmp_path / "a.laz")
    info = coords.crs_info(io.read(tmp_path / "a.laz").crs)
    assert (info.epsg, info.vertical_epsg) == (7855, 5711)


def test_las_geotiff_keys(tmp_path):
    laspy = pytest.importorskip("laspy")
    from laspy.vlrs.known import GeoKeyDirectoryVlr, GeoKeyEntryStruct

    vlr = GeoKeyDirectoryVlr()
    vlr.geo_keys_header.number_of_keys = 3
    vlr.geo_keys = [GeoKeyEntryStruct(1024, 0, 1, 1), GeoKeyEntryStruct(3072, 0, 1, 28355),
                    GeoKeyEntryStruct(4096, 0, 1, 5711)]
    header = laspy.LasHeader(point_format=1, version="1.2")
    header.vlrs.append(vlr)
    las = laspy.LasData(header)
    las.x, las.y, las.z = np.array([512_345.0]), np.array([5_412_345.0]), np.array([1.0])
    las.write(tmp_path / "gk.las")
    assert io.read(tmp_path / "gk.las").crs == "EPSG:28355+5711"


def test_las_without_crs_and_other_formats(tmp_path, projected):
    c = cloud_at(projected.xyz)
    io.write(c, tmp_path / "a.laz")
    assert io.read(tmp_path / "a.laz").crs is None
    io.write(projected, tmp_path / "a.ply")
    assert io.read(tmp_path / "a.ply").crs is None           # PLY has no place for it


def test_geographic_las_keeps_precision(tmp_path, projected):
    g = coords.reproject(projected, 7844)
    io.write(g, tmp_path / "g.laz")
    back = io.read(tmp_path / "g.laz")
    np.testing.assert_allclose(back.xyz[:, :2], g.xyz[:, :2], rtol=0, atol=6e-8)   # 1e-7 degree steps
    np.testing.assert_allclose(back.z, g.z, atol=6e-4)


def test_uninterpreted_wkt_is_stored_as_given(tmp_path, projected):
    wkt = ('PROJCS["World_Robinson",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",'
           '6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],'
           'PROJECTION["Robinson"],PARAMETER["Central_Meridian",0.0],UNIT["Meter",1.0]]')
    with pytest.raises(ValueError, match="Robinson"):
        coords.crs_info(wkt)
    c = projected.copy()
    c.crs = wkt
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        io.write(c, tmp_path / "r.las")
    assert io.read(tmp_path / "r.las").crs == wkt


def test_proj_string_crs_is_not_written(tmp_path, projected):
    c = projected.copy()
    c.crs = "+proj=utm +zone=55 +south +ellps=GRS80 +units=m +no_defs"
    with pytest.warns(UserWarning, match="no WKT"):
        io.write(c, tmp_path / "a.las")
    assert io.read(tmp_path / "a.las").crs is None


# --------------------------------------------------------------------------- #
# CRS parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("crs", [7855, "EPSG:7855", "epsg:7855", "7855", "urn:ogc:def:crs:EPSG::7855"])
def test_crs_spellings(crs):
    info = coords.crs_info(crs)
    assert info.epsg == 7855 and info.label == "EPSG:7855" and info.name == "GDA2020 / MGA zone 55"
    assert info.geographic is False


def test_crs_info_geographic_and_proj_string():
    assert coords.crs_info(4326).geographic is True
    info = coords.crs_info("+proj=utm +zone=55 +south +datum=WGS84 +units=m +no_defs")
    assert info.epsg is None and info.wkt is None


@pytest.mark.parametrize("bad", ["", "EPSG:99999", "EPSG:abc", "nonsense", "+proj=nonsense", None, True,
                                 "PROJCS[\"x\""])
def test_invalid_crs(bad):
    with pytest.raises(ValueError):
        coords.crs_info(bad)


def test_esri_wkt_without_codes():
    # An ESRI .prj for MGA zone 55 has no EPSG codes; it is converted to PROJ.
    prj = ('PROJCS["GDA2020_MGA_Zone_55",GEOGCS["GCS_GDA2020",DATUM["D_GDA2020",'
           'SPHEROID["GRS_1980",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],'
           'UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
           'PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",10000000.0],'
           'PARAMETER["Central_Meridian",147.0],PARAMETER["Scale_Factor",0.9996],'
           'PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')
    assert coords.crs_info(prj).epsg is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", coords.ApproximateTransformationWarning)
        xy = coords.reproject(np.array([FLINDERS_LONLAT]), prj, "+proj=longlat +ellps=GRS80")
    np.testing.assert_allclose(xy[0], FLINDERS_MGA55, atol=1e-3)


# --------------------------------------------------------------------------- #
# reproject: known coordinates
# --------------------------------------------------------------------------- #


def test_mga55_against_published_coordinates():
    # GDA2020 geographic -> MGA zone 55: the manual gives the grid to 1 mm.
    xy = coords.reproject(np.array([FLINDERS_LONLAT]), 7855, 7844)
    np.testing.assert_allclose(xy[0], FLINDERS_MGA55, atol=6e-4)
    lonlat = coords.reproject(xy, 7844, 7855)
    np.testing.assert_allclose(lonlat[0], FLINDERS_LONLAT, rtol=0, atol=1e-10)
    # GDA94 / MGA94 zone 55 is the same projection on a GRS80 datum.
    np.testing.assert_allclose(coords.reproject(np.array([FLINDERS_LONLAT]), 28355, 4283)[0],
                               FLINDERS_MGA55, atol=6e-4)


@pytest.mark.parametrize("lonlat, epsg, expected", [
    # References from PROJ 9.8 (pyproj 3.8).
    ((147.5, -42.9), 32755, (540820.2432126359, 5250168.690354232)),
    ((15.0, 52.0), 32633, (500000.0000000011, 5761038.212590415)),
    ((-73.9857, 40.7484), 32618, (585628.4090877832, 4511322.447496134)),
    ((147.5, -42.9), 3577, (1302654.6514726398, -4768012.545205606)),
])
def test_wgs84_projections_against_proj(lonlat, epsg, expected):
    xy = coords.reproject(np.array([lonlat]), epsg, 4326)
    np.testing.assert_allclose(xy[0], expected, rtol=0, atol=1e-6)


def test_helmert_against_proj():
    # OSGB36 / British National Grid -> WGS 84 by the 7-parameter towgs84 of
    # the EPSG definition; PROJ 9.8 with the same towgs84 gives
    # (-0.1283539680960025, 51.50399082304035).
    osgb = cloud_at([[530_000.0, 180_000.0, 50.0]], crs=27700)
    assert coords.transformation(27700, 4326).kind == "helmert"
    out = coords.reproject(osgb, 4326)
    np.testing.assert_allclose(out.xyz[0, :2], (-0.1283539680960025, 51.50399082304035),
                               rtol=0, atol=1e-9)
    assert out.z[0] != 50.0                            # ellipsoidal height changes
    # The reverse applies the negated parameters (as PROJ.4 did), which
    # undoes the forward transformation to well under a millimetre.
    back = coords.reproject(out, 27700)
    np.testing.assert_allclose(back.xyz, osgb.xyz, rtol=0, atol=1e-4)


def test_gda94_to_gda2020_helmert_against_proj():
    # PROJ 9.8, "GDA94 to GDA2020 (1)": (144.42487390469094, -37.951020218729184).
    out = coords.reproject(np.array([[*FLINDERS_LONLAT, 0.0]]), GDA2020_HELMERT, 4283)
    np.testing.assert_allclose(out[0, :2], (144.42487390469094, -37.951020218729184),
                               rtol=0, atol=2e-10)


# --------------------------------------------------------------------------- #
# reproject: behaviour
# --------------------------------------------------------------------------- #


def test_reproject_cloud_sets_crs_and_keeps_attrs(projected):
    g = coords.reproject(projected, 7844)
    assert g.crs == "EPSG:7844" and set(g.attrs) == set(projected.attrs)
    np.testing.assert_array_equal(g.z, projected.z)             # same datum: heights unchanged
    assert np.all((g.x > 147) & (g.x < 148) & (g.y < -41) & (g.y > -42))
    back = coords.reproject(g, "EPSG:7855")
    np.testing.assert_allclose(back.xyz, projected.xyz, rtol=0, atol=1e-6)
    assert back.crs == "EPSG:7855"


def test_reproject_identity_and_src_override(projected):
    same = coords.reproject(projected, "epsg:7855")
    np.testing.assert_array_equal(same.xyz, projected.xyz)
    unlabelled = cloud_at(projected.xyz)
    with pytest.raises(ValueError, match="no crs"):
        coords.reproject(unlabelled, 7844)
    np.testing.assert_allclose(coords.reproject(unlabelled, 7844, src_crs=7855).xyz,
                               coords.reproject(projected, 7844).xyz)


def test_approximate_transformations_warn(projected):
    t = coords.transformation(7855, 32755)
    assert t.kind == "null datum" and not t.exact
    with pytest.warns(coords.ApproximateTransformationWarning, match="null transformation"):
        coords.reproject(projected, 32755)
    with pytest.warns(coords.ApproximateTransformationWarning, match="vertical"):
        coords.reproject(np.array([[512_000.0, 5_412_000.0, 10.0]]), "EPSG:7855+9458", "EPSG:7855+5711")
    exact = coords.transformation(28355, 32755)                # GDA94 is tied to WGS 84
    assert exact.kind == "conversion" and exact.exact


def test_grid_transformations_are_refused():
    with pytest.raises(ValueError, match="grid"):
        coords.reproject(np.array([[-100.0, 40.0]]), 4326, 4267)   # NAD27 needs NADCON


def test_reproject_edge_cases(projected):
    empty = PointCloud(np.zeros((0, 3)), crs=7855)
    out = coords.reproject(empty, 7844)
    assert len(out) == 0 and out.crs == "EPSG:7844"
    xyz = np.array([[np.nan, 5_412_000.0, 0.0], [512_000.0, 5_412_000.0, 1.0]])
    out = coords.reproject(xyz, 7844, 7855)
    assert np.isnan(out[0]).all() and np.isfinite(out[1]).all()
    assert coords.reproject([[512_000.0, 5_412_000.0]], 7844, 7855).shape == (1, 2)
    with pytest.raises(ValueError, match="shape"):
        coords.reproject(np.zeros((3, 4)), 7844, 7855)
    with pytest.raises(ValueError, match="src_crs"):
        coords.reproject(np.zeros((3, 3)), 7844)
    with pytest.raises(TypeError, match="rasters"):
        coords.reproject(Raster(np.zeros((2, 2)), 0, 0, 1, crs="EPSG:7855"), 7844)
    with pytest.raises(TypeError):
        coords.reproject("plot.laz", 7844)


def test_reproject_many_points_is_deterministic():
    rng = np.random.default_rng(0)
    xyz = np.column_stack([rng.uniform(3e5, 6e5, 300_000), rng.uniform(1e5, 9e5, 300_000),
                           rng.uniform(0, 100, 300_000)])
    a = coords.reproject(xyz, 4326, 27700)
    b = coords.reproject(xyz, 4326, 27700)
    np.testing.assert_array_equal(a, b)
    for i in (0, 12345, 299_999):
        np.testing.assert_array_equal(a[i], coords.reproject(xyz[i:i + 1], 4326, 27700)[0])


# --------------------------------------------------------------------------- #
# apply_transforms
# --------------------------------------------------------------------------- #


@pytest.fixture
def scans(tmp_path):
    rng = np.random.default_rng(7)
    paths = []
    for k in range(3):
        c = PointCloud(np.round(rng.uniform(-5, 5, (200, 3)), 3),
                       {"intensity": rng.integers(0, 1000, 200).astype(np.uint16)})
        p = tmp_path / "scans" / f"ScanPos00{k + 1}_2cm.laz"
        p.parent.mkdir(exist_ok=True)
        io.write(c, p)
        paths.append(p)
    mats = {f"ScanPos00{k + 1}": coords.rotation_matrix(30 * k, "z") @ coords.translation_matrix(10 * k, 5, 0)
            for k in range(3)}
    return paths, mats


def _expected(paths, mats):
    return [io.read(p).transform(mats[f"ScanPos00{k + 1}"]).xyz for k, p in enumerate(paths)]


def test_apply_transforms_by_name_list_and_mapping(scans):
    paths, mats = scans
    expected = _expected(paths, mats)
    for got in (coords.apply_transforms(paths, mats),
                coords.apply_transforms(list(reversed(paths)), mats)[::-1],
                coords.apply_transforms(paths[0].parent, mats),
                coords.apply_transforms(paths, [mats[k] for k in sorted(mats)]),
                coords.apply_transforms({f"ScanPos00{k + 1}": io.read(p) for k, p in enumerate(paths)}, mats)):
        for g, e in zip(got, expected, strict=True):
            np.testing.assert_array_equal(g.xyz, e)


def test_apply_transforms_from_files(tmp_path, scans):
    paths, mats = scans
    expected = _expected(paths, mats)
    # A transforms.json as SurveyResult.save writes it (unregistered scans skipped).
    payload = {"scans": [{"name": n, "registered": True, "world_from_scan": m.tolist()} for n, m in mats.items()]
               + [{"name": "ScanPos009", "registered": False, "world_from_scan": np.eye(4).tolist()}]}
    (tmp_path / "transforms.json").write_text(json.dumps(payload))
    # A directory of RiSCAN .DAT matrices.
    dat = tmp_path / "DAT"
    dat.mkdir()
    for n, m in mats.items():
        np.savetxt(dat / f"{n}.DAT", m, fmt="%.17g")
    for tf in (tmp_path / "transforms.json", dat):
        for g, e in zip(coords.apply_transforms(paths, tf), expected, strict=True):
            np.testing.assert_allclose(g.xyz, e, rtol=0, atol=1e-9)
    one = coords.apply_transforms(paths, dat / "ScanPos002.DAT")
    np.testing.assert_allclose(one[0].xyz, io.read(paths[0]).transform(mats["ScanPos002"]).xyz, atol=1e-9)


def test_apply_transforms_riscan_project(tmp_path, scans):
    paths, mats = scans
    root = tmp_path / "Demo.RiSCAN"
    root.mkdir()
    rows = "".join(
        f'<scanposition name="{n}" kind="PositionX"><singlescans/><sop name="SOP" kind="SOP">'
        f'<matrix rows="4" cols="4">{" ".join(f"{v:.17g}" for v in m.ravel())}</matrix></sop>'
        f"</scanposition>" for n, m in mats.items())
    (root / "project.rsp").write_text(
        f'<?xml version="1.0"?><project><name>Demo</name><scanpositions name="SCANS" kind="SCANS">'
        f"{rows}</scanpositions></project>")
    for g, e in zip(coords.apply_transforms(paths, root), _expected(paths, mats), strict=True):
        np.testing.assert_allclose(g.xyz, e, rtol=0, atol=1e-9)


def test_apply_transforms_merge_and_write(tmp_path, scans):
    paths, mats = scans
    merged = coords.apply_transforms(paths, mats, out=tmp_path / "out" / "merged.laz", merge=True)
    assert len(merged) == 600 and merged.attrs["scan_id"].dtype == np.int32
    assert np.bincount(merged.attrs["scan_id"]).tolist() == [200, 200, 200]
    np.testing.assert_allclose(io.read(tmp_path / "out" / "merged.laz").xyz, merged.xyz, atol=1e-3)
    clouds = coords.apply_transforms(paths, mats, out=tmp_path / "each")
    assert sorted(p.name for p in (tmp_path / "each").iterdir()) == [p.name for p in paths]
    np.testing.assert_allclose(io.read(tmp_path / "each" / paths[1].name).xyz, clouds[1].xyz, atol=1e-3)
    written = coords.apply_transforms([io.read(p) for p in paths], [np.eye(4)] * 3, out=tmp_path / "anon")
    assert len(written) == 3
    assert sorted(p.name for p in (tmp_path / "anon").iterdir()) == ["scan000.laz", "scan001.laz", "scan002.laz"]


def test_apply_transforms_errors(tmp_path, scans):
    paths, mats = scans
    with pytest.raises(ValueError, match="no transform"):
        coords.apply_transforms(paths, {"ScanPos001": np.eye(4)})
    with pytest.raises(ValueError, match="3 scans but 2"):
        coords.apply_transforms(paths, [np.eye(4)] * 2)
    with pytest.raises(ValueError, match="without names"):
        coords.apply_transforms([io.read(paths[0])], mats)
    with pytest.raises(ValueError, match="4x4"):
        coords.apply_transforms(paths[:1], {"ScanPos001": np.eye(3)})
    with pytest.raises(ValueError, match="no scans"):
        coords.apply_transforms([], [])
    with pytest.raises(ValueError, match="several"):
        coords.apply_transforms(paths[:1], {"ScanPos001": np.eye(4), "scanpos001": np.eye(4)})
    # ScanPos0011 must not take ScanPos001's matrix.
    p = paths[0].with_name("ScanPos0011.laz")
    paths[0].rename(p)
    with pytest.raises(ValueError, match="no transform"):
        coords.apply_transforms([p], {"ScanPos001": np.eye(4)})


def test_public_api_is_documented():
    import inspect

    names = list(coords.__all__) + ["PointCloud.translate", "PointCloud.rotate", "PointCloud.recentre"]
    for name in names:
        obj = getattr(PointCloud, name.split(".")[1]) if "." in name else getattr(coords, name)
        doc = inspect.getdoc(obj)
        assert doc, name
        if inspect.isfunction(obj) and len(inspect.signature(obj).parameters) > 1:
            assert "Parameters" in doc and "Returns" in doc, name


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def test_cli_reproject(tmp_path, projected, capsys):
    src = tmp_path / "plot.laz"
    io.write(projected, src)
    cli.main(["--no-progress", "reproject", str(src), str(tmp_path / "ll.laz"), "--to", "EPSG:7844"])
    assert "conversion" in capsys.readouterr().out
    out = io.read(tmp_path / "ll.laz")
    assert coords.crs_info(out.crs).epsg == 7844
    np.testing.assert_allclose(out.xyz[:, :2], coords.reproject(projected, 7844).xyz[:, :2], atol=1e-6)
    io.write(cloud_at(projected.xyz), tmp_path / "plain.ply")
    with pytest.raises(SystemExit):
        cli.main(["--no-progress", "reproject", str(tmp_path / "plain.ply"), str(tmp_path / "x.laz"),
                  "--to", "EPSG:7844"])
    assert "--from" in capsys.readouterr().err
    cli.main(["--no-progress", "reproject", str(tmp_path / "plain.ply"), str(tmp_path / "x.laz"),
              "--to", "EPSG:7844", "--from", "EPSG:7855"])
    cli.main(["--no-progress", "info", str(tmp_path / "x.laz")])
    assert "crs: EPSG:7844 (GDA2020)" in capsys.readouterr().out


def test_cli_transform(tmp_path, projected, capsys):
    src = tmp_path / "plot.laz"
    io.write(projected, src)
    ref = io.read(src)
    m = coords.rotation_matrix(20, "z") @ coords.translation_matrix(-512_000, -5_412_000, 0)
    np.savetxt(tmp_path / "m.DAT", m, fmt="%.17g")
    cases = {
        "matrix": (["--matrix", str(tmp_path / "m.DAT")], ref.transform(m)),
        "shift": (["--translate", "-512000", "-5412000", "-100"], ref.translate(-512_000, -5_412_000, -100)),
        "turn": (["--rotate", "90", "--axis", "z", "--about", "512000", "5412000", "0"],
                 ref.rotate(90, about=(512_000, 5_412_000, 0))),
    }
    for name, (opts, expected) in cases.items():
        cli.main(["--no-progress", "transform", str(src), str(tmp_path / f"{name}.laz"), *opts])
        got = io.read(tmp_path / f"{name}.laz")
        np.testing.assert_allclose(got.xyz, expected.xyz, atol=1e-3)
        assert got.crs == ref.crs
    assert "rotation by 90 degrees" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["transform", str(src), str(tmp_path / "x.laz")])       # no operation given
    with pytest.raises(SystemExit):
        cli.main(["transform", str(src), str(tmp_path / "x.laz"), "--rotate", "5", "--translate", "1", "2", "3"])
