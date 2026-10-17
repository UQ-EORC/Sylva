# Sylva: LiDAR processing for forest ecology and remote sensing research.
# Copyright (C) 2026 Tim Devereux, The University of Queensland.
# Free software under the GNU General Public License v3.0 or later;
# see the LICENSE file. There is no warranty, to the extent permitted by law.
"""RIEGL ``.rdbx`` reading checked against librdb itself.

The reference drives the same library through its C interface (``rdb_*``)
with ctypes: a select query with one buffer per attribute, in the type the
database suggests, read in the library's own order. The tests need the RDB
SDK and a database, which cannot be distributed: give the database with
``SYLVA_TEST_RDBX``, or the folder of RiSCAN projects with ``SYLVA_TERN_DIR``
(the smallest found there is used). Without either they are skipped.
"""

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from sylva import cli, io

# RDB buffer data type codes (dataTypes.h) as numpy dtypes.
DTYPES = {1: "u1", 2: "i1", 3: "u2", 4: "i2", 5: "u4", 6: "i4", 7: "u8", 8: "i8", 9: "f4", 10: "f8"}
# What read_rdbx returns for each (int16 and uint64 are widened).
RETURNED = {"int16": "int32", "uint64": "int64"}
RENAMES = {"riegl.target_index": "return_number", "riegl.target_count": "number_of_returns",
           "riegl.timestamp": "gps_time", "riegl.class": "classification", "riegl.id": "id"}
CHUNK = 1 << 18


def _find_db() -> Path | None:
    if os.environ.get("SYLVA_TEST_RDBX"):
        return Path(os.environ["SYLVA_TEST_RDBX"])
    tern = Path(os.environ.get("SYLVA_TERN_DIR", "TERN_TLS_RAW"))
    found = list(tern.glob("*.RiSCAN/project.rdb/SCANS/*/SINGLESCANS/*/*.rdbx"))
    return min(found, key=lambda p: p.stat().st_size) if found else None


def _find_library() -> Path | None:
    try:
        return io.find_rdblib()
    except ValueError:
        return None


DB = _find_db()
LIBRARY = _find_library()
pytestmark = [
    pytest.mark.skipif(LIBRARY is None,
                       reason="the RDB SDK (librdb) is not installed; set RDBLIB_PATH"),
    pytest.mark.skipif(DB is None or not DB.is_file(),
                       reason="no .rdbx: set SYLVA_TEST_RDBX or SYLVA_TERN_DIR"),
]


def sylva_name(rdb_name: str, length: int, k: int) -> str:
    """The column name read_rdbx documents for element ``k`` of an attribute."""
    if rdb_name in RENAMES:
        return RENAMES[rdb_name]
    if rdb_name in ("riegl.rgba", "riegl.rgb") and k < 4:
        return ["red", "green", "blue", "alpha"][k]
    base = rdb_name.removeprefix("riegl.").replace(".", "_")
    return f"{base}_{k}" if length > 1 else base


class Rdb:
    """An ``.rdbx`` opened through librdb's C interface with ctypes."""

    def __init__(self, library: Path, path: Path):
        self.lib = ctypes.CDLL(str(library))
        self.ctx = ctypes.c_void_p()
        self._ok(self.lib.rdb_context_new(ctypes.byref(self.ctx), b"", b""))
        self.pc = ctypes.c_void_p()
        self._ok(self.lib.rdb_pointcloud_new(self.ctx, ctypes.byref(self.pc)))
        self.settings = ctypes.c_void_p()
        self._ok(self.lib.rdb_pointcloud_open_settings_new(self.ctx, ctypes.byref(self.settings)))
        self._ok(self.lib.rdb_pointcloud_open(self.ctx, self.pc, str(path).encode(), self.settings))

    def _ok(self, rc):
        if rc != 1:
            code, text, details = ctypes.c_int32(), ctypes.c_char_p(), ctypes.c_char_p()
            self.lib.rdb_context_get_last_error(self.ctx, ctypes.byref(code), ctypes.byref(text),
                                                ctypes.byref(details))
            raise RuntimeError(f"librdb error {code.value}: {text.value} {details.value}")

    def _names(self, count: int, packed: ctypes.c_char_p) -> list[str]:
        address, out = ctypes.cast(packed, ctypes.c_void_p).value, []
        for _ in range(count):
            s = ctypes.string_at(address)
            out.append(s.decode())
            address += len(s) + 1
        return out

    def attributes(self) -> dict[str, tuple[int, int]]:
        """``name -> (length, data type code)`` in the database's order."""
        n, packed = ctypes.c_uint32(), ctypes.c_char_p()
        self._ok(self.lib.rdb_pointcloud_point_attributes_list(self.ctx, self.pc, ctypes.byref(n),
                                                               ctypes.byref(packed)))
        out = {}
        for name in self._names(n.value, packed):
            a = ctypes.c_void_p()
            self._ok(self.lib.rdb_pointcloud_point_attribute_new(self.ctx, ctypes.byref(a)))
            self._ok(self.lib.rdb_pointcloud_point_attributes_get(self.ctx, self.pc,
                                                                  name.encode(), a))
            length, code = ctypes.c_uint32(), ctypes.c_uint32()
            self._ok(self.lib.rdb_pointcloud_point_attribute_get_length_u32(
                self.ctx, a, ctypes.byref(length)))
            self._ok(self.lib.rdb_pointcloud_point_attribute_data_type(self.ctx, a,
                                                                       ctypes.byref(code)))
            self.lib.rdb_pointcloud_point_attribute_delete(self.ctx, ctypes.byref(a))
            out[name] = (length.value, code.value)
        return out

    def metadata(self) -> dict[str, str]:
        n, packed = ctypes.c_uint32(), ctypes.c_char_p()
        self._ok(self.lib.rdb_pointcloud_meta_data_list(self.ctx, self.pc, ctypes.byref(n),
                                                        ctypes.byref(packed)))
        out = {}
        for name in self._names(n.value, packed):
            value = ctypes.c_char_p()
            self._ok(self.lib.rdb_pointcloud_meta_data_get(self.ctx, self.pc, name.encode(),
                                                           ctypes.byref(value), b""))
            out[name] = value.value.decode()
        return out

    def count(self) -> int:
        class Node(ctypes.Structure):
            _pack_ = 1
            _fields_ = [("id", ctypes.c_uint32), ("revision", ctypes.c_uint32),
                        ("children", ctypes.c_void_p), ("child_count", ctypes.c_uint32),
                        ("total", ctypes.c_uint64), ("node", ctypes.c_uint64)]

        stat, root = ctypes.c_void_p(), ctypes.POINTER(Node)()
        self._ok(self.lib.rdb_pointcloud_query_stat_new(self.ctx, self.pc, ctypes.byref(stat)))
        self._ok(self.lib.rdb_pointcloud_graph_node_new(self.ctx, ctypes.byref(root)))
        self._ok(self.lib.rdb_pointcloud_query_stat_index(self.ctx, stat, root))
        total = root.contents.total
        self.lib.rdb_pointcloud_graph_node_delete(self.ctx, ctypes.byref(root))
        self.lib.rdb_pointcloud_query_stat_delete(self.ctx, ctypes.byref(stat))
        return total

    def head(self, n: int, names=None, filter: str = "") -> dict[str, np.ndarray]:
        """The first ``n`` points (at least) of a select, ``name -> (n, length)`` arrays."""
        schema = self.attributes()
        names = list(schema) if names is None else names
        query = ctypes.c_void_p()
        self._ok(self.lib.rdb_pointcloud_query_select_new(self.ctx, self.pc, 0, filter.encode(),
                                                          ctypes.byref(query)))
        buffers = {name: np.zeros((CHUNK, schema[name][0]), DTYPES[schema[name][1]])
                   for name in names}
        for name, buf in buffers.items():
            self._ok(self.lib.rdb_pointcloud_query_select_bind(
                self.ctx, query, name.encode(), schema[name][1],
                buf.ctypes.data_as(ctypes.c_void_p), 0))
        parts, got, total = {name: [] for name in names}, ctypes.c_uint32(), 0
        while total < n:
            self._ok(self.lib.rdb_pointcloud_query_select_next(self.ctx, query, CHUNK,
                                                               ctypes.byref(got)))
            if got.value == 0:
                break
            for name, buf in buffers.items():
                parts[name].append(buf[:got.value].copy())
            total += got.value
        self.lib.rdb_pointcloud_query_select_delete(self.ctx, ctypes.byref(query))
        return {name: np.concatenate(p) for name, p in parts.items()}

    def close(self):
        self.lib.rdb_pointcloud_close(self.ctx, self.pc)
        self.lib.rdb_pointcloud_open_settings_delete(self.ctx, ctypes.byref(self.settings))
        self.lib.rdb_pointcloud_delete(self.ctx, ctypes.byref(self.pc))
        self.lib.rdb_context_delete(ctypes.byref(self.ctx))


@pytest.fixture(scope="module")
def rdb():
    db = Rdb(LIBRARY, DB)
    yield db
    db.close()


@pytest.fixture(scope="module")
def head(rdb):
    """The first million points of the database, every attribute."""
    return rdb.head(1_000_000)


def test_info_matches_librdb(rdb):
    info = io.rdbx_info(DB)
    assert info.point_count == rdb.count() > 0
    assert info.primary_attribute == "riegl.xyz"
    schema = rdb.attributes()
    assert list(info.attributes) == list(schema)
    for name, (length, code) in schema.items():
        a = info.attributes[name]
        assert (a.length, a.dtype) == (length, np.dtype(DTYPES[code]).name), name
    xyz = info.attributes["riegl.xyz"]
    assert (xyz.unit, xyz.length, xyz.dtype) == ("m", 3, "float64")
    assert info.metadata == rdb.metadata()
    assert "device_type" in info.json("riegl.device")
    assert "0" in info.attributes["riegl.class"].named_values
    lo, hi = info.bounds
    assert lo.shape == hi.shape == (3,) and np.all(lo < hi)
    assert info.library_version.startswith("2.")


def test_every_attribute_as_librdb_reads_it(head):
    n = 300_000
    cloud = io.read_rdbx(DB, max_points=n)
    assert len(cloud) == n
    np.testing.assert_array_equal(cloud.xyz, head["riegl.xyz"][:n])
    expected = set()
    for name, values in head.items():
        if name == "riegl.xyz":
            continue
        length = values.shape[1]
        for k in range(length):
            column = sylva_name(name, length, k)
            expected.add(column)
            got = cloud.attrs[column]
            assert got.dtype == RETURNED.get(values.dtype.name, values.dtype.name), column
            np.testing.assert_array_equal(got, values[:n, k], err_msg=column)
    assert set(cloud.attrs) == expected
    # The LAS names are in use for the attributes LAS has a place for.
    las_names = {"return_number", "number_of_returns", "gps_time", "reflectance", "deviation"}
    assert las_names <= expected


def test_attributes_are_selected_by_either_name(head):
    names = ["reflectance", "riegl.deviation", "return_number", "riegl.xyz"]
    cloud = io.read_rdbx(DB, attributes=names, max_points=1000)
    assert set(cloud.attrs) == {"reflectance", "deviation", "return_number"}
    assert cloud.attrs["return_number"].dtype == np.uint8
    np.testing.assert_array_equal(cloud.attrs["return_number"],
                                  head["riegl.target_index"][:1000, 0])
    with pytest.raises(ValueError, match='no point attribute "riegl.nope"'):
        io.read_rdbx(DB, attributes=["nope"], max_points=10)


def test_filter_stride_and_max_points(rdb, head):
    ids = head["riegl.id"][:, 0]
    thinned = io.read_rdbx(DB, attributes=["id"], stride=7, max_points=1000)
    np.testing.assert_array_equal(thinned.attrs["id"], ids[6::7][:1000])
    assert len(io.read_rdbx(DB, attributes=["id"], max_points=12345)) == 12345

    # A filter changes the order the index is walked in, so the reference is
    # librdb's own answer to the same filter, not a mask on the unfiltered head.
    expr = "riegl.reflectance > 0"
    bright = io.read_rdbx(DB, attributes=["id", "reflectance"], filter=expr, max_points=5000)
    ref = rdb.head(5000, ["riegl.id", "riegl.reflectance"], filter=expr)
    assert len(bright) == 5000 and np.all(bright.attrs["reflectance"] > 0)
    np.testing.assert_array_equal(bright.attrs["id"], ref["riegl.id"][:5000, 0])
    np.testing.assert_array_equal(bright.attrs["reflectance"], ref["riegl.reflectance"][:5000, 0])
    with pytest.raises(ValueError, match="selecting points with filter"):
        io.read_rdbx(DB, filter="riegl.reflectance >>> 0", max_points=10)


def test_laz_round_trip_keeps_the_las_dimensions_and_the_rest_as_extra_bytes(tmp_path):
    cloud = io.read_rdbx(DB, max_points=20_000)
    path = tmp_path / "scan.laz"
    io.write(cloud, path)
    back = io.read(path)
    assert len(back) == len(cloud)
    np.testing.assert_allclose(back.xyz, cloud.xyz, atol=1e-3)
    for name in ("return_number", "number_of_returns", "classification"):
        np.testing.assert_array_equal(back.attrs[name], cloud.attrs[name])
    np.testing.assert_allclose(back.attrs["gps_time"], cloud.attrs["gps_time"])
    for name in ("reflectance", "amplitude", "deviation", "mta_zone", "id"):
        assert back.attrs[name].dtype == cloud.attrs[name].dtype, name
        np.testing.assert_array_equal(back.attrs[name], cloud.attrs[name])


def test_cli_info_reads_the_index_not_the_points(capsys):
    cli.main(["info", str(DB)])
    out = capsys.readouterr().out
    info = io.rdbx_info(DB)
    assert f"{info.point_count:,} points (RDB 2" in out
    assert "riegl.xyz[3]: float64" in out and "riegl.mta_settings" in out
