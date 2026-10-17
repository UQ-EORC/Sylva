// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! RIEGL `.rdbx` (RDB 2) reading through RIEGL's `librdb`, loaded at runtime.
//!
//! RDB 2 is the point database RiSCAN PRO writes when it imports a scan: the
//! echoes of one `.rxp` after MTA resolution and the import filters, stored
//! as typed columns (`riegl.xyz`, `riegl.reflectance`, `riegl.deviation`,
//! ...) with the processing settings beside them as JSON metadata
//! (`riegl.mta_settings`, `riegl.notch_filter`, `riegl.device`, ...).
//!
//! `librdb` is proprietary and cannot be redistributed. It is located from
//! `RDBLIB_PATH` / `RDBLIB_HOME`, an explicit path, or the usual install
//! locations, and driven through its C interface (`rdb_*` in `riegl/rdb.h`
//! of the RDB SDK). Points come back in the frame the database stores them
//! in: the scanner's own coordinate system for a scan position's database.

use std::collections::BTreeMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr::null_mut;
use std::sync::{Arc, Mutex, OnceLock};

use libloading::{Library, Symbol};

use crate::error::{Error, Result};
use crate::pointcloud::Attr;
use crate::{Point, PointCloud};

// Opaque handles of the C interface: a library context, a point cloud and
// the settings, attribute and query objects. `librdb` allocates them and
// hands back pointers we only pass around and finally delete.
type Ctx = *mut c_void;
type Pc = *mut c_void;
type Handle = *mut c_void;
/// `RDBString`: a NUL-terminated UTF-8 string owned by the library.
type RdbString = *const c_char;

/// Every function returns this on success; on failure the context holds the reason.
const RDB_SUCCESS: i32 = 1;

/// Buffer data types (`dataTypes.h`), as bound to a query and as the type
/// `librdb` suggests for an attribute from its resolution and range.
const DT_UINT8: u32 = 1;
const DT_INT8: u32 = 2;
const DT_UINT16: u32 = 3;
const DT_INT16: u32 = 4;
const DT_UINT32: u32 = 5;
const DT_INT32: u32 = 6;
const DT_UINT64: u32 = 7;
const DT_INT64: u32 = 8;
const DT_SINGLE: u32 = 9;
const DT_DOUBLE: u32 = 10;

/// Points read per `select_next` call.
const CHUNK: usize = 1 << 18;

/// Index graph node (`graphNode.h`). `#[repr(C, packed)]` because the header
/// declares it with `#pragma pack(1)`; the root node's `point_count_total`
/// is the database's point count. `children` is never followed.
#[repr(C, packed)]
struct GraphNode {
    _id: u32,
    _revision: u32,
    _children: *mut *mut GraphNode,
    _child_count: u32,
    point_count_total: u64,
    _point_count_node: u64,
}

// The function pointer types below are `librdb`'s C signatures written for
// Rust; nothing can check them against `riegl/rdb.h` but us. `unsafe extern
// "C" fn` is a pointer to a function with C's calling convention. The macro
// declares the struct holding them and the loader that resolves each by name.
macro_rules! api {
    ($($field:ident : $ty:ty = $sym:literal;)*) => {
        /// `librdb`'s C functions, resolved once per process.
        struct Api {
            _lib: Library,
            $($field: $ty,)*
        }

        impl Api {
            /// Resolve every function. SAFETY: the declared signatures must
            /// match `riegl/rdb.h`; they were checked against RDB SDK 2.4.1.
            unsafe fn load(lib: Library) -> Result<Api> {
                $(let $field: $ty = {
                    let s: Symbol<$ty> = lib.get(concat!($sym, "\0").as_bytes()).map_err(|e| Error::Rdb(format!("librdb has no {}: {e}", $sym)))?;
                    *s
                };)*
                Ok(Api { _lib: lib, $($field,)* })
            }
        }
    };
}

api! {
    context_new: unsafe extern "C" fn(*mut Ctx, RdbString, RdbString) -> i32 = "rdb_context_new";
    context_delete: unsafe extern "C" fn(*mut Ctx) -> i32 = "rdb_context_delete";
    context_get_last_error: unsafe extern "C" fn(Ctx, *mut i32, *mut RdbString, *mut RdbString) -> i32 = "rdb_context_get_last_error";
    library_version: unsafe extern "C" fn(Ctx, *mut RdbString) -> i32 = "rdb_library_version";
    pointcloud_new: unsafe extern "C" fn(Ctx, *mut Pc) -> i32 = "rdb_pointcloud_new";
    pointcloud_delete: unsafe extern "C" fn(Ctx, *mut Pc) -> i32 = "rdb_pointcloud_delete";
    pointcloud_open: unsafe extern "C" fn(Ctx, Pc, RdbString, Handle) -> i32 = "rdb_pointcloud_open";
    pointcloud_close: unsafe extern "C" fn(Ctx, Pc) -> i32 = "rdb_pointcloud_close";
    open_settings_new: unsafe extern "C" fn(Ctx, *mut Handle) -> i32 = "rdb_pointcloud_open_settings_new";
    open_settings_delete: unsafe extern "C" fn(Ctx, *mut Handle) -> i32 = "rdb_pointcloud_open_settings_delete";
    attributes_list: unsafe extern "C" fn(Ctx, Pc, *mut u32, *mut RdbString) -> i32 = "rdb_pointcloud_point_attributes_list";
    attributes_get: unsafe extern "C" fn(Ctx, Pc, RdbString, Handle) -> i32 = "rdb_pointcloud_point_attributes_get";
    attributes_primary: unsafe extern "C" fn(Ctx, Pc, *mut RdbString) -> i32 = "rdb_pointcloud_point_attributes_primary_attribute_name";
    attribute_new: unsafe extern "C" fn(Ctx, *mut Handle) -> i32 = "rdb_pointcloud_point_attribute_new";
    attribute_delete: unsafe extern "C" fn(Ctx, *mut Handle) -> i32 = "rdb_pointcloud_point_attribute_delete";
    attribute_data_type: unsafe extern "C" fn(Ctx, Handle, *mut u32) -> i32 = "rdb_pointcloud_point_attribute_data_type";
    attribute_length: unsafe extern "C" fn(Ctx, Handle, *mut u32) -> i32 = "rdb_pointcloud_point_attribute_get_length_u32";
    attribute_title: unsafe extern "C" fn(Ctx, Handle, *mut RdbString) -> i32 = "rdb_pointcloud_point_attribute_get_title";
    attribute_unit: unsafe extern "C" fn(Ctx, Handle, *mut RdbString) -> i32 = "rdb_pointcloud_point_attribute_get_unit_symbol";
    attribute_named_values: unsafe extern "C" fn(Ctx, Handle, *mut RdbString) -> i32 = "rdb_pointcloud_point_attribute_get_named_values";
    attribute_resolution: unsafe extern "C" fn(Ctx, Handle, *mut f64) -> i32 = "rdb_pointcloud_point_attribute_get_resolution";
    attribute_minimum: unsafe extern "C" fn(Ctx, Handle, *mut f64) -> i32 = "rdb_pointcloud_point_attribute_get_minimum_value";
    attribute_maximum: unsafe extern "C" fn(Ctx, Handle, *mut f64) -> i32 = "rdb_pointcloud_point_attribute_get_maximum_value";
    attribute_invalid: unsafe extern "C" fn(Ctx, Handle, *mut f64) -> i32 = "rdb_pointcloud_point_attribute_get_invalid_value";
    meta_data_list: unsafe extern "C" fn(Ctx, Pc, *mut u32, *mut RdbString) -> i32 = "rdb_pointcloud_meta_data_list";
    meta_data_get: unsafe extern "C" fn(Ctx, Pc, RdbString, *mut RdbString, RdbString) -> i32 = "rdb_pointcloud_meta_data_get";
    stat_new: unsafe extern "C" fn(Ctx, Pc, *mut Handle) -> i32 = "rdb_pointcloud_query_stat_new";
    stat_delete: unsafe extern "C" fn(Ctx, *mut Handle) -> i32 = "rdb_pointcloud_query_stat_delete";
    stat_index: unsafe extern "C" fn(Ctx, Handle, *mut GraphNode) -> i32 = "rdb_pointcloud_query_stat_index";
    stat_minimum: unsafe extern "C" fn(Ctx, Handle, u32, RdbString, u32, *mut c_void) -> i32 = "rdb_pointcloud_query_stat_minimum";
    stat_maximum: unsafe extern "C" fn(Ctx, Handle, u32, RdbString, u32, *mut c_void) -> i32 = "rdb_pointcloud_query_stat_maximum";
    graph_node_new: unsafe extern "C" fn(Ctx, *mut *mut GraphNode) -> i32 = "rdb_pointcloud_graph_node_new";
    graph_node_delete: unsafe extern "C" fn(Ctx, *mut *mut GraphNode) -> i32 = "rdb_pointcloud_graph_node_delete";
    select_new: unsafe extern "C" fn(Ctx, Pc, u32, RdbString, *mut Handle) -> i32 = "rdb_pointcloud_query_select_new";
    select_delete: unsafe extern "C" fn(Ctx, *mut Handle) -> i32 = "rdb_pointcloud_query_select_delete";
    select_bind: unsafe extern "C" fn(Ctx, Handle, RdbString, u32, *mut c_void, i32) -> i32 = "rdb_pointcloud_query_select_bind";
    select_next: unsafe extern "C" fn(Ctx, Handle, u32, *mut u32) -> i32 = "rdb_pointcloud_query_select_next";
}

/// The library, loaded once and kept for the life of the process.
///
/// Unlike RiVLib's stream reader, the resolved functions are shared through
/// an `Arc` and the lock is held only while loading, so scans can be read on
/// several threads at once: `librdb` allows one point cloud object per
/// thread, and each read here opens its own.
static API: OnceLock<Mutex<Option<Arc<Api>>>> = OnceLock::new();

/// Directories searched for `librdb`, in order.
pub fn search_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for var in ["RDBLIB_PATH", "RDBLIB_HOME"] {
        if let Ok(v) = std::env::var(var) {
            roots.push(PathBuf::from(v));
        }
    }
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        roots.push(home.join(".local/lib"));
        roots.push(home.join(".local"));
        roots.push(home.join("lib"));
        roots.push(home.join("opt"));
        roots.push(home);
    }
    roots.push(PathBuf::from("/opt"));
    roots.push(PathBuf::from("/usr/local/lib"));
    roots.push(PathBuf::from("/usr/local"));
    roots.push(PathBuf::from("/usr/lib"));
    roots
}

const LIB_NAMES: &[&str] = &["librdb.so", "librdb.so.2", "rdblib.dll", "librdb.dylib"];

fn search_dir(dir: &Path, depth: usize) -> Option<PathBuf> {
    if !dir.is_dir() {
        return None;
    }
    for name in LIB_NAMES {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    if depth == 0 {
        return None;
    }
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    entries.sort();
    for e in entries {
        let n = e.file_name().and_then(|n| n.to_str()).unwrap_or("").to_ascii_lowercase();
        // The SDK unpacks to rdblib-<version>-<platform>/library/librdb.so.
        if n.contains("rdb") || n == "lib" || n == "library" || depth >= 2 {
            if let Some(p) = search_dir(&e, depth - 1) {
                return Some(p);
            }
        }
    }
    None
}

/// Locate `librdb`.
pub fn find_rdblib(hint: Option<&Path>) -> Result<PathBuf> {
    if let Some(h) = hint {
        if h.is_file() {
            return Ok(h.to_path_buf());
        }
        if let Some(p) = search_dir(h, 3) {
            return Ok(p);
        }
    }
    for root in search_roots() {
        if root.is_file() {
            return Ok(root);
        }
        if let Some(p) = search_dir(&root, 3) {
            return Ok(p);
        }
    }
    Err(Error::Rdb(
        "librdb not found. Download the RDB SDK (rdblib) from RIEGL and set RDBLIB_PATH to the extracted \
         directory (containing library/librdb.so)"
            .into(),
    ))
}

/// The loaded library, loading it the first time.
fn api(hint: Option<&Path>) -> Result<Arc<Api>> {
    let cell = API.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap();
    if guard.is_none() {
        let path = find_rdblib(hint)?;
        // SAFETY: librdb is a well-behaved C library with no init side effects.
        let lib = unsafe { Library::new(&path) }.map_err(|e| Error::Rdb(format!("{}: {e}", path.display())))?;
        // SAFETY: see `Api::load`.
        *guard = Some(Arc::new(unsafe { Api::load(lib) }?));
    }
    Ok(guard.as_ref().unwrap().clone())
}

/// Copy a library-owned string (null is empty).
fn text(p: RdbString) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: librdb returns NUL-terminated strings that live until the next call.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// Copy a list of `count` strings packed one after the other, each NUL-terminated.
fn text_list(p: RdbString, count: u32) -> Vec<String> {
    let mut out = Vec::with_capacity(count as usize);
    if p.is_null() {
        return out;
    }
    let mut cursor = p;
    for _ in 0..count {
        // SAFETY: the library packs exactly `count` NUL-terminated strings.
        let s = unsafe { CStr::from_ptr(cursor) };
        out.push(s.to_string_lossy().into_owned());
        cursor = cursor.wrapping_add(s.to_bytes_with_nul().len());
    }
    out
}

fn c_string(s: &str, what: &str) -> Result<CString> {
    CString::new(s).map_err(|_| Error::Rdb(format!("{what} contains a NUL byte: {s:?}")))
}

/// numpy-style name of an RDB buffer data type.
fn dtype_name(code: u32) -> &'static str {
    match code {
        DT_UINT8 => "uint8",
        DT_INT8 => "int8",
        DT_UINT16 => "uint16",
        DT_INT16 => "int16",
        DT_UINT32 => "uint32",
        DT_INT32 => "int32",
        DT_UINT64 => "uint64",
        DT_INT64 => "int64",
        DT_SINGLE => "float32",
        DT_DOUBLE => "float64",
        _ => "unknown",
    }
}

/// One point attribute as the database describes it.
#[derive(Debug, Clone, PartialEq)]
pub struct RdbxAttribute {
    /// Unique name with its namespace, e.g. `riegl.reflectance`.
    pub name: String,
    /// Display title, e.g. `Reflectance`.
    pub title: String,
    /// Physical unit symbol (`m`, `dB`, `s`, or empty).
    pub unit: String,
    /// Elements per point: 1 for a scalar, 3 for `riegl.xyz`.
    pub length: u32,
    /// Data type `librdb` suggests for a buffer, numpy style (`float32`,
    /// `int16`, ...). The column [`read_rdbx`] returns may be wider, see there.
    pub dtype: &'static str,
    /// Resolution the values are stored with.
    pub resolution: f64,
    pub minimum: f64,
    pub maximum: f64,
    /// Value standing for "no value", if the attribute defines one.
    pub invalid: Option<f64>,
    /// `(value, name)` pairs of an enumerated attribute, e.g. `riegl.class`.
    pub named_values: Vec<(String, String)>,
    /// The suggested type's RDB code, kept for binding buffers.
    code: u32,
}

/// What a database holds besides its points: see [`rdbx_info`].
#[derive(Debug, Clone)]
pub struct RdbxInfo {
    /// Version string of the `librdb` that read the file.
    pub library_version: String,
    pub point_count: u64,
    /// The attribute the spatial index is built on, usually `riegl.xyz`.
    pub primary_attribute: String,
    /// Every point attribute, in the database's own group order.
    pub attributes: Vec<RdbxAttribute>,
    /// Metadata entries by name; the values of RIEGL's entries are JSON.
    pub metadata: BTreeMap<String, String>,
    /// Extent of the coordinates, from the index statistics.
    pub bounds: Option<(Point, Point)>,
}

/// An open database: a context and a point cloud object, released in order
/// when dropped, so an early `?` leaves nothing behind.
struct Db {
    api: Arc<Api>,
    ctx: Ctx,
    pc: Pc,
    settings: Handle,
    path: PathBuf,
}

impl Db {
    fn open(api: Arc<Api>, path: &Path) -> Result<Db> {
        let mut ctx: Ctx = null_mut();
        // SAFETY: out-pointer; empty log level and path select the defaults (context.h).
        if unsafe { (api.context_new)(&mut ctx, b"\0".as_ptr() as RdbString, b"\0".as_ptr() as RdbString) } != RDB_SUCCESS || ctx.is_null() {
            return Err(Error::Rdb("could not create a library context".into()));
        }
        let mut db = Db { api, ctx, pc: null_mut(), settings: null_mut(), path: path.to_path_buf() };
        // SAFETY: valid context and out-pointers; the handles are deleted in Drop.
        let rc = unsafe { (db.api.pointcloud_new)(db.ctx, &mut db.pc) };
        db.check(rc, "creating the point cloud object")?;
        // SAFETY: as above.
        let rc = unsafe { (db.api.open_settings_new)(db.ctx, &mut db.settings) };
        db.check(rc, "creating the open settings")?;
        let location = c_string(&path.to_string_lossy(), "the path")?;
        // SAFETY: valid handles and C string.
        db.check(unsafe { (db.api.pointcloud_open)(db.ctx, db.pc, location.as_ptr(), db.settings) }, "opening")?;
        Ok(db)
    }

    /// `Ok` for a successful return code, otherwise the context's last error
    /// as a message naming the file and the step.
    fn check(&self, rc: i32, what: &str) -> Result<()> {
        if rc == RDB_SUCCESS {
            return Ok(());
        }
        let (mut code, mut msg, mut details) = (0i32, null_mut::<c_char>() as RdbString, null_mut::<c_char>() as RdbString);
        // SAFETY: valid context and out-pointers.
        unsafe { (self.api.context_get_last_error)(self.ctx, &mut code, &mut msg, &mut details) };
        let (msg, details) = (text(msg), text(details));
        let details = if details.is_empty() { String::new() } else { format!(": {details}") };
        Err(Error::Rdb(format!("{}: {what}: {msg}{details} (error {code})", self.path.display())))
    }

    fn library_version(&self) -> Result<String> {
        let mut v: RdbString = null_mut();
        // SAFETY: valid context and out-pointer.
        self.check(unsafe { (self.api.library_version)(self.ctx, &mut v) }, "querying the library version")?;
        Ok(text(v))
    }

    fn primary_attribute(&self) -> Result<String> {
        let mut v: RdbString = null_mut();
        // SAFETY: valid handles and out-pointer.
        self.check(unsafe { (self.api.attributes_primary)(self.ctx, self.pc, &mut v) }, "querying the primary attribute")?;
        Ok(text(v))
    }

    /// Every point attribute, described.
    fn attributes(&self) -> Result<Vec<RdbxAttribute>> {
        let (mut count, mut list): (u32, RdbString) = (0, null_mut());
        // SAFETY: valid handles and out-pointers.
        self.check(unsafe { (self.api.attributes_list)(self.ctx, self.pc, &mut count, &mut list) }, "listing the point attributes")?;
        let names = text_list(list, count);
        let mut out = Vec::with_capacity(names.len());
        for name in names {
            let mut h: Handle = null_mut();
            // SAFETY: valid context and out-pointer; deleted below on both paths.
            self.check(unsafe { (self.api.attribute_new)(self.ctx, &mut h) }, "creating an attribute object")?;
            let described = self.describe(h, &name);
            // SAFETY: `h` came from attribute_new above.
            unsafe { (self.api.attribute_delete)(self.ctx, &mut h) };
            out.push(described?);
        }
        Ok(out)
    }

    fn describe(&self, h: Handle, name: &str) -> Result<RdbxAttribute> {
        let cname = c_string(name, "the attribute name")?;
        let what = format!("describing {name}");
        // SAFETY: valid handles, C string and out-pointers, each of the declared type.
        unsafe {
            self.check((self.api.attributes_get)(self.ctx, self.pc, cname.as_ptr(), h), &what)?;
            let (mut code, mut length) = (0u32, 0u32);
            let (mut title, mut unit, mut named): (RdbString, RdbString, RdbString) = (null_mut(), null_mut(), null_mut());
            let (mut resolution, mut minimum, mut maximum, mut invalid) = (0f64, 0f64, 0f64, f64::NAN);
            self.check((self.api.attribute_data_type)(self.ctx, h, &mut code), &what)?;
            self.check((self.api.attribute_length)(self.ctx, h, &mut length), &what)?;
            self.check((self.api.attribute_title)(self.ctx, h, &mut title), &what)?;
            let title = text(title);
            self.check((self.api.attribute_unit)(self.ctx, h, &mut unit), &what)?;
            let unit = text(unit);
            self.check((self.api.attribute_named_values)(self.ctx, h, &mut named), &what)?;
            let named = text(named);
            self.check((self.api.attribute_resolution)(self.ctx, h, &mut resolution), &what)?;
            self.check((self.api.attribute_minimum)(self.ctx, h, &mut minimum), &what)?;
            self.check((self.api.attribute_maximum)(self.ctx, h, &mut maximum), &what)?;
            self.check((self.api.attribute_invalid)(self.ctx, h, &mut invalid), &what)?;
            // "VALUE=NAME" pairs, one per line.
            let named_values = named.lines().filter_map(|l| l.split_once('=')).map(|(v, n)| (v.trim().to_string(), n.trim().to_string())).collect();
            Ok(RdbxAttribute { name: name.to_string(), title, unit, length: length.max(1), dtype: dtype_name(code), resolution, minimum, maximum, invalid: (!invalid.is_nan()).then_some(invalid), named_values, code })
        }
    }

    fn metadata(&self) -> Result<BTreeMap<String, String>> {
        let (mut count, mut list): (u32, RdbString) = (0, null_mut());
        // SAFETY: valid handles and out-pointers.
        self.check(unsafe { (self.api.meta_data_list)(self.ctx, self.pc, &mut count, &mut list) }, "listing the metadata")?;
        let mut out = BTreeMap::new();
        for name in text_list(list, count) {
            let cname = c_string(&name, "the metadata name")?;
            let mut value: RdbString = null_mut();
            // SAFETY: valid handles, C strings and out-pointer; "" is the default for a missing entry.
            self.check(unsafe { (self.api.meta_data_get)(self.ctx, self.pc, cname.as_ptr(), &mut value, b"\0".as_ptr() as RdbString) }, &format!("reading metadata {name}"))?;
            out.insert(name, text(value));
        }
        Ok(out)
    }

    /// Point count and, for a 3-vector attribute, its extent, from the index statistics.
    fn statistics(&self, extent_of: Option<&str>) -> Result<(u64, Option<(Point, Point)>)> {
        let mut stat: Handle = null_mut();
        let mut root: *mut GraphNode = null_mut();
        // SAFETY: valid handles and out-pointers; both objects are deleted below on every path.
        let result = unsafe {
            self.check((self.api.stat_new)(self.ctx, self.pc, &mut stat), "creating the statistics query").and_then(|_| {
                self.check((self.api.graph_node_new)(self.ctx, &mut root), "creating the index root node")?;
                self.check((self.api.stat_index)(self.ctx, stat, root), "reading the index statistics")?;
                let count = (*root).point_count_total;
                let bounds = match extent_of {
                    Some(name) => {
                        let cname = c_string(name, "the attribute name")?;
                        let (mut lo, mut hi) = ([0f64; 3], [0f64; 3]);
                        self.check((self.api.stat_minimum)(self.ctx, stat, 0, cname.as_ptr(), DT_DOUBLE, lo.as_mut_ptr() as *mut c_void), &format!("reading the minimum of {name}"))?;
                        self.check((self.api.stat_maximum)(self.ctx, stat, 0, cname.as_ptr(), DT_DOUBLE, hi.as_mut_ptr() as *mut c_void), &format!("reading the maximum of {name}"))?;
                        Some((lo, hi))
                    }
                    None => None,
                };
                Ok((count, bounds))
            })
        };
        // SAFETY: null handles are skipped; the others came from the calls above.
        unsafe {
            if !root.is_null() {
                (self.api.graph_node_delete)(self.ctx, &mut root);
            }
            if !stat.is_null() {
                (self.api.stat_delete)(self.ctx, &mut stat);
            }
        }
        result
    }

    fn select(&self, filter: &str) -> Result<Select<'_>> {
        let cfilter = c_string(filter, "the filter")?;
        let what = if filter.is_empty() { "selecting points".to_string() } else { format!("selecting points with filter {filter:?}") };
        let mut h: Handle = null_mut();
        // SAFETY: valid handles and C string; node 0 is the whole index.
        self.check(unsafe { (self.api.select_new)(self.ctx, self.pc, 0, cfilter.as_ptr(), &mut h) }, &what)?;
        Ok(Select { db: self, h })
    }
}

impl Drop for Db {
    fn drop(&mut self) {
        // SAFETY: each handle is either null or was created by the matching `_new`.
        unsafe {
            if !self.pc.is_null() {
                (self.api.pointcloud_close)(self.ctx, self.pc);
                (self.api.pointcloud_delete)(self.ctx, &mut self.pc);
            }
            if !self.settings.is_null() {
                (self.api.open_settings_delete)(self.ctx, &mut self.settings);
            }
            (self.api.context_delete)(&mut self.ctx);
        }
    }
}

/// A select query: buffers bound with [`Select::bind`] are filled by
/// [`Select::next`], a chunk at a time, in index order.
struct Select<'a> {
    db: &'a Db,
    h: Handle,
}

impl Select<'_> {
    /// Bind `buffer` (room for `CHUNK` points of `data_type`, times the
    /// attribute's length) to `attribute`. The buffer must stay where it is
    /// until the query is dropped: the library keeps only the pointer.
    fn bind(&self, attribute: &str, data_type: u32, buffer: *mut c_void) -> Result<()> {
        let cname = c_string(attribute, "the attribute name")?;
        // SAFETY: the caller keeps `buffer` alive and large enough; stride 0 is "packed".
        self.db.check(unsafe { (self.db.api.select_bind)(self.db.ctx, self.h, cname.as_ptr(), data_type, buffer, 0) }, &format!("binding {attribute}"))
    }

    /// Fill the bound buffers with up to `CHUNK` points; 0 when there are no more.
    fn next(&self) -> Result<usize> {
        let mut got = 0u32;
        // SAFETY: valid handles; the bound buffers hold CHUNK points each.
        self.db.check(unsafe { (self.db.api.select_next)(self.db.ctx, self.h, CHUNK as u32, &mut got) }, "reading points")?;
        Ok(got as usize)
    }
}

impl Drop for Select<'_> {
    fn drop(&mut self) {
        // SAFETY: `h` came from select_new.
        unsafe { (self.db.api.select_delete)(self.db.ctx, &mut self.h) };
    }
}

/// One bound attribute while reading: the chunk buffer `librdb` fills and
/// the columns kept so far, one per vector element.
struct Column<T> {
    buf: Vec<T>,
    out: Vec<Vec<T>>,
    len: usize,
}

impl<T: Copy + Default> Column<T> {
    fn new(len: usize, capacity: usize) -> Self {
        Column { buf: vec![T::default(); CHUNK * len], out: (0..len).map(|_| Vec::with_capacity(capacity)).collect(), len }
    }

    /// Keep the given rows of the current chunk.
    fn take(&mut self, rows: &[usize]) {
        for &i in rows {
            for k in 0..self.len {
                self.out[k].push(self.buf[i * self.len + k]);
            }
        }
    }
}

/// A column of whichever type the attribute suggests. Each case holds a
/// whole [`Column`]; the `match` in every method is the one place the
/// types are told apart.
enum Col {
    U8(Column<u8>),
    I8(Column<i8>),
    U16(Column<u16>),
    I16(Column<i16>),
    U32(Column<u32>),
    I32(Column<i32>),
    U64(Column<u64>),
    I64(Column<i64>),
    F32(Column<f32>),
    F64(Column<f64>),
}

macro_rules! col_dispatch {
    ($self:expr, $c:ident => $body:expr) => {
        match $self {
            Col::U8($c) => $body,
            Col::I8($c) => $body,
            Col::U16($c) => $body,
            Col::I16($c) => $body,
            Col::U32($c) => $body,
            Col::I32($c) => $body,
            Col::U64($c) => $body,
            Col::I64($c) => $body,
            Col::F32($c) => $body,
            Col::F64($c) => $body,
        }
    };
}

impl Col {
    fn new(code: u32, len: usize, capacity: usize) -> Col {
        match code {
            DT_UINT8 => Col::U8(Column::new(len, capacity)),
            DT_INT8 => Col::I8(Column::new(len, capacity)),
            DT_UINT16 => Col::U16(Column::new(len, capacity)),
            DT_INT16 => Col::I16(Column::new(len, capacity)),
            DT_UINT32 => Col::U32(Column::new(len, capacity)),
            DT_INT32 => Col::I32(Column::new(len, capacity)),
            DT_UINT64 => Col::U64(Column::new(len, capacity)),
            DT_INT64 => Col::I64(Column::new(len, capacity)),
            DT_SINGLE => Col::F32(Column::new(len, capacity)),
            // DOUBLE, and anything unknown: the widest type loses nothing.
            _ => Col::F64(Column::new(len, capacity)),
        }
    }

    /// The RDB data type the buffer is bound as.
    fn data_type(&self) -> u32 {
        match self {
            Col::U8(_) => DT_UINT8,
            Col::I8(_) => DT_INT8,
            Col::U16(_) => DT_UINT16,
            Col::I16(_) => DT_INT16,
            Col::U32(_) => DT_UINT32,
            Col::I32(_) => DT_INT32,
            Col::U64(_) => DT_UINT64,
            Col::I64(_) => DT_INT64,
            Col::F32(_) => DT_SINGLE,
            Col::F64(_) => DT_DOUBLE,
        }
    }

    fn buffer(&mut self) -> *mut c_void {
        col_dispatch!(self, c => c.buf.as_mut_ptr() as *mut c_void)
    }

    fn take(&mut self, rows: &[usize]) {
        col_dispatch!(self, c => c.take(rows))
    }

    /// The kept columns as attributes, one per vector element. `int16`
    /// widens to `int32` and `uint64` to `int64`, the nearest types a
    /// [`PointCloud`] carries (RIEGL's 64-bit ids stay below 2^63).
    fn into_attrs(self) -> Vec<Attr> {
        match self {
            Col::U8(c) => c.out.into_iter().map(Attr::U8).collect(),
            Col::I8(c) => c.out.into_iter().map(Attr::I8).collect(),
            Col::U16(c) => c.out.into_iter().map(Attr::U16).collect(),
            Col::I16(c) => c.out.into_iter().map(|v| Attr::I32(v.into_iter().map(i32::from).collect())).collect(),
            Col::U32(c) => c.out.into_iter().map(Attr::U32).collect(),
            Col::I32(c) => c.out.into_iter().map(Attr::I32).collect(),
            Col::U64(c) => c.out.into_iter().map(|v| Attr::I64(v.into_iter().map(|x| x as i64).collect())).collect(),
            Col::I64(c) => c.out.into_iter().map(Attr::I64).collect(),
            Col::F32(c) => c.out.into_iter().map(Attr::F32).collect(),
            Col::F64(c) => c.out.into_iter().map(Attr::F64).collect(),
        }
    }
}

/// RDB attributes that are LAS standard dimensions, and the names the LAS
/// reader and writer use for them, so a converted file gets proper return
/// numbers and GPS times rather than extra bytes.
const RENAMES: &[(&str, &str)] = &[
    ("riegl.target_index", "return_number"),
    ("riegl.target_count", "number_of_returns"),
    ("riegl.timestamp", "gps_time"),
    ("riegl.class", "classification"),
    ("riegl.id", "id"),
];

/// Attribute name on the Sylva side: the [`RENAMES`] above, `riegl.rgba` as
/// `red`/`green`/`blue`/`alpha`, otherwise the RDB name without its `riegl.`
/// namespace and with dots as underscores, plus `_k` for element `k` of a vector.
pub fn attribute_name(rdb_name: &str, length: u32, k: usize) -> String {
    if let Some((_, name)) = RENAMES.iter().find(|(r, _)| *r == rdb_name) {
        return name.to_string();
    }
    if matches!(rdb_name, "riegl.rgba" | "riegl.rgb") {
        if let Some(c) = ["red", "green", "blue", "alpha"].get(k) {
            return c.to_string();
        }
    }
    let base = rdb_name.strip_prefix("riegl.").unwrap_or(rdb_name).replace('.', "_");
    if length > 1 {
        format!("{base}_{k}")
    } else {
        base
    }
}

/// The RDB name of an attribute asked for by either name: `reflectance`,
/// `riegl.reflectance` and `return_number` all resolve.
pub fn rdb_attribute_name(name: &str) -> String {
    if let Some((rdb, _)) = RENAMES.iter().find(|(_, s)| *s == name) {
        return rdb.to_string();
    }
    if name.contains('.') {
        name.to_string()
    } else {
        format!("riegl.{name}")
    }
}

/// Options for [`read_rdbx`].
#[derive(Debug, Clone)]
pub struct RdbxOptions {
    /// Explicit `librdb` directory or library file.
    pub library: Option<PathBuf>,
    /// Attributes to read besides the coordinates, by RDB or Sylva name
    /// (see [`rdb_attribute_name`]); `None` reads them all.
    pub attributes: Option<Vec<String>>,
    /// RDB filter expression in the library's own syntax, with RDB names,
    /// e.g. `riegl.reflectance > -10 && riegl.deviation < 20`.
    pub filter: Option<String>,
    /// Keep every `stride`-th point (1 = all).
    pub stride: usize,
    /// Read at most this many points.
    pub max_points: Option<usize>,
}

impl Default for RdbxOptions {
    fn default() -> Self {
        RdbxOptions { library: None, attributes: None, filter: None, stride: 1, max_points: None }
    }
}

/// The attribute holding the coordinates: `riegl.xyz`, or `riegl.xyz_socs`
/// in a database that keeps only the scanner frame.
fn coordinates<'a>(schema: &'a [RdbxAttribute]) -> Option<&'a RdbxAttribute> {
    ["riegl.xyz", "riegl.xyz_socs"].iter().find_map(|n| schema.iter().find(|a| a.name == *n && a.length == 3))
}

fn available(schema: &[RdbxAttribute]) -> String {
    schema.iter().map(|a| a.name.as_str()).collect::<Vec<_>>().join(", ")
}

/// Read an `.rdbx` database into a point cloud: the coordinates of
/// `riegl.xyz` and every other attribute as a column named by
/// [`attribute_name`], in the type the database suggests (see
/// [`Col::into_attrs`] for the two that widen).
pub fn read_rdbx(path: impl AsRef<Path>, opts: &RdbxOptions) -> Result<PointCloud> {
    let path = path.as_ref();
    let db = Db::open(api(opts.library.as_deref())?, path)?;
    let schema = db.attributes()?;
    let coords = coordinates(&schema).ok_or_else(|| Error::Rdb(format!("{}: no riegl.xyz point attribute (has {})", path.display(), available(&schema))))?;
    let selected: Vec<&RdbxAttribute> = match &opts.attributes {
        None => schema.iter().filter(|a| a.name != coords.name).collect(),
        Some(wanted) => {
            let mut out: Vec<&RdbxAttribute> = Vec::new();
            for w in wanted {
                let full = rdb_attribute_name(w);
                let a = schema.iter().find(|a| a.name == full).ok_or_else(|| Error::Rdb(format!("{}: no point attribute {full:?} (has {})", path.display(), available(&schema))))?;
                if a.name != coords.name && !out.iter().any(|o| o.name == a.name) {
                    out.push(a);
                }
            }
            out
        }
    };
    let stride = opts.stride.max(1);
    let (total, _) = db.statistics(None)?;
    // Room for what will be kept; a filter leaves the count unknown, so the
    // columns grow as they go rather than reserving the whole file.
    let mut capacity = if opts.filter.is_some() { 0 } else { (total as usize).div_ceil(stride) };
    if let Some(m) = opts.max_points {
        capacity = capacity.min(m);
    }
    let mut xyz: Vec<Point> = Vec::with_capacity(capacity);
    let mut cols: Vec<Col> = selected.iter().map(|a| Col::new(a.code, a.length as usize, capacity)).collect();
    let mut xyz_buf = vec![0f64; CHUNK * 3];
    {
        let query = db.select(opts.filter.as_deref().unwrap_or(""))?;
        query.bind(&coords.name, DT_DOUBLE, xyz_buf.as_mut_ptr() as *mut c_void)?;
        for (a, c) in selected.iter().zip(cols.iter_mut()) {
            query.bind(&a.name, c.data_type(), c.buffer())?;
        }
        let mut rows: Vec<usize> = Vec::with_capacity(CHUNK);
        let mut counter = 0usize;
        let mut done = false;
        while !done {
            let got = query.next()?;
            if got == 0 {
                break;
            }
            rows.clear();
            for i in 0..got {
                counter += 1;
                if stride > 1 && counter % stride != 0 {
                    continue;
                }
                rows.push(i);
                if opts.max_points.is_some_and(|m| xyz.len() + rows.len() >= m) {
                    done = true;
                    break;
                }
            }
            for &i in &rows {
                xyz.push([xyz_buf[3 * i], xyz_buf[3 * i + 1], xyz_buf[3 * i + 2]]);
            }
            for c in &mut cols {
                c.take(&rows);
            }
        }
    }
    let mut attrs = BTreeMap::new();
    for (a, c) in selected.iter().zip(cols) {
        for (k, col) in c.into_attrs().into_iter().enumerate() {
            let mut name = attribute_name(&a.name, a.length, k);
            if attrs.contains_key(&name) {
                // Two namespaces mapping to one name: keep the second under its full name.
                let element = if a.length > 1 { format!("_{k}") } else { String::new() };
                name = format!("{}{element}", a.name.replace('.', "_"));
            }
            attrs.insert(name, col);
        }
    }
    Ok(PointCloud { xyz, attrs })
}

/// Describe an `.rdbx` database without reading its points: point count,
/// attributes, metadata and extent.
pub fn rdbx_info(path: impl AsRef<Path>, library: Option<&Path>) -> Result<RdbxInfo> {
    let db = Db::open(api(library)?, path.as_ref())?;
    let attributes = db.attributes()?;
    let extent_of = coordinates(&attributes).map(|a| a.name.clone());
    let (point_count, bounds) = db.statistics(extent_of.as_deref())?;
    Ok(RdbxInfo { library_version: db.library_version()?, point_count, primary_attribute: db.primary_attribute()?, attributes, metadata: db.metadata()?, bounds })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_follow_las_where_there_is_a_match_and_the_rdb_name_otherwise() {
        assert_eq!(attribute_name("riegl.target_index", 1, 0), "return_number");
        assert_eq!(attribute_name("riegl.target_count", 1, 0), "number_of_returns");
        assert_eq!(attribute_name("riegl.timestamp", 1, 0), "gps_time");
        assert_eq!(attribute_name("riegl.class", 1, 0), "classification");
        assert_eq!(attribute_name("riegl.id", 1, 0), "id");
        assert_eq!(attribute_name("riegl.reflectance", 1, 0), "reflectance");
        assert_eq!(attribute_name("riegl.mta_zone", 1, 0), "mta_zone");
        assert_eq!(attribute_name("riegl.direction_coarse", 3, 2), "direction_coarse_2");
        assert_eq!(attribute_name("riegl.rgba", 4, 1), "green");
        assert_eq!(attribute_name("riegl.rgba", 4, 3), "alpha");
        assert_eq!(attribute_name("acme.score", 1, 0), "acme_score");
        for (rdb, sylva) in RENAMES {
            assert_eq!(rdb_attribute_name(sylva), *rdb);
            assert_eq!(rdb_attribute_name(rdb), *rdb);
        }
        assert_eq!(rdb_attribute_name("reflectance"), "riegl.reflectance");
        assert_eq!(rdb_attribute_name("acme.score"), "acme.score");
    }

    #[test]
    fn columns_keep_rows_and_widen_only_int16_and_uint64() {
        let mut c = Col::new(DT_INT16, 1, 4);
        if let Col::I16(col) = &mut c {
            col.buf[..4].copy_from_slice(&[-1, 5, 7, 9]);
        }
        c.take(&[0, 2]);
        assert_eq!(c.into_attrs(), [Attr::I32(vec![-1, 7])]);

        let mut c = Col::new(DT_UINT64, 1, 2);
        if let Col::U64(col) = &mut c {
            col.buf[..2].copy_from_slice(&[1, 1_000_000_000_000]);
        }
        c.take(&[0, 1]);
        assert_eq!(c.into_attrs(), [Attr::I64(vec![1, 1_000_000_000_000])]);

        // A 3-vector becomes three columns, element k of row i at buf[3 i + k].
        let mut c = Col::new(DT_SINGLE, 3, 2);
        if let Col::F32(col) = &mut c {
            col.buf[..6].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        }
        c.take(&[1, 0]);
        assert_eq!(c.into_attrs(), [Attr::F32(vec![4.0, 1.0]), Attr::F32(vec![5.0, 2.0]), Attr::F32(vec![6.0, 3.0])]);

        let types: Vec<&str> = [DT_UINT8, DT_INT8, DT_UINT16, DT_UINT32, DT_INT32, DT_INT64, DT_DOUBLE, 99].iter().map(|&t| Col::new(t, 1, 0).into_attrs()[0].dtype()).collect();
        assert_eq!(types, ["uint8", "int8", "uint16", "uint32", "int32", "int64", "float64", "float64"]);
        assert_eq!(dtype_name(DT_INT16), "int16");
        assert_eq!(dtype_name(0), "unknown");
    }

    #[test]
    fn missing_library_says_what_to_set() {
        // Only when no librdb is installed where the search looks; otherwise the
        // hint still has to be honoured when it is a file.
        if let Err(e) = find_rdblib(Some(Path::new("/nonexistent/rdblib"))) {
            assert!(e.to_string().contains("RDBLIB_PATH"), "{e}");
        }
        let text_list_empty = text_list(std::ptr::null(), 3);
        assert!(text_list_empty.is_empty());
        let packed = b"riegl.xyz\0riegl.id\0";
        assert_eq!(text_list(packed.as_ptr() as RdbString, 2), ["riegl.xyz", "riegl.id"]);
        assert_eq!(text(std::ptr::null()), "");
    }
}
