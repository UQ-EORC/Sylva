// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! RIEGL `.rxp` reading through RiVLib's `libscanifc`, loaded at runtime.
//!
//! RiVLib is proprietary and cannot be redistributed. The library is located
//! from `RIVLIB_PATH` / `RIVLIB_HOME`, an explicit path, or the usual install
//! locations (`~/.local/lib`, `~/opt`, `/opt`, `/usr/local/lib`, ...).
//!
//! Points come back in the scanner's own coordinate system (SOCS) with the
//! scanner at the origin. Runs of echoes sharing a timestamp form one pulse,
//! which is how [`Shots`] are recovered.

use std::ffi::{c_char, c_void, CString};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use libloading::{Library, Symbol};

use crate::error::{Error, Result};
use crate::pointcloud::Attr;
use crate::{Point, PointCloud, Shots};

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Xyz32 {
    x: f32,
    y: f32,
    z: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Attributes {
    amplitude: f32,
    reflectance: f32,
    deviation: u16,
    flags: u16,
    background_radiation: f32,
}

type Handle = *mut c_void;
type OpenFn = unsafe extern "C" fn(*const c_char, i32, *mut Handle) -> i32;
type ReadFn = unsafe extern "C" fn(Handle, u32, *mut Xyz32, *mut Attributes, *mut u64, *mut u32, *mut i32) -> i32;
type CloseFn = unsafe extern "C" fn(Handle) -> i32;
type LastErrorFn = unsafe extern "C" fn(*mut c_char, u32, *mut u32) -> i32;

/// Echo type stored in the low two bits of `flags`.
pub const ECHO_SINGLE: u16 = 0;
pub const ECHO_FIRST: u16 = 1;
pub const ECHO_INTERIOR: u16 = 2;
pub const ECHO_LAST: u16 = 3;
const FLAG_PSEUDO_ECHO: u16 = 1 << 4;

static LIBRARY: OnceLock<Mutex<Option<Library>>> = OnceLock::new();

/// Directories searched for `libscanifc`, in order.
pub fn search_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for var in ["RIVLIB_PATH", "RIVLIB_HOME"] {
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

const LIB_NAMES: &[&str] = &["libscanifc-mt.so", "libscanifc.so", "scanifc-mt.dll", "libscanifc-mt.dylib"];

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
        let n = e.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if n.to_ascii_lowercase().contains("rivlib") || n == "lib" || depth >= 2 {
            if let Some(p) = search_dir(&e, depth - 1) {
                return Some(p);
            }
        }
    }
    None
}

/// Locate `libscanifc`.
pub fn find_rivlib(hint: Option<&Path>) -> Result<PathBuf> {
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
    Err(Error::Riegl(
        "libscanifc not found. Download RiVLib from RIEGL and set RIVLIB_PATH to the extracted \
         directory (containing lib/libscanifc-mt.so)"
            .into(),
    ))
}

fn with_library<T>(hint: Option<&Path>, f: impl FnOnce(&Library) -> Result<T>) -> Result<T> {
    let cell = LIBRARY.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().unwrap();
    if guard.is_none() {
        let path = find_rivlib(hint)?;
        // SAFETY: RiVLib is a well-behaved C library with no init side effects.
        let lib = unsafe { Library::new(&path) }.map_err(|e| Error::Riegl(format!("{}: {e}", path.display())))?;
        *guard = Some(lib);
    }
    f(guard.as_ref().unwrap())
}

/// Options for [`read_rxp`] / [`read_rxp_shots`].
#[derive(Debug, Clone)]
pub struct RxpOptions {
    /// Explicit RiVLib directory or library file.
    pub library: Option<PathBuf>,
    /// Drop pseudo echoes (flag bit 4).
    pub drop_pseudo_echoes: bool,
    /// Minimum range (m); RiSCAN's default export drops `< 0.5`.
    pub min_range: f64,
    pub max_range: f64,
    /// Keep every `stride`-th point (1 = all). Splits multi-echo pulses;
    /// prefer `shot_stride` for pulse data.
    pub stride: usize,
    /// Keep every `shot_stride`-th pulse with all its echoes (1 = all).
    pub shot_stride: usize,
    /// Read at most this many points.
    pub max_points: Option<usize>,
    /// Echo selection: `all`, `first`, `last`, `single`.
    pub echoes: String,
}

impl Default for RxpOptions {
    fn default() -> Self {
        RxpOptions {
            library: None,
            drop_pseudo_echoes: true,
            min_range: 0.5,
            max_range: f64::INFINITY,
            stride: 1,
            shot_stride: 1,
            max_points: None,
            echoes: "all".into(),
        }
    }
}

struct RawRxp {
    xyz: Vec<Point>,
    amplitude: Vec<f32>,
    reflectance: Vec<f32>,
    deviation: Vec<u16>,
    flags: Vec<u16>,
    time_ns: Vec<u64>,
}

fn read_raw(path: &Path, opts: &RxpOptions) -> Result<RawRxp> {
    with_library(opts.library.as_deref(), |lib| {
        // SAFETY: symbol signatures follow scanifc.h from RiVLib.
        let (open, read, close, last_error) = unsafe {
            let open: Symbol<OpenFn> = lib.get(b"scanifc_point3dstream_open\0").map_err(|e| Error::Riegl(e.to_string()))?;
            let read: Symbol<ReadFn> = lib.get(b"scanifc_point3dstream_read\0").map_err(|e| Error::Riegl(e.to_string()))?;
            let close: Symbol<CloseFn> = lib.get(b"scanifc_point3dstream_close\0").map_err(|e| Error::Riegl(e.to_string()))?;
            let last_error: Symbol<LastErrorFn> = lib.get(b"scanifc_get_last_error\0").map_err(|e| Error::Riegl(e.to_string()))?;
            (open, read, close, last_error)
        };
        let describe = |code: i32| -> Error {
            let mut buf = vec![0u8; 1024];
            let mut size = 0u32;
            // SAFETY: buffer sizes are passed explicitly.
            unsafe { last_error(buf.as_mut_ptr() as *mut c_char, buf.len() as u32, &mut size) };
            let msg = String::from_utf8_lossy(&buf[..(size as usize).min(buf.len())]).trim_end_matches('\0').to_string();
            Error::Riegl(format!("{path:?}: scanifc error {code}: {msg}", path = path.display()))
        };
        let uri = CString::new(format!("file:{}", path.canonicalize().unwrap_or(path.to_path_buf()).display()))
            .map_err(|e| Error::Riegl(e.to_string()))?;
        let mut handle: Handle = std::ptr::null_mut();
        // SAFETY: valid C string and out-pointer.
        let rc = unsafe { open(uri.as_ptr(), 0, &mut handle) };
        if rc != 0 {
            return Err(describe(rc));
        }
        const CHUNK: usize = 1 << 18;
        let mut xyz_buf = vec![Xyz32::default(); CHUNK];
        let mut attr_buf = vec![Attributes::default(); CHUNK];
        let mut time_buf = vec![0u64; CHUNK];
        let mut raw = RawRxp { xyz: Vec::new(), amplitude: Vec::new(), reflectance: Vec::new(), deviation: Vec::new(), flags: Vec::new(), time_ns: Vec::new() };
        let mut counter = 0usize;
        // Pulses are runs of echoes sharing a timestamp.
        let mut pulse = 0usize;
        let mut last_time = u64::MAX;
        loop {
            let mut got = 0u32;
            let mut eof = 0i32;
            // SAFETY: buffers are CHUNK long and `want` == CHUNK.
            let rc = unsafe { read(handle, CHUNK as u32, xyz_buf.as_mut_ptr(), attr_buf.as_mut_ptr(), time_buf.as_mut_ptr(), &mut got, &mut eof) };
            if rc != 0 {
                // SAFETY: handle was opened above.
                unsafe { close(handle) };
                return Err(describe(rc));
            }
            let n = got as usize;
            for i in 0..n {
                counter += 1;
                if time_buf[i] != last_time {
                    last_time = time_buf[i];
                    pulse += 1;
                }
                if opts.shot_stride > 1 && pulse % opts.shot_stride != 0 {
                    continue;
                }
                if opts.stride > 1 && counter % opts.stride != 0 {
                    continue;
                }
                let a = attr_buf[i];
                if opts.drop_pseudo_echoes && a.flags & FLAG_PSEUDO_ECHO != 0 {
                    continue;
                }
                let p = xyz_buf[i];
                let r = ((p.x as f64).powi(2) + (p.y as f64).powi(2) + (p.z as f64).powi(2)).sqrt();
                if r < opts.min_range || r > opts.max_range {
                    continue;
                }
                let echo = a.flags & 3;
                let keep = match opts.echoes.as_str() {
                    "first" => echo == ECHO_FIRST || echo == ECHO_SINGLE,
                    "last" => echo == ECHO_LAST || echo == ECHO_SINGLE,
                    "single" => echo == ECHO_SINGLE,
                    _ => true,
                };
                if !keep {
                    continue;
                }
                raw.xyz.push([p.x as f64, p.y as f64, p.z as f64]);
                raw.amplitude.push(a.amplitude);
                raw.reflectance.push(a.reflectance);
                raw.deviation.push(a.deviation);
                raw.flags.push(a.flags);
                raw.time_ns.push(time_buf[i]);
                if opts.max_points.map(|m| raw.xyz.len() >= m).unwrap_or(false) {
                    break;
                }
            }
            if n == 0 && eof == 0 {
                break;
            }
            if opts.max_points.map(|m| raw.xyz.len() >= m).unwrap_or(false) {
                break;
            }
        }
        // SAFETY: handle was opened above.
        unsafe { close(handle) };
        Ok(raw)
    })
}

/// Read an `.rxp` file into a point cloud (SOCS frame) with `amplitude`,
/// `reflectance`, `deviation`, `echo_type` and `gps_time` (seconds) attributes.
pub fn read_rxp(path: impl AsRef<Path>, opts: &RxpOptions) -> Result<PointCloud> {
    let raw = read_raw(path.as_ref(), opts)?;
    let mut cloud = PointCloud::new(raw.xyz);
    cloud.attrs.insert("amplitude".into(), Attr::F32(raw.amplitude));
    cloud.attrs.insert("reflectance".into(), Attr::F32(raw.reflectance));
    cloud.attrs.insert("deviation".into(), Attr::U16(raw.deviation));
    cloud.attrs.insert("echo_type".into(), Attr::U8(raw.flags.iter().map(|f| (f & 3) as u8).collect()));
    cloud.attrs.insert("gps_time".into(), Attr::F64(raw.time_ns.iter().map(|&t| t as f64 * 1e-9).collect()));
    Ok(cloud)
}

/// Read an `.rxp` file as pulses. Echoes sharing a timestamp form one shot.
pub fn read_rxp_shots(path: impl AsRef<Path>, opts: &RxpOptions) -> Result<Shots> {
    let cloud = read_rxp(path, opts)?;
    Ok(Shots::from_pointcloud(&cloud, [0.0, 0.0, 0.0]))
}
