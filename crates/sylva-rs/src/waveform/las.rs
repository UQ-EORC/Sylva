// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! LAS 1.3 / 1.4 waveform data packets (ASPRS LAS 1.4 R15 specification,
//! point formats 4, 5, 9 and 10).
//!
//! A waveform point carries the index of a wave packet descriptor (a
//! `LASF_Spec` VLR with record id 99 + index: bits per sample, number of
//! samples, temporal spacing in ps, digitiser gain and offset), the byte
//! offset and size of its packet, the "return point waveform location" `L`
//! (ps from the first sample to the point) and a vector `v = (X(t), Y(t),
//! Z(t))` in metres per picosecond. The packets are stored in an EVLR
//! (record id 65535) of the file itself or in a `.wdp` file next to it, whose
//! first 60 bytes repeat the EVLR header; offsets count from the start of the
//! EVLR header or of the `.wdp` file.
//!
//! Sample `i` of a packet lies at `P + (L - i dt) v`: `v` points from the
//! point back towards the scanner. This is how RIEGL's exports and
//! LAStools' PulseWaves converter read the vector; the specification's own
//! wording is ambiguous about the sign.
//!
//! Points are read in chunks and the packets they refer to are read with
//! them, so a file of any size streams through a fixed amount of memory.
//! Returns of one pulse share a packet; consecutive records that point to
//! the same packet become one waveform.

#![allow(clippy::neg_cmp_op_on_partial_ord)]

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use byteorder::{ByteOrder, LittleEndian as LE, WriteBytesExt};

use super::Waveforms;
use crate::error::{Error, Result};
use crate::pointcloud::Attr;
use crate::transform::{norm, scale};
use crate::Point;

const HEADER_SIZE_14: usize = 375;
const VLR_HEADER: usize = 54;
const EVLR_HEADER: usize = 60;
const WDP_RECORD_ID: u16 = 65535;

/// A wave packet descriptor (LAS VLR record id 100..=354).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WavePacketDescriptor {
    pub bits_per_sample: u8,
    pub compression: u8,
    pub n_samples: u32,
    /// Temporal sample spacing (ps).
    pub spacing_ps: u32,
    pub gain: f64,
    pub offset: f64,
}

/// Where the waveform packets are.
#[derive(Debug, Clone, PartialEq)]
pub enum PacketStore {
    /// In this file, offsets counted from `base` (the start of the EVLR header).
    Internal(u64),
    /// In an auxiliary `.wdp` file.
    External(PathBuf),
    /// Neither in the file nor next to it.
    Missing,
}

/// What a LAS/LAZ header says about its waveforms.
#[derive(Debug, Clone)]
pub struct LasWaveInfo {
    pub version: (u8, u8),
    pub point_format: u8,
    pub record_length: u16,
    pub n_points: u64,
    pub offset_to_points: u64,
    pub scale: [f64; 3],
    pub offset: [f64; 3],
    /// `[min_x, min_y, min_z, max_x, max_y, max_z]`.
    pub bounds: [f64; 6],
    pub global_encoding: u16,
    pub compressed: bool,
    /// Descriptor by packet index (1..=255); index 0 means no waveform.
    pub descriptors: BTreeMap<u8, WavePacketDescriptor>,
    pub store: PacketStore,
}

fn waveform_field_offset(format: u8) -> Option<usize> {
    match format {
        4 => Some(28),
        5 => Some(34),
        9 => Some(30),
        10 => Some(38),
        _ => None,
    }
}

fn file_err(path: &Path, msg: impl Into<String>) -> Error {
    Error::file(path, msg)
}

fn wdp_path(path: &Path) -> Option<PathBuf> {
    for ext in ["wdp", "WDP"] {
        let p = path.with_extension(ext);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// Read the header, the wave packet descriptors and where the packets are.
pub fn read_info(path: impl AsRef<Path>) -> Result<LasWaveInfo> {
    let path = path.as_ref();
    let mut f = File::open(path).map_err(|e| file_err(path, e.to_string()))?;
    let file_len = f.metadata()?.len();
    let mut h = vec![0u8; HEADER_SIZE_14];
    let got = read_up_to(&mut f, &mut h)?;
    if got < 227 || &h[0..4] != b"LASF" {
        return Err(file_err(path, "not a LAS file (no LASF signature)"));
    }
    let version = (h[24], h[25]);
    let header_size = LE::read_u16(&h[94..]) as u64;
    let offset_to_points = LE::read_u32(&h[96..]) as u64;
    let n_vlrs = LE::read_u32(&h[100..]);
    let raw_format = h[104];
    let compressed = raw_format & 0x80 != 0 || path.extension().is_some_and(|e| e.eq_ignore_ascii_case("laz"));
    let point_format = raw_format & 0x3f;
    let record_length = LE::read_u16(&h[105..]);
    let legacy_n = LE::read_u32(&h[107..]) as u64;
    let rd = |o: usize| LE::read_f64(&h[o..]);
    let scale = [rd(131), rd(139), rd(147)];
    let offset = [rd(155), rd(163), rd(171)];
    let bounds = [rd(187), rd(203), rd(219), rd(179), rd(195), rd(211)];
    let global_encoding = LE::read_u16(&h[6..]);
    let wdp_start = if version.1 >= 3 && got >= 235 { LE::read_u64(&h[227..]) } else { 0 };
    let (evlr_start, n_evlrs, n_points) = if version.1 >= 4 && got >= 255 {
        let n64 = LE::read_u64(&h[247..]);
        (LE::read_u64(&h[235..]), LE::read_u32(&h[243..]), if n64 > 0 { n64 } else { legacy_n })
    } else {
        (0, 0, legacy_n)
    };
    if waveform_field_offset(point_format).is_none() {
        return Err(file_err(path, format!("point format {point_format} has no waveform data (waveform formats are 4, 5, 9 and 10)")));
    }

    let mut descriptors = BTreeMap::new();
    f.seek(SeekFrom::Start(header_size))?;
    let mut pos = header_size;
    for _ in 0..n_vlrs {
        let mut vh = [0u8; VLR_HEADER];
        if pos + VLR_HEADER as u64 > offset_to_points || read_up_to(&mut f, &mut vh)? < VLR_HEADER {
            break;
        }
        let user = String::from_utf8_lossy(&vh[2..18]).trim_end_matches('\0').to_string();
        let record_id = LE::read_u16(&vh[18..]);
        let len = LE::read_u16(&vh[20..]) as usize;
        let mut data = vec![0u8; len];
        f.read_exact(&mut data).map_err(|e| file_err(path, format!("truncated VLR: {e}")))?;
        pos += (VLR_HEADER + len) as u64;
        if user == "LASF_Spec" && (100..=354).contains(&record_id) && len >= 26 {
            descriptors.insert(
                (record_id - 99) as u8,
                WavePacketDescriptor {
                    bits_per_sample: data[0],
                    compression: data[1],
                    n_samples: LE::read_u32(&data[2..]),
                    spacing_ps: LE::read_u32(&data[6..]),
                    gain: LE::read_f64(&data[10..]),
                    offset: LE::read_f64(&data[18..]),
                },
            );
        }
    }

    // Packets: the header's pointer, else an EVLR with record id 65535, else a .wdp file.
    let external = global_encoding & 0b100 != 0;
    let mut store = PacketStore::Missing;
    if !external && wdp_start > 0 && wdp_start < file_len {
        store = PacketStore::Internal(wdp_start);
    } else if !external && evlr_start > 0 && !compressed {
        let mut p = evlr_start;
        for _ in 0..n_evlrs {
            let mut eh = [0u8; EVLR_HEADER];
            f.seek(SeekFrom::Start(p))?;
            if read_up_to(&mut f, &mut eh)? < EVLR_HEADER {
                break;
            }
            if LE::read_u16(&eh[18..]) == WDP_RECORD_ID {
                store = PacketStore::Internal(p);
                break;
            }
            p += EVLR_HEADER as u64 + LE::read_u64(&eh[20..]);
        }
    }
    if store == PacketStore::Missing {
        if let Some(w) = wdp_path(path) {
            store = PacketStore::External(w);
        }
    }
    Ok(LasWaveInfo { version, point_format, record_length, n_points, offset_to_points, scale, offset, bounds, global_encoding, compressed, descriptors, store })
}

fn read_up_to(f: &mut impl Read, buf: &mut [u8]) -> Result<usize> {
    let mut got = 0;
    while got < buf.len() {
        let n = f.read(&mut buf[got..])?;
        if n == 0 {
            break;
        }
        got += n;
    }
    Ok(got)
}

/// One decoded point record with a waveform.
#[derive(Debug, Clone, Copy)]
struct Rec {
    xyz: Point,
    intensity: u16,
    return_number: u8,
    n_returns: u8,
    classification: u8,
    point_source_id: u16,
    gps_time: f64,
    index: u8,
    byte_offset: u64,
    size: u32,
    location: f32,
    v: [f32; 3],
}

/// One point record, picked apart field by field.
///
/// `rec` is the raw bytes of the record, and every number is read from a fixed
/// offset into it, as the LAS specification lays them out: `LE::read_i32(&rec[4..])`
/// is "the little-endian 32-bit integer starting at byte 4" (`&rec[4..]` is a
/// view from byte 4 onwards, not a copy). Coordinates are stored as integers
/// and scaled on the way out, and the flag byte packs several fields, which is
/// what the shifts and masks undo. Where the offsets move between point
/// formats, the two cases are spelled out rather than computed.
fn decode(rec: &[u8], info: &LasWaveInfo) -> Rec {
    let fmt = info.point_format;
    let xyz = [
        LE::read_i32(&rec[0..]) as f64 * info.scale[0] + info.offset[0],
        LE::read_i32(&rec[4..]) as f64 * info.scale[1] + info.offset[1],
        LE::read_i32(&rec[8..]) as f64 * info.scale[2] + info.offset[2],
    ];
    let intensity = LE::read_u16(&rec[12..]);
    let (return_number, n_returns, classification, point_source_id, gps_time) = if fmt >= 6 {
        (rec[14] & 0x0f, rec[14] >> 4, rec[16], LE::read_u16(&rec[20..]), LE::read_f64(&rec[22..]))
    } else {
        (rec[14] & 0x07, (rec[14] >> 3) & 0x07, rec[15] & 0x1f, LE::read_u16(&rec[18..]), LE::read_f64(&rec[20..]))
    };
    let w = waveform_field_offset(fmt).expect("waveform format");
    Rec {
        xyz,
        intensity,
        return_number,
        n_returns,
        classification,
        point_source_id,
        gps_time,
        index: rec[w],
        byte_offset: LE::read_u64(&rec[w + 1..]),
        size: LE::read_u32(&rec[w + 9..]),
        location: LE::read_f32(&rec[w + 13..]),
        v: [LE::read_f32(&rec[w + 17..]), LE::read_f32(&rec[w + 21..]), LE::read_f32(&rec[w + 25..])],
    }
}

/// Raw point records `[start, start + n)` (fewer at the end of the file).
struct RecordSource {
    info: LasWaveInfo,
    file: Option<File>,
    laz: Option<las::Reader>,
    rl: usize,
}

impl RecordSource {
    fn open(path: &Path, info: LasWaveInfo) -> Result<Self> {
        let rl = info.record_length as usize;
        if info.compressed {
            let laz = las::Reader::from_path(path)?;
            Ok(RecordSource { info, file: None, laz: Some(laz), rl })
        } else {
            let file = File::open(path).map_err(|e| file_err(path, e.to_string()))?;
            Ok(RecordSource { info, file: Some(file), laz: None, rl })
        }
    }

    fn read(&mut self, start: u64, n: u64) -> Result<Vec<u8>> {
        let n = n.min(self.info.n_points.saturating_sub(start));
        if n == 0 {
            return Ok(Vec::new());
        }
        if let Some(f) = &mut self.file {
            let mut buf = vec![0u8; n as usize * self.rl];
            f.seek(SeekFrom::Start(self.info.offset_to_points + start * self.rl as u64))?;
            let got = read_up_to(f, &mut buf)?;
            buf.truncate(got / self.rl * self.rl);
            Ok(buf)
        } else {
            let r = self.laz.as_mut().expect("reader");
            r.seek(start)?;
            let pd = r.read_points(n)?;
            let rl = pd.record_len();
            if rl != self.rl {
                // The decoder's record may drop extra bytes; keep the fields we read.
                self.rl = rl;
            }
            Ok(pd.raw_bytes().to_vec())
        }
    }
}

/// Options for [`read_waveforms`].
#[derive(Debug, Clone)]
pub struct LasReadOptions {
    /// First point record.
    pub start: u64,
    /// Number of point records to read (more are read to finish the last pulse).
    pub count: Option<u64>,
    /// Join consecutive records that share a packet into one waveform.
    pub dedupe: bool,
}

impl Default for LasReadOptions {
    fn default() -> Self {
        LasReadOptions { start: 0, count: None, dedupe: true }
    }
}

/// Read the waveforms of point records `start .. start + count`. Returns
/// the waveforms and the record index to start the next chunk at.
///
/// A pulse belongs to the chunk it starts in: records at the start of the
/// range that share the packet of the record before it are skipped, and the
/// range is extended over the records that continue its last packet.
/// `pulse` is the index of the pulse's first record.
pub fn read_waveforms(path: impl AsRef<Path>, opts: &LasReadOptions) -> Result<(Waveforms, u64)> {
    let path = path.as_ref();
    let info = read_info(path)?;
    let n_total = info.n_points;
    let start = opts.start.min(n_total);
    let count = opts.count.unwrap_or(n_total - start).min(n_total - start);
    let mut src = RecordSource::open(path, info.clone())?;
    let key = |r: &Rec| (r.index, r.byte_offset);

    let buf = src.read(start, count)?;
    let rl = src.rl;
    let mut recs: Vec<Rec> = buf.chunks_exact(rl).map(|c| decode(c, &info)).collect();
    let mut end = start + recs.len() as u64;
    // Extend over the last pulse.
    if opts.dedupe && !recs.is_empty() {
        let last = key(recs.last().unwrap());
        'extend: while end < n_total {
            let more = src.read(end, 64)?;
            if more.is_empty() {
                break;
            }
            for c in more.chunks_exact(src.rl) {
                let r = decode(c, &info);
                if key(&r) != last || r.index == 0 {
                    break 'extend;
                }
                recs.push(r);
                end += 1;
            }
        }
    }
    // Skip records that continue the previous chunk's last pulse.
    let mut skip = 0usize;
    if opts.dedupe && start > 0 && !recs.is_empty() {
        let prev = src.read(start - 1, 1)?;
        if let Some(c) = prev.chunks_exact(src.rl).next() {
            let p = decode(c, &info);
            if p.index != 0 {
                while skip < recs.len() && key(&recs[skip]) == key(&p) {
                    skip += 1;
                }
            }
        }
    }

    // Group into waveforms.
    let mut groups: Vec<(usize, usize)> = Vec::new();
    let mut i = skip;
    while i < recs.len() {
        let mut j = i + 1;
        if opts.dedupe {
            while j < recs.len() && key(&recs[j]) == key(&recs[i]) {
                j += 1;
            }
        }
        if recs[i].index != 0 && recs[i].size > 0 {
            groups.push((i, j));
        }
        i = j;
    }
    let wf = build(path, &info, &recs, &groups, start)?;
    Ok((wf, end))
}

fn build(path: &Path, info: &LasWaveInfo, recs: &[Rec], groups: &[(usize, usize)], start: u64) -> Result<Waveforms> {
    let mut wf = Waveforms::default();
    let (mut intensity, mut rn, mut nr, mut class, mut psid, mut n_rec, mut desc_idx) = (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    let mut packets: Vec<(u64, u32, WavePacketDescriptor)> = Vec::with_capacity(groups.len());
    for &(a, b) in groups {
        let r = &recs[a];
        let d = *info.descriptors.get(&r.index).ok_or_else(|| file_err(path, format!("point {} refers to wave packet descriptor {} which the file does not define", start + a as u64, r.index)))?;
        if d.compression != 0 {
            return Err(file_err(path, format!("wave packet descriptor {} uses compression type {}, which is not supported", r.index, d.compression)));
        }
        if !matches!(d.bits_per_sample, 8 | 16 | 32) {
            return Err(file_err(path, format!("wave packet descriptor {} has {} bits per sample; 8, 16 and 32 are supported", r.index, d.bits_per_sample)));
        }
        let v = [r.v[0] as f64, r.v[1] as f64, r.v[2] as f64];
        let speed = norm(&v); // m/ps
        if !(speed > 0.0) || !speed.is_finite() || d.spacing_ps == 0 {
            continue;
        }
        wf.pulse.push((start + a as u64) as i64);
        wf.gps_time.push(r.gps_time);
        wf.origin.push([f64::NAN; 3]);
        wf.anchor.push(r.xyz);
        wf.direction.push(scale(&v, -1.0 / speed));
        wf.offset.push(-(r.location as f64) / 1000.0);
        wf.interval.push(d.spacing_ps as f64 / 1000.0);
        wf.metres_per_ns.push(speed * 1000.0);
        intensity.push(r.intensity);
        rn.push(r.return_number);
        nr.push(r.n_returns);
        class.push(r.classification);
        psid.push(r.point_source_id);
        n_rec.push((b - a) as u32);
        desc_idx.push(r.index);
        packets.push((r.byte_offset, r.size, d));
    }
    wf.attrs.insert("intensity".into(), Attr::U16(intensity));
    wf.attrs.insert("return_number".into(), Attr::U8(rn));
    wf.attrs.insert("number_of_returns".into(), Attr::U8(nr));
    wf.attrs.insert("classification".into(), Attr::U8(class));
    wf.attrs.insert("point_source_id".into(), Attr::U16(psid));
    wf.attrs.insert("n_records".into(), Attr::U32(n_rec));
    wf.attrs.insert("descriptor".into(), Attr::U8(desc_idx));
    read_packets(path, info, &packets, &mut wf)?;
    Ok(wf)
}

/// Read packets in file order through one buffered reader, skipping forward
/// within the buffer where packets are close together.
fn read_packets(path: &Path, info: &LasWaveInfo, packets: &[(u64, u32, WavePacketDescriptor)], wf: &mut Waveforms) -> Result<()> {
    if packets.is_empty() {
        return Ok(());
    }
    let (file_path, base) = match &info.store {
        PacketStore::Internal(b) => (path.to_path_buf(), *b),
        PacketStore::External(p) => (p.clone(), 0),
        PacketStore::Missing => return Err(file_err(path, "the waveform packets are neither in the file nor in a .wdp file next to it")),
    };
    let f = File::open(&file_path).map_err(|e| file_err(&file_path, e.to_string()))?;
    let flen = f.metadata()?.len();
    let mut reader = BufReader::with_capacity(1 << 20, f);
    let mut order: Vec<usize> = (0..packets.len()).collect();
    order.sort_by_key(|&k| packets[k].0);
    let mut decoded: Vec<Vec<f32>> = vec![Vec::new(); packets.len()];
    let mut pos: u64 = 0;
    reader.seek(SeekFrom::Start(0))?;
    let mut raw = Vec::new();
    for k in order {
        let (off, size, d) = packets[k];
        let at = base + off;
        if at + size as u64 > flen {
            return Err(file_err(&file_path, format!("waveform packet at byte {at} ({size} bytes) runs past the end of the file")));
        }
        if at >= pos && at - pos < (1 << 20) {
            reader.seek_relative((at - pos) as i64)?;
        } else {
            reader.seek(SeekFrom::Start(at))?;
        }
        raw.resize(size as usize, 0);
        reader.read_exact(&mut raw)?;
        pos = at + size as u64;
        let bytes = (d.bits_per_sample / 8) as usize;
        let n = (d.n_samples as usize).min(raw.len() / bytes);
        let (g, o) = (d.gain, d.offset);
        let conv = |x: f64| (o + g * x) as f32;
        decoded[k] = match bytes {
            1 => raw[..n].iter().map(|&x| conv(x as f64)).collect(),
            2 => raw[..2 * n].as_chunks::<2>().0.iter().map(|c| conv(LE::read_u16(c) as f64)).collect(),
            _ => raw[..4 * n].as_chunks::<4>().0.iter().map(|c| conv(LE::read_u32(c) as f64)).collect(),
        };
    }
    for s in decoded {
        wf.sample_start.push(wf.samples.len());
        wf.sample_count.push(s.len() as u32);
        wf.samples.extend(s);
    }
    Ok(())
}

/// Options for [`write_waveforms`].
#[derive(Debug, Clone)]
pub struct LasWriteWaveOptions {
    /// Coordinate resolution (m).
    pub scale: f64,
    /// 8 or 16 bits per sample.
    pub bits: u8,
    /// Write the packets to a `.wdp` file next to the LAS file instead of an EVLR.
    pub external: bool,
    /// OGC WKT of the coordinate reference system.
    pub crs_wkt: Option<String>,
}

impl Default for LasWriteWaveOptions {
    fn default() -> Self {
        LasWriteWaveOptions { scale: 0.001, bits: 16, external: false, crs_wkt: None }
    }
}

/// Digitiser gain and offset that store `samples` in `bits` without loss
/// when they are integers in range, else spread over the full range.
pub fn quantisation(samples: &[f32], bits: u8) -> (f64, f64) {
    let top = ((1u64 << bits) - 1) as f64;
    let finite = samples.iter().filter(|v| v.is_finite());
    let (lo, hi) = finite.fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v as f64), b.max(v as f64)));
    if !lo.is_finite() {
        return (1.0, 0.0);
    }
    if samples.iter().all(|&v| v.fract() == 0.0) && lo >= 0.0 && hi <= top {
        return (1.0, 0.0);
    }
    let span = hi - lo;
    (if span > 0.0 { span / top } else { 1.0 }, lo)
}

fn attr_u(wf: &Waveforms, name: &str, default: f64) -> Vec<f64> {
    wf.attrs.get(name).map(|a| a.to_f64()).unwrap_or_else(|| vec![default; wf.len()])
}

/// Write waveforms as LAS 1.4 point format 9: one point per waveform at its
/// anchor, the packets in an EVLR (or a `.wdp` file). Samples are stored as
/// `bits`-bit integers with the gain and offset of [`quantisation`]; the
/// sampling interval is rounded to whole picoseconds.
#[allow(clippy::needless_range_loop)]
pub fn write_waveforms(wf: &Waveforms, path: impl AsRef<Path>, opts: &LasWriteWaveOptions) -> Result<()> {
    let path = path.as_ref();
    wf.validate()?;
    if !matches!(opts.bits, 8 | 16) {
        return Err(Error::invalid(format!("bits must be 8 or 16, got {}", opts.bits)));
    }
    if !(opts.scale > 0.0) {
        return Err(Error::invalid("scale must be positive"));
    }
    let n = wf.len();
    let (gain, dig_offset) = quantisation(&wf.samples, opts.bits);
    let top = ((1u64 << opts.bits) - 1) as f64;

    // Descriptors by (number of samples, spacing in ps).
    let mut desc_of: BTreeMap<(u32, u32), u8> = BTreeMap::new();
    let mut row_desc = vec![0u8; n];
    for i in 0..n {
        if wf.sample_count[i] == 0 {
            continue;
        }
        let ps = (wf.interval[i] * 1000.0).round();
        if !(ps >= 1.0 && ps <= u32::MAX as f64) {
            return Err(Error::invalid(format!("sampling interval {} ns of waveform {i} cannot be stored in whole picoseconds", wf.interval[i])));
        }
        let k = (wf.sample_count[i], ps as u32);
        row_desc[i] = match desc_of.get(&k) {
            Some(&d) => d,
            None => {
                if desc_of.len() >= 255 {
                    return Err(Error::invalid("more than 255 distinct (samples, interval) combinations; LAS allows 255 wave packet descriptors"));
                }
                let d = (desc_of.len() + 1) as u8;
                desc_of.insert(k, d);
                d
            }
        };
    }

    // Bounds and quantisation of coordinates.
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for a in &wf.anchor {
        for k in 0..3 {
            if !a[k].is_finite() {
                return Err(Error::invalid("anchors must be finite to be written to LAS"));
            }
            lo[k] = lo[k].min(a[k]);
            hi[k] = hi[k].max(a[k]);
        }
    }
    if n == 0 {
        lo = [0.0; 3];
        hi = [0.0; 3];
    }
    let off: [f64; 3] = [(lo[0] / 1000.0).floor() * 1000.0, (lo[1] / 1000.0).floor() * 1000.0, (lo[2] / 1000.0).floor() * 1000.0];
    let q = |v: f64, k: usize| -> Result<i32> {
        let x = ((v - off[k]) / opts.scale).round();
        if x.abs() > i32::MAX as f64 {
            return Err(Error::invalid(format!("coordinates span too far for scale {}", opts.scale)));
        }
        Ok(x as i32)
    };

    // VLRs.
    let mut vlrs: Vec<u8> = Vec::new();
    let mut n_vlrs = 0u32;
    let mut put_vlr = |user: &str, id: u16, desc: &str, data: &[u8]| {
        vlrs.write_u16::<LE>(0).unwrap();
        let mut u = [0u8; 16];
        u[..user.len()].copy_from_slice(user.as_bytes());
        vlrs.extend_from_slice(&u);
        vlrs.write_u16::<LE>(id).unwrap();
        vlrs.write_u16::<LE>(data.len() as u16).unwrap();
        let mut d = [0u8; 32];
        d[..desc.len().min(32)].copy_from_slice(&desc.as_bytes()[..desc.len().min(32)]);
        vlrs.extend_from_slice(&d);
        vlrs.extend_from_slice(data);
        n_vlrs += 1;
    };
    let mut by_index: Vec<((u32, u32), u8)> = desc_of.iter().map(|(k, v)| (*k, *v)).collect();
    by_index.sort_by_key(|x| x.1);
    for ((ns, ps), idx) in &by_index {
        let mut d = Vec::with_capacity(26);
        d.push(opts.bits);
        d.push(0);
        d.write_u32::<LE>(*ns).unwrap();
        d.write_u32::<LE>(*ps).unwrap();
        d.write_f64::<LE>(gain).unwrap();
        d.write_f64::<LE>(dig_offset).unwrap();
        put_vlr("LASF_Spec", 99 + *idx as u16, "Waveform packet descriptor", &d);
    }
    if let Some(wkt) = &opts.crs_wkt {
        let mut d = wkt.as_bytes().to_vec();
        d.push(0);
        if d.len() > u16::MAX as usize {
            return Err(Error::invalid("CRS WKT too long for a VLR"));
        }
        put_vlr("LASF_Projection", 2112, "OGC WKT", &d);
    }
    let offset_to_points = (HEADER_SIZE_14 + vlrs.len()) as u64;
    let rl: u16 = 59;

    // Packet offsets.
    let bytes = (opts.bits / 8) as u64;
    let mut packet_off = vec![0u64; n];
    let mut acc = EVLR_HEADER as u64;
    for i in 0..n {
        if row_desc[i] != 0 {
            packet_off[i] = acc;
            acc += bytes * wf.sample_count[i] as u64;
        }
    }
    let packet_bytes = acc - EVLR_HEADER as u64;
    let evlr_start = offset_to_points + n as u64 * rl as u64;

    // Header.
    let mut h = vec![0u8; HEADER_SIZE_14];
    h[0..4].copy_from_slice(b"LASF");
    let mut ge: u16 = 1 << 4; // WKT
    ge |= if opts.external { 1 << 2 } else { 1 << 1 };
    LE::write_u16(&mut h[6..], ge);
    h[24] = 1;
    h[25] = 4;
    let sys = b"Sylva";
    h[26..26 + sys.len()].copy_from_slice(sys);
    let sw = b"Sylva waveform writer";
    h[58..58 + sw.len()].copy_from_slice(sw);
    LE::write_u16(&mut h[94..], HEADER_SIZE_14 as u16);
    LE::write_u32(&mut h[96..], offset_to_points as u32);
    LE::write_u32(&mut h[100..], n_vlrs);
    h[104] = 9;
    LE::write_u16(&mut h[105..], rl);
    for k in 0..3 {
        LE::write_f64(&mut h[131 + 8 * k..], opts.scale);
        LE::write_f64(&mut h[155 + 8 * k..], off[k]);
    }
    LE::write_f64(&mut h[179..], hi[0]);
    LE::write_f64(&mut h[187..], lo[0]);
    LE::write_f64(&mut h[195..], hi[1]);
    LE::write_f64(&mut h[203..], lo[1]);
    LE::write_f64(&mut h[211..], hi[2]);
    LE::write_f64(&mut h[219..], lo[2]);
    let has_packets = !opts.external && packet_bytes > 0;
    LE::write_u64(&mut h[227..], if has_packets { evlr_start } else { 0 });
    LE::write_u64(&mut h[235..], if has_packets { evlr_start } else { 0 });
    LE::write_u32(&mut h[243..], if has_packets { 1 } else { 0 });
    LE::write_u64(&mut h[247..], n as u64);
    let rn = attr_u(wf, "return_number", 1.0);
    let nr = attr_u(wf, "number_of_returns", 1.0);
    let mut by_return = [0u64; 15];
    for &r in &rn {
        let r = (r as usize).clamp(1, 15);
        by_return[r - 1] += 1;
    }
    for (k, c) in by_return.iter().enumerate() {
        LE::write_u64(&mut h[255 + 8 * k..], *c);
    }

    let file = File::create(path).map_err(|e| file_err(path, e.to_string()))?;
    let mut out = BufWriter::with_capacity(1 << 20, file);
    out.write_all(&h)?;
    out.write_all(&vlrs)?;
    let intensity = attr_u(wf, "intensity", 0.0);
    let class = attr_u(wf, "classification", 0.0);
    let psid = attr_u(wf, "point_source_id", 0.0);
    let mut rec = [0u8; 59];
    for i in 0..n {
        rec.fill(0);
        let a = wf.anchor[i];
        LE::write_i32(&mut rec[0..], q(a[0], 0)?);
        LE::write_i32(&mut rec[4..], q(a[1], 1)?);
        LE::write_i32(&mut rec[8..], q(a[2], 2)?);
        LE::write_u16(&mut rec[12..], intensity[i].clamp(0.0, 65535.0) as u16);
        let r = (rn[i] as u8).clamp(1, 15);
        let t = (nr[i] as u8).clamp(r, 15);
        rec[14] = r | (t << 4);
        rec[16] = class[i].clamp(0.0, 255.0) as u8;
        LE::write_u16(&mut rec[20..], psid[i].clamp(0.0, 65535.0) as u16);
        LE::write_f64(&mut rec[22..], wf.gps_time[i]);
        rec[30] = row_desc[i];
        if row_desc[i] != 0 {
            let speed = wf.metres_per_ns[i] / 1000.0; // m/ps
            let d = wf.direction[i];
            let dn = norm(&d);
            LE::write_u64(&mut rec[31..], packet_off[i]);
            LE::write_u32(&mut rec[39..], (bytes * wf.sample_count[i] as u64) as u32);
            LE::write_f32(&mut rec[43..], (-wf.offset[i] * 1000.0) as f32);
            for k in 0..3 {
                LE::write_f32(&mut rec[47 + 4 * k..], (-d[k] / dn * speed) as f32);
            }
        }
        out.write_all(&rec)?;
    }

    // Packets.
    let write_packets = |w: &mut dyn Write| -> Result<()> {
        let mut eh = [0u8; EVLR_HEADER];
        eh[2..11].copy_from_slice(b"LASF_Spec");
        LE::write_u16(&mut eh[18..], WDP_RECORD_ID);
        LE::write_u64(&mut eh[20..], packet_bytes);
        let desc = b"Waveform Data Packets";
        eh[28..28 + desc.len()].copy_from_slice(desc);
        w.write_all(&eh)?;
        for i in 0..n {
            if row_desc[i] == 0 {
                continue;
            }
            for &v in wf.samples_of(i) {
                let x = if v.is_finite() { ((v as f64 - dig_offset) / gain).round().clamp(0.0, top) } else { 0.0 };
                if bytes == 1 {
                    w.write_u8(x as u8)?;
                } else {
                    w.write_u16::<LE>(x as u16)?;
                }
            }
        }
        Ok(())
    };
    if opts.external {
        out.flush()?;
        let wp = path.with_extension("wdp");
        let wfile = File::create(&wp).map_err(|e| file_err(&wp, e.to_string()))?;
        let mut w = BufWriter::with_capacity(1 << 20, wfile);
        write_packets(&mut w)?;
        w.flush()?;
    } else if has_packets {
        write_packets(&mut out)?;
    }
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::waveform::{Waveforms, C_HALF};

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sylva_wf_las_{}_{}", std::process::id(), name));
        std::fs::create_dir_all(&d).unwrap();
        d.join("t.las")
    }

    fn sample() -> Waveforms {
        let mut w = Waveforms::default();
        for i in 0..5 {
            let n = 20 + (i % 2) * 10;
            w.pulse.push(i as i64);
            w.gps_time.push(100.0 + i as f64);
            w.origin.push([f64::NAN; 3]);
            w.anchor.push([1000.0 + i as f64, 2000.5, 30.25]);
            let d = crate::transform::normalize(&[0.1, -0.2, -1.0]);
            w.direction.push(d);
            w.offset.push(-3.5 + i as f64);
            w.interval.push(if i == 3 { 2.0 } else { 1.0 });
            w.metres_per_ns.push(C_HALF);
            w.sample_start.push(w.samples.len());
            w.sample_count.push(n);
            w.samples.extend((0..n).map(|k| ((k * 7 + i as u32) % 50) as f32));
        }
        w.attrs.insert("intensity".into(), Attr::U16(vec![1, 2, 3, 4, 5]));
        w
    }

    #[test]
    fn round_trip_internal_and_external() {
        for external in [false, true] {
            let p = tmp(if external { "ext" } else { "int" });
            let w = sample();
            write_waveforms(&w, &p, &LasWriteWaveOptions { external, ..Default::default() }).unwrap();
            let info = read_info(&p).unwrap();
            assert_eq!(info.n_points, 5);
            assert_eq!(info.descriptors.len(), 3);
            assert_eq!(matches!(info.store, PacketStore::External(_)), external);
            let (r, next) = read_waveforms(&p, &LasReadOptions::default()).unwrap();
            assert_eq!(next, 5);
            assert_eq!(r.len(), 5);
            assert_eq!(r.samples, w.samples);
            assert_eq!(r.sample_count, w.sample_count);
            let (pa, pb) = (w.sample_positions(), r.sample_positions());
            for (a, b) in pa.iter().zip(&pb) {
                for k in 0..3 {
                    assert!((a[k] - b[k]).abs() < 1e-3, "{a:?} {b:?}");
                }
            }
            assert_eq!(r.gps_time, w.gps_time);
            assert_eq!(r.attrs["intensity"].to_f64(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
            // Chunks: every waveform exactly once.
            let mut start = 0;
            let mut seen = 0;
            while start < info.n_points {
                let (c, nx) = read_waveforms(&p, &LasReadOptions { start, count: Some(2), dedupe: true }).unwrap();
                seen += c.len();
                start = nx;
            }
            assert_eq!(seen, 5);
            // The las crate reads the points.
            let cloud = crate::io::las::read_las(&p).unwrap();
            assert_eq!(cloud.len(), 5);
            assert!((cloud.xyz[2][0] - 1002.0).abs() < 1e-9);
        }
    }

    #[test]
    fn float_samples_are_quantised() {
        let mut w = sample();
        for (k, s) in w.samples.iter_mut().enumerate() {
            *s = 0.37 * k as f32 - 3.0;
        }
        let p = tmp("float");
        write_waveforms(&w, &p, &LasWriteWaveOptions::default()).unwrap();
        let (r, _) = read_waveforms(&p, &LasReadOptions::default()).unwrap();
        let (g, _) = quantisation(&w.samples, 16);
        for (a, b) in w.samples.iter().zip(&r.samples) {
            assert!(((a - b).abs() as f64) <= 0.5 * g + 1e-4);
        }
    }

    #[test]
    fn shared_packets_become_one_waveform() {
        // Two records pointing to the same packet: write, then duplicate a record by hand.
        let w = sample().take(&[0]);
        let p = tmp("dup");
        write_waveforms(&w, &p, &LasWriteWaveOptions { external: true, ..Default::default() }).unwrap();
        let mut bytes = std::fs::read(&p).unwrap();
        let off = LE::read_u32(&bytes[96..]) as usize;
        let rec = bytes[off..off + 59].to_vec();
        bytes.splice(off..off, rec.iter().cloned());
        LE::write_u64(&mut bytes[247..], 2);
        std::fs::write(&p, &bytes).unwrap();
        let (r, _) = read_waveforms(&p, &LasReadOptions::default()).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r.attrs["n_records"].to_f64(), vec![2.0]);
        let (r, _) = read_waveforms(&p, &LasReadOptions { dedupe: false, ..Default::default() }).unwrap();
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn rejects_non_waveform_formats() {
        let p = tmp("nowave");
        let cloud = crate::PointCloud::new(vec![[0.0, 0.0, 0.0]]);
        crate::io::las::write_las(&cloud, &p, &Default::default()).unwrap();
        assert!(read_info(&p).is_err());
    }
}
