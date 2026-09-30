// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Block tracing against the whole-grid trace on a scanned synthetic forest.
use sylva_rs::io::shots::{write_shots, ShotsFile, ShotsWriteOptions};
use sylva_rs::voxel::{self, Attenuation, BeamSpec, BlockOptions, BlockedGrid, EchoLabels, Pulses, RayVoxels, VoxelParams, WriteOptions};
use sylva_rs::{synthetic, Raster, Shots};

/// Two scans of a four-tree forest on sloping ground, misses included.
fn scene() -> Shots {
    let trees = [(5.0, 5.0, 0.3, 9.0), (12.0, 6.0, 0.25, 7.0), (7.0, 13.0, 0.35, 10.0), (14.0, 14.0, 0.2, 6.0)];
    let cloud = synthetic::forest(&trees, 18.0, 6000, 2.0, 3);
    let mut out = Shots::default();
    for origin in [[9.0, 9.0, 1.6], [3.0, 15.0, 1.4]] {
        let s = synthetic::scan(&cloud, origin, 1.5, 125.0, 3, 0.4);
        let offset = out.echo_range.len();
        out.origin.extend(s.origin);
        out.direction.extend(s.direction);
        out.echo_start.extend(s.echo_start.iter().map(|e| e + offset));
        out.echo_count.extend(s.echo_count);
        out.echo_range.extend(s.echo_range);
        for (k, a) in s.echo_attrs {
            match out.echo_attrs.get_mut(&k) {
                Some(o) => o.extend(&a).unwrap(),
                None => {
                    out.echo_attrs.insert(k, a);
                }
            }
        }
    }
    assert!(out.echo_count.contains(&0), "the scene has misses");
    assert!(out.echo_count.iter().any(|&c| c > 1), "the scene has multi-echo pulses");
    out
}

fn dtm() -> Raster {
    let mut r = Raster::filled(24, 24, -3.0, -3.0, 1.0, 0.0);
    for row in 0..24 {
        for col in 0..24 {
            let (x, y) = r.cell_center(row, col);
            r.set(row, col, synthetic::terrain_height(x, y, 0.05));
        }
    }
    r
}

fn labels() -> EchoLabels {
    EchoLabels { ground_class: Some(2), leaf_classes: vec![4], wood_classes: vec![5], ..Default::default() }
}

fn params(occlusion: bool) -> VoxelParams {
    VoxelParams {
        voxel_size: 0.5,
        bounds: Some(([-1.0, -1.0, -1.5], [19.0, 19.0, 11.5])),
        occlusion,
        beam: Some(BeamSpec { diameter: 0.007, divergence: 0.00027 }),
        attenuation: vec![Attenuation::Fpl, Attenuation::Ppl],
        subvoxel_split: 2,
        ..Default::default()
    }
}

fn one_thread<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(f)
}

fn assert_same(a: &RayVoxels, b: &RayVoxels) {
    assert_eq!(a.shape, b.shape);
    assert_eq!(a.origin, b.origin);
    assert_eq!(a.i, b.i);
    for (x, y) in a.f.iter().zip(&b.f) {
        assert_eq!(x.len(), y.len());
        assert!(x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()), "sums differ");
    }
    assert_eq!(a.ppl_lambda, b.ppl_lambda);
    assert_eq!(a.subvoxel_counts, b.subvoxel_counts);
    assert_eq!(a.ground_height, b.ground_height);
    assert_eq!((a.has_leaf, a.has_wood), (b.has_leaf, b.has_wood));
}

fn whole(shots: &Shots, p: &VoxelParams, dtm: Option<&Raster>) -> RayVoxels {
    one_thread(|| sylva_rs::voxel_grid::voxelize_labelled(shots, p, &labels(), None, None, dtm).unwrap())
}

fn blocked(shots: &Shots, p: &VoxelParams, dtm: Option<&Raster>, block: [usize; 3], workers: usize) -> RayVoxels {
    let notes = labels().annotate(shots, dtm).unwrap();
    let opts = BlockOptions { block, workers, ..Default::default() };
    let (g, stats) = voxel::voxelize_blocks(&Pulses::Memory(notes.inputs(shots, dtm)), p, dtm, &opts, None).unwrap();
    assert!(stats.block_pulses > 0);
    g.unwrap()
}

#[test]
fn blocks_equal_the_whole_grid_bit_for_bit() {
    let shots = scene();
    let dtm = dtm();
    for occlusion in [false, true] {
        let p = params(occlusion);
        for d in [None, Some(&dtm)] {
            let reference = whole(&shots, &p, d);
            assert!(reference.i[voxel::I::NumHitWood as usize].iter().any(|&v| v > 0));
            assert!(reference.i[voxel::I::NumHitLeaf as usize].iter().any(|&v| v > 0));
            assert_eq!(reference.i[voxel::I::NumRaysOccluded as usize].iter().any(|&v| v > 0), occlusion);
            for (block, workers) in [([7, 7, 7], 1), ([7, 7, 7], 4), ([5, 11, 3], 3), ([64, 64, 64], 2), ([40, 1, 26], 4)] {
                assert_same(&blocked(&shots, &p, d, block, workers), &reference);
            }
        }
    }
}

#[test]
fn blocks_agree_with_a_parallel_whole_grid() {
    // Several threads add to a voxel in any order: the double-precision sums
    // differ by rounding, which a single-precision result rarely shows (and
    // sums that cancel, such as the azimuth sines, show in absolute terms).
    let shots = scene();
    let p = params(true);
    let notes = labels().annotate(&shots, None).unwrap();
    let reference = voxel::voxelize(&notes.inputs(&shots, None), &p).unwrap();
    let b = blocked(&shots, &p, None, [9, 9, 9], 0);
    assert_eq!(b.i, reference.i);
    for (x, y) in b.f.iter().zip(&reference.f) {
        for (p, q) in x.iter().zip(y) {
            assert!((p - q).abs() <= 2.0 * f32::EPSILON * p.abs().max(q.abs()) + 1e-12, "{p} vs {q}");
        }
    }
}

#[test]
fn other_options_and_whole_grid_refinements() {
    let shots = scene();
    let dtm = dtm();
    for p in [
        VoxelParams { flat_top: true, weighting: voxel::WeightMethod::First, ..params(true) },
        VoxelParams { weighting: voxel::WeightMethod::Full, unbounded_range: 6.0, attenuation: vec![Attenuation::Transmittance], beam: None, subvoxel_split: 0, ..params(true) },
        VoxelParams { neighbour_prior_min_rays: 5, inclination: true, ..params(false) },
    ] {
        let reference = whole(&shots, &p, Some(&dtm));
        let b = blocked(&shots, &p, Some(&dtm), [6, 8, 5], 3);
        assert_same(&b, &reference);
        assert_eq!(b.predominant_tree, reference.predominant_tree);
        assert_eq!(b.tree_iad.len(), reference.tree_iad.len());
    }
}

fn temp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("sylva_blocks_{name}_{}", std::process::id()))
}

/// Tracing state per voxel: 24 sums of 8 bytes, 8 counts of 4, and the sub-voxel cells.
fn voxel_bytes(p: &VoxelParams) -> u64 {
    (24 * 8 + 8 * 4 + p.subvoxel_split.pow(3)) as u64
}

#[test]
fn streamed_file_and_blocks_on_disk() {
    let shots = scene();
    let dtm = dtm();
    let p = params(true);
    let path = temp("shots.parquet");
    write_shots(&shots, &path, &ShotsWriteOptions { double: true, row_group_size: 9000, origin_tolerance: 0.0, ..Default::default() }).unwrap();
    let file = ShotsFile::open(&path).unwrap();
    assert!(file.n_groups() > 4);
    let lab = labels();
    let reference = one_thread(|| voxel::voxelize_file(&file, &p, &lab, Some(&dtm)).unwrap());

    // Streamed from the file, assembled in memory.
    let opts = BlockOptions { block: [8, 8, 8], workers: 3, ..Default::default() };
    let src = Pulses::File { file: &file, labels: &lab };
    let (g, _) = voxel::voxelize_blocks(&src, &p, Some(&dtm), &opts, None).unwrap();
    assert_same(&g.unwrap(), &reference);

    // Streamed, written block by block, with a small memory allowance: many passes.
    let dir = temp("grid");
    let small = BlockOptions { block: [8, 8, 8], workers: 2, max_memory: Some(3 * 8 * 8 * 8 * voxel_bytes(&p)) };
    let (none, stats) = voxel::voxelize_blocks(&src, &p, Some(&dtm), &small, Some(&dir)).unwrap();
    assert!(none.is_none());
    assert!(stats.n_passes >= 10, "{stats:?}");
    assert!(stats.peak_voxels <= 3 * 512, "{stats:?}");
    assert!(stats.peak_voxels * 10 < reference.n_voxels());
    let bg = BlockedGrid::open(&dir).unwrap();
    assert_eq!(bg.shape, reference.shape);
    assert_same(&bg.to_grid().unwrap(), &reference);
    assert_eq!(stats.blocks_written, bg.blocks_present().len());

    // Summaries over slabs of blocks equal the whole grid's.
    assert_eq!(bg.profile("free_path_length", 1.0).unwrap(), reference.profile("free_path_length", 1.0).unwrap());
    assert_eq!(bg.profile("num_hits", 3.0).unwrap(), reference.profile("num_hits", 3.0).unwrap());
    assert_eq!(bg.occlusion_profile(0.5, None).unwrap(), reference.occlusion_profile(0.5, None).unwrap());
    assert_eq!(bg.observed_map(0.0, Some(8.0)).unwrap(), reference.observed_map(0.0, Some(8.0)).unwrap());
    assert_eq!(bg.metric("pad_fpl").unwrap(), reference.metric("pad_fpl").unwrap());
    let pts = shots.echo_xyz();
    let tid: Vec<i64> = shots.echo_attrs["tree_id"].to_f64().iter().map(|&v| if v > 0.0 { v as i64 } else { -1 }).collect();
    // Debug text, as a tree without space above its top has a NaN share.
    assert_eq!(format!("{:?}", bg.tree_sampling(&pts, &tid, 5.0, 2.0).unwrap()), format!("{:?}", reference.tree_sampling(&pts, &tid, 5.0, 2.0).unwrap()));

    // The .vox file is the same text.
    let (a, b) = (temp("a.vox"), temp("b.vox"));
    reference.write_vox(&a, WriteOptions::default()).unwrap();
    bg.write(&b, false, WriteOptions::default()).unwrap();
    assert_eq!(std::fs::read_to_string(&a).unwrap(), std::fs::read_to_string(&b).unwrap());

    // A block read alone matches the same box of the whole grid.
    let blk = bg.block([1, 2, 0]).unwrap();
    assert_eq!(blk.shape, [8, 8, 8]);
    let idx = |g: &RayVoxels, i: usize, j: usize, k: usize| i + g.shape[0] * (j + g.shape[1] * k);
    for (i, j, k) in [(0, 0, 0), (3, 5, 7), (7, 7, 2)] {
        assert_eq!(blk.get_i(voxel::I::NumBeams, idx(&blk, i, j, k)), reference.get_i(voxel::I::NumBeams, idx(&reference, 8 + i, 16 + j, k)));
        assert_eq!(blk.get_f(voxel::F::FreePathLength, idx(&blk, i, j, k)), reference.get_f(voxel::F::FreePathLength, idx(&reference, 8 + i, 16 + j, k)));
    }
    for f in [&path, &a, &b] {
        let _ = std::fs::remove_file(f);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn refusals() {
    let shots = scene();
    let notes = labels().annotate(&shots, None).unwrap();
    let src = Pulses::Memory(notes.inputs(&shots, None));
    let dir = temp("refused");
    let incl = VoxelParams { inclination: true, ..params(false) };
    assert!(voxel::voxelize_blocks(&src, &incl, None, &BlockOptions::default(), Some(&dir)).is_err());
    let zero = BlockOptions { block: [0, 4, 4], ..Default::default() };
    assert!(voxel::voxelize_blocks(&src, &params(false), None, &zero, None).is_err());
    let tight = BlockOptions { block: [16, 16, 16], max_memory: Some(1000), ..Default::default() };
    assert!(voxel::voxelize_blocks(&src, &params(false), None, &tight, None).is_err());
    assert!(BlockedGrid::open(&dir).is_err());
}
