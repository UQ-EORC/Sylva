// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
use sylva_rs::qsm::Cylinder;
use sylva_rs::voxel::{self, foliage, Attenuation, BeamSpec, VoxelInputs, VoxelParams, VoxelState, WeightMethod, WriteOptions, F, I};
use sylva_rs::Shots;

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 11) as f64 / (1u64 << 53) as f64
}

/// Shots straight down from z = 10 at the given `(x, y)`, with echo ranges.
fn down_shots(rays: &[(f64, f64, Vec<f64>)]) -> Shots {
    let mut s = Shots::default();
    for (x, y, ranges) in rays {
        s.origin.push([*x, *y, 10.0]);
        s.direction.push([0.0, 0.0, -1.0]);
        s.echo_start.push(s.echo_range.len());
        s.echo_count.push(ranges.len() as u32);
        s.echo_range.extend(ranges);
    }
    s
}

fn params(bounds: ([f64; 3], [f64; 3])) -> VoxelParams {
    VoxelParams { voxel_size: 1.0, bounds: Some(bounds), average_leaf_area: 0.0, ..Default::default() }
}

#[test]
fn single_pulse_accumulates_chords_and_free_path() {
    // Echo at z = 1.25: crosses voxels k = 3, 2 fully and stops 0.75 m into k = 1.
    let shots = down_shots(&[(0.5, 0.5, vec![8.75])]);
    let p = VoxelParams { occlusion: true, ..params(([0.0; 3], [1.0, 1.0, 4.0])) };
    let v = voxel::voxelize(&VoxelInputs::new(&shots), &p).unwrap();
    assert_eq!(v.shape, [1, 1, 4]);
    for k in [3, 2] {
        assert_eq!(v.state(k), VoxelState::Empty);
        assert_eq!(v.get_i(I::NumBeams, k), 1);
        assert_eq!(v.get_i(I::NumMissRays, k), 1);
        assert!((v.get_f(F::FreePathLength, k) - 1.0).abs() < 1e-5);
    }
    assert_eq!(v.state(1), VoxelState::Filled);
    assert_eq!(v.get_i(I::NumHits, 1), 1);
    assert_eq!(v.get_i(I::NumHitPlant, 1), 1);
    assert_eq!(v.get_i(I::NumMissRays, 1), 0);
    assert!((v.get_f(F::PathLength, 1) - 1.0).abs() < 1e-5, "potential path is the full chord");
    assert!((v.get_f(F::FreePathLength, 1) - 0.75).abs() < 1e-5);
    assert!((v.get_f(F::SumHitDelta, 1) - 1.0).abs() < 1e-5);
    assert!((v.mean_zenith(1).to_degrees() - 180.0).abs() < 1e-3);
    // Beyond the echo: the rest of k = 1 and all of k = 0.
    assert_eq!(v.state(0), VoxelState::Occluded);
    assert!((v.get_f(F::PathLengthOccluded, 0) - 1.0).abs() < 1e-5);
    assert!((v.get_f(F::PathLengthOccluded, 1) - 0.25).abs() < 1e-5);
}

#[test]
fn echo_weights_share_the_pulse() {
    let shots = down_shots(&[(0.5, 0.5, vec![6.5, 8.5])]); // echoes in k = 3 and k = 1
    let v = voxel::voxelize(&VoxelInputs::new(&shots), &params(([0.0; 3], [1.0, 1.0, 4.0]))).unwrap();
    // The second segment starts at the first echo, carrying half the pulse; as in
    // rayvoxel it counts again in the voxel it starts in.
    assert!((v.get_f(F::NumBeamsWeighted, 3) - 1.5).abs() < 1e-6);
    assert!((v.get_f(F::NumBeamsWeighted, 2) - 0.5).abs() < 1e-6);
    assert!((v.get_f(F::FreePathLength, 3) - (0.5 + 0.5 * 0.5)).abs() < 1e-5);
    assert_eq!(v.get_i(I::NumBeams, 3), 1, "one pulse, however many echoes");

    let first = VoxelParams { weighting: WeightMethod::First, ..params(([0.0; 3], [1.0, 1.0, 4.0])) };
    let v = voxel::voxelize(&VoxelInputs::new(&shots), &first).unwrap();
    assert_eq!(v.get_i(I::NumHits, 3), 1);
    assert_eq!(v.get_i(I::NumHits, 1), 0, "the pulse is spent at its first echo");
    assert_eq!(v.state(2), VoxelState::Unobserved);
}

#[test]
fn ground_echoes_and_misses_are_traversed_but_never_hits() {
    let shots = down_shots(&[(0.5, 0.5, vec![9.5]), (0.5, 0.5, vec![])]);
    let ground = [true];
    let inputs = VoxelInputs { ground: Some(&ground), ..VoxelInputs::new(&shots) };
    let v = voxel::voxelize(&inputs, &params(([0.0; 3], [1.0, 1.0, 4.0]))).unwrap();
    assert_eq!(v.i[I::NumHits as usize].iter().sum::<i32>(), 0);
    assert_eq!(v.get_i(I::NumBeams, 3), 2);
    assert_eq!(v.get_i(I::NumUnboundRays, 3), 2);
    assert_eq!(v.get_i(I::NumBeams, 0), 2, "the echo-less pulse runs to the grid floor, the ground one to z = 0.5");
    assert!((v.get_f(F::PathLengthUnbound, 0) - 1.5).abs() < 1e-5);
}

/// Pulses through a 1 m slab of a turbid medium with attenuation `lambda`.
fn turbid(lambda: f64, n: usize) -> Shots {
    let mut seed = 42;
    let rays: Vec<_> = (0..n)
        .map(|_| {
            let (x, y) = (lcg(&mut seed), lcg(&mut seed));
            let free = -(1.0 - lcg(&mut seed)).ln() / lambda;
            // Slab spans z in [1, 2]; pulses that get through hit the floor at z = 0.
            (x, y, vec![if free < 1.0 { 8.0 + free } else { 10.0 }])
        })
        .collect();
    down_shots(&rays)
}

#[test]
fn attenuation_estimators_recover_a_turbid_medium() {
    let lambda = 0.8;
    let shots = turbid(lambda, 20_000);
    let ground: Vec<bool> = shots.echo_range.iter().map(|&r| r >= 10.0).collect();
    let inputs = VoxelInputs { ground: Some(&ground), ..VoxelInputs::new(&shots) };
    let p = VoxelParams {
        attenuation: vec![Attenuation::Fpl, Attenuation::Ppl, Attenuation::Transmittance],
        beam: Some(BeamSpec { diameter: 0.007, divergence: 0.00035 }),
        ..params(([0.0; 3], [1.0, 1.0, 3.0]))
    };
    let v = voxel::voxelize(&inputs, &p).unwrap();
    let slab = 1;
    for m in [Attenuation::Fpl, Attenuation::Ppl, Attenuation::Transmittance] {
        let est = v.attenuation(slab, m);
        assert!((est - lambda).abs() < 0.03, "{}: {est}", m.name());
    }
    assert!(v.fpl_bias(slab) > 0.0 && v.fpl_bias(slab) < 1e-3);
    assert!((v.transmittance(slab) - (-lambda).exp()).abs() < 0.02);
    // Spherical G = 0.5.
    assert!((v.area_density(slab, Attenuation::Ppl).pad - lambda / 0.5).abs() < 0.06);
    assert!((v.pad_g_corrected(slab) - lambda / 0.5).abs() < 0.06);
    assert_eq!(v.attenuation(2, Attenuation::Fpl), 0.0);
    assert!(v.metric("pad_ppl").unwrap()[slab] > 1.0);
    assert!(v.metric("nonsense").is_err());
}

#[test]
fn leaf_and_wood_split_the_density() {
    let shots = turbid(1.0, 8_000);
    let ground: Vec<bool> = shots.echo_range.iter().map(|&r| r >= 10.0).collect();
    let fol: Vec<u8> = (0..shots.n_echoes()).map(|e| if e % 4 == 0 { foliage::WOOD } else { foliage::LEAF }).collect();
    let inputs = VoxelInputs { ground: Some(&ground), foliage: Some(&fol), ..VoxelInputs::new(&shots) };
    let v = voxel::voxelize(&inputs, &params(([0.0; 3], [1.0, 1.0, 3.0]))).unwrap();
    assert!(v.has_leaf && v.has_wood);
    let d = v.area_density(1, Attenuation::Transmittance);
    assert!((d.lad + d.wad - d.pad).abs() < 1e-9);
    assert!((d.wad / d.pad - 0.25).abs() < 0.03);
}

#[test]
fn inclination_of_a_horizontal_sheet_is_planophile() {
    // Echoes on the plane z = 1.5, scanned from above.
    let mut seed = 3;
    let rays: Vec<_> = (0..3000).map(|_| (4.0 * lcg(&mut seed), 4.0 * lcg(&mut seed), vec![8.5])).collect();
    let shots = down_shots(&rays);
    let p = VoxelParams { inclination: true, ..params(([0.0; 3], [4.0, 4.0, 3.0])) };
    let v = voxel::voxelize(&VoxelInputs::new(&shots), &p).unwrap();
    let iad = &v.tree_iad[&0];
    assert!(iad.piad[0] > 0.99, "flat leaves fall in the first inclination bin");
    assert_eq!(iad.piad_de_wit, Some("planophile"));
    // Horizontal leaves under a vertical beam project fully.
    assert!(iad.g_plant > 0.95, "{}", iad.g_plant);
    let hit = (0..v.n_voxels()).find(|&i| v.get_i(I::NumHits, i) > 0).unwrap();
    assert_eq!(v.predominant_tree.as_ref().unwrap()[hit], 0);
}

#[test]
fn subvoxels_priors_and_wood_volume() {
    let mut seed = 9;
    let rays: Vec<_> = (0..4000).map(|_| (3.0 * lcg(&mut seed), 3.0 * lcg(&mut seed), vec![10.0])).collect();
    let shots = down_shots(&rays);
    let ground = vec![true; shots.n_echoes()];
    let inputs = VoxelInputs { ground: Some(&ground), ..VoxelInputs::new(&shots) };
    let p = VoxelParams { subvoxel_split: 2, subvoxel_min_beams: 5, ..params(([0.0; 3], [3.0, 3.0, 3.0])) };
    let mut v = voxel::voxelize(&inputs, &p).unwrap();
    assert_eq!(v.exploration(13), (1.0, 0xff));

    let beams = v.get_f(F::NumBeamsWeighted, 13);
    let primed = voxel::voxelize(&inputs, &VoxelParams { neighbour_prior_min_rays: 2 * beams as u32, ..p.clone() }).unwrap();
    assert!((primed.get_f(F::NumBeamsWeighted, 13) - 2.0 * beams.floor()).abs() < 1.0);
    assert_eq!(primed.get_f(F::NumBeamsWeighted, 0), v.get_f(F::NumBeamsWeighted, 0), "the border is untouched");

    let cyl = Cylinder { start: [1.5, 1.5, 0.2], axis: [0.0, 0.0, 1.0], length: 2.5, radius: 0.3, parent: -1, branch_order: 0, branch_id: 0, n_points: 0 };
    v.add_wood_volume(std::slice::from_ref(&cyl));
    let total: f64 = v.wood_volume.as_ref().unwrap().iter().map(|&w| w as f64).sum();
    assert!((total - cyl.volume()).abs() < 1e-4);
    assert!(v.metric("wood_volume_density").unwrap()[13] > 0.2);

    let dir = std::env::temp_dir().join(format!("sylva_voxel_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let n = v.write_vox(dir.join("a.vox"), WriteOptions::default()).unwrap();
    assert_eq!(n, 27);
    let text = std::fs::read_to_string(dir.join("a.vox")).unwrap();
    assert!(text.starts_with("VOXEL SPACE\n#g_correction:analytic LAD (spherical)\n"));
    assert!(text.contains("#split:3 3 3\n"));
    let header = text.lines().find(|l| l.starts_with("i j k")).unwrap();
    assert!(header.ends_with("pad_g_corrected wood_volume wood_volume_density"));
    assert_eq!(text.lines().last().unwrap().split(' ').count(), header.split(' ').count());
    assert_eq!(v.write_text(dir.join("a.txt"), WriteOptions { filled_only: true, ..Default::default() }).unwrap(), 0);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn multi_echo_pulses_count_hits_by_their_share() {
    // Two echoes of one pulse in the same voxel (z = 1.5 and 1.2): two hits, one pulse.
    let shots = down_shots(&[(0.5, 0.5, vec![8.5, 8.8])]);
    let v = voxel::voxelize(&VoxelInputs::new(&shots), &params(([0.0; 3], [1.0, 1.0, 4.0]))).unwrap();
    assert_eq!(v.get_i(I::NumHits, 1), 2);
    assert!((v.get_f(F::HitsWeighted, 1) - 1.0).abs() < 1e-6);

    // Pulses with up to three echoes in a 1 m slab. Without beam geometry the
    // FPL estimate uses the weighted hits, so it must equal the beam-section
    // estimate when every section is the same (no divergence) and there is no
    // leaf-size correction.
    let mut seed = 7;
    let rays: Vec<_> = (0..20_000)
        .map(|_| {
            let (x, y) = (lcg(&mut seed), lcg(&mut seed));
            let mut r = 8.0;
            let mut echoes = Vec::new();
            while echoes.len() < 3 {
                r += -(1.0 - lcg(&mut seed)).ln() / 0.8;
                if r >= 9.0 {
                    break;
                }
                echoes.push(r);
            }
            (x, y, echoes)
        })
        .collect();
    let shots = down_shots(&rays);
    let multi = shots.echo_count.iter().filter(|&&c| c > 1).count();
    assert!(multi > 2_000, "{multi} multi-echo pulses");
    let bounds = ([0.0; 3], [1.0, 1.0, 3.0]);
    let plain = voxel::voxelize(&VoxelInputs::new(&shots), &params(bounds)).unwrap();
    let beam = VoxelParams { beam: Some(BeamSpec { diameter: 0.01, divergence: 0.0 }), ..params(bounds) };
    let beam = voxel::voxelize(&VoxelInputs::new(&shots), &beam).unwrap();
    let (a, b) = (plain.attenuation(1, Attenuation::Fpl), beam.attenuation(1, Attenuation::Fpl));
    assert!((a - b).abs() < 1e-4 * b, "no beam {a} vs beam {b}");
    // Counting every echo as a whole hit would put the plain estimate well above.
    let counts = plain.get_i(I::NumHits, 1) as f64 / plain.get_f(F::FreePathLength, 1) as f64;
    assert!(counts > 1.2 * a, "counts {counts} vs weighted {a}");
    // A pulse is intercepted at most once: the contact-frequency density cannot
    // exceed 2 x (pulses) / path.
    let bound = 2.0 * plain.get_i(I::NumBeams, 1) as f64 / plain.get_f(F::PathLength, 1) as f64;
    assert!(plain.pad_g0_5(1) <= bound);
}
