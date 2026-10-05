// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
use sylva_rs::{filters, ground, qsm, registration, trees, PointCloud, Transform};

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 11) as f64 / (1u64 << 53) as f64
}

fn stem(seed: &mut u64, cx: f64, cy: f64, r: f64, h: f64, n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|_| {
            let t = lcg(seed) * std::f64::consts::TAU;
            [cx + r * t.cos(), cy + r * t.sin(), lcg(seed) * h]
        })
        .collect()
}

#[test]
fn circle_fit_recovers_radius() {
    let mut s = 7;
    let xy: Vec<[f64; 2]> = (0..300)
        .map(|_| {
            let t = lcg(&mut s) * std::f64::consts::PI; // half circle
            [3.0 + 0.2 * t.cos(), -1.0 + 0.2 * t.sin()]
        })
        .collect();
    let (cx, cy, r, rmse) = trees::fit_circle(&xy).unwrap();
    assert!((cx - 3.0).abs() < 1e-6 && (cy + 1.0).abs() < 1e-6 && (r - 0.2).abs() < 1e-6);
    assert!(rmse < 1e-9);
}

#[test]
fn kabsch_recovers_rigid_transform() {
    let mut s = 1;
    let src: Vec<[f64; 3]> = (0..50).map(|_| [lcg(&mut s), lcg(&mut s), lcg(&mut s)]).collect();
    let t = Transform::rotation_z(20.0).compose(&Transform::translation(1.0, -2.0, 0.5));
    let dst: Vec<[f64; 3]> = src.iter().map(|p| t.apply(p)).collect();
    let est = registration::kabsch(&src, &dst).unwrap();
    for (a, b) in est.to_row_major().iter().zip(t.to_row_major()) {
        assert!((a - b).abs() < 1e-9);
    }
}

#[test]
fn cylinder_fit_and_qsm() {
    let mut s = 3;
    let pts = stem(&mut s, 0.0, 0.0, 0.15, 4.0, 4000);
    let fit = qsm::fit_cylinder(&pts, None).unwrap();
    assert!((fit.radius - 0.15).abs() < 1e-3);
    assert!(fit.axis[2] > 0.999);
    let model = qsm::build_qsm(&pts, None, &qsm::QsmParams { bin_length: 0.5, ..Default::default() }).unwrap();
    let expected = std::f64::consts::PI * 0.15 * 0.15 * 4.0;
    assert!((model.total_volume() - expected).abs() / expected < 0.2, "{}", model.total_volume());
}

#[test]
fn ground_and_stems_pipeline() {
    let mut s = 11;
    let mut xyz: Vec<[f64; 3]> = (0..20000)
        .map(|_| {
            let x = lcg(&mut s) * 20.0;
            let y = lcg(&mut s) * 20.0;
            [x, y, 0.05 * x + (lcg(&mut s) - 0.5) * 0.02]
        })
        .collect();
    xyz.extend(stem(&mut s, 5.0, 5.0, 0.15, 8.0, 20000).into_iter().map(|p| [p[0], p[1], p[2] + 0.25]));
    let cloud = PointCloud::new(xyz);
    let mask = ground::csf_ground_mask(&cloud.xyz, &ground::CsfParams::default());
    let ground_pts: Vec<_> = cloud.xyz.iter().zip(&mask).filter(|(_, &m)| m).map(|(p, _)| *p).collect();
    assert!(ground_pts.len() > 18000 && ground_pts.len() < 22000, "{}", ground_pts.len());
    let dtm = ground::make_dtm(&ground_pts, 0.5, None).unwrap();
    let h = ground::heights_above(&cloud.xyz, &dtm);
    let found = trees::detect_stems(&cloud.xyz, &h, &trees::StemParams::default());
    assert_eq!(found.len(), 1);
    assert!((found[0].dbh - 0.3).abs() < 0.01, "{}", found[0].dbh);
    let thinned = filters::voxel_downsample(&cloud, 0.5, false);
    assert!(thinned.len() < cloud.len());
}
