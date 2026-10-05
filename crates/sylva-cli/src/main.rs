// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! `sylva` command-line tool (Rust build; the Python package ships an
//! equivalent `sylva` entry point).

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use sylva_rs::pointcloud::Attr;
use sylva_rs::{canopy, filters, ground, io, qsm, trees, voxel, PointCloud, Raster, Shots};

#[derive(Parser)]
#[command(name = "sylva", version, about = "TLS processing for forest ecology")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print point count, bounds and attributes.
    Info { input: PathBuf },
    /// Convert between formats, optionally voxel-thinning.
    Convert {
        input: PathBuf,
        output: PathBuf,
        #[arg(long)]
        voxel: Option<f64>,
    },
    /// Classify ground, build a DTM and add a height attribute.
    Ground {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value = "csf")]
        method: String,
        #[arg(long, default_value_t = 0.5)]
        resolution: f64,
        /// Also write the DTM as an ESRI ASCII grid.
        #[arg(long)]
        dtm: Option<PathBuf>,
    },
    /// Detect stems and DBH from a height-normalised cloud.
    Trees {
        input: PathBuf,
        #[arg(short, long)]
        output: Option<PathBuf>,
        #[arg(long, default_value_t = 0.05)]
        min_dbh: f64,
        /// Write a cloud with a tree_id attribute to this path.
        #[arg(long)]
        segment: Option<PathBuf>,
    },
    /// Canopy height model (ESRI ASCII grid) from a height-normalised cloud.
    Chm {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value_t = 0.5)]
        resolution: f64,
    },
    /// Plant area density profile (voxel contact-frequency method).
    Pad {
        input: PathBuf,
        #[arg(long, default_value_t = 0.5)]
        voxel: f64,
    },
    /// Convert a ray cloud (sx,sy,sz or nx,ny,nz attributes) to a sylva shots
    /// file (.parquet): one row per pulse, misses without far points.
    Shots {
        input: PathBuf,
        output: PathBuf,
        /// Store angles and ranges in double precision.
        #[arg(long)]
        double: bool,
    },
    /// Ray-traced voxel grid (AMAPVox-style) from a shots file (.parquet,
    /// streamed) or a ray cloud. Writes .vox, or a text table otherwise.
    Voxel {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value_t = 0.1)]
        voxel: f64,
        /// Grid corners: x0 y0 z0 x1 y1 z1.
        #[arg(long, num_args = 6, allow_negative_numbers = true)]
        bounds: Option<Vec<f64>>,
        /// ESRI ASCII grid of terrain heights.
        #[arg(long)]
        dtm: Option<PathBuf>,
        /// Classification code of ground echoes.
        #[arg(long)]
        ground_class: Option<u8>,
        /// Echoes this close above the DTM are ground.
        #[arg(long, default_value_t = 0.2)]
        ground_distance: f64,
        #[arg(long, num_args = 0..)]
        leaf_classes: Vec<u8>,
        #[arg(long, num_args = 0..)]
        wood_classes: Vec<u8>,
        #[arg(long, default_value = "equal")]
        weighting: String,
        /// fpl, ppl, transmittance and/or bailey.
        #[arg(long, num_args = 1.., default_value = "fpl")]
        attenuation: Vec<String>,
        /// Scanner name (e.g. VZ-400) for beam-section metrics.
        #[arg(long)]
        laser: Option<String>,
        #[arg(long, default_value = "spherical")]
        lad: String,
        #[arg(long, num_args = 0..)]
        lad_params: Vec<f64>,
        /// Estimate per-tree inclination angle distributions.
        #[arg(long)]
        inclination: bool,
        /// Write the per-tree inclination distributions to this CSV.
        #[arg(long)]
        iad: Option<PathBuf>,
        #[arg(long)]
        occlusion: bool,
        #[arg(long)]
        flat_top: bool,
        #[arg(long, default_value_t = 0)]
        neighbour_priors: u32,
        #[arg(long, default_value_t = 0)]
        subvoxel_split: usize,
        /// Also write unobserved voxels.
        #[arg(long)]
        write_empty: bool,
        #[arg(long)]
        filled_only: bool,
    },
    /// Build a cylinder model of a single tree.
    Qsm {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value_t = 0.3)]
        bin_length: f64,
    },
}

fn is_shots_file(path: &std::path::Path) -> bool {
    path.extension().is_some_and(|e| e.eq_ignore_ascii_case("parquet"))
}

fn heights(cloud: &PointCloud) -> Vec<f64> {
    cloud.heights("height")
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Info { input } => {
            let c = io::read(&input).with_context(|| format!("reading {}", input.display()))?;
            println!("{}: {} points", input.display(), c.len());
            if let Some((lo, hi)) = c.bounds() {
                println!("  min: {lo:?}\n  max: {hi:?}");
            }
            for (k, a) in &c.attrs {
                let v = a.to_f64();
                let (mn, mx) = v.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
                println!("  {k}: {} [{mn} .. {mx}]", a.dtype());
            }
        }
        Command::Convert { input, output, voxel } => {
            let mut c = io::read(&input)?;
            if let Some(v) = voxel {
                c = filters::voxel_downsample(&c, v, false);
            }
            io::write(&c, &output)?;
            println!("wrote {} points to {}", c.len(), output.display());
        }
        Command::Ground { input, output, method, resolution, dtm } => {
            let mut c = io::read(&input)?;
            let mask = match method.as_str() {
                "csf" => ground::csf_ground_mask(&c.xyz, &ground::CsfParams { cloth_resolution: resolution, ..Default::default() }),
                "pmf" => ground::pmf_ground_mask(&c.xyz, &ground::PmfParams { cell_size: resolution, ..Default::default() })?,
                other => anyhow::bail!("unknown method {other:?} (csf|pmf)"),
            };
            c.attrs.insert("classification".into(), ground::classification_from_mask(&mask));
            let ground_pts: Vec<_> = c.xyz.iter().zip(&mask).filter(|(_, &m)| m).map(|(p, _)| *p).collect();
            let dtm_r = ground::make_dtm(&ground_pts, resolution, None)?;
            let h = ground::heights_above(&c.xyz, &dtm_r);
            c.attrs.insert("height".into(), Attr::F64(h));
            io::write(&c, &output)?;
            if let Some(p) = dtm {
                dtm_r.write_ascii_grid(&p, -9999.0)?;
            }
            println!("{} ground points; wrote {}", ground_pts.len(), output.display());
        }
        Command::Trees { input, output, min_dbh, segment } => {
            let c = io::read(&input)?;
            let h = heights(&c);
            let mut found = trees::detect_stems(&c.xyz, &h, &trees::StemParams { min_radius: min_dbh / 2.0, ..Default::default() });
            if let Some(p) = segment {
                let labels = trees::segment_trees(&c.xyz, &h, &found, &trees::SegmentParams::default());
                trees::tree_heights(&h, &labels, &mut found, 100.0);
                let mut out = c.clone();
                out.attrs.insert("tree_id".into(), Attr::I32(labels.iter().map(|&l| l as i32).collect()));
                io::write(&out, &p)?;
            }
            let mut text = String::from("tree_id,x,y,dbh,height,n_points,coverage,n_slices,quality\n");
            for t in &found {
                text += &format!("{},{:.4},{:.4},{:.4},{:.3},{},{:.3},{},{:.3}\n", t.tree_id, t.x, t.y, t.dbh, t.height, t.n_points, t.inlier_fraction, t.n_slices, t.quality);
            }
            match output {
                Some(p) => {
                    std::fs::write(&p, text)?;
                    println!("{} trees -> {}", found.len(), p.display());
                }
                None => print!("{text}"),
            }
        }
        Command::Chm { input, output, resolution } => {
            let c = io::read(&input)?;
            let h = heights(&c);
            let chm = ground::make_chm(&c.xyz, &h, resolution, None, 0.0)?;
            chm.write_ascii_grid(&output, -9999.0)?;
            println!("CHM {}x{} -> {}; cover(>2m)={:.2}", chm.nrows, chm.ncols, output.display(), canopy::canopy_cover(&chm.data, 2.0));
        }
        Command::Pad { input, voxel } => {
            let c = io::read(&input)?;
            let h = heights(&c);
            let (z, pad) = canopy::pad_profile_voxel(&c.xyz, &h, voxel, None, 1.0)?;
            println!("height,pad");
            for (zz, p) in z.iter().zip(&pad) {
                println!("{zz:.2},{p:.4}");
            }
            eprintln!("# PAI = {:.3}", pad.iter().sum::<f64>() * voxel);
        }
        Command::Shots { input, output, double } => {
            let shots = Shots::from_ray_cloud(&io::read(&input)?)?;
            io::shots::write_shots(&shots, &output, &io::shots::ShotsWriteOptions { double, ..Default::default() })?;
            println!("{} pulses, {} echoes -> {}", shots.n_shots(), shots.n_echoes(), output.display());
        }
        Command::Voxel { input, output, voxel: voxel_size, bounds, dtm, ground_class, ground_distance, leaf_classes, wood_classes, weighting, attenuation, laser, lad, lad_params, inclination, iad, occlusion, flat_top, neighbour_priors, subvoxel_split, write_empty, filled_only } => {
            let dtm = dtm.map(Raster::read_ascii_grid).transpose()?;
            let labels = voxel::EchoLabels { ground_class: ground_class.map(i64::from), ground_distance, leaf_classes: leaf_classes.iter().map(|&c| c as i64).collect(), wood_classes: wood_classes.iter().map(|&c| c as i64).collect(), ..Default::default() };
            let beam = match laser {
                Some(name) => {
                    let (diameter, divergence) = voxel::laser_spec(&name).with_context(|| format!("unknown laser {name:?}"))?;
                    Some(voxel::BeamSpec { diameter, divergence })
                }
                None => None,
            };
            let params = voxel::VoxelParams {
                voxel_size,
                bounds: bounds.map(|b| ([b[0], b[1], b[2]], [b[3], b[4], b[5]])),
                weighting: voxel::WeightMethod::parse(&weighting)?,
                occlusion,
                flat_top,
                neighbour_prior_min_rays: neighbour_priors,
                beam,
                subvoxel_split,
                lad: voxel::Lad::parse(&lad, &lad_params)?,
                attenuation: attenuation.iter().map(|m| voxel::Attenuation::parse(m)).collect::<Result<_, _>>()?,
                inclination,
                ..Default::default()
            };
            // Shots files are streamed; ray clouds have to be loaded whole.
            let (grid, n_shots) = if is_shots_file(&input) {
                let file = io::shots::ShotsFile::open(&input)?;
                (voxel::voxelize_file(&file, &params, &labels, dtm.as_ref())?, file.n_shots)
            } else {
                let shots = Shots::from_ray_cloud(&io::read(&input)?)?;
                let notes = labels.annotate(&shots, dtm.as_ref())?;
                (voxel::voxelize(&notes.inputs(&shots, dtm.as_ref()), &params)?, shots.n_shots())
            };
            let opts = voxel::WriteOptions { include_unobserved: write_empty, filled_only };
            let n = if output.extension().is_some_and(|e| e.eq_ignore_ascii_case("vox")) { grid.write_vox(&output, opts)? } else { grid.write_text(&output, opts)? };
            if let Some(p) = iad {
                grid.write_iad_csv(&p)?;
            }
            println!("{n_shots} pulses -> {}x{}x{} voxels @ {} m; wrote {n} to {}", grid.shape[0], grid.shape[1], grid.shape[2], grid.voxel_size, output.display());
        }
        Command::Qsm { input, output, bin_length } => {
            let c = io::read(&input)?;
            let model = qsm::build_qsm(&c.xyz, None, &qsm::QsmParams { bin_length, ..Default::default() })?;
            model.write_csv(&output)?;
            println!("n_cylinders: {}", model.len());
            println!("total_volume_m3: {:.5}", model.total_volume());
            println!("stem_volume_m3: {:.5}", model.stem_volume());
            println!("branch_volume_m3: {:.5}", model.branch_volume());
            println!("total_length_m: {:.3}", model.total_length());
            println!("max_branch_order: {}", model.max_branch_order());
            println!("dbh_m: {:.4}", model.dbh());
        }
    }
    Ok(())
}
