// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Internal plumbing shared across domains: numerics, numpy-compatible RNG, JSON, progress and limits.

pub mod json;
pub mod limits;
pub mod numeric;
pub mod nprandom;
pub mod optim;
pub mod progress;
pub mod pyformat;
pub mod relay;
pub mod spatial;
