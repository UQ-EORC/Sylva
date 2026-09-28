// Sylva: terrestrial laser scanning processing for forest ecology.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Fusion of terrestrial (TLS) and airborne (ALS) lidar over the same forest.
//!
//! * [`register`]: a TLS plot placed on an ALS survey, by a search over
//!   heading and horizontal shift on the canopy height and terrain models,
//!   then a robust ICP, with the residuals and their uncertainty;
//! * [`link`]: TLS stems linked to ALS trees, with the stems under another
//!   tree's crown reported, and one table of TLS diameters and ALS heights;
//! * [`merge`]: one point cloud from both (TLS below, ALS above) and one
//!   plant area density profile taking each height from the instrument that
//!   sampled it better;
//! * [`upscale`]: plot values from TLS regressed on ALS area-based metrics,
//!   with leave-one-out cross-validation, and predicted wall to wall;
//! * [`synthetic`]: a terrestrial scanner that sees a synthetic scene as the
//!   airborne simulator does (every point a small sphere, the analytic
//!   terrain), so both instruments can be checked against one known forest.

#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod link;
pub mod merge;
pub mod register;
pub mod synthetic;
pub mod upscale;
