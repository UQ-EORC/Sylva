// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! A learned MTA decision rule: gradient-boosted trees trained on rimta5's
//! own output (`riegl-data/mta/train_mta.py`), evaluated here on the evidence
//! the three passes of [`super::mta`] gather per echo and zone.
//!
//! The model scores every (echo, zone) pair; the best-scoring zone is taken
//! and the echo kept when its score reaches the model's threshold. The
//! deterministic filters (amplitude 0, dynamic trigger, range gate, notch)
//! stay outside the model. The JSON format is plain enough to produce from
//! any tree library: per tree, parallel arrays `value`, `feature`,
//! `threshold`, `missing_left`, `left`, `right`, `leaf` (scikit-learn's
//! `HistGradientBoostingClassifier` node table), plus `baseline` (log-odds)
//! and the `features` list in the order [`row_features`] produces them.

use std::path::Path;

use crate::error::{Error, Result};

use super::mta::{Evidence, MAXZ};

/// Feature names, in the order the model expects them.
pub const FEATURE_NAMES: [&str; 23] = [
    "zone", "range_k", "refl_k", "votes_own_k", "votes_nb_k", "anchors_k", "density_k",
    "votes_own_other", "votes_nb_other", "anchors_other", "density_other",
    "votes_own_sum", "anchors_sum", "density_sum",
    "amplitude", "deviation", "r1", "late", "n_echoes", "idx_in_shot", "n_zones",
    "prev_amp", "prev_gap",
];

struct Node {
    value: f32,
    feature: u16,
    threshold: f32,
    missing_left: bool,
    left: u32,
    right: u32,
    leaf: bool,
}

struct Tree {
    nodes: Vec<Node>,
}

/// A boosted-tree decision rule loaded from JSON.
pub struct MtaModel {
    kmax: usize,
    baseline: f64,
    threshold: f32,
    trees: Vec<Tree>,
}

/// What the model needs to know about one echo besides the per-zone evidence.
#[derive(Clone, Copy, Debug)]
pub struct EchoInfo {
    pub amplitude: f32,
    pub deviation: f32,
    pub r1: f32,
    pub late: bool,
    pub n_echoes: u8,
    pub idx_in_shot: u8,
    /// Strongest amplitude among the closer echoes of the same pulse (-1 if none).
    pub prev_amp: f32,
    /// Range gap to the closest closer echo of the same pulse (-1 if none).
    pub prev_gap: f32,
}

impl MtaModel {
    /// Parse the JSON produced by `train_mta.py`.
    pub fn from_json(text: &str) -> Result<MtaModel> {
        let v: serde_json::Value = serde_json::from_str(text).map_err(|e| Error::Riegl(format!("MTA model: {e}")))?;
        let bad = |what: &str| Error::Riegl(format!("MTA model: missing or malformed {what}"));
        if v.get("format").and_then(|f| f.as_str()) != Some("sylva-mta-model/1") {
            return Err(bad("format (expected sylva-mta-model/1)"));
        }
        let names: Vec<&str> = v.get("features").and_then(|f| f.as_array()).ok_or_else(|| bad("features"))?.iter().filter_map(|s| s.as_str()).collect();
        if names != FEATURE_NAMES {
            return Err(Error::Riegl(format!("MTA model: feature list {names:?} does not match this build ({FEATURE_NAMES:?})")));
        }
        let kmax = v.get("kmax").and_then(|k| k.as_u64()).ok_or_else(|| bad("kmax"))? as usize;
        let baseline = v.get("baseline").and_then(|b| b.as_f64()).ok_or_else(|| bad("baseline"))?;
        let threshold = v.get("threshold").and_then(|b| b.as_f64()).ok_or_else(|| bad("threshold"))? as f32;
        let mut trees = Vec::new();
        for t in v.get("trees").and_then(|t| t.as_array()).ok_or_else(|| bad("trees"))? {
            let arr = |key: &str| -> Result<Vec<f64>> {
                t.get(key).and_then(|a| a.as_array()).ok_or_else(|| bad(key))?.iter().map(|x| x.as_f64().ok_or_else(|| bad(key))).collect()
            };
            let (value, feature, threshold, missing, left, right, leaf) =
                (arr("value")?, arr("feature")?, arr("threshold")?, arr("missing_left")?, arr("left")?, arr("right")?, arr("leaf")?);
            let n = value.len();
            if [feature.len(), threshold.len(), missing.len(), left.len(), right.len(), leaf.len()].iter().any(|&l| l != n) {
                return Err(bad("tree arrays of unequal length"));
            }
            let nodes = (0..n)
                .map(|i| Node {
                    value: value[i] as f32,
                    feature: feature[i] as u16,
                    threshold: threshold[i] as f32,
                    missing_left: missing[i] != 0.0,
                    left: left[i] as u32,
                    right: right[i] as u32,
                    leaf: leaf[i] != 0.0,
                })
                .collect();
            trees.push(Tree { nodes });
        }
        Ok(MtaModel { kmax: kmax.clamp(1, MAXZ), baseline, threshold, trees })
    }

    /// Load from a file.
    pub fn load(path: &Path) -> Result<MtaModel> {
        let text = std::fs::read_to_string(path).map_err(|e| Error::Riegl(format!("{}: {e}", path.display())))?;
        Self::from_json(&text)
    }

    /// Zones considered.
    pub fn kmax(&self) -> usize {
        self.kmax
    }

    /// Keep threshold on the probability.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Probability that rimta5 keeps this (echo, zone) pair.
    pub fn score(&self, x: &[f32; FEATURE_NAMES.len()]) -> f32 {
        let mut raw = self.baseline;
        for t in &self.trees {
            let mut i = 0usize;
            loop {
                let n = &t.nodes[i];
                if n.leaf {
                    raw += n.value as f64;
                    break;
                }
                let v = x[n.feature as usize];
                let go_left = if v.is_nan() { n.missing_left } else { v <= n.threshold };
                i = if go_left { n.left as usize } else { n.right as usize };
            }
        }
        (1.0 / (1.0 + (-raw).exp())) as f32
    }

    /// Best zone and its score for one echo (`zr`: hypothesis ranges, NaN = none).
    pub fn decide(&self, zr: &[f32; MAXZ], zrefl: &[f32; MAXZ], ev: &Evidence, info: EchoInfo) -> Option<(u8, f32)> {
        let mut best: Option<(u8, f32)> = None;
        for k in 0..self.kmax {
            if !zr[k].is_finite() {
                continue;
            }
            let x = row_features(k, zr, zrefl, ev, info, self.kmax);
            let p = self.score(&x);
            if best.map(|(_, b)| p > b).unwrap_or(true) {
                best = Some((k as u8 + 1, p));
            }
        }
        best
    }
}

/// The feature row for hypothesis `k` (0-based), exactly as `train_mta.py` builds it.
pub fn row_features(k: usize, zr: &[f32; MAXZ], zrefl: &[f32; MAXZ], ev: &Evidence, info: EchoInfo, kmax: usize) -> [f32; FEATURE_NAMES.len()] {
    let other_max = |a: &[f32; MAXZ]| -> f32 {
        let mut m = -1.0f32;
        for j in 0..kmax {
            if j != k && zr[j].is_finite() && a[j] > m {
                m = a[j];
            }
        }
        m
    };
    let sum_pos = |a: &[f32; MAXZ]| -> f32 { (0..kmax).map(|j| a[j].max(0.0)).sum() };
    let n_zones = (0..kmax).filter(|&j| zr[j].is_finite()).count() as f32;
    [
        k as f32 + 1.0,
        zr[k],
        zrefl[k],
        ev.votes_own[k],
        ev.votes_nb[k],
        ev.anchors[k],
        ev.density[k],
        other_max(&ev.votes_own),
        other_max(&ev.votes_nb),
        other_max(&ev.anchors),
        other_max(&ev.density),
        sum_pos(&ev.votes_own),
        sum_pos(&ev.anchors),
        sum_pos(&ev.density),
        info.amplitude,
        info.deviation,
        info.r1,
        info.late as u8 as f32,
        info.n_echoes as f32,
        info.idx_in_shot as f32,
        n_zones,
        info.prev_amp,
        info.prev_gap,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY: &str = r#"{"format":"sylva-mta-model/1","features":["zone","range_k","refl_k","votes_own_k","votes_nb_k","anchors_k","density_k","votes_own_other","votes_nb_other","anchors_other","density_other","votes_own_sum","anchors_sum","density_sum","amplitude","deviation","r1","late","n_echoes","idx_in_shot","n_zones","prev_amp","prev_gap"],"kmax":3,"baseline":-1.0,"threshold":0.5,
      "trees":[{"value":[0,2.0,-2.0],"feature":[3,0,0],"threshold":[5.0,0,0],"missing_left":[0,0,0],"left":[1,0,0],"right":[2,0,0],"leaf":[0,1,1]}]}"#;

    #[test]
    fn evaluates_a_hand_made_tree() {
        let m = MtaModel::from_json(TINY).unwrap();
        let mut x = [0f32; FEATURE_NAMES.len()];
        x[3] = 3.0; // votes_own_k <= 5 -> left leaf +2 -> raw 1.0
        assert!((m.score(&x) - 1.0 / (1.0 + (-1.0f32).exp())).abs() < 1e-6);
        x[3] = 9.0; // right leaf -2 -> raw -3
        assert!((m.score(&x) - 1.0 / (1.0 + (3.0f32).exp())).abs() < 1e-6);
    }

    #[test]
    fn decides_the_best_zone() {
        let m = MtaModel::from_json(TINY).unwrap();
        let mut zr = [f32::NAN; MAXZ];
        zr[0] = 20.0;
        zr[1] = 88.0;
        let mut ev = Evidence::default();
        ev.votes_own[0] = 9.0; // zone 1 scores low
        ev.votes_own[1] = 2.0; // zone 2 scores high
        let info = EchoInfo { amplitude: 10.0, deviation: 1.0, r1: 20.0, late: false, n_echoes: 1, idx_in_shot: 1, prev_amp: -1.0, prev_gap: -1.0 };
        let (zone, p) = m.decide(&zr, &zr, &ev, info).unwrap();
        assert_eq!(zone, 2);
        assert!(p > m.threshold());
    }

    #[test]
    fn rejects_a_foreign_feature_list() {
        let bad = TINY.replace("\"zone\",", "\"other\",");
        assert!(MtaModel::from_json(&bad).is_err());
    }
}
