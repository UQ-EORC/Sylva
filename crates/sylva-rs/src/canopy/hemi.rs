// Sylva: LiDAR processing for forest ecology and remote sensing research.
// Copyright (C) 2026 Tim Devereux, The University of Queensland.
// Free software under the GNU General Public License v3.0 or later;
// see the LICENSE file. There is no warranty, to the extent permitted by law.
//! Synthetic hemispherical photographs from pulses or points.
//!
//! A hemispherical photograph looks up from the scanner through a fisheye
//! lens: the zenith is the image centre, the horizon its rim, and every pixel
//! is sky or canopy. From pulses that include their misses, a pixel's gap
//! fraction is the share of its pulses that returned nothing above a height,
//! which is the photograph the scanner itself took. From points alone a pixel
//! is canopy where a point lies in its direction, which is what thresholding
//! a photograph gives. Either way the image feeds the same ring gap fractions
//! and PAI estimators as a real photograph.

use crate::error::{Error, Result};
use crate::transform::{norm, sub};
use crate::{Point, Shots};

/// Fisheye projections: how the zenith angle maps to the distance from the
/// image centre.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lens {
    /// `r ∝ θ` (equiangular): most hemispherical-photography lenses and the
    /// assumption of most analysis software.
    Equidistant,
    /// `r ∝ 2 sin(θ/2)` (equal area; Sigma 8 mm, Nikon FC-E8).
    EquisolidAngle,
    /// `r ∝ sin θ`; sees at most 90 degrees.
    Orthographic,
    /// `r ∝ 2 tan(θ/2)`.
    Stereographic,
}

impl Lens {
    /// A lens by name: `equidistant` (or `equiangular`), `equisolid` (or
    /// `equal_area`), `orthographic`, `stereographic`.
    pub fn parse(name: &str) -> Result<Lens> {
        match name.trim().to_ascii_lowercase().replace(['-', ' '], "_").as_str() {
            "equidistant" | "equiangular" | "linear" => Ok(Lens::Equidistant),
            "equisolid" | "equisolid_angle" | "equal_area" => Ok(Lens::EquisolidAngle),
            "orthographic" => Ok(Lens::Orthographic),
            "stereographic" => Ok(Lens::Stereographic),
            other => Err(Error::invalid(format!("unknown lens {other:?}; use equidistant, equisolid, orthographic or stereographic"))),
        }
    }

    fn raw(self, theta: f64) -> f64 {
        match self {
            Lens::Equidistant => theta,
            Lens::EquisolidAngle => 2.0 * (theta / 2.0).sin(),
            Lens::Orthographic => theta.sin(),
            Lens::Stereographic => 2.0 * (theta / 2.0).tan(),
        }
    }

    /// Distance from the image centre of zenith `theta`, as a fraction of the
    /// distance of `theta_max` (both radians).
    pub fn radius(self, theta: f64, theta_max: f64) -> f64 {
        self.raw(theta) / self.raw(theta_max)
    }

    /// The zenith (radians) at a fraction `r` of the image radius: the inverse
    /// of [`Lens::radius`].
    pub fn zenith(self, r: f64, theta_max: f64) -> f64 {
        let x = r * self.raw(theta_max);
        match self {
            Lens::Equidistant => x,
            Lens::EquisolidAngle => 2.0 * (x / 2.0).clamp(-1.0, 1.0).asin(),
            Lens::Orthographic => x.clamp(-1.0, 1.0).asin(),
            Lens::Stereographic => 2.0 * (x / 2.0).atan(),
        }
    }
}

/// Where each direction lands in a square hemispherical image.
///
/// The image is `size` pixels square, the zenith at its centre and the
/// zenith angle `max_zenith` on the inscribed circle; rows run down the
/// image and columns right. The azimuth `north` points to the top. With
/// `mirror` the image is as a camera on the ground sees the sky, east on
/// the left; without it east is on the right, as on a map.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HemiGeometry {
    pub size: usize,
    pub lens: Lens,
    /// Zenith angle on the image rim (radians).
    pub max_zenith: f64,
    /// Azimuth at the top of the image (radians, clockwise from +y).
    pub north: f64,
    pub mirror: bool,
}

impl HemiGeometry {
    /// Angles in degrees; `size` at least 2, `max_zenith_deg` in (0, 180),
    /// at most 90 for an orthographic lens.
    pub fn new(size: usize, lens: Lens, max_zenith_deg: f64, north_deg: f64, mirror: bool) -> Result<HemiGeometry> {
        if size < 2 {
            return Err(Error::invalid("size must be at least 2 pixels"));
        }
        if !(max_zenith_deg > 0.0 && max_zenith_deg < 180.0) {
            return Err(Error::invalid(format!("max_zenith must be between 0 and 180 degrees, got {max_zenith_deg}")));
        }
        if lens == Lens::Orthographic && max_zenith_deg > 90.0 {
            return Err(Error::invalid("an orthographic lens sees at most 90 degrees"));
        }
        Ok(HemiGeometry { size, lens, max_zenith: max_zenith_deg.to_radians(), north: north_deg.to_radians(), mirror })
    }

    /// Half the image side: the radius of the image circle in pixels.
    pub fn radius_px(&self) -> f64 {
        self.size as f64 / 2.0
    }

    /// Zenith (radians) and azimuth (radians clockwise from +y) of a direction.
    fn zen_az(d: &Point) -> Option<(f64, f64)> {
        let n = norm(d);
        if !(n > 0.0) {
            return None;
        }
        let zen = (d[2] / n).clamp(-1.0, 1.0).acos();
        let az = d[0].atan2(d[1]);
        Some((zen, az))
    }

    /// Continuous image coordinates `(x, y)` of a direction (x right, y down,
    /// origin at the top-left corner) and its zenith; None beyond the rim.
    pub fn xy(&self, d: &Point) -> Option<(f64, f64, f64)> {
        let (zen, az) = Self::zen_az(d)?;
        if zen > self.max_zenith {
            return None;
        }
        let r = self.lens.radius(zen, self.max_zenith) * self.radius_px();
        let a = az - self.north;
        let c = self.radius_px();
        let x = if self.mirror { c - r * a.sin() } else { c + r * a.sin() };
        Some((x, c - r * a.cos(), zen))
    }

    /// Pixel `(row, col)` of a direction; None beyond the rim.
    pub fn pixel(&self, d: &Point) -> Option<(usize, usize)> {
        let (x, y, _) = self.xy(d)?;
        Some(self.clamp(y, x))
    }

    fn clamp(&self, y: f64, x: f64) -> (usize, usize) {
        let m = (self.size - 1) as f64;
        (y.floor().clamp(0.0, m) as usize, x.floor().clamp(0.0, m) as usize)
    }

    /// Zenith and azimuth (degrees) at the centre of a pixel; None outside
    /// the image circle.
    pub fn angles(&self, row: usize, col: usize) -> Option<(f64, f64)> {
        let c = self.radius_px();
        let x = col as f64 + 0.5 - c;
        let y = row as f64 + 0.5 - c;
        let rr = x.hypot(y) / c;
        if rr > 1.0 {
            return None;
        }
        let zen = self.lens.zenith(rr, self.max_zenith);
        let sx = if self.mirror { -x } else { x };
        let az = (sx.atan2(-y) + self.north).to_degrees().rem_euclid(360.0);
        Some((zen.to_degrees(), az))
    }

    /// Unit direction at the centre of a pixel; None outside the image circle.
    pub fn direction(&self, row: usize, col: usize) -> Option<Point> {
        let (zen, az) = self.angles(row, col)?;
        let (zen, az) = (zen.to_radians(), az.to_radians());
        Some([zen.sin() * az.sin(), zen.sin() * az.cos(), zen.cos()])
    }

    /// Pixels per radian of zenith at `theta`, the local scale of the lens.
    pub fn pixels_per_radian(&self, theta: f64) -> f64 {
        let h = 1e-4;
        let a = (theta - h).max(0.0);
        let b = (theta + h).min(self.max_zenith);
        if b <= a {
            return 0.0;
        }
        (self.lens.radius(b, self.max_zenith) - self.lens.radius(a, self.max_zenith)) / (b - a) * self.radius_px()
    }

    /// Zenith and azimuth (degrees) of every pixel, row-major, NaN outside the
    /// image circle.
    pub fn pixel_angles(&self) -> (Vec<f64>, Vec<f64>) {
        let n = self.size * self.size;
        let (mut zen, mut az) = (vec![f64::NAN; n], vec![f64::NAN; n]);
        for row in 0..self.size {
            for col in 0..self.size {
                if let Some((z, a)) = self.angles(row, col) {
                    zen[row * self.size + col] = z;
                    az[row * self.size + col] = a;
                }
            }
        }
        (zen, az)
    }
}

/// Per-pixel counts of a hemispherical image, row-major `size × size`.
///
/// `shots` and `ground` are only filled from pulses: with points they stay
/// zero and a pixel is canopy wherever `hits` is positive.
#[derive(Debug, Clone, PartialEq)]
pub struct HemiImage {
    pub geometry: HemiGeometry,
    /// Pulses fired through the pixel.
    pub shots: Vec<u32>,
    /// Pulses (or points) that met canopy in the pixel.
    pub hits: Vec<u32>,
    /// Pulses that returned only from at or below the canopy's lowest
    /// height, or beyond the range limit: terrain, the understorey, or a
    /// hillside above the horizon.
    pub ground: Vec<u32>,
    /// Range to the nearest hit (m); NaN without one.
    pub range: Vec<f64>,
    /// Sum and count of the optional per-hit value, for its mean.
    pub value_sum: Vec<f64>,
    pub value_n: Vec<u32>,
}

impl HemiImage {
    fn new(geometry: HemiGeometry) -> HemiImage {
        let n = geometry.size * geometry.size;
        HemiImage { geometry, shots: vec![0; n], hits: vec![0; n], ground: vec![0; n], range: vec![f64::NAN; n], value_sum: vec![0.0; n], value_n: vec![0; n] }
    }

    fn hit(&mut self, p: usize, range: f64, value: Option<f64>) {
        self.hits[p] += 1;
        if !(self.range[p] <= range) {
            self.range[p] = range;
        }
        if let Some(v) = value {
            self.value_sum[p] += v;
            self.value_n[p] += 1;
        }
    }
}

/// The photograph the scanner took: every pulse through a pixel counts as a
/// shot, as a hit when it has an echo above `min_height` (heights per echo,
/// any height when None) and within `max_range`, and as ground when it has
/// echoes but none that qualify. The nearest qualifying echo gives the
/// pixel's range and, with `value` (one per echo), its mean value. Needs
/// the pulses that returned nothing, so use a ray cloud or shots read with
/// their misses filled.
pub fn from_shots(shots: &Shots, geometry: HemiGeometry, echo_heights: Option<&[f64]>, min_height: f64, max_range: Option<f64>, value: Option<&[f64]>) -> Result<HemiImage> {
    for (name, v) in [("echo_heights", echo_heights), ("value", value)] {
        if let Some(v) = v {
            if v.len() != shots.n_echoes() {
                return Err(Error::invalid(format!("{name} has {} values for {} echoes", v.len(), shots.n_echoes())));
            }
        }
    }
    let mut img = HemiImage::new(geometry);
    let size = geometry.size;
    for s in 0..shots.n_shots() {
        let Some((row, col)) = geometry.pixel(&shots.direction[s]) else { continue };
        let p = row * size + col;
        img.shots[p] += 1;
        let a = shots.echo_start[s];
        let b = a + shots.echo_count[s] as usize;
        let hit = (a..b).find(|&e| echo_heights.is_none_or(|h| h[e] > min_height) && max_range.is_none_or(|m| shots.echo_range[e] <= m));
        if let Some(e) = hit {
            img.hit(p, shots.echo_range[e], value.map(|v| v[e]));
        } else if b > a {
            img.ground[p] += 1;
        }
    }
    Ok(img)
}

/// A photograph rendered from points seen from `origin`: every point within
/// `max_range` paints the pixels within half its angular width of its
/// direction, and its range and `value` (one per point) go to each. The
/// width is the larger of `footprint` (radians, e.g. the beam divergence)
/// and the angle `spacing` (metres, the cloud's point spacing) subtends at
/// the point's range, so that a thinned cloud does not show its sampling
/// up close. The footprint is painted as a disc in the image, which is
/// exact for an equidistant lens and close for the others.
pub fn from_points(points: &[Point], origin: Point, geometry: HemiGeometry, footprint: f64, spacing: f64, max_range: Option<f64>, value: Option<&[f64]>) -> Result<HemiImage> {
    if let Some(v) = value {
        if v.len() != points.len() {
            return Err(Error::invalid(format!("value has {} entries for {} points", v.len(), points.len())));
        }
    }
    if !(footprint >= 0.0) || !(spacing >= 0.0) {
        return Err(Error::invalid("footprint and spacing must not be negative"));
    }
    let mut img = HemiImage::new(geometry);
    let size = geometry.size;
    let c = geometry.radius_px();
    for (i, p) in points.iter().enumerate() {
        let d = sub(p, &origin);
        let r = norm(&d);
        if !(r > 0.0) || max_range.is_some_and(|m| r > m) {
            continue;
        }
        let Some((x, y, zen)) = geometry.xy(&d) else { continue };
        let v = value.map(|v| v[i]);
        let width = footprint.max(2.0 * (0.5 * spacing / r).atan());
        let rad = 0.5 * width * geometry.pixels_per_radian(zen);
        if rad < 0.5 {
            let (row, col) = geometry.clamp(y, x);
            img.hit(row * size + col, r, v);
            continue;
        }
        let m = (size - 1) as f64;
        let (x0, x1) = ((x - rad).floor().clamp(0.0, m) as usize, (x + rad).floor().clamp(0.0, m) as usize);
        let (y0, y1) = ((y - rad).floor().clamp(0.0, m) as usize, (y + rad).floor().clamp(0.0, m) as usize);
        for row in y0..=y1 {
            let cy = row as f64 + 0.5;
            for col in x0..=x1 {
                let cx = col as f64 + 0.5;
                if (cx - x).hypot(cy - y) <= rad && (cx - c).hypot(cy - c) <= c {
                    img.hit(row * size + col, r, v);
                }
            }
        }
    }
    Ok(img)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::f64::consts::PI;

    use super::*;

    fn geom(lens: Lens, mirror: bool) -> HemiGeometry {
        HemiGeometry::new(200, lens, 90.0, 0.0, mirror).unwrap()
    }

    fn dir(zen_deg: f64, az_deg: f64) -> Point {
        let (z, a) = (zen_deg.to_radians(), az_deg.to_radians());
        [z.sin() * a.sin(), z.sin() * a.cos(), z.cos()]
    }

    #[test]
    fn lenses_invert_and_are_named() {
        for lens in [Lens::Equidistant, Lens::EquisolidAngle, Lens::Orthographic, Lens::Stereographic] {
            for t in [0.0, 0.3, 1.0, PI / 2.0] {
                let r = lens.radius(t, PI / 2.0);
                assert!((lens.zenith(r, PI / 2.0) - t).abs() < 1e-9, "{lens:?} {t}");
            }
            assert_eq!(lens.radius(PI / 2.0, PI / 2.0), 1.0);
        }
        // Equidistant is linear, equisolid compresses the rim, stereographic stretches it.
        assert_eq!(Lens::Equidistant.radius(PI / 4.0, PI / 2.0), 0.5);
        assert!(Lens::EquisolidAngle.radius(PI / 4.0, PI / 2.0) > 0.5);
        assert!(Lens::Stereographic.radius(PI / 4.0, PI / 2.0) < 0.5);
        assert_eq!(Lens::parse("Equal-Area").unwrap(), Lens::EquisolidAngle);
        assert_eq!(Lens::parse("fish").unwrap_err().to_string(), "unknown lens \"fish\"; use equidistant, equisolid, orthographic or stereographic");
        assert!(HemiGeometry::new(200, Lens::Orthographic, 100.0, 0.0, true).is_err());
        assert!(HemiGeometry::new(1, Lens::Equidistant, 90.0, 0.0, true).is_err());
        assert!(HemiGeometry::new(200, Lens::Equidistant, 180.0, 0.0, true).is_err());
    }

    #[test]
    fn directions_land_where_a_photograph_puts_them() {
        let g = geom(Lens::Equidistant, true);
        // The zenith is the centre, north the top, and with `mirror` east is
        // on the left as in an upward photograph.
        assert_eq!(g.pixel(&[0.0, 0.0, 1.0]), Some((100, 100)));
        assert_eq!(g.pixel(&dir(45.0, 0.0)), Some((50, 100)));
        assert_eq!(g.pixel(&dir(45.0, 90.0)), Some((100, 50)));
        assert_eq!(geom(Lens::Equidistant, false).pixel(&dir(45.0, 90.0)), Some((100, 150)));
        // Beyond the rim is nothing; on the rim is the last pixel.
        assert_eq!(g.pixel(&dir(91.0, 0.0)), None);
        assert_eq!(g.pixel(&dir(90.0, 180.0)).unwrap().0, 199);
        // Turning north rotates the image.
        let r = HemiGeometry::new(200, Lens::Equidistant, 90.0, 90.0, false).unwrap();
        assert_eq!(r.pixel(&dir(45.0, 90.0)), Some((50, 100)));
    }

    #[test]
    fn pixel_angles_round_trip() {
        for lens in [Lens::Equidistant, Lens::EquisolidAngle, Lens::Stereographic] {
            let g = HemiGeometry::new(64, lens, 100.0, 30.0, true).unwrap();
            for (row, col) in [(0, 32), (32, 0), (8, 48), (40, 41), (31, 31)] {
                let d = g.direction(row, col).unwrap();
                assert_eq!(g.pixel(&d), Some((row, col)), "{lens:?} {row} {col}");
            }
            assert_eq!(g.angles(0, 0), None);
            let (zen, az) = g.pixel_angles();
            assert_eq!(zen.len(), 64 * 64);
            assert!(zen[0].is_nan() && az[0].is_nan());
            assert!(zen.iter().filter(|z| z.is_finite()).all(|&z| z <= 100.0 + 1e-9));
        }
    }

    #[test]
    fn shots_count_gaps_and_hits_per_pixel() {
        let g = geom(Lens::Equidistant, true);
        let direction = vec![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], dir(45.0, 0.0), dir(120.0, 0.0)];
        let counts = [0u32, 1, 2, 1, 1];
        let mut start = Vec::new();
        let mut s = 0;
        for &c in &counts {
            start.push(s);
            s += c as usize;
        }
        let shots = Shots { origin: vec![[0.0; 3]; 5], direction, echo_start: start, echo_count: counts.to_vec(), echo_range: vec![8.0, 3.0, 6.0, 12.0, 2.0], echo_attrs: BTreeMap::new() };
        let heights = [7.5, 0.2, 5.5, 8.0, 0.0];
        let img = from_shots(&shots, g, Some(&heights), 0.5, None, Some(&[1.0, 2.0, 3.0, 4.0, 5.0])).unwrap();
        let centre = 100 * 200 + 100;
        // Three pulses through the centre: a miss, a hit at 8 m, and one whose
        // first echo is below min_height so its second (6 m) is the hit.
        assert_eq!((img.shots[centre], img.hits[centre], img.ground[centre]), (3, 2, 0));
        assert_eq!(img.range[centre], 6.0);
        assert_eq!((img.value_sum[centre], img.value_n[centre]), (4.0, 2));
        assert_eq!((img.shots[50 * 200 + 100], img.hits[50 * 200 + 100]), (1, 1));
        // The downward pulse is beyond the 90 degree rim.
        assert_eq!(img.shots.iter().sum::<u32>(), 4);
        // A range limit turns the far hit into a gap.
        let near = from_shots(&shots, g, None, 0.0, Some(5.0), None).unwrap();
        assert_eq!((near.hits[centre], near.range[centre], near.ground[centre]), (1, 3.0, 1));
        // A pulse whose echoes are all below min_height is ground, not a miss.
        let low = from_shots(&shots, g, Some(&heights), 9.0, None, None).unwrap();
        assert_eq!((low.shots[centre], low.hits[centre], low.ground[centre]), (3, 0, 2));
        assert!(from_shots(&shots, g, Some(&[0.0; 2]), 0.0, None, None).is_err());
    }

    #[test]
    fn points_paint_their_footprint() {
        let g = geom(Lens::Equidistant, true);
        let pts = [[0.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, -5.0]];
        let one = from_points(&pts, [0.0; 3], g, 0.0, 0.0, None, Some(&[1.0, 2.0, 3.0])).unwrap();
        assert_eq!(one.hits.iter().sum::<u32>(), 2);
        assert_eq!((one.shots.iter().sum::<u32>(), one.ground.iter().sum::<u32>()), (0, 0));
        assert_eq!(one.range[100 * 200 + 100], 10.0);
        let (row, col) = g.pixel(&[0.0, 10.0, 10.0]).unwrap();
        assert_eq!(one.value_sum[row * 200 + col], 2.0);
        // A footprint of 4 degrees is 4/90 of the radius: about 4.4 pixels
        // across, so a disc of a dozen or so pixels.
        let wide = from_points(&pts, [0.0; 3], g, 4.0_f64.to_radians(), 0.0, None, None).unwrap();
        let painted = wide.hits.iter().filter(|&&h| h > 0).count();
        assert!((20..40).contains(&painted), "{painted}");
        // A spacing of 0.7 m subtends the same 4 degrees at 10 m, less at the
        // further point, so a smaller disc there.
        let spaced = from_points(&pts, [0.0; 3], g, 0.0, 10.0 * 2.0 * 2.0_f64.to_radians().tan(), None, None).unwrap();
        let n_spaced = spaced.hits.iter().filter(|&&h| h > 0).count();
        assert!(n_spaced > 12 && n_spaced < painted, "{n_spaced} vs {painted}");
        assert!(from_points(&pts, [0.0; 3], g, 0.0, 0.0, Some(5.0), None).unwrap().hits.iter().all(|&h| h == 0));
        assert!(from_points(&pts, [0.0; 3], g, -1.0, 0.0, None, None).is_err());
    }
}
