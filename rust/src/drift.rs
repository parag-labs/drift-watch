//! PSI, KL-divergence, and severity classification over binned distributions.

const EPS: f64 = 1e-6;

/// The default bin count used when a caller passes `None`, mirroring the Python
/// reference default of 10.
const DEFAULT_BINS: usize = 10;

/// How far a distribution has drifted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// No meaningful drift (PSI < 0.10).
    Stable,
    /// Moderate drift worth watching (0.10 <= PSI < 0.25).
    Minor,
    /// Significant drift that should trigger an alert (PSI >= 0.25).
    Major,
}

/// The two drift metrics plus the derived severity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriftResult {
    /// Population Stability Index.
    pub psi: f64,
    /// KL(actual || expected) over the binned distributions.
    pub kl: f64,
    /// Severity derived from `psi`.
    pub severity: Severity,
}

fn round6(v: f64) -> f64 {
    (v * 1e6).round() / 1e6
}

/// Builds `bins + 1` equal-width edges spanning the range of `values`. A
/// degenerate (zero-width) range is widened by one so binning never divides by
/// zero.
fn bin_edges(values: &[f64], bins: usize) -> Vec<f64> {
    let lo = values.iter().copied().fold(values[0], f64::min);
    let mut hi = values.iter().copied().fold(values[0], f64::max);
    if hi == lo {
        hi = lo + 1.0;
    }
    let width = (hi - lo) / bins as f64;
    (0..=bins).map(|i| lo + i as f64 * width).collect()
}

/// Returns the fraction of `values` in each bin, clamping out-of-range values
/// into the first or last bin.
fn proportions(values: &[f64], edges: &[f64]) -> Vec<f64> {
    let bins = edges.len() - 1;
    let span = edges[edges.len() - 1] - edges[0];
    let mut counts = vec![0_i64; bins];
    for &v in values {
        let raw = ((v - edges[0]) / span * bins as f64) as i64;
        let idx = raw.clamp(0, bins as i64 - 1) as usize;
        counts[idx] += 1;
    }
    let total = if values.is_empty() { 1 } else { values.len() };
    counts.iter().map(|&c| c as f64 / total as f64).collect()
}

/// Population Stability Index between the expected and actual samples. 0 means
/// identical distributions; higher means more drift. `bins` defaults to 10 when
/// `None`.
pub fn psi(expected: &[f64], actual: &[f64], bins: Option<usize>) -> f64 {
    let bins = bins.unwrap_or(DEFAULT_BINS);
    let edges = bin_edges(expected, bins);
    let e = proportions(expected, &edges);
    let a = proportions(actual, &edges);
    let total: f64 = e
        .iter()
        .zip(a.iter())
        .map(|(&ei, &ai)| {
            let ei = ei.max(EPS);
            let ai = ai.max(EPS);
            (ai - ei) * (ai / ei).ln()
        })
        .sum();
    round6(total)
}

/// KL(actual || expected) over the binned distributions. `bins` defaults to 10
/// when `None`.
pub fn kl_divergence(expected: &[f64], actual: &[f64], bins: Option<usize>) -> f64 {
    let bins = bins.unwrap_or(DEFAULT_BINS);
    let edges = bin_edges(expected, bins);
    let e = proportions(expected, &edges);
    let a = proportions(actual, &edges);
    let total: f64 = e
        .iter()
        .zip(a.iter())
        .map(|(&ei, &ai)| {
            let ei = ei.max(EPS);
            let ai = ai.max(EPS);
            ai * (ai / ei).ln()
        })
        .sum();
    round6(total)
}

/// Maps a PSI value to a [`Severity`] using the industry-standard thresholds
/// (0.10 and 0.25).
pub fn classify(psi_value: f64) -> Severity {
    if psi_value < 0.1 {
        Severity::Stable
    } else if psi_value < 0.25 {
        Severity::Minor
    } else {
        Severity::Major
    }
}

/// Computes PSI and KL-divergence and classifies the result. `bins` defaults to
/// 10 when `None`.
pub fn detect_drift(expected: &[f64], actual: &[f64], bins: Option<usize>) -> DriftResult {
    let p = psi(expected, actual, bins);
    DriftResult {
        psi: p,
        kl: kl_divergence(expected, actual, bins),
        severity: classify(p),
    }
}
