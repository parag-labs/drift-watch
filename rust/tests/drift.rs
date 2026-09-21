//! DriftWatch detector tests: PSI is zero for identical data, rises with drift,
//! and classifies by the standard thresholds.

use drift_watch::{classify, detect_drift, kl_divergence, psi, Severity};

/// A tiny deterministic xorshift64* PRNG so the tests are dependency-free.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    fn next_unit(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (v >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Draws `n` normal samples via Box-Muller from a seeded generator.
fn gauss(n: usize, mean: f64, std: f64, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let u1 = 1.0 - rng.next_unit();
            let u2 = 1.0 - rng.next_unit();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            mean + std * z
        })
        .collect()
}

#[test]
fn psi_zero_for_identical() {
    let data: Vec<f64> = (0..1000).map(|i| (i % 10) as f64).collect();
    assert!(psi(&data, &data, None) < 1e-3);
}

#[test]
fn psi_increases_with_shift() {
    let base = gauss(2000, 0.0, 1.0, 1);
    let small = gauss(2000, 0.2, 1.0, 2);
    let large = gauss(2000, 3.0, 1.0, 3);
    assert!(psi(&base, &small, None) < psi(&base, &large, None));
}

#[test]
fn classify_thresholds() {
    assert_eq!(classify(0.05), Severity::Stable);
    assert_eq!(classify(0.15), Severity::Minor);
    assert_eq!(classify(0.40), Severity::Major);
}

#[test]
fn detect_drift_major_on_big_shift() {
    let base = gauss(3000, 0.0, 1.0, 2);
    let shifted = gauss(3000, 5.0, 1.0, 5);
    let result = detect_drift(&base, &shifted, None);
    assert_eq!(result.severity, Severity::Major);
    assert!(result.psi > 0.25);
}

#[test]
fn detect_drift_stable_on_same_distribution() {
    let base = gauss(5000, 0.0, 1.0, 3);
    let similar = gauss(5000, 0.0, 1.0, 33);
    let result = detect_drift(&base, &similar, None);
    assert_eq!(result.severity, Severity::Stable);
}

#[test]
fn kl_nonnegative() {
    let base = gauss(1000, 0.0, 1.0, 4);
    let other = gauss(1000, 1.0, 1.0, 44);
    assert!(kl_divergence(&base, &other, None) >= -1e-6);
}

#[test]
fn constant_feature_does_not_crash() {
    let const_data = vec![5.0_f64; 500];
    let result = detect_drift(&const_data, &const_data, None);
    assert_eq!(result.severity, Severity::Stable);
}

#[test]
fn default_bins_match_explicit_ten() {
    let base = gauss(1500, 0.0, 1.0, 7);
    let other = gauss(1500, 1.5, 1.0, 8);
    assert_eq!(psi(&base, &other, None), psi(&base, &other, Some(10)));
    assert_eq!(
        kl_divergence(&base, &other, None),
        kl_divergence(&base, &other, Some(10))
    );
}
