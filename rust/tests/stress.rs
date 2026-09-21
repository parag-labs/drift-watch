//! Stress suite: drift math must stay sane on identical, extreme, and degenerate
//! inputs. Drift statistics quietly return NaN or infinity on an edge case and
//! poison a monitoring dashboard, so every result must be finite, non-negative,
//! and monotonic in the obvious way.

use drift_watch::{classify, detect_drift, kl_divergence, psi, Severity};

/// A tiny deterministic xorshift64* PRNG so the soak test is reproducible.
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

/// Sum-of-uniforms approximation of a normal - no dependencies, keeping the
/// stress suite portable across every language port.
fn normalish(rng: &mut Rng, n: usize, center: f64, spread: f64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            let s: f64 = (0..6).map(|_| rng.next_unit()).sum();
            center + spread * (s - 3.0)
        })
        .collect()
}

#[test]
fn identical_distributions_show_no_drift() {
    let mut rng = Rng::new(0);
    for _ in 0..200 {
        let sample = normalish(&mut rng, 500, 50.0, 10.0);
        let other = normalish(&mut rng, 500, 50.0, 10.0);
        let result = detect_drift(&sample, &other, None);
        assert!(result.psi.is_finite() && result.psi >= 0.0);
        assert!(result.psi < 0.25);
    }
}

#[test]
fn large_shift_is_flagged_major_and_finite() {
    let mut rng = Rng::new(1);
    let expected = normalish(&mut rng, 1000, 0.0, 5.0);
    let actual = normalish(&mut rng, 1000, 100.0, 5.0);
    let result = detect_drift(&expected, &actual, None);
    assert!(result.psi.is_finite());
    assert_eq!(result.severity, Severity::Major);
}

#[test]
fn psi_and_kl_are_never_negative_or_nan_under_fuzz() {
    let mut rng = Rng::new(2);
    for _ in 0..2000 {
        let n_e = 10 + (rng.next_unit() * 291.0) as usize;
        let n_a = 10 + (rng.next_unit() * 291.0) as usize;
        let expected: Vec<f64> = (0..n_e).map(|_| rng.next_unit() * 200.0 - 100.0).collect();
        let actual: Vec<f64> = (0..n_a).map(|_| rng.next_unit() * 200.0 - 100.0).collect();
        let p = psi(&expected, &actual, None);
        let k = kl_divergence(&expected, &actual, None);
        assert!(p.is_finite() && p >= 0.0, "bad psi: {p}");
        assert!(k.is_finite() && k >= 0.0, "bad kl: {k}");
    }
}

#[test]
fn constant_sample_does_not_blow_up() {
    let a = vec![5.0_f64; 100];
    assert!(detect_drift(&a, &a, None).psi.is_finite());
    let b = vec![9.0_f64; 100];
    assert!(detect_drift(&a, &b, None).psi.is_finite());
}

#[test]
fn bigger_shift_gives_bigger_psi() {
    let mut rng = Rng::new(3);
    let base = normalish(&mut rng, 800, 0.0, 5.0);
    let small = detect_drift(&base, &normalish(&mut rng, 800, 5.0, 5.0), None).psi;
    let large = detect_drift(&base, &normalish(&mut rng, 800, 40.0, 5.0), None).psi;
    assert!(large > small);
}

#[test]
fn classify_thresholds_are_ordered() {
    assert_eq!(classify(0.0), Severity::Stable);
    assert_eq!(classify(1.0), Severity::Major);
    assert_eq!(classify(1e9), Severity::Major);
}
