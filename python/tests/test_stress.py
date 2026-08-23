"""Stress suite: drift math must stay sane on identical, extreme, and degenerate inputs.

Drift statistics are the kind of thing that quietly returns NaN or infinity on an edge
case and poisons a monitoring dashboard. This suite feeds identical distributions
(drift must be ~0), wildly shifted ones (drift must be large and finite), constant and
tiny samples, and thousands of random pairs - asserting every result is finite,
non-negative, and monotonic in the obvious way.
"""

from __future__ import annotations

import math
import random

from drift import Severity, classify, detect_drift, kl_divergence, psi


def _normalish(rng: random.Random, n: int, center: float, spread: float) -> list[float]:
    # Sum-of-uniforms approximation of a normal - no numpy, keeps it portable.
    return [center + spread * (sum(rng.random() for _ in range(6)) - 3) for _ in range(n)]


def test_identical_distributions_show_no_drift():
    rng = random.Random(0)
    for _ in range(200):
        sample = _normalish(rng, 500, 50, 10)
        # Compare a sample to an independent draw from the same distribution.
        other = _normalish(rng, 500, 50, 10)
        result = detect_drift(sample, other)
        assert math.isfinite(result.psi) and result.psi >= 0
        assert result.psi < 0.25  # same distribution -> stable-ish, never "major"


def test_large_shift_is_flagged_major_and_finite():
    rng = random.Random(1)
    expected = _normalish(rng, 1000, 0, 5)
    actual = _normalish(rng, 1000, 100, 5)  # shifted far away
    result = detect_drift(expected, actual)
    assert math.isfinite(result.psi)
    assert result.severity is Severity.MAJOR


def test_psi_and_kl_are_never_negative_or_nan_under_fuzz():
    rng = random.Random(2)
    for _ in range(2000):
        n_e = rng.randint(10, 300)
        n_a = rng.randint(10, 300)
        expected = [rng.uniform(-100, 100) for _ in range(n_e)]
        actual = [rng.uniform(-100, 100) for _ in range(n_a)]
        p = psi(expected, actual)
        k = kl_divergence(expected, actual)
        assert math.isfinite(p) and p >= 0, f"bad psi: {p}"
        assert math.isfinite(k) and k >= 0, f"bad kl: {k}"


def test_constant_sample_does_not_blow_up():
    # A degenerate reference (all one value) must not divide by a zero-width range.
    result = detect_drift([5.0] * 100, [5.0] * 100)
    assert math.isfinite(result.psi)
    result2 = detect_drift([5.0] * 100, [9.0] * 100)
    assert math.isfinite(result2.psi)


def test_bigger_shift_gives_bigger_psi():
    rng = random.Random(3)
    base = _normalish(rng, 800, 0, 5)
    small = detect_drift(base, _normalish(rng, 800, 5, 5)).psi
    large = detect_drift(base, _normalish(rng, 800, 40, 5)).psi
    assert large > small


def test_classify_thresholds_are_ordered():
    # Severity must be monotonic in the PSI value.
    assert classify(0.0) is Severity.STABLE
    assert classify(1.0) is Severity.MAJOR
    # And never crash on extremes.
    assert classify(1e9) is Severity.MAJOR
