"""DriftWatch tests: PSI is zero for identical data, rises with drift, classifies."""

import random

from drift import Severity, classify, detect_drift, kl_divergence, psi


def test_psi_zero_for_identical():
    data = [float(i % 10) for i in range(1000)]
    assert psi(data, data) < 1e-3


def test_psi_increases_with_shift():
    random.seed(1)
    base = [random.gauss(0, 1) for _ in range(2000)]
    small = [random.gauss(0.2, 1) for _ in range(2000)]
    large = [random.gauss(3.0, 1) for _ in range(2000)]
    assert psi(base, small) < psi(base, large)


def test_classify_thresholds():
    assert classify(0.05) is Severity.STABLE
    assert classify(0.15) is Severity.MINOR
    assert classify(0.40) is Severity.MAJOR


def test_detect_drift_major_on_big_shift():
    random.seed(2)
    base = [random.gauss(0, 1) for _ in range(3000)]
    shifted = [random.gauss(5.0, 1) for _ in range(3000)]
    result = detect_drift(base, shifted)
    assert result.severity is Severity.MAJOR
    assert result.psi > 0.25


def test_detect_drift_stable_on_same_distribution():
    random.seed(3)
    base = [random.gauss(0, 1) for _ in range(5000)]
    similar = [random.gauss(0, 1) for _ in range(5000)]
    result = detect_drift(base, similar)
    assert result.severity is Severity.STABLE


def test_kl_nonnegative():
    random.seed(4)
    base = [random.gauss(0, 1) for _ in range(1000)]
    other = [random.gauss(1, 1) for _ in range(1000)]
    assert kl_divergence(base, other) >= -1e-6


def test_constant_feature_does_not_crash():
    const = [5.0] * 500
    result = detect_drift(const, const)
    assert result.severity is Severity.STABLE
