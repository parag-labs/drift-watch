"""Data-drift detection: PSI, KL-divergence, and severity classification.

Pure numeric logic (no numpy) so it ports identically to C# and Java. Given a
reference ("expected") sample and a live ("actual") sample of a numeric feature,
we bin both over the reference range and quantify how much the distribution moved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

_EPS = 1e-6


class Severity(str, Enum):
    STABLE = "stable"
    MINOR = "minor"
    MAJOR = "major"


@dataclass
class DriftResult:
    psi: float
    kl: float
    severity: Severity


def _bin_edges(values: list[float], bins: int) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        hi = lo + 1.0  # avoid zero-width range
    width = (hi - lo) / bins
    return [lo + i * width for i in range(bins + 1)]


def _proportions(values: list[float], edges: list[float]) -> list[float]:
    bins = len(edges) - 1
    counts = [0] * bins
    for v in values:
        # clamp into [0, bins-1]
        idx = int((v - edges[0]) / (edges[-1] - edges[0]) * bins)
        if idx < 0:
            idx = 0
        elif idx >= bins:
            idx = bins - 1
        counts[idx] += 1
    total = len(values) or 1
    return [c / total for c in counts]


def psi(expected: list[float], actual: list[float], bins: int = 10) -> float:
    """Population Stability Index. 0 = identical; higher = more drift."""
    edges = _bin_edges(expected, bins)
    e = _proportions(expected, edges)
    a = _proportions(actual, edges)
    total = 0.0
    for ei, ai in zip(e, a):
        ei = max(ei, _EPS)
        ai = max(ai, _EPS)
        total += (ai - ei) * math.log(ai / ei)
    return round(total, 6)


def kl_divergence(expected: list[float], actual: list[float], bins: int = 10) -> float:
    """KL(actual || expected) over the binned distributions."""
    edges = _bin_edges(expected, bins)
    e = _proportions(expected, edges)
    a = _proportions(actual, edges)
    total = 0.0
    for ei, ai in zip(e, a):
        ei = max(ei, _EPS)
        ai = max(ai, _EPS)
        total += ai * math.log(ai / ei)
    return round(total, 6)


def classify(psi_value: float) -> Severity:
    # Industry-standard PSI thresholds.
    if psi_value < 0.1:
        return Severity.STABLE
    if psi_value < 0.25:
        return Severity.MINOR
    return Severity.MAJOR


def detect_drift(expected: list[float], actual: list[float], bins: int = 10) -> DriftResult:
    p = psi(expected, actual, bins)
    return DriftResult(psi=p, kl=kl_divergence(expected, actual, bins), severity=classify(p))
