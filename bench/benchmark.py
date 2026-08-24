"""Benchmark harness for drift-watch.

Drift detection is O(n) in the sample size - it bins two samples and sums a few terms
per bin - so the useful thing to measure is that throughput stays flat as samples grow,
i.e. detecting drift on a big monitoring window costs the same per sample as a small
one. Also measures how bin count affects cost (it barely does).

Run: python bench/benchmark.py
"""

from __future__ import annotations

import gc
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from drift import detect_drift

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


def sample(rng: random.Random, n: int, center: float) -> list[float]:
    return [center + 10 * (sum(rng.random() for _ in range(6)) - 3) for _ in range(n)]


def bench_throughput_by_size() -> dict:
    rng = random.Random(0)
    sizes = [1_000, 5_000, 20_000, 50_000, 100_000, 250_000]
    rates = []
    for n in sizes:
        expected = sample(rng, n, 0)
        actual = sample(rng, n, 5)
        gc.disable()
        start = time.perf_counter()
        detect_drift(expected, actual, bins=10)
        elapsed = time.perf_counter() - start
        gc.enable()
        rates.append((2 * n) / elapsed)  # both samples are processed

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot([s / 1000 for s in sizes], [r / 1e6 for r in rates], "o-", color="tab:blue")
    ax.set_xlabel("sample size per side (thousands)")
    ax.set_ylabel("throughput (million values/sec)")
    ax.set_title("drift-watch: detection throughput vs sample size")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "throughput_by_size.png", dpi=110)
    plt.close(fig)
    return {"sizes": sizes, "values_per_sec": [int(r) for r in rates]}


def bench_cost_by_bins() -> dict:
    rng = random.Random(1)
    expected = sample(rng, 50_000, 0)
    actual = sample(rng, 50_000, 5)
    bin_counts = [5, 10, 20, 50, 100, 200]
    times_ms = []
    for b in bin_counts:
        gc.disable()
        start = time.perf_counter()
        detect_drift(expected, actual, bins=b)
        times_ms.append((time.perf_counter() - start) * 1000)
        gc.enable()

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(bin_counts, times_ms, "o-", color="tab:purple")
    ax.set_xlabel("number of bins")
    ax.set_ylabel("time for 50k-vs-50k detection (ms)")
    ax.set_title("drift-watch: bin count barely moves the cost")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "cost_by_bins.png", dpi=110)
    plt.close(fig)
    return {"bins": bin_counts, "time_ms": [round(t, 2) for t in times_ms]}


def main() -> None:
    summary = {
        "throughput_by_size": bench_throughput_by_size(),
        "cost_by_bins": bench_cost_by_bins(),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
