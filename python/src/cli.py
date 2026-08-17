"""Command line front end for DriftWatch.

Give it two files of numbers (one value per line) -- a reference sample and the
current one -- and it tells you how far the distribution has moved. Exits 1 on
major drift so you can wire it into a monitoring cron or CI check.

    python -m src.cli reference.txt current.txt --bins 10
"""

from __future__ import annotations

import argparse
import sys

from drift import Severity, detect_drift


def read_numbers(path: str) -> list[float]:
    values = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(float(line))
    if not values:
        raise ValueError(f"no numeric values found in {path}")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="driftwatch")
    parser.add_argument("reference", help="baseline sample, one number per line")
    parser.add_argument("current", help="live sample, one number per line")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args(argv)

    result = detect_drift(read_numbers(args.reference), read_numbers(args.current), args.bins)

    print(f"PSI      {result.psi}")
    print(f"KL       {result.kl}")
    print(f"severity {result.severity.value}")

    # Major drift is the one you actually want to page on.
    return 1 if result.severity is Severity.MAJOR else 0


if __name__ == "__main__":
    sys.exit(main())
