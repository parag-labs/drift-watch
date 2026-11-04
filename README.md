# DriftWatch

**Catch model degradation before your users do.**

Models silently rot when production data drifts away from what they were trained on. DriftWatch quantifies that drift with the two industry-standard metrics - **PSI (Population Stability Index)** and **KL-divergence** - and classifies severity so you can alert or block. Same math in **Python, C#, Java, Go, Rust, and TypeScript**.

## The problem

You ship a model, it works, everyone moves on. Six weeks later inputs have shifted (a new user segment, a changed upstream feature) and accuracy quietly craters. Nobody notices until it's a customer complaint. DriftWatch turns "silent rot" into a measurable, alertable signal.

## What it does

- **PSI** between a reference sample and a live sample of a numeric feature.
- **KL-divergence** over the same binned distributions.
- **Severity classification** using standard PSI thresholds:
  - `< 0.10` → **stable**
  - `0.10-0.25` → **minor** drift
  - `> 0.25` → **major** drift (alert / retrain)

## Use it (Python)

```python
from drift import detect_drift

result = detect_drift(reference_sample, live_sample)
print(result.psi, result.severity)   # e.g. 0.83 Severity.MAJOR
```

Or from the command line (two files, one number per line; exits 1 on major drift):

```bash
cd python
python src/cli.py reference.txt current.txt
```

## Six languages, one result

| Language | Tests | Run |
|----------|:-----:|-----|
| Python (3.15) | 13 | `cd python && pytest -q` |
| C# (.NET 10) | 6 | `cd csharp && dotnet test` |
| Java (25+) | 6 | `cd java && mvn test` |
| Go (1.27+) | 14 | `cd go && go test ./...` |
| Rust (1.99+, edition 2024) | 14 | `cd rust && cargo test` |
| TypeScript (Node 26 LTS) | 14 | `cd ts && npm install && npm test` |

The binning + PSI/KL math is pure numeric logic (no numpy/ML.NET/DL4J), so all six agree on the core detector. Each language also carries a **stress suite** (`test_stress.py` / `DriftStress*` / `stress_test.go` / `tests/stress.rs` / `stress.test.ts`) that hammers the load-bearing properties in that port: identical distributions stay stable, large shifts read as major, and PSI/KL stay finite and non-negative under fuzzed input. Because each language uses its own PRNG the exact scores differ, so the ports assert *properties* (finite, non-negative, monotonic in the shift, correct severity band) rather than exact values.

## Known limitations / next

- Numeric features only - categorical drift (chi-square) is an obvious extension.
- Equal-width binning; quantile binning is more robust for skewed data.
- For unstructured/LLM inputs, compute drift over **embeddings** instead of raw values (a planned addition).

## Design notes and numbers

- **[DESIGN.md](DESIGN.md)** - why bin over the reference range, the epsilon smoothing
  that keeps it finite, the no-numpy/three-language parity trade-off, and the non-goals.
- **[BENCHMARKS.md](BENCHMARKS.md)** - detection throughput vs sample size and cost vs
  bin count, with graphs. Reproduce with `python bench/benchmark.py`.

## How it works

```mermaid
flowchart LR
  classDef proc fill:#eff6ff,stroke:#3b82f6,color:#1e3a8a
  classDef good fill:#f0fdf4,stroke:#22c55e,color:#14532d
  classDef bad fill:#fef2f2,stroke:#ef4444,color:#7f1d1d
  classDef work fill:#faf5ff,stroke:#a855f7,color:#581c87
  REF["Reference"]:::proc
  LIVE["Live"]:::proc
  BIN["Bin on same edges"]:::work
  CMP["PSI + KL"]:::work
  CLS{"PSI level"}:::work
  STABLE["Stable"]:::good
  MAJOR["Major → alert"]:::bad
  REF --> BIN
  LIVE --> BIN
  BIN --> CMP --> CLS
  CLS -->|low| STABLE
  CLS -->|high| MAJOR
```

## Layout

```
drift-watch/
├── python/         reference implementation (PSI + KL) + pytest suite
├── csharp/         .NET 10 port - Drift.cs + tests
├── java/           JDK 25+ port (Maven)
├── go/             Go 1.27+ port (go test)
├── rust/           Rust port (cargo test) - lib + integration tests
├── ts/             TypeScript port (vitest) - strict tsconfig, ESM
├── bench/          benchmark.py - drift-score cost vs feature count
├── DESIGN.md       why PSI and KL, bucketing choices, weighting by importance
└── BENCHMARKS.md   reproducible numbers
```


Part of [parag-labs](https://github.com/parag-labs) - small, focused tools for building AI systems you can trust.
