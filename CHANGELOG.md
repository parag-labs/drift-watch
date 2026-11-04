# Changelog

All notable changes to this project are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses [semantic versioning](https://semver.org/).

## [Unreleased]

### Changed

- Toolchains: Python 3.15, Java 25 (LTS), Go 1.27, Rust 1.99 / edition 2024, Node 26 LTS; .NET stays on 10 (LTS) and is pinned via `csharp/global.json`. CI actions moved to their latest majors.
- Test and tooling dependencies: TypeScript 7, Vitest 5 (TS target/lib ES2024), xunit.v3 on Microsoft.Testing.Platform (opted in via `csharp/global.json`) with Microsoft.NET.Test.Sdk 18, JUnit 6 with Surefire 3.6 and Compiler plugin 3.16, matplotlib >= 3.11 for the benchmark.

## [0.1.0] - 2026-08-24

First public release. PSI and KL-divergence data-drift detection with epsilon smoothing, in Python, C#, and Java.

[0.1.0]: https://github.com/parag-labs/drift-watch/releases/tag/v0.1.0
