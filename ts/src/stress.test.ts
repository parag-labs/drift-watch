import { describe, expect, it } from "vitest";

import { Severity, classify, detectDrift, klDivergence, psi } from "./drift";

// Stress suite: drift math must stay sane on identical, extreme, and degenerate
// inputs. Drift statistics quietly return NaN or infinity on an edge case and
// poison a monitoring dashboard, so every result must be finite, non-negative,
// and monotonic in the obvious way.

/** A tiny deterministic mulberry32 PRNG so the soak test is reproducible. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Sum-of-uniforms approximation of a normal - keeps the suite portable. */
function normalish(rng: () => number, n: number, center: number, spread: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < 6; j++) {
      s += rng();
    }
    out.push(center + spread * (s - 3));
  }
  return out;
}

describe("stress", () => {
  it("shows no drift for identical distributions", () => {
    const rng = mulberry32(0);
    for (let i = 0; i < 200; i++) {
      const sample = normalish(rng, 500, 50, 10);
      const other = normalish(rng, 500, 50, 10);
      const result = detectDrift(sample, other);
      expect(Number.isFinite(result.psi)).toBe(true);
      expect(result.psi).toBeGreaterThanOrEqual(0);
      expect(result.psi).toBeLessThan(0.25);
    }
  });

  it("flags a large shift as major and finite", () => {
    const rng = mulberry32(1);
    const expected = normalish(rng, 1000, 0, 5);
    const actual = normalish(rng, 1000, 100, 5);
    const result = detectDrift(expected, actual);
    expect(Number.isFinite(result.psi)).toBe(true);
    expect(result.severity).toBe(Severity.Major);
  });

  it("never returns a negative or NaN PSI/KL under fuzz", () => {
    const rng = mulberry32(2);
    for (let i = 0; i < 2000; i++) {
      const nE = 10 + Math.trunc(rng() * 291);
      const nA = 10 + Math.trunc(rng() * 291);
      const expected = Array.from({ length: nE }, () => rng() * 200 - 100);
      const actual = Array.from({ length: nA }, () => rng() * 200 - 100);
      const p = psi(expected, actual);
      const k = klDivergence(expected, actual);
      expect(Number.isFinite(p) && p >= 0).toBe(true);
      expect(Number.isFinite(k) && k >= 0).toBe(true);
    }
  });

  it("does not blow up on a constant sample", () => {
    const a = new Array<number>(100).fill(5.0);
    expect(Number.isFinite(detectDrift(a, a).psi)).toBe(true);
    const b = new Array<number>(100).fill(9.0);
    expect(Number.isFinite(detectDrift(a, b).psi)).toBe(true);
  });

  it("gives a bigger PSI for a bigger shift", () => {
    const rng = mulberry32(3);
    const base = normalish(rng, 800, 0, 5);
    const small = detectDrift(base, normalish(rng, 800, 5, 5)).psi;
    const large = detectDrift(base, normalish(rng, 800, 40, 5)).psi;
    expect(large).toBeGreaterThan(small);
  });

  it("keeps classify thresholds ordered and crash-free at extremes", () => {
    expect(classify(0.0)).toBe(Severity.Stable);
    expect(classify(1.0)).toBe(Severity.Major);
    expect(classify(1e9)).toBe(Severity.Major);
  });
});
