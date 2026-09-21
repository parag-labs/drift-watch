import { describe, expect, it } from "vitest";

import { Severity, classify, detectDrift, klDivergence, psi } from "./drift";

/** A tiny deterministic mulberry32 PRNG so the tests are dependency-free. */
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

/** Draws n normal samples via Box-Muller from a seeded generator. */
function gauss(n: number, mean: number, std: number, seed: number): number[] {
  const rng = mulberry32(seed);
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const u1 = 1.0 - rng();
    const u2 = 1.0 - rng();
    const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    out.push(mean + std * z);
  }
  return out;
}

describe("drift", () => {
  it("has ~zero PSI for identical data", () => {
    const data = Array.from({ length: 1000 }, (_, i) => i % 10);
    expect(psi(data, data)).toBeLessThan(1e-3);
  });

  it("raises PSI as the shift grows", () => {
    const base = gauss(2000, 0, 1, 1);
    const small = gauss(2000, 0.2, 1, 2);
    const large = gauss(2000, 3.0, 1, 3);
    expect(psi(base, small)).toBeLessThan(psi(base, large));
  });

  it("classifies by the standard thresholds", () => {
    expect(classify(0.05)).toBe(Severity.Stable);
    expect(classify(0.15)).toBe(Severity.Minor);
    expect(classify(0.4)).toBe(Severity.Major);
  });

  it("flags a big shift as major", () => {
    const base = gauss(3000, 0, 1, 2);
    const shifted = gauss(3000, 5.0, 1, 5);
    const result = detectDrift(base, shifted);
    expect(result.severity).toBe(Severity.Major);
    expect(result.psi).toBeGreaterThan(0.25);
  });

  it("reports stable for two draws of the same distribution", () => {
    const base = gauss(5000, 0, 1, 3);
    const similar = gauss(5000, 0, 1, 33);
    expect(detectDrift(base, similar).severity).toBe(Severity.Stable);
  });

  it("keeps KL non-negative", () => {
    const base = gauss(1000, 0, 1, 4);
    const other = gauss(1000, 1, 1, 44);
    expect(klDivergence(base, other)).toBeGreaterThanOrEqual(-1e-6);
  });

  it("does not crash on a constant feature", () => {
    const constData = new Array<number>(500).fill(5.0);
    expect(detectDrift(constData, constData).severity).toBe(Severity.Stable);
  });

  it("treats the default bin count as an explicit 10", () => {
    const base = gauss(1500, 0, 1, 7);
    const other = gauss(1500, 1.5, 1, 8);
    expect(psi(base, other)).toBe(psi(base, other, 10));
    expect(klDivergence(base, other)).toBe(klDivergence(base, other, 10));
  });
});
