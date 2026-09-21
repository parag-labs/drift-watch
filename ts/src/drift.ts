/**
 * Data-drift detection: PSI, KL-divergence, and severity classification.
 *
 * Pure numeric logic (no dependencies) so it ports identically to the Python,
 * C#, Java, Go, and Rust siblings. Given a reference ("expected") sample and a
 * live ("actual") sample of a numeric feature, both are binned over the reference
 * range and the amount the distribution moved is quantified.
 */

const EPS = 1e-6;

/** How far a distribution has drifted. */
export enum Severity {
  Stable = "stable",
  Minor = "minor",
  Major = "major",
}

/** The two drift metrics plus the derived severity. */
export interface DriftResult {
  psi: number;
  kl: number;
  severity: Severity;
}

function round6(v: number): number {
  return Math.round(v * 1e6) / 1e6;
}

/**
 * Builds `bins + 1` equal-width edges spanning the range of `values`. A
 * degenerate (zero-width) range is widened by one so binning never divides by
 * zero.
 */
function binEdges(values: number[], bins: number): number[] {
  let lo = values[0];
  let hi = values[0];
  for (const v of values) {
    if (v < lo) {
      lo = v;
    }
    if (v > hi) {
      hi = v;
    }
  }
  if (hi === lo) {
    hi = lo + 1.0;
  }
  const width = (hi - lo) / bins;
  const edges: number[] = [];
  for (let i = 0; i <= bins; i++) {
    edges.push(lo + i * width);
  }
  return edges;
}

/**
 * Returns the fraction of `values` in each bin, clamping out-of-range values
 * into the first or last bin.
 */
function proportions(values: number[], edges: number[]): number[] {
  const bins = edges.length - 1;
  const counts = new Array<number>(bins).fill(0);
  const span = edges[edges.length - 1] - edges[0];
  for (const v of values) {
    let idx = Math.trunc(((v - edges[0]) / span) * bins);
    if (idx < 0) {
      idx = 0;
    } else if (idx >= bins) {
      idx = bins - 1;
    }
    counts[idx]++;
  }
  const total = values.length === 0 ? 1 : values.length;
  return counts.map((c) => c / total);
}

/**
 * Population Stability Index between the expected and actual samples. 0 means
 * identical distributions; higher means more drift. `bins` defaults to 10.
 */
export function psi(expected: number[], actual: number[], bins = 10): number {
  const edges = binEdges(expected, bins);
  const e = proportions(expected, edges);
  const a = proportions(actual, edges);
  let total = 0;
  for (let i = 0; i < e.length; i++) {
    const ei = Math.max(e[i], EPS);
    const ai = Math.max(a[i], EPS);
    total += (ai - ei) * Math.log(ai / ei);
  }
  return round6(total);
}

/** KL(actual || expected) over the binned distributions. `bins` defaults to 10. */
export function klDivergence(expected: number[], actual: number[], bins = 10): number {
  const edges = binEdges(expected, bins);
  const e = proportions(expected, edges);
  const a = proportions(actual, edges);
  let total = 0;
  for (let i = 0; i < e.length; i++) {
    const ei = Math.max(e[i], EPS);
    const ai = Math.max(a[i], EPS);
    total += ai * Math.log(ai / ei);
  }
  return round6(total);
}

/**
 * Maps a PSI value to a {@link Severity} using the industry-standard thresholds
 * (0.10 and 0.25).
 */
export function classify(psiValue: number): Severity {
  if (psiValue < 0.1) {
    return Severity.Stable;
  }
  if (psiValue < 0.25) {
    return Severity.Minor;
  }
  return Severity.Major;
}

/** Computes PSI and KL-divergence and classifies the result. `bins` defaults to 10. */
export function detectDrift(expected: number[], actual: number[], bins = 10): DriftResult {
  const p = psi(expected, actual, bins);
  return { psi: p, kl: klDivergence(expected, actual, bins), severity: classify(p) };
}
