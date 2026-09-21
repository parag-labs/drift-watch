// Package driftwatch detects data drift with PSI, KL-divergence, and severity
// classification.
//
// The logic is pure numeric code (no third-party math) so it ports identically
// to the Python, C#, Java, Rust, and TypeScript siblings. Given a reference
// ("expected") sample and a live ("actual") sample of a numeric feature, both are
// binned over the reference range and the amount the distribution moved is
// quantified.
package driftwatch

import "math"

const eps = 1e-6

// defaultBins is the bin count used when a caller omits the optional bins
// argument, mirroring the Python reference default of 10.
const defaultBins = 10

// Severity classifies how far a distribution has drifted.
type Severity string

const (
	// Stable means no meaningful drift (PSI < 0.10).
	Stable Severity = "stable"
	// Minor means moderate drift worth watching (0.10 <= PSI < 0.25).
	Minor Severity = "minor"
	// Major means significant drift that should trigger an alert (PSI >= 0.25).
	Major Severity = "major"
)

// DriftResult bundles the two drift metrics with the derived severity.
type DriftResult struct {
	PSI      float64
	KL       float64
	Severity Severity
}

// resolveBins returns the caller-supplied bin count or the default when omitted.
func resolveBins(bins []int) int {
	if len(bins) > 0 {
		return bins[0]
	}
	return defaultBins
}

// binEdges builds bins+1 equal-width edges spanning the range of values. A
// degenerate (zero-width) range is widened by one so binning never divides by
// zero.
func binEdges(values []float64, bins int) []float64 {
	lo, hi := values[0], values[0]
	for _, v := range values {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	if hi == lo {
		hi = lo + 1.0
	}
	width := (hi - lo) / float64(bins)
	edges := make([]float64, bins+1)
	for i := 0; i <= bins; i++ {
		edges[i] = lo + float64(i)*width
	}
	return edges
}

// proportions returns the fraction of values falling in each bin, clamping
// out-of-range values into the first or last bin.
func proportions(values []float64, edges []float64) []float64 {
	bins := len(edges) - 1
	counts := make([]int, bins)
	span := edges[len(edges)-1] - edges[0]
	for _, v := range values {
		idx := int((v - edges[0]) / span * float64(bins))
		if idx < 0 {
			idx = 0
		} else if idx >= bins {
			idx = bins - 1
		}
		counts[idx]++
	}
	total := len(values)
	if total == 0 {
		total = 1
	}
	props := make([]float64, bins)
	for i, c := range counts {
		props[i] = float64(c) / float64(total)
	}
	return props
}

func round6(v float64) float64 {
	return math.Round(v*1e6) / 1e6
}

// PSI returns the Population Stability Index between the expected and actual
// samples. 0 means identical distributions; higher means more drift. The
// optional bins argument defaults to 10.
func PSI(expected, actual []float64, bins ...int) float64 {
	b := resolveBins(bins)
	edges := binEdges(expected, b)
	e := proportions(expected, edges)
	a := proportions(actual, edges)
	total := 0.0
	for i := range e {
		ei := math.Max(e[i], eps)
		ai := math.Max(a[i], eps)
		total += (ai - ei) * math.Log(ai/ei)
	}
	return round6(total)
}

// KLDivergence returns KL(actual || expected) over the binned distributions.
// The optional bins argument defaults to 10.
func KLDivergence(expected, actual []float64, bins ...int) float64 {
	b := resolveBins(bins)
	edges := binEdges(expected, b)
	e := proportions(expected, edges)
	a := proportions(actual, edges)
	total := 0.0
	for i := range e {
		ei := math.Max(e[i], eps)
		ai := math.Max(a[i], eps)
		total += ai * math.Log(ai/ei)
	}
	return round6(total)
}

// Classify maps a PSI value to a Severity using the industry-standard
// thresholds (0.10 and 0.25).
func Classify(psiValue float64) Severity {
	if psiValue < 0.1 {
		return Stable
	}
	if psiValue < 0.25 {
		return Minor
	}
	return Major
}

// DetectDrift computes PSI and KL-divergence and classifies the result. The
// optional bins argument defaults to 10.
func DetectDrift(expected, actual []float64, bins ...int) DriftResult {
	b := resolveBins(bins)
	p := PSI(expected, actual, b)
	return DriftResult{PSI: p, KL: KLDivergence(expected, actual, b), Severity: Classify(p)}
}
