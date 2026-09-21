package driftwatch

import (
	"math"
	"math/rand"
	"testing"
)

// Stress suite: drift math must stay sane on identical, extreme, and degenerate
// inputs. Drift statistics quietly return NaN or infinity on an edge case and
// poison a monitoring dashboard, so every result must be finite, non-negative,
// and monotonic in the obvious way.

// normalish is a sum-of-uniforms approximation of a normal - no dependencies,
// keeping the stress suite portable across every language port.
func normalish(rng *rand.Rand, n int, center, spread float64) []float64 {
	out := make([]float64, n)
	for i := range out {
		s := 0.0
		for j := 0; j < 6; j++ {
			s += rng.Float64()
		}
		out[i] = center + spread*(s-3)
	}
	return out
}

func TestIdenticalDistributionsShowNoDrift(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	for i := 0; i < 200; i++ {
		sample := normalish(rng, 500, 50, 10)
		other := normalish(rng, 500, 50, 10)
		result := DetectDrift(sample, other)
		if !isFinite(result.PSI) || result.PSI < 0 {
			t.Fatalf("PSI must be finite and non-negative, got %v", result.PSI)
		}
		if result.PSI >= 0.25 {
			t.Fatalf("same distribution should never be major, got %v", result.PSI)
		}
	}
}

func TestLargeShiftIsFlaggedMajorAndFinite(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	expected := normalish(rng, 1000, 0, 5)
	actual := normalish(rng, 1000, 100, 5)
	result := DetectDrift(expected, actual)
	if !isFinite(result.PSI) {
		t.Fatalf("PSI must stay finite even for a huge shift, got %v", result.PSI)
	}
	if result.Severity != Major {
		t.Fatalf("a shift of 100 must be major, got %v", result.Severity)
	}
}

func TestPSIAndKLNeverNegativeOrNaNUnderFuzz(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for i := 0; i < 2000; i++ {
		nE := rng.Intn(291) + 10
		nA := rng.Intn(291) + 10
		expected := make([]float64, nE)
		for j := range expected {
			expected[j] = rng.Float64()*200 - 100
		}
		actual := make([]float64, nA)
		for j := range actual {
			actual[j] = rng.Float64()*200 - 100
		}
		p := PSI(expected, actual)
		k := KLDivergence(expected, actual)
		if !isFinite(p) || p < 0 {
			t.Fatalf("bad psi: %v", p)
		}
		if !isFinite(k) || k < 0 {
			t.Fatalf("bad kl: %v", k)
		}
	}
}

func TestConstantSampleDoesNotBlowUp(t *testing.T) {
	a := make([]float64, 100)
	for i := range a {
		a[i] = 5.0
	}
	if !isFinite(DetectDrift(a, a).PSI) {
		t.Fatal("a degenerate reference must not divide by a zero-width range")
	}
	b := make([]float64, 100)
	for i := range b {
		b[i] = 9.0
	}
	if !isFinite(DetectDrift(a, b).PSI) {
		t.Fatal("constant reference vs constant actual must stay finite")
	}
}

func TestBiggerShiftGivesBiggerPSI(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	base := normalish(rng, 800, 0, 5)
	small := DetectDrift(base, normalish(rng, 800, 5, 5)).PSI
	large := DetectDrift(base, normalish(rng, 800, 40, 5)).PSI
	if large <= small {
		t.Fatalf("a bigger shift should give a bigger PSI: small=%v large=%v", small, large)
	}
}

func TestClassifyThresholdsAreOrdered(t *testing.T) {
	if Classify(0.0) != Stable {
		t.Fatal("0.0 should be stable")
	}
	if Classify(1.0) != Major {
		t.Fatal("1.0 should be major")
	}
	if Classify(math.MaxFloat64) != Major {
		t.Fatal("extreme PSI should still be major, not crash")
	}
}
