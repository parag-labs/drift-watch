package driftwatch

import (
	"math"
	"math/rand"
	"testing"
)

// gauss draws n normal samples with the given mean/std from a seeded generator.
func gauss(n int, mean, std float64, seed int64) []float64 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]float64, n)
	for i := range out {
		out[i] = mean + std*rng.NormFloat64()
	}
	return out
}

func isFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

func TestPSIZeroForIdentical(t *testing.T) {
	data := make([]float64, 1000)
	for i := range data {
		data[i] = float64(i % 10)
	}
	if PSI(data, data) >= 1e-3 {
		t.Fatalf("identical data should have ~0 PSI, got %v", PSI(data, data))
	}
}

func TestPSIIncreasesWithShift(t *testing.T) {
	base := gauss(2000, 0, 1, 1)
	small := gauss(2000, 0.2, 1, 2)
	large := gauss(2000, 3.0, 1, 3)
	if PSI(base, small) >= PSI(base, large) {
		t.Fatalf("a larger shift should raise PSI: small=%v large=%v", PSI(base, small), PSI(base, large))
	}
}

func TestClassifyThresholds(t *testing.T) {
	if Classify(0.05) != Stable {
		t.Fatal("0.05 should be stable")
	}
	if Classify(0.15) != Minor {
		t.Fatal("0.15 should be minor")
	}
	if Classify(0.40) != Major {
		t.Fatal("0.40 should be major")
	}
}

func TestDetectDriftMajorOnBigShift(t *testing.T) {
	base := gauss(3000, 0, 1, 2)
	shifted := gauss(3000, 5.0, 1, 5)
	result := DetectDrift(base, shifted)
	if result.Severity != Major {
		t.Fatalf("a 5-sigma shift should be major, got %v", result.Severity)
	}
	if result.PSI <= 0.25 {
		t.Fatalf("major drift should exceed the 0.25 threshold, got %v", result.PSI)
	}
}

func TestDetectDriftStableOnSameDistribution(t *testing.T) {
	base := gauss(5000, 0, 1, 3)
	similar := gauss(5000, 0, 1, 33)
	result := DetectDrift(base, similar)
	if result.Severity != Stable {
		t.Fatalf("independent draws from one distribution should be stable, got %v", result.Severity)
	}
}

func TestKLNonnegative(t *testing.T) {
	base := gauss(1000, 0, 1, 4)
	other := gauss(1000, 1, 1, 44)
	if KLDivergence(base, other) < -1e-6 {
		t.Fatalf("KL should be non-negative, got %v", KLDivergence(base, other))
	}
}

func TestConstantFeatureDoesNotCrash(t *testing.T) {
	constData := make([]float64, 500)
	for i := range constData {
		constData[i] = 5.0
	}
	result := DetectDrift(constData, constData)
	if result.Severity != Stable {
		t.Fatalf("a constant feature vs itself is stable, got %v", result.Severity)
	}
}

func TestDefaultBinsMatchExplicitTen(t *testing.T) {
	base := gauss(1500, 0, 1, 7)
	other := gauss(1500, 1.5, 1, 8)
	if PSI(base, other) != PSI(base, other, 10) {
		t.Fatal("the default bin count must equal an explicit 10")
	}
	if KLDivergence(base, other) != KLDivergence(base, other, 10) {
		t.Fatal("the default bin count must equal an explicit 10 for KL too")
	}
}
