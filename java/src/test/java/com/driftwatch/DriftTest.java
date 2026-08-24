package com.driftwatch;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.Test;

import com.driftwatch.Drift.DriftResult;
import com.driftwatch.Drift.Severity;

class DriftTest {

    private static List<Double> gauss(int n, double mean, double std, long seed) {
        Random rng = new Random(seed);
        List<Double> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            out.add(mean + std * rng.nextGaussian());
        }
        return out;
    }

    @Test
    void psiZeroForIdentical() {
        List<Double> data = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            data.add((double) (i % 10));
        }
        assertTrue(Drift.psi(data, data, 10) < 1e-3);
    }

    @Test
    void psiIncreasesWithShift() {
        List<Double> base = gauss(2000, 0, 1, 1);
        List<Double> small = gauss(2000, 0.2, 1, 2);
        List<Double> large = gauss(2000, 3.0, 1, 3);
        assertTrue(Drift.psi(base, small, 10) < Drift.psi(base, large, 10));
    }

    @Test
    void classifyThresholds() {
        assertEquals(Severity.STABLE, Drift.classify(0.05));
        assertEquals(Severity.MINOR, Drift.classify(0.15));
        assertEquals(Severity.MAJOR, Drift.classify(0.40));
    }

    @Test
    void detectDriftMajorOnBigShift() {
        List<Double> base = gauss(3000, 0, 1, 2);
        List<Double> shifted = gauss(3000, 5.0, 1, 5);
        DriftResult result = Drift.detectDrift(base, shifted, 10);
        assertEquals(Severity.MAJOR, result.severity());
        assertTrue(result.psi() > 0.25);
    }

    @Test
    void detectDriftStableOnSameDistribution() {
        List<Double> base = gauss(5000, 0, 1, 3);
        List<Double> similar = gauss(5000, 0, 1, 33);
        DriftResult result = Drift.detectDrift(base, similar, 10);
        assertEquals(Severity.STABLE, result.severity());
    }

    @Test
    void constantFeatureDoesNotCrash() {
        List<Double> constData = new ArrayList<>();
        for (int i = 0; i < 500; i++) {
            constData.add(5.0);
        }
        DriftResult result = Drift.detectDrift(constData, constData, 10);
        assertEquals(Severity.STABLE, result.severity());
    }
}
