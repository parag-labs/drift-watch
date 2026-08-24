package com.driftwatch;

import java.util.List;

/** Data-drift detection: PSI, KL-divergence, and severity classification. */
public final class Drift {

    private static final double EPS = 1e-6;

    public enum Severity {
        STABLE, MINOR, MAJOR
    }

    public record DriftResult(double psi, double kl, Severity severity) {
    }

    public static double psi(List<Double> expected, List<Double> actual, int bins) {
        double[] edges = binEdges(expected, bins);
        double[] e = proportions(expected, edges);
        double[] a = proportions(actual, edges);
        double total = 0.0;
        for (int i = 0; i < e.length; i++) {
            double ei = Math.max(e[i], EPS);
            double ai = Math.max(a[i], EPS);
            total += (ai - ei) * Math.log(ai / ei);
        }
        return round6(total);
    }

    public static double klDivergence(List<Double> expected, List<Double> actual, int bins) {
        double[] edges = binEdges(expected, bins);
        double[] e = proportions(expected, edges);
        double[] a = proportions(actual, edges);
        double total = 0.0;
        for (int i = 0; i < e.length; i++) {
            double ei = Math.max(e[i], EPS);
            double ai = Math.max(a[i], EPS);
            total += ai * Math.log(ai / ei);
        }
        return round6(total);
    }

    public static Severity classify(double psi) {
        if (psi < 0.1) {
            return Severity.STABLE;
        }
        if (psi < 0.25) {
            return Severity.MINOR;
        }
        return Severity.MAJOR;
    }

    public static DriftResult detectDrift(List<Double> expected, List<Double> actual, int bins) {
        double p = psi(expected, actual, bins);
        return new DriftResult(p, klDivergence(expected, actual, bins), classify(p));
    }

    private static double[] binEdges(List<Double> values, int bins) {
        double lo = Double.POSITIVE_INFINITY;
        double hi = Double.NEGATIVE_INFINITY;
        for (double v : values) {
            lo = Math.min(lo, v);
            hi = Math.max(hi, v);
        }
        if (hi == lo) {
            hi = lo + 1.0;
        }
        double width = (hi - lo) / bins;
        double[] edges = new double[bins + 1];
        for (int i = 0; i <= bins; i++) {
            edges[i] = lo + i * width;
        }
        return edges;
    }

    private static double[] proportions(List<Double> values, double[] edges) {
        int bins = edges.length - 1;
        int[] counts = new int[bins];
        double span = edges[edges.length - 1] - edges[0];
        for (double v : values) {
            int idx = (int) ((v - edges[0]) / span * bins);
            if (idx < 0) {
                idx = 0;
            } else if (idx >= bins) {
                idx = bins - 1;
            }
            counts[idx]++;
        }
        int total = values.isEmpty() ? 1 : values.size();
        double[] props = new double[bins];
        for (int i = 0; i < bins; i++) {
            props[i] = (double) counts[i] / total;
        }
        return props;
    }

    static double round6(double v) {
        return Math.round(v * 1_000_000.0) / 1_000_000.0;
    }

    private Drift() {
    }
}
