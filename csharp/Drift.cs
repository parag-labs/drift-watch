namespace DriftWatch;

public enum Severity
{
    Stable,
    Minor,
    Major,
}

public sealed record DriftResult(double Psi, double Kl, Severity Severity);

/// <summary>Data-drift detection: PSI, KL-divergence, and severity classification.</summary>
public static class Drift
{
    private const double Eps = 1e-6;

    public static double Psi(IReadOnlyList<double> expected, IReadOnlyList<double> actual, int bins = 10)
    {
        var edges = BinEdges(expected, bins);
        var e = Proportions(expected, edges);
        var a = Proportions(actual, edges);
        var total = 0.0;
        for (var i = 0; i < e.Length; i++)
        {
            var ei = Math.Max(e[i], Eps);
            var ai = Math.Max(a[i], Eps);
            total += (ai - ei) * Math.Log(ai / ei);
        }

        return Math.Round(total, 6);
    }

    public static double KlDivergence(IReadOnlyList<double> expected, IReadOnlyList<double> actual, int bins = 10)
    {
        var edges = BinEdges(expected, bins);
        var e = Proportions(expected, edges);
        var a = Proportions(actual, edges);
        var total = 0.0;
        for (var i = 0; i < e.Length; i++)
        {
            var ei = Math.Max(e[i], Eps);
            var ai = Math.Max(a[i], Eps);
            total += ai * Math.Log(ai / ei);
        }

        return Math.Round(total, 6);
    }

    public static Severity Classify(double psi) => psi switch
    {
        < 0.1 => Severity.Stable,
        < 0.25 => Severity.Minor,
        _ => Severity.Major,
    };

    public static DriftResult DetectDrift(IReadOnlyList<double> expected, IReadOnlyList<double> actual, int bins = 10)
    {
        var p = Psi(expected, actual, bins);
        return new DriftResult(p, KlDivergence(expected, actual, bins), Classify(p));
    }

    private static double[] BinEdges(IReadOnlyList<double> values, int bins)
    {
        var lo = values.Min();
        var hi = values.Max();
        if (hi == lo)
        {
            hi = lo + 1.0;
        }

        var width = (hi - lo) / bins;
        var edges = new double[bins + 1];
        for (var i = 0; i <= bins; i++)
        {
            edges[i] = lo + i * width;
        }

        return edges;
    }

    private static double[] Proportions(IReadOnlyList<double> values, double[] edges)
    {
        var bins = edges.Length - 1;
        var counts = new int[bins];
        var span = edges[^1] - edges[0];
        foreach (var v in values)
        {
            var idx = (int)((v - edges[0]) / span * bins);
            if (idx < 0)
            {
                idx = 0;
            }
            else if (idx >= bins)
            {
                idx = bins - 1;
            }

            counts[idx]++;
        }

        var total = values.Count == 0 ? 1 : values.Count;
        return counts.Select(c => (double)c / total).ToArray();
    }
}
