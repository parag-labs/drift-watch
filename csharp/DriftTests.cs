using Xunit;

namespace DriftWatch.Tests;

public class DriftTests
{
    private static List<double> Gauss(int n, double mean, double std, int seed)
    {
        var rng = new Random(seed);
        var outp = new List<double>(n);
        for (var i = 0; i < n; i++)
        {
            // Box-Muller
            var u1 = 1.0 - rng.NextDouble();
            var u2 = 1.0 - rng.NextDouble();
            var z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            outp.Add(mean + std * z);
        }

        return outp;
    }

    [Fact]
    public void PsiZeroForIdentical()
    {
        var data = Enumerable.Range(0, 1000).Select(i => (double)(i % 10)).ToList();
        Assert.True(Drift.Psi(data, data) < 1e-3);
    }

    [Fact]
    public void PsiIncreasesWithShift()
    {
        var baseData = Gauss(2000, 0, 1, 1);
        var small = Gauss(2000, 0.2, 1, 2);
        var large = Gauss(2000, 3.0, 1, 3);
        Assert.True(Drift.Psi(baseData, small) < Drift.Psi(baseData, large));
    }

    [Fact]
    public void ClassifyThresholds()
    {
        Assert.Equal(Severity.Stable, Drift.Classify(0.05));
        Assert.Equal(Severity.Minor, Drift.Classify(0.15));
        Assert.Equal(Severity.Major, Drift.Classify(0.40));
    }

    [Fact]
    public void DetectDriftMajorOnBigShift()
    {
        var baseData = Gauss(3000, 0, 1, 2);
        var shifted = Gauss(3000, 5.0, 1, 5);
        var result = Drift.DetectDrift(baseData, shifted);
        Assert.Equal(Severity.Major, result.Severity);
        Assert.True(result.Psi > 0.25);
    }

    [Fact]
    public void DetectDriftStableOnSameDistribution()
    {
        var baseData = Gauss(5000, 0, 1, 3);
        var similar = Gauss(5000, 0, 1, 33);
        var result = Drift.DetectDrift(baseData, similar);
        Assert.Equal(Severity.Stable, result.Severity);
    }

    [Fact]
    public void ConstantFeatureDoesNotCrash()
    {
        var constData = Enumerable.Repeat(5.0, 500).ToList();
        var result = Drift.DetectDrift(constData, constData);
        Assert.Equal(Severity.Stable, result.Severity);
    }
}
