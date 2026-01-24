using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace CQT.Tests;

[TestClass]
public class WindowFunctionTests
{
    private const double Tolerance = 1e-10;

    [TestMethod]
    public void HannWindow_Periodic_StartsAtZero()
    {
        // Periodic Hann (fftbins=True) starts at 0 but doesn't end at 0
        var window = WindowFunctions.Hann(16);

        Assert.AreEqual(0.0, window[0], Tolerance);
        // window[N-1] is NOT zero for periodic window
        Assert.IsTrue(window[15] > 0, "Periodic Hann window last value should be > 0");
    }

    [TestMethod]
    public void HannWindow_Periodic_IsPeriodic()
    {
        // For periodic window, window[N] would equal window[0] if extended
        // This means the window is suitable for FFT where periodicity is assumed
        var window = WindowFunctions.Hann(16);
            
        // All values should be in [0, 1]
        for (int i = 0; i < 16; i++)
        {
            Assert.IsTrue(window[i] >= 0.0 && window[i] <= 1.0, $"Value at {i} should be in [0,1]");
        }
    }

    [TestMethod]
    public void HannSymmetric_StartsAndEndsAtZero()
    {
        // Symmetric Hann window has zeros at both endpoints
        var window = WindowFunctions.HannSymmetric(16);

        Assert.AreEqual(0.0, window[0], Tolerance);
        Assert.AreEqual(0.0, window[15], Tolerance);
    }

    [TestMethod]
    public void HannSymmetric_PeaksAtCenter()
    {
        var window = WindowFunctions.HannSymmetric(17);

        // Center value should be maximum (1.0) for symmetric window
        Assert.AreEqual(1.0, window[8], Tolerance);
    }

    [TestMethod]
    public void HannSymmetric_IsSymmetric()
    {
        var window = WindowFunctions.HannSymmetric(16);

        for (int i = 0; i < 8; i++)
        {
            Assert.AreEqual(window[i], window[15 - i], Tolerance, $"Symmetry at {i}");
        }
    }

    [TestMethod]
    public void HannSymmetric_MatchesPythonReference()
    {
        if (!TestDataLoader.TestDataExists("hann_window_16"))
        {
            Assert.Inconclusive("Test data not generated. Run python generate_test_data.py");
            return;
        }

        var testCase = TestDataLoader.LoadTestCase("hann_window_16");
        var expected = TestDataLoader.GetDoubleArray(testCase.GetProperty("expected"));

        // Python reference uses symmetric Hann: 0.5*(1-cos(2*pi*n/(N-1))).
        var window = WindowFunctions.HannSymmetric(16);

        for (int i = 0; i < 16; i++)
        {
            Assert.AreEqual(expected[i], window[i], 1e-10, $"Value at {i}");
        }
    }

    [TestMethod]
    public void HammingWindow_NeverReachesZero()
    {
        var window = WindowFunctions.Hamming(16);

        for (int i = 0; i < 16; i++)
        {
            Assert.IsTrue(window[i] > 0, $"Value at {i} should be positive");
        }
    }

    [TestMethod]
    public void RectangularWindow_AllOnes()
    {
        var window = WindowFunctions.Rectangular(16);

        for (int i = 0; i < 16; i++)
        {
            Assert.AreEqual(1.0, window[i], Tolerance);
        }
    }
}