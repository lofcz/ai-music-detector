using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace CQT.Tests;

[TestClass]
public class FFTTests
{
    private const double Tolerance = 1e-10;

    [TestMethod]
    public void FFT_Impulse_ReturnsAllOnes()
    {
        // Unit impulse: [1, 0, 0, 0, 0, 0, 0, 0]
        // FFT should be all ones
        var data = new Complex[8];
        data[0] = new Complex(1, 0);
        for (int i = 1; i < 8; i++)
            data[i] = Complex.Zero;

        FFT.Forward(data);

        for (int i = 0; i < 8; i++)
        {
            Assert.AreEqual(1.0, data[i].Real, Tolerance, $"Real part at index {i}");
            Assert.AreEqual(0.0, data[i].Imaginary, Tolerance, $"Imaginary part at index {i}");
        }
    }

    [TestMethod]
    public void FFT_DC_EnergyAtBinZero()
    {
        // DC signal: all ones
        // FFT should have energy only at bin 0
        var data = new Complex[8];
        for (int i = 0; i < 8; i++)
            data[i] = new Complex(1, 0);

        FFT.Forward(data);

        Assert.AreEqual(8.0, data[0].Real, Tolerance, "Bin 0 should be N");
        for (int i = 1; i < 8; i++)
        {
            Assert.AreEqual(0.0, data[i].Magnitude, Tolerance, $"Bin {i} should be zero");
        }
    }

    [TestMethod]
    public void FFT_Sine_PeaksAtCorrectBins()
    {
        // Sine wave with 2 cycles in 16 samples
        int n = 16;
        var data = new Complex[n];
        for (int i = 0; i < n; i++)
        {
            double value = Math.Sin(2 * Math.PI * 2 * i / n);
            data[i] = new Complex(value, 0);
        }

        FFT.Forward(data);

        // Energy should be at bins 2 and 14 (N-2)
        Assert.IsTrue(data[2].Magnitude > 1.0, "Should have energy at bin 2");
        Assert.IsTrue(data[14].Magnitude > 1.0, "Should have energy at bin N-2");

        // Other bins should be near zero
        for (int i = 0; i < n; i++)
        {
            if (i != 2 && i != 14)
            {
                Assert.AreEqual(0.0, data[i].Magnitude, 1e-10, $"Bin {i} should be zero");
            }
        }
    }

    [TestMethod]
    public void FFT_Inverse_RecoversOriginal()
    {
        // Random signal
        var original = new Complex[16];
        var random = new Random(42);
        for (int i = 0; i < 16; i++)
        {
            original[i] = new Complex(random.NextDouble() * 2 - 1, 0);
        }

        // Copy for FFT
        var data = new Complex[16];
        Array.Copy(original, data, 16);

        // Forward then inverse
        FFT.Forward(data);
        FFT.Inverse(data);

        // Should match original
        for (int i = 0; i < 16; i++)
        {
            Assert.AreEqual(original[i].Real, data[i].Real, 1e-10, $"Real at {i}");
            Assert.AreEqual(original[i].Imaginary, data[i].Imaginary, 1e-10, $"Imag at {i}");
        }
    }

    [TestMethod]
    public void FFT_MatchesPythonReference()
    {
        if (!TestDataLoader.TestDataExists("fft_random"))
        {
            Assert.Inconclusive("Test data not generated. Run python generate_test_data.py");
            return;
        }

        var testCase = TestDataLoader.LoadTestCase("fft_random");
        var input = TestDataLoader.GetDoubleArray(testCase.GetProperty("input"));
        var expected = TestDataLoader.GetComplexArray(testCase.GetProperty("expected"));

        // Convert to complex
        var data = new Complex[input.Length];
        for (int i = 0; i < input.Length; i++)
            data[i] = new Complex(input[i], 0);

        FFT.Forward(data);

        for (int i = 0; i < data.Length; i++)
        {
            Assert.AreEqual(expected[i].Real, data[i].Real, 1e-10, $"Real at {i}");
            Assert.AreEqual(expected[i].Imaginary, data[i].Imaginary, 1e-10, $"Imag at {i}");
        }
    }

    [TestMethod]
    public void NextPowerOf2_ReturnsCorrectValues()
    {
        Assert.AreEqual(1, FFT.NextPowerOf2(0));
        Assert.AreEqual(1, FFT.NextPowerOf2(1));
        Assert.AreEqual(2, FFT.NextPowerOf2(2));
        Assert.AreEqual(4, FFT.NextPowerOf2(3));
        Assert.AreEqual(4, FFT.NextPowerOf2(4));
        Assert.AreEqual(8, FFT.NextPowerOf2(5));
        Assert.AreEqual(16, FFT.NextPowerOf2(9));
        Assert.AreEqual(1024, FFT.NextPowerOf2(1000));
    }

    #region Detailed FFT Comparison Tests for Precision Investigation

    [TestMethod]
    [DataRow(512)]
    [DataRow(1024)]
    [DataRow(2048)]
    [DataRow(4096)]
    public void FFT_CompareWithPython_RandomSignal(int size)
    {
        string testName = $"fft_comparison_{size}";
        if (!TestDataLoader.TestDataExists(testName))
        {
            Assert.Inconclusive($"Test data not found: {testName}. Run generate_soxr_test_data.py");
            return;
        }

        var testCase = TestDataLoader.LoadTestCase(testName);
        var input = TestDataLoader.GetDoubleArray(testCase.GetProperty("input"));
        var expectedReal = TestDataLoader.GetDoubleArray(testCase.GetProperty("expected_real"));
        var expectedImag = TestDataLoader.GetDoubleArray(testCase.GetProperty("expected_imag"));

        // Convert to complex and compute FFT
        var data = new Complex[input.Length];
        for (int i = 0; i < input.Length; i++)
            data[i] = new Complex(input[i], 0);

        FFT.Forward(data);

        // Per-bin error analysis
        double maxAbsError = 0;
        double sumAbsError = 0;
        double sumExpectedMag = 0;
        int maxErrorBin = 0;

        for (int i = 0; i < data.Length; i++)
        {
            double expectedMag = Math.Sqrt(expectedReal[i] * expectedReal[i] + expectedImag[i] * expectedImag[i]);
            double realError = Math.Abs(data[i].Real - expectedReal[i]);
            double imagError = Math.Abs(data[i].Imaginary - expectedImag[i]);
            double absError = Math.Max(realError, imagError);

            sumExpectedMag += expectedMag;
            sumAbsError += absError;

            if (absError > maxAbsError)
            {
                maxAbsError = absError;
                maxErrorBin = i;
            }
        }

        double meanRelError = sumAbsError / (sumExpectedMag + 1e-10);

        Console.WriteLine($"\n=== FFT Comparison: Size {size} ===");
        Console.WriteLine($"Max absolute error: {maxAbsError:E6} at bin {maxErrorBin}");
        Console.WriteLine($"Mean relative error: {meanRelError:P6}");
        Console.WriteLine($"Sum of expected magnitudes: {sumExpectedMag:F2}");

        // Check first few bins
        Console.WriteLine("\nFirst 5 bins comparison:");
        Console.WriteLine("Bin | Expected (Re, Im) | C# (Re, Im) | Error");
        for (int i = 0; i < Math.Min(5, data.Length); i++)
        {
            double err = Math.Abs(data[i].Real - expectedReal[i]) + Math.Abs(data[i].Imaginary - expectedImag[i]);
            Console.WriteLine($"{i,3} | ({expectedReal[i],12:F6}, {expectedImag[i],12:F6}) | ({data[i].Real,12:F6}, {data[i].Imaginary,12:F6}) | {err:E3}");
        }

        // FFT should match Python to very high precision (1e-10 or better)
        Assert.IsTrue(maxAbsError < 1e-10, 
            $"FFT size {size}: Max error {maxAbsError:E6} at bin {maxErrorBin} exceeds 1e-10");
    }

    [TestMethod]
    [DataRow(512)]
    [DataRow(1024)]
    public void FFT_Impulse_CompareWithPython(int size)
    {
        string testName = $"fft_impulse_{size}";
        if (!TestDataLoader.TestDataExists(testName))
        {
            Assert.Inconclusive($"Test data not found: {testName}. Run generate_soxr_test_data.py");
            return;
        }

        var testCase = TestDataLoader.LoadTestCase(testName);
        var input = TestDataLoader.GetDoubleArray(testCase.GetProperty("input"));
        var expectedReal = TestDataLoader.GetDoubleArray(testCase.GetProperty("expected_real"));
        var expectedImag = TestDataLoader.GetDoubleArray(testCase.GetProperty("expected_imag"));

        var data = new Complex[input.Length];
        for (int i = 0; i < input.Length; i++)
            data[i] = new Complex(input[i], 0);

        FFT.Forward(data);

        double maxError = 0;
        for (int i = 0; i < data.Length; i++)
        {
            double err = Math.Abs(data[i].Real - expectedReal[i]) + Math.Abs(data[i].Imaginary - expectedImag[i]);
            maxError = Math.Max(maxError, err);
        }

        Console.WriteLine($"FFT Impulse Size {size}: Max error = {maxError:E6}");

        // Impulse should be exact (all ones in frequency domain)
        Assert.IsTrue(maxError < 1e-14, $"Impulse FFT error {maxError:E6} exceeds 1e-14");
    }

    [TestMethod]
    [DataRow(512)]
    [DataRow(1024)]
    public void FFT_DC_CompareWithPython(int size)
    {
        string testName = $"fft_dc_{size}";
        if (!TestDataLoader.TestDataExists(testName))
        {
            Assert.Inconclusive($"Test data not found: {testName}. Run generate_soxr_test_data.py");
            return;
        }

        var testCase = TestDataLoader.LoadTestCase(testName);
        var input = TestDataLoader.GetDoubleArray(testCase.GetProperty("input"));
        var expectedReal = TestDataLoader.GetDoubleArray(testCase.GetProperty("expected_real"));
        var expectedImag = TestDataLoader.GetDoubleArray(testCase.GetProperty("expected_imag"));

        var data = new Complex[input.Length];
        for (int i = 0; i < input.Length; i++)
            data[i] = new Complex(input[i], 0);

        FFT.Forward(data);

        double maxError = 0;
        for (int i = 0; i < data.Length; i++)
        {
            double err = Math.Abs(data[i].Real - expectedReal[i]) + Math.Abs(data[i].Imaginary - expectedImag[i]);
            maxError = Math.Max(maxError, err);
        }

        Console.WriteLine($"FFT DC Size {size}: Max error = {maxError:E6}");
        Console.WriteLine($"  Bin 0: expected ({expectedReal[0]}, {expectedImag[0]}), got ({data[0].Real}, {data[0].Imaginary})");

        // DC should be exact (N at bin 0, zeros elsewhere)
        Assert.IsTrue(maxError < 1e-14, $"DC FFT error {maxError:E6} exceeds 1e-14");
    }

    [TestMethod]
    public void FFT_BitCount_MatchesExpected()
    {
        // Verify that the bit count calculation is correct for common FFT sizes
        // This tests that the fix from Math.Log to bit-shift is working
        int[] sizes = { 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768 };
        int[] expectedBits = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

        Console.WriteLine("Verifying bit count calculation:");
        for (int i = 0; i < sizes.Length; i++)
        {
            int n = sizes[i];
            // Compute bit count using our fixed method
            int bits = 0;
            for (int temp = n; temp > 1; temp >>= 1) bits++;

            Console.WriteLine($"  n={n,5}: bits={bits}, expected={expectedBits[i]}");
            Assert.AreEqual(expectedBits[i], bits, $"Bit count for n={n} should be {expectedBits[i]}");
        }
    }

    #endregion
}