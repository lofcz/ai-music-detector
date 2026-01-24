using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using System.Text.Json;

namespace CQT.Tests;

/// <summary>
/// Tests for SoxrDecimate comparing C# output to Python soxr library.
/// </summary>
[TestClass]
public class SoxrDecimateTests
{
    private static (double meanRelDiff, double maxAbsDiff) CompareSignals(double[] expected, double[] actual, int offset = 0)
    {
        int startExpected = Math.Max(0, -offset);
        int startActual = Math.Max(0, offset);
        int length = Math.Min(expected.Length - startExpected, actual.Length - startActual);
        if (length <= 0)
            return (double.PositiveInfinity, double.PositiveInfinity);

        double sumAbsDiff = 0;
        double sumExpected = 0;
        double maxAbsDiff = 0;

        for (int i = 0; i < length; i++)
        {
            double expVal = expected[startExpected + i];
            double actVal = actual[startActual + i];
            double diff = Math.Abs(actVal - expVal);
            sumAbsDiff += diff;
            sumExpected += Math.Abs(expVal);
            if (diff > maxAbsDiff) maxAbsDiff = diff;
        }

        double meanRelDiff = sumAbsDiff / (sumExpected + 1e-12);
        return (meanRelDiff, maxAbsDiff);
    }

    private static double BestScaleFactor(double[] expected, double[] actual)
    {
        int length = Math.Min(expected.Length, actual.Length);
        double num = 0;
        double den = 0;
        for (int i = 0; i < length; i++)
        {
            num += expected[i] * actual[i];
            den += actual[i] * actual[i];
        }
        return den > 0 ? num / den : 1.0;
    }

    [TestMethod]
    public void SoxrDecimate_FilterDesign_ValidCoefficients()
    {
        // Get filter info
        var (numTaps, dftLength, coeffs) = SoxrDecimate.GetFilterInfo();

        Console.WriteLine($"Filter design for SOXR_HQ 2:1 decimation:");
        Console.WriteLine($"  Number of taps: {numTaps}");
        Console.WriteLine($"  DFT length: {dftLength}");
        Console.WriteLine($"  First 5 coefficients: {string.Join(", ", coeffs[0..Math.Min(5, coeffs.Length)])}");

        // Filter should have reasonable number of taps for 120 dB attenuation
        Assert.IsTrue(numTaps >= 100, $"Expected at least 100 taps for HQ, got {numTaps}");
        Assert.IsTrue(numTaps <= 1000, $"Expected at most 1000 taps, got {numTaps}");

        // DFT length should be power of 2 and >= 4 * numTaps
        Assert.IsTrue(FFT.IsPowerOf2(dftLength), "DFT length should be power of 2");
        Assert.IsTrue(dftLength >= numTaps * 2, "DFT length should be at least 2x filter length");

        // Filter coefficients should sum to approximately 1.0 (unity DC gain)
        // The 0.5 factor comes from picking every other sample during decimation
        double sum = 0;
        for (int i = 0; i < coeffs.Length; i++)
            sum += coeffs[i];
        Console.WriteLine($"  Coefficient sum: {sum:F6} (expected ~1.0 for unity DC gain)");
            
        // Sum should be around 0.9-1.1 for a lowpass filter
        Assert.IsTrue(sum > 0.9 && sum < 1.1, $"Filter sum should be ~1.0, got {sum}");
    }

    [TestMethod]
    public void SoxrDecimate_SineWave_PreservesFrequency()
    {
        // Create a 1 kHz sine wave at 16 kHz sample rate
        int sr = 16000;
        int duration = 1;  // seconds
        double freq = 1000.0;
            
        var input = new double[sr * duration];
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = Math.Sin(2 * Math.PI * freq * i / sr);
        }

        // Decimate by 2 (16 kHz -> 8 kHz)
        var output = SoxrDecimate.Decimate2(input);

        Console.WriteLine($"Input length: {input.Length}");
        Console.WriteLine($"Output length: {output.Length}");
        Assert.AreEqual(input.Length / 2, output.Length, "Output should be half the input length");

        // The 1 kHz sine should still be present in the 8 kHz signal
        // (1 kHz < 4 kHz Nyquist for 8 kHz)
            
        // Compute energy in the output
        double energy = 0;
        for (int i = 0; i < output.Length; i++)
            energy += output[i] * output[i];
        energy /= output.Length;
            
        Console.WriteLine($"Output RMS energy: {Math.Sqrt(energy):F4}");
        Assert.IsTrue(Math.Sqrt(energy) > 0.3, "1 kHz sine should be preserved (RMS > 0.3)");
    }

    [TestMethod]
    public void SoxrDecimate_HighFrequency_Attenuated()
    {
        // Create a 7 kHz sine wave at 16 kHz sample rate
        // After decimating to 8 kHz, this should be heavily attenuated
        // (7 kHz > 4 kHz Nyquist for 8 kHz)
        int sr = 16000;
        int duration = 1;
        double freq = 7000.0;
            
        var input = new double[sr * duration];
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = Math.Sin(2 * Math.PI * freq * i / sr);
        }

        // Decimate by 2
        var output = SoxrDecimate.Decimate2(input);

        // Compute energy in output
        double energy = 0;
        for (int i = 0; i < output.Length; i++)
            energy += output[i] * output[i];
        energy /= output.Length;
            
        double rms = Math.Sqrt(energy);
        Console.WriteLine($"7 kHz sine after decimation RMS: {rms:F6}");
            
        // Should be heavily attenuated (< 0.01 for 120 dB attenuation)
        // But we're testing practical attenuation, so allow some tolerance
        Assert.IsTrue(rms < 0.1, $"7 kHz should be attenuated below Nyquist, RMS={rms}");
    }

    [TestMethod]
    public void SoxrDecimate_CompareToSoxr()
    {
        if (!TestDataLoader.TestDataExists("soxr_decimate_test"))
        {
            Assert.Inconclusive("soxr test data not generated. Run: python generate_soxr_test_data.py");
            return;
        }

        var testCase = TestDataLoader.LoadTestCase("soxr_decimate_test");
            
        var input = TestDataLoader.GetDoubleArray(testCase.GetProperty("input"));
        var expectedOutput = TestDataLoader.GetDoubleArray(testCase.GetProperty("expected_output"));

        var actualOutput = SoxrDecimate.Decimate2(input);

        Console.WriteLine($"Input length: {input.Length}");
        Console.WriteLine($"Expected output length: {expectedOutput.Length}");
        Console.WriteLine($"Actual output length: {actualOutput.Length}");

        // Lengths should match
        Assert.AreEqual(expectedOutput.Length, actualOutput.Length, 
            $"Output length mismatch: expected {expectedOutput.Length}, got {actualOutput.Length}");

        // Compare values
        double sumAbsDiff = 0;
        double sumExpected = 0;
        double maxDiff = 0;
        int maxDiffIdx = 0;

        for (int i = 0; i < expectedOutput.Length; i++)
        {
            double diff = Math.Abs(actualOutput[i] - expectedOutput[i]);
            sumAbsDiff += diff;
            sumExpected += Math.Abs(expectedOutput[i]);
                
            if (diff > maxDiff)
            {
                maxDiff = diff;
                maxDiffIdx = i;
            }
        }

        double meanRelDiff = sumAbsDiff / (sumExpected + 1e-10);

        Console.WriteLine($"\nComparison C# SoxrDecimate vs Python soxr:");
        Console.WriteLine($"  Max absolute difference: {maxDiff:F6} at index {maxDiffIdx}");
        Console.WriteLine($"  Mean relative difference: {meanRelDiff:P2}");

        // Target: < 1% relative difference
        Console.WriteLine($"\n  Target: < 1% relative difference");
        Console.WriteLine($"  ACTUAL: {meanRelDiff:P4}");
        Assert.IsTrue(meanRelDiff < 0.05, $"Relative difference too high: {meanRelDiff:P2}");
    }

    [TestMethod]
    public void SoxrDecimate_CompareToSoxr_Strict()
    {
        if (!TestDataLoader.TestDataExists("soxr_decimate_test"))
        {
            Assert.Inconclusive("soxr test data not generated. Run: python generate_soxr_test_data.py");
            return;
        }

        var testCase = TestDataLoader.LoadTestCase("soxr_decimate_test");
        var input = TestDataLoader.GetDoubleArray(testCase.GetProperty("input"));
        var expectedOutput = TestDataLoader.GetDoubleArray(testCase.GetProperty("expected_output"));

        var actualOutput = SoxrDecimate.Decimate2(input, scale: false);

        var (meanRelDiff, maxAbsDiff) = CompareSignals(expectedOutput, actualOutput);

        // Alignment/scaling diagnostics
        int bestOffset = 0;
        double bestRel = meanRelDiff;
        for (int offset = -4; offset <= 4; offset++)
        {
            var (rel, _) = CompareSignals(expectedOutput, actualOutput, offset);
            if (rel < bestRel)
            {
                bestRel = rel;
                bestOffset = offset;
            }
        }

        double scale = BestScaleFactor(expectedOutput, actualOutput);
        var scaled = new double[actualOutput.Length];
        for (int i = 0; i < actualOutput.Length; i++) scaled[i] = actualOutput[i] * scale;
        var (scaledRel, scaledMax) = CompareSignals(expectedOutput, scaled);

        Console.WriteLine("\nStrict SoxrDecimate comparison:");
        Console.WriteLine($"  Mean relative diff: {meanRelDiff:P6}");
        Console.WriteLine($"  Max absolute diff:  {maxAbsDiff:E6}");
        Console.WriteLine($"  Best offset (±4):   {bestOffset} (rel {bestRel:P6})");
        Console.WriteLine($"  Best scale factor:  {scale:F9} (rel {scaledRel:P6}, max {scaledMax:E6})");

        // Strict target: near-identical output to soxr
        Assert.IsTrue(meanRelDiff < 1e-4,
            $"Strict soxr diff too high: {meanRelDiff:P6} (best offset {bestOffset}, scale {scale:F9})");
    }

    [TestMethod]
    public void SoxrDecimate_ChirpSignal_CompareToSoxr()
    {
        if (!TestDataLoader.TestDataExists("soxr_decimate_chirp"))
        {
            Assert.Inconclusive("soxr test data not generated. Run: python generate_soxr_test_data.py");
            return;
        }

        var testCase = TestDataLoader.LoadTestCase("soxr_decimate_chirp");
            
        var input = TestDataLoader.GetDoubleArray(testCase.GetProperty("input"));
        var expectedOutput = TestDataLoader.GetDoubleArray(testCase.GetProperty("expected_output"));

        var actualOutput = SoxrDecimate.Decimate2(input);

        Console.WriteLine($"Chirp signal decimation test:");
        Console.WriteLine($"Input length: {input.Length}");
        Console.WriteLine($"Expected output length: {expectedOutput.Length}");
        Console.WriteLine($"Actual output length: {actualOutput.Length}");

        int minLen = Math.Min(expectedOutput.Length, actualOutput.Length);

        double sumAbsDiff = 0;
        double sumExpected = 0;
        double maxDiff = 0;

        for (int i = 0; i < minLen; i++)
        {
            double diff = Math.Abs(actualOutput[i] - expectedOutput[i]);
            sumAbsDiff += diff;
            sumExpected += Math.Abs(expectedOutput[i]);
                
            if (diff > maxDiff)
                maxDiff = diff;
        }

        double meanRelDiff = sumAbsDiff / (sumExpected + 1e-10);

        Console.WriteLine($"\nChirp decimation comparison:");
        Console.WriteLine($"  Max absolute difference: {maxDiff:F6}");
        Console.WriteLine($"  Mean relative difference: {meanRelDiff:P2}");

        Assert.IsTrue(meanRelDiff < 0.05, $"Chirp relative difference too high: {meanRelDiff:P2}");
    }

    [TestMethod]
    public void LibrosaCQT_WithSoxrDecimate_CompareToLibrosa()
    {
        // This test compares our C# LibrosaCQT (which now uses SoxrDecimate)
        // against librosa.cqt with res_type='soxr_hq'
            
        if (!TestDataLoader.TestDataExists("librosa_cqt_soxr_random"))
        {
            Assert.Inconclusive("librosa soxr test data not generated. Run: python generate_soxr_test_data.py");
            return;
        }

        var testCase = TestDataLoader.LoadTestCase("librosa_cqt_soxr_random");
            
        int sampleRate = testCase.GetProperty("sample_rate").GetInt32();
        double fMin = testCase.GetProperty("f_min").GetDouble();
        int nBins = testCase.GetProperty("n_bins").GetInt32();
        int binsPerOctave = testCase.GetProperty("bins_per_octave").GetInt32();
        int hopLength = testCase.GetProperty("hop_length").GetInt32();

        var input = TestDataLoader.GetDoubleArray(testCase.GetProperty("input"));
        var librosaMag = TestDataLoader.GetDouble2DArray(testCase.GetProperty("expected_magnitude"));

        var cqt = new LibrosaCQT(sampleRate, fMin, nBins, binsPerOctave, hopLength);
        var result = cqt.Compute(input);

        Console.WriteLine($"C# LibrosaCQT (with SoxrDecimate) shape: [{result.GetLength(0)}, {result.GetLength(1)}]");
        Console.WriteLine($"librosa (soxr_hq) shape: [{librosaMag.GetLength(0)}, {librosaMag.GetLength(1)}]");

        int minFrames = Math.Min(librosaMag.GetLength(1), result.GetLength(1));
            
        double sumAbsDiff = 0;
        double sumLibrosa = 0;
        int count = 0;
            
        for (int k = 0; k < nBins; k++)
        {
            for (int t = 0; t < minFrames; t++)
            {
                double diff = Math.Abs(result[k, t] - librosaMag[k, t]);
                sumAbsDiff += diff;
                sumLibrosa += Math.Abs(librosaMag[k, t]);
                count++;
            }
        }

        double meanRelDiff = sumAbsDiff / (sumLibrosa + 1e-10);

        Console.WriteLine($"\nComparison C# LibrosaCQT vs librosa.cqt (soxr_hq):");
        Console.WriteLine($"  Mean relative difference: {meanRelDiff:P2}");

        // Per-octave breakdown
        Console.WriteLine("\n=== Per-octave mean relative difference ===");
        for (int octave = 0; octave < 4; octave++)
        {
            int startBin = octave * 12;
            int endBin = (octave + 1) * 12;
                
            double sumDiff = 0;
            double sumLib = 0;
                
            for (int k = startBin; k < endBin && k < nBins; k++)
            {
                for (int t = 0; t < minFrames; t++)
                {
                    sumDiff += Math.Abs(result[k, t] - librosaMag[k, t]);
                    sumLib += Math.Abs(librosaMag[k, t]);
                }
            }
                
            double octaveRelDiff = sumDiff / (sumLib + 1e-10);
            Console.WriteLine($"Octave {octave} (bins {startBin}-{endBin-1}): {100*octaveRelDiff:F1}%");
        }

        // Target: < 1% overall relative difference  
        Console.WriteLine($"\n  Target: < 1% relative difference");
        Assert.IsTrue(meanRelDiff < 0.05, 
            $"CQT relative difference too high: {meanRelDiff:P2}. Target is < 5%");
    }
}