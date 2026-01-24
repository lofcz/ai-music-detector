using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace CQT.Tests;

[TestClass]
public class ConstantQTransformTests
{
    [TestMethod]
    public void CQT_Frequencies_AreLogarithmicallySpaced()
    {
        var cqt = new ConstantQTransform(
            sampleRate: 16000,
            fMin: 500,
            nBins: 48,
            binsPerOctave: 12,
            hopLength: 512
        );

        var freqs = cqt.Frequencies;

        // Check first and last frequency
        Assert.AreEqual(500.0, freqs[0], 0.01, "First frequency");
            
        // 48 bins = 4 octaves, so last freq = 500 * 2^4 = 8000 Hz
        // Actually, last bin is at index 47, so: 500 * 2^(47/12)
        double expectedLast = 500.0 * Math.Pow(2.0, 47.0 / 12.0);
        Assert.AreEqual(expectedLast, freqs[47], 0.01, "Last frequency");

        // Check octave relationship: every 12 bins should double
        Assert.AreEqual(freqs[0] * 2, freqs[12], 0.01, "One octave up");
        Assert.AreEqual(freqs[0] * 4, freqs[24], 0.01, "Two octaves up");
    }

    [TestMethod]
    public void CQT_OutputShape_IsCorrect()
    {
        var cqt = new ConstantQTransform(
            sampleRate: 16000,
            fMin: 500,
            nBins: 48,
            binsPerOctave: 12,
            hopLength: 512
        );

        // 1 second of audio = 16000 samples
        var audio = new double[16000];
        var result = cqt.Compute(audio);

        // Should have 48 frequency bins
        Assert.AreEqual(48, result.GetLength(0), "Number of frequency bins");

        // Number of frames = 16000 / 512 = 31
        Assert.AreEqual(31, result.GetLength(1), "Number of time frames");
    }

    [TestMethod]
    public void CQT_SineWave_PeaksAtCorrectBin()
    {
        int sampleRate = 16000;
        var cqt = new ConstantQTransform(
            sampleRate: sampleRate,
            fMin: 500,
            nBins: 48,
            binsPerOctave: 12,
            hopLength: 512
        );

        // Generate 1 kHz sine wave
        double frequency = 1000.0;
        var audio = new double[sampleRate]; // 1 second
        for (int i = 0; i < audio.Length; i++)
        {
            audio[i] = Math.Sin(2 * Math.PI * frequency * i / sampleRate);
        }

        var result = cqt.Compute(audio);

        // Find bin with maximum energy (average across frames)
        int maxBin = 0;
        double maxEnergy = 0;
        for (int k = 0; k < 48; k++)
        {
            double energy = 0;
            for (int t = 0; t < result.GetLength(1); t++)
                energy += result[k, t];
                
            if (energy > maxEnergy)
            {
                maxEnergy = energy;
                maxBin = k;
            }
        }

        // 1000 Hz should be at bin: log2(1000/500) * 12 = 12
        // (one octave above 500 Hz)
        int expectedBin = (int)Math.Round(Math.Log(frequency / 500.0, 2) * 12);
            
        Assert.AreEqual(expectedBin, maxBin, 1, $"Peak should be near bin {expectedBin}, got {maxBin}");
    }

    [TestMethod]
    public void CQT_Silence_ReturnsNearZero()
    {
        var cqt = new ConstantQTransform(
            sampleRate: 16000,
            fMin: 500,
            nBins: 48,
            binsPerOctave: 12,
            hopLength: 512
        );

        var silence = new double[8000];
        var result = cqt.Compute(silence);

        for (int k = 0; k < result.GetLength(0); k++)
        {
            for (int t = 0; t < result.GetLength(1); t++)
            {
                Assert.AreEqual(0.0, result[k, t], 1e-10, $"Should be zero at [{k},{t}]");
            }
        }
    }

    [TestMethod]
    public void CQT_SingleFrame_MatchesPythonReference()
    {
        if (!TestDataLoader.TestDataExists("cqt_single_frame"))
        {
            Assert.Inconclusive("Test data not generated. Run python generate_test_data.py");
            return;
        }

        var testCase = TestDataLoader.LoadTestCase("cqt_single_frame");
            
        int sampleRate = testCase.GetProperty("sample_rate").GetInt32();
        double fMin = testCase.GetProperty("f_min").GetDouble();
        int nBins = testCase.GetProperty("n_bins").GetInt32();
        int binsPerOctave = testCase.GetProperty("bins_per_octave").GetInt32();
        int hopLength = testCase.GetProperty("hop_length").GetInt32();

        var input = TestDataLoader.GetDoubleArray(testCase.GetProperty("input"));
        var expectedMag = TestDataLoader.GetDoubleArray(testCase.GetProperty("expected_magnitude"));

        var cqt = new ConstantQTransform(sampleRate, fMin, nBins, binsPerOctave, hopLength);
        var result = cqt.ComputeFrame(input);

        for (int i = 0; i < nBins; i++)
        {
            Assert.AreEqual(expectedMag[i], result[i].Magnitude, 1e-3, $"Magnitude at bin {i}");
        }
    }

    [TestMethod]
    public void CQT_STCQT_MatchesPythonReference()
    {
        if (!TestDataLoader.TestDataExists("cqt_random"))
        {
            Assert.Inconclusive("Test data not generated. Run python generate_test_data.py");
            return;
        }

        var testCase = TestDataLoader.LoadTestCase("cqt_random");
            
        int sampleRate = testCase.GetProperty("sample_rate").GetInt32();
        double fMin = testCase.GetProperty("f_min").GetDouble();
        int nBins = testCase.GetProperty("n_bins").GetInt32();
        int binsPerOctave = testCase.GetProperty("bins_per_octave").GetInt32();
        int hopLength = testCase.GetProperty("hop_length").GetInt32();

        var input = TestDataLoader.GetDoubleArray(testCase.GetProperty("input"));
        var expected = TestDataLoader.GetDouble2DArray(testCase.GetProperty("expected"));

        var cqt = new ConstantQTransform(sampleRate, fMin, nBins, binsPerOctave, hopLength);
        var result = cqt.Compute(input);

        int nRows = expected.GetLength(0);
        int nCols = expected.GetLength(1);

        Assert.AreEqual(nRows, result.GetLength(0), "Number of bins");
        Assert.AreEqual(nCols, result.GetLength(1), "Number of frames");

        for (int i = 0; i < nRows; i++)
        {
            for (int j = 0; j < nCols; j++)
            {
                Assert.AreEqual(expected[i, j], result[i, j], 1e-3, $"Value at [{i},{j}]");
            }
        }
    }
}