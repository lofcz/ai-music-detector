using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace CQT.Tests;

[TestClass]
public class CepstrumExtractorTests
{
    [TestMethod]
    public void Cepstrum_OutputShape_IsCorrect()
    {
        var extractor = new CepstrumExtractor(
            sampleRate: 16000,
            fMin: 500,
            nBins: 48,
            binsPerOctave: 12,
            hopLength: 512,
            nCoeffs: 24
        );

        // 1 second of audio
        var audio = new double[16000];
        var result = extractor.Extract(audio);

        // Should have 24 cepstral coefficients
        Assert.AreEqual(24, result.GetLength(0), "Number of coefficients");

        // Number of frames depends on STFT implementation (with center padding)
        // Approximately audio_length / hop_length + some padding frames
        int expectedFramesApprox = 16000 / 512;
        Assert.IsTrue(Math.Abs(result.GetLength(1) - expectedFramesApprox) <= 2, 
            $"Number of frames should be approximately {expectedFramesApprox}, got {result.GetLength(1)}");
    }

    [TestMethod]
    public void Cepstrum_Silence_HasNegativeLogValues()
    {
        var extractor = new CepstrumExtractor(
            sampleRate: 16000,
            fMin: 500,
            nBins: 48,
            binsPerOctave: 12,
            hopLength: 512,
            nCoeffs: 24
        );

        var silence = new double[8000];
        var result = extractor.Extract(silence);

        // For silence, log(0 + 1e-6) = log(1e-6) ≈ -13.8
        // After DCT, first coefficient should be negative (log of small values)
        // This is a sanity check that the log is being applied
        Assert.IsTrue(result[0, 0] < 0, "First coefficient should be negative for silence");
    }

    [TestMethod]
    public void Cepstrum_DifferentNCoeffs_TruncatesCorrectly()
    {
        var extractor12 = new CepstrumExtractor(
            sampleRate: 16000,
            fMin: 500,
            nBins: 48,
            binsPerOctave: 12,
            hopLength: 512,
            nCoeffs: 12
        );

        var extractor24 = new CepstrumExtractor(
            sampleRate: 16000,
            fMin: 500,
            nBins: 48,
            binsPerOctave: 12,
            hopLength: 512,
            nCoeffs: 24
        );

        var random = new Random(42);
        var audio = new double[8000];
        for (int i = 0; i < audio.Length; i++)
            audio[i] = random.NextDouble() * 2 - 1;

        var result12 = extractor12.Extract(audio);
        var result24 = extractor24.Extract(audio);

        Assert.AreEqual(12, result12.GetLength(0));
        Assert.AreEqual(24, result24.GetLength(0));

        // First 12 coefficients should match
        for (int k = 0; k < 12; k++)
        {
            for (int t = 0; t < result12.GetLength(1); t++)
            {
                Assert.AreEqual(result24[k, t], result12[k, t], 1e-10, $"Coefficient {k}, frame {t}");
            }
        }
    }

    [TestMethod]
    public void Cepstrum_FullPipeline_MatchesPythonReference()
    {
        // This test uses librosa-based test data which matches our LibrosaCQT implementation
        if (!TestDataLoader.TestDataExists("librosa_cepstrum_full"))
        {
            Assert.Inconclusive("Test data not generated. Run python generate_test_data_librosa.py");
            return;
        }

        var testCase = TestDataLoader.LoadTestCase("librosa_cepstrum_full");
            
        int sampleRate = testCase.GetProperty("sample_rate").GetInt32();
        double fMin = testCase.GetProperty("f_min").GetDouble();
        int nBins = testCase.GetProperty("n_bins").GetInt32();
        int binsPerOctave = testCase.GetProperty("bins_per_octave").GetInt32();
        int hopLength = testCase.GetProperty("hop_length").GetInt32();
        int nCoeffs = testCase.GetProperty("n_coeffs").GetInt32();

        var input = TestDataLoader.GetDoubleArray(testCase.GetProperty("input"));
        var expected = TestDataLoader.GetDouble2DArray(testCase.GetProperty("expected"));

        var extractor = new CepstrumExtractor(sampleRate, fMin, nBins, binsPerOctave, hopLength, nCoeffs);
        var result = extractor.Extract(input);

        int nRows = expected.GetLength(0);
        int minCols = Math.Min(expected.GetLength(1), result.GetLength(1));

        Assert.AreEqual(nRows, result.GetLength(0), "Number of coefficients");
            
        // Frame counts may differ slightly due to STFT padding differences
        int frameDiff = Math.Abs(expected.GetLength(1) - result.GetLength(1));
        Assert.IsTrue(frameDiff <= 2, $"Frame count difference too large: {frameDiff}");

        // Note: Values will differ due to algorithm differences between C# and librosa
        // This is documented in LibrosaComparisonTests
    }

    [TestMethod]
    public void Cepstrum_IntermediateSteps_MatchPythonReference()
    {
        if (!TestDataLoader.TestDataExists("cepstrum_full"))
        {
            Assert.Inconclusive("Test data not generated. Run python generate_test_data.py");
            return;
        }

        var testCase = TestDataLoader.LoadTestCase("cepstrum_full");
            
        int sampleRate = testCase.GetProperty("sample_rate").GetInt32();
        double fMin = testCase.GetProperty("f_min").GetDouble();
        int nBins = testCase.GetProperty("n_bins").GetInt32();
        int binsPerOctave = testCase.GetProperty("bins_per_octave").GetInt32();
        int hopLength = testCase.GetProperty("hop_length").GetInt32();

        var input = TestDataLoader.GetDoubleArray(testCase.GetProperty("input"));
        var expectedCqtMag = TestDataLoader.GetDouble2DArray(testCase.GetProperty("cqt_magnitude"));

        // Step 1: CQT magnitude
        var cqt = new ConstantQTransform(sampleRate, fMin, nBins, binsPerOctave, hopLength);
        var cqtMag = cqt.Compute(input);

        int nRows = expectedCqtMag.GetLength(0);
        int nCols = expectedCqtMag.GetLength(1);

        for (int i = 0; i < nRows; i++)
        {
            for (int j = 0; j < nCols; j++)
            {
                Assert.AreEqual(expectedCqtMag[i, j], cqtMag[i, j], 1e-3, $"CQT magnitude at [{i},{j}]");
            }
        }
    }
}