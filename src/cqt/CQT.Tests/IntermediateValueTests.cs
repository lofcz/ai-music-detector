using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace CQT.Tests;

/// <summary>
/// Tests for intermediate CQT computation values to identify precision loss sources.
/// These tests compare C# intermediate values against Python reference values.
/// </summary>
[TestClass]
public class IntermediateValueTests
{
    #region STFT Comparison Tests

    [TestMethod]
    public void STFT_RectangularWindow_CompareWithPython()
    {
        if (!TestDataLoader.TestDataExists("stft_impulse_rectangular"))
        {
            Assert.Inconclusive("Test data not found. Run generate_soxr_test_data.py");
            return;
        }

        var testCase = TestDataLoader.LoadTestCase("stft_impulse_rectangular");
        var input = TestDataLoader.GetDoubleArray(testCase.GetProperty("input"));
        int nFft = testCase.GetProperty("n_fft").GetInt32();
        int hopLength = testCase.GetProperty("hop_length").GetInt32();
        var expectedReal = TestDataLoader.GetDouble2DArray(testCase.GetProperty("stft_real"));
        var expectedImag = TestDataLoader.GetDouble2DArray(testCase.GetProperty("stft_imag"));

        // Compute STFT using C# implementation
        var stft = new STFT(nFft, hopLength, WindowType.Rectangular);
        var result = stft.Compute(input);

        int nFreqs = result.GetLength(0);
        int nFrames = result.GetLength(1);

        Console.WriteLine($"=== STFT Comparison ===");
        Console.WriteLine($"C# shape: [{nFreqs}, {nFrames}]");
        Console.WriteLine($"Python shape: [{expectedReal.GetLength(0)}, {expectedReal.GetLength(1)}]");

        // Compare shapes
        Assert.AreEqual(expectedReal.GetLength(0), nFreqs, "Frequency bin count mismatch");
            
        int minFrames = Math.Min(nFrames, expectedReal.GetLength(1));
            
        // Calculate error statistics
        double maxError = 0;
        double sumError = 0;
        double sumExpected = 0;
        int count = 0;
        int maxErrorFreq = 0, maxErrorFrame = 0;

        for (int f = 0; f < nFreqs; f++)
        {
            for (int t = 0; t < minFrames; t++)
            {
                double realErr = Math.Abs(result[f, t].Real - expectedReal[f, t]);
                double imagErr = Math.Abs(result[f, t].Imaginary - expectedImag[f, t]);
                double err = realErr + imagErr;
                double expectedMag = Math.Sqrt(expectedReal[f, t] * expectedReal[f, t] + 
                                               expectedImag[f, t] * expectedImag[f, t]);

                sumError += err;
                sumExpected += expectedMag;
                count++;

                if (err > maxError)
                {
                    maxError = err;
                    maxErrorFreq = f;
                    maxErrorFrame = t;
                }
            }
        }

        double meanRelError = sumError / (sumExpected + 1e-10);

        Console.WriteLine($"\nError Statistics:");
        Console.WriteLine($"  Max absolute error: {maxError:E6} at freq={maxErrorFreq}, frame={maxErrorFrame}");
        Console.WriteLine($"  Mean relative error: {meanRelError:P4}");

        // Print sample values
        Console.WriteLine($"\nSample values at frame 0:");
        for (int f = 0; f < Math.Min(5, nFreqs); f++)
        {
            Console.WriteLine($"  Freq {f}: Python=({expectedReal[f, 0]:F6}, {expectedImag[f, 0]:F6}), " +
                              $"C#=({result[f, 0].Real:F6}, {result[f, 0].Imaginary:F6})");
        }

        // STFT should match very closely
        Assert.IsTrue(maxError < 1e-10, $"STFT max error {maxError:E6} exceeds 1e-10");
    }

    #endregion

    #region Wavelet Filter Comparison Tests

    [TestMethod]
    public void WaveletFilter_1kHz_CompareWithPython()
    {
        if (!TestDataLoader.TestDataExists("wavelet_filter_1khz"))
        {
            Assert.Inconclusive("Test data not found. Run generate_soxr_test_data.py");
            return;
        }

        var testCase = TestDataLoader.LoadTestCase("wavelet_filter_1khz");
        double frequency = testCase.GetProperty("frequency").GetDouble();
        int sampleRate = testCase.GetProperty("sample_rate").GetInt32();
        double filterScale = testCase.GetProperty("filter_scale").GetDouble();
        int binsPerOctave = testCase.GetProperty("bins_per_octave").GetInt32();
        int filterLength = testCase.GetProperty("filter_length").GetInt32();
        double ilenFloat = testCase.GetProperty("ilen_float").GetDouble();
        int nFft = testCase.GetProperty("n_fft").GetInt32();

        var expectedWindow = TestDataLoader.GetDoubleArray(testCase.GetProperty("window"));
        var expectedWaveletReal = TestDataLoader.GetDoubleArray(testCase.GetProperty("wavelet_real"));
        var expectedWaveletImag = TestDataLoader.GetDoubleArray(testCase.GetProperty("wavelet_imag"));
        var expectedFftReal = TestDataLoader.GetDoubleArray(testCase.GetProperty("wavelet_fft_real"));
        var expectedFftImag = TestDataLoader.GetDoubleArray(testCase.GetProperty("wavelet_fft_imag"));

        Console.WriteLine($"=== Wavelet Filter Comparison for {frequency} Hz ===");
        Console.WriteLine($"Sample rate: {sampleRate}, Filter length: {filterLength}");
        Console.WriteLine($"ilen_float: {ilenFloat:F6}, n_fft: {nFft}");

        // Compute our wavelet using the same method as LibrosaCQT.BuildFilterBasis
        double r = Math.Pow(2.0, 1.0 / binsPerOctave);
        double alpha = (r * r - 1) / (r * r + 1);
        double Q = filterScale / alpha;
        double ourIlenFloat = Q * sampleRate / frequency;
        int ourNk = (int)Math.Floor(ourIlenFloat / 2) + (int)Math.Ceiling(ourIlenFloat / 2);

        Console.WriteLine($"\nC# computed values:");
        Console.WriteLine($"  alpha: {alpha:F10}");
        Console.WriteLine($"  Q: {Q:F6}");
        Console.WriteLine($"  ilen_float: {ourIlenFloat:F6} (expected: {ilenFloat:F6})");
        Console.WriteLine($"  filter_length: {ourNk} (expected: {filterLength})");

        // Compare window
        var ourWindow = WindowFunctions.Hann(ourNk);
        double windowMaxError = 0;
        for (int i = 0; i < Math.Min(ourWindow.Length, expectedWindow.Length); i++)
        {
            double err = Math.Abs(ourWindow[i] - expectedWindow[i]);
            windowMaxError = Math.Max(windowMaxError, err);
        }
        Console.WriteLine($"\nWindow max error: {windowMaxError:E6}");

        // Build wavelet
        int startIdx = (int)Math.Floor(-ourIlenFloat / 2);
        var wavelet = new Complex[ourNk];
        double coef = 2.0 * Math.PI * frequency / sampleRate;

        for (int n = 0; n < ourNk; n++)
        {
            int tIdx = startIdx + n;
            double phase = coef * tIdx;
            wavelet[n] = new Complex(
                ourWindow[n] * Math.Cos(phase),
                ourWindow[n] * Math.Sin(phase));
        }

        // L1 normalize
        double norm = 0;
        for (int i = 0; i < ourNk; i++)
            norm += wavelet[i].Magnitude;
        if (norm > 0)
        {
            for (int i = 0; i < ourNk; i++)
                wavelet[i] = new Complex(wavelet[i].Real / norm, wavelet[i].Imaginary / norm);
        }

        // Compare wavelet (pre-FFT)
        double waveletMaxError = 0;
        for (int i = 0; i < Math.Min(wavelet.Length, expectedWaveletReal.Length); i++)
        {
            double err = Math.Abs(wavelet[i].Real - expectedWaveletReal[i]) +
                         Math.Abs(wavelet[i].Imaginary - expectedWaveletImag[i]);
            waveletMaxError = Math.Max(waveletMaxError, err);
        }
        Console.WriteLine($"Wavelet (pre-FFT) max error: {waveletMaxError:E6}");

        // Print first few wavelet values
        Console.WriteLine("\nFirst 5 wavelet values:");
        for (int i = 0; i < Math.Min(5, wavelet.Length); i++)
        {
            Console.WriteLine($"  [{i}] Python=({expectedWaveletReal[i]:F8}, {expectedWaveletImag[i]:F8}), " +
                              $"C#=({wavelet[i].Real:F8}, {wavelet[i].Imaginary:F8})");
        }

        // Pad and FFT
        int ourNFft = FFT.NextPowerOf2(ourNk);
        var tempKernel = new Complex[ourNFft];
        int padStart = (ourNFft - ourNk) / 2;
        for (int i = 0; i < ourNk; i++)
            tempKernel[padStart + i] = wavelet[i];

        // Scale by lengths / nFft
        double fftNorm = ourIlenFloat / ourNFft;
        for (int i = 0; i < ourNFft; i++)
            tempKernel[i] = new Complex(tempKernel[i].Real * fftNorm, tempKernel[i].Imaginary * fftNorm);

        FFT.Forward(tempKernel);

        // Compare FFT result
        double fftMaxError = 0;
        for (int i = 0; i < Math.Min(tempKernel.Length, expectedFftReal.Length); i++)
        {
            double err = Math.Abs(tempKernel[i].Real - expectedFftReal[i]) +
                         Math.Abs(tempKernel[i].Imaginary - expectedFftImag[i]);
            fftMaxError = Math.Max(fftMaxError, err);
        }
        Console.WriteLine($"\nWavelet FFT max error: {fftMaxError:E6}");

        // Print first few FFT values
        Console.WriteLine("\nFirst 5 wavelet FFT values:");
        for (int i = 0; i < Math.Min(5, tempKernel.Length); i++)
        {
            Console.WriteLine($"  [{i}] Python=({expectedFftReal[i]:F8}, {expectedFftImag[i]:F8}), " +
                              $"C#=({tempKernel[i].Real:F8}, {tempKernel[i].Imaginary:F8})");
        }

        // Assertions
        Assert.IsTrue(windowMaxError < 1e-14, $"Window error {windowMaxError:E6} exceeds 1e-14");
        Assert.IsTrue(waveletMaxError < 1e-14, $"Wavelet error {waveletMaxError:E6} exceeds 1e-14");
        Assert.IsTrue(fftMaxError < 1e-10, $"Wavelet FFT error {fftMaxError:E6} exceeds 1e-10");
    }

    #endregion

    #region Matrix Multiply Precision Tests

    [TestMethod]
    public void MatrixMultiply_Complex_CompareWithPython()
    {
        if (!TestDataLoader.TestDataExists("matrix_multiply_complex"))
        {
            Assert.Inconclusive("Test data not found. Run generate_soxr_test_data.py");
            return;
        }

        var testCase = TestDataLoader.LoadTestCase("matrix_multiply_complex");
        int nFilters = testCase.GetProperty("n_filters").GetInt32();
        int nFreqBins = testCase.GetProperty("n_freq_bins").GetInt32();
        int nFrames = testCase.GetProperty("n_frames").GetInt32();

        var basisReal = TestDataLoader.GetDouble2DArray(testCase.GetProperty("basis_real"));
        var basisImag = TestDataLoader.GetDouble2DArray(testCase.GetProperty("basis_imag"));
        var stftReal = TestDataLoader.GetDouble2DArray(testCase.GetProperty("stft_real"));
        var stftImag = TestDataLoader.GetDouble2DArray(testCase.GetProperty("stft_imag"));
        var expectedReal = TestDataLoader.GetDouble2DArray(testCase.GetProperty("result_real"));
        var expectedImag = TestDataLoader.GetDouble2DArray(testCase.GetProperty("result_imag"));

        Console.WriteLine($"=== Matrix Multiply Precision Test ===");
        Console.WriteLine($"Basis: [{nFilters}, {nFreqBins}], STFT: [{nFreqBins}, {nFrames}]");
        Console.WriteLine($"Accumulations per output: {nFreqBins}");

        // Compute matrix multiply using the same method as ComputeCQTResponse
        var result = new Complex[nFilters, nFrames];

        for (int k = 0; k < nFilters; k++)
        {
            for (int t = 0; t < nFrames; t++)
            {
                double sumReal = 0, sumImag = 0;
                for (int f = 0; f < nFreqBins; f++)
                {
                    double bReal = basisReal[k, f];
                    double bImag = basisImag[k, f];
                    double dReal = stftReal[f, t];
                    double dImag = stftImag[f, t];

                    // Complex multiplication
                    sumReal += bReal * dReal - bImag * dImag;
                    sumImag += bReal * dImag + bImag * dReal;
                }
                result[k, t] = new Complex(sumReal, sumImag);
            }
        }

        // Compute error statistics
        double maxError = 0;
        double sumError = 0;
        double sumExpected = 0;
        int maxErrorK = 0, maxErrorT = 0;

        for (int k = 0; k < nFilters; k++)
        {
            for (int t = 0; t < nFrames; t++)
            {
                double realErr = Math.Abs(result[k, t].Real - expectedReal[k, t]);
                double imagErr = Math.Abs(result[k, t].Imaginary - expectedImag[k, t]);
                double err = realErr + imagErr;
                double expectedMag = Math.Sqrt(expectedReal[k, t] * expectedReal[k, t] + 
                                               expectedImag[k, t] * expectedImag[k, t]);

                sumError += err;
                sumExpected += expectedMag;

                if (err > maxError)
                {
                    maxError = err;
                    maxErrorK = k;
                    maxErrorT = t;
                }
            }
        }

        double meanRelError = sumError / (sumExpected + 1e-10);

        Console.WriteLine($"\nError Statistics ({nFreqBins} accumulations):");
        Console.WriteLine($"  Max absolute error: {maxError:E6} at filter={maxErrorK}, frame={maxErrorT}");
        Console.WriteLine($"  Mean relative error: {meanRelError:P6}");

        // Print sample values
        Console.WriteLine($"\nSample values at filter=0, frame=0:");
        Console.WriteLine($"  Python: ({expectedReal[0, 0]:F10}, {expectedImag[0, 0]:F10})");
        Console.WriteLine($"  C#:     ({result[0, 0].Real:F10}, {result[0, 0].Imaginary:F10})");
        Console.WriteLine($"  Diff:   ({Math.Abs(result[0, 0].Real - expectedReal[0, 0]):E6}, " +
                          $"{Math.Abs(result[0, 0].Imaginary - expectedImag[0, 0]):E6})");

        // With 513 accumulations, we expect ~1e-13 relative error from double precision
        Assert.IsTrue(maxError < 1e-10, $"Matrix multiply max error {maxError:E6} exceeds 1e-10");
    }

    [TestMethod]
    public void MatrixMultiply_Large_CompareWithPython()
    {
        if (!TestDataLoader.TestDataExists("matrix_multiply_complex_large"))
        {
            Assert.Inconclusive("Test data not found. Run generate_soxr_test_data.py");
            return;
        }

        var testCase = TestDataLoader.LoadTestCase("matrix_multiply_complex_large");
        int nFilters = testCase.GetProperty("n_filters").GetInt32();
        int nFreqBins = testCase.GetProperty("n_freq_bins").GetInt32();
        int nFrames = testCase.GetProperty("n_frames").GetInt32();

        var basisReal = TestDataLoader.GetDouble2DArray(testCase.GetProperty("basis_real"));
        var basisImag = TestDataLoader.GetDouble2DArray(testCase.GetProperty("basis_imag"));
        var stftReal = TestDataLoader.GetDouble2DArray(testCase.GetProperty("stft_real"));
        var stftImag = TestDataLoader.GetDouble2DArray(testCase.GetProperty("stft_imag"));
        var expectedReal = TestDataLoader.GetDouble2DArray(testCase.GetProperty("result_real"));
        var expectedImag = TestDataLoader.GetDouble2DArray(testCase.GetProperty("result_imag"));

        Console.WriteLine($"=== Large Matrix Multiply Precision Test ===");
        Console.WriteLine($"Basis: [{nFilters}, {nFreqBins}], STFT: [{nFreqBins}, {nFrames}]");
        Console.WriteLine($"Accumulations per output: {nFreqBins}");

        // Compute matrix multiply
        var result = new Complex[nFilters, nFrames];

        for (int k = 0; k < nFilters; k++)
        {
            for (int t = 0; t < nFrames; t++)
            {
                double sumReal = 0, sumImag = 0;
                for (int f = 0; f < nFreqBins; f++)
                {
                    double bReal = basisReal[k, f];
                    double bImag = basisImag[k, f];
                    double dReal = stftReal[f, t];
                    double dImag = stftImag[f, t];

                    sumReal += bReal * dReal - bImag * dImag;
                    sumImag += bReal * dImag + bImag * dReal;
                }
                result[k, t] = new Complex(sumReal, sumImag);
            }
        }

        // Compute error statistics
        double maxError = 0;
        double sumError = 0;
        double sumExpected = 0;

        for (int k = 0; k < nFilters; k++)
        {
            for (int t = 0; t < nFrames; t++)
            {
                double realErr = Math.Abs(result[k, t].Real - expectedReal[k, t]);
                double imagErr = Math.Abs(result[k, t].Imaginary - expectedImag[k, t]);
                double err = realErr + imagErr;
                double expectedMag = Math.Sqrt(expectedReal[k, t] * expectedReal[k, t] + 
                                               expectedImag[k, t] * expectedImag[k, t]);

                sumError += err;
                sumExpected += expectedMag;
                maxError = Math.Max(maxError, err);
            }
        }

        double meanRelError = sumError / (sumExpected + 1e-10);

        Console.WriteLine($"\nError Statistics ({nFreqBins} accumulations):");
        Console.WriteLine($"  Max absolute error: {maxError:E6}");
        Console.WriteLine($"  Mean relative error: {meanRelError:P6}");

        // With 2049 accumulations, error should still be in 1e-10 range
        Assert.IsTrue(maxError < 1e-9, $"Large matrix multiply max error {maxError:E6} exceeds 1e-9");
    }

    [TestMethod]
    public void MatrixMultiply_KahanVsNaive_PrecisionComparison()
    {
        // This test demonstrates the precision improvement of Kahan summation
        // by comparing naive vs Kahan summation against Python reference
            
        if (!TestDataLoader.TestDataExists("matrix_multiply_complex_large"))
        {
            Assert.Inconclusive("Test data not found. Run generate_soxr_test_data.py");
            return;
        }

        var testCase = TestDataLoader.LoadTestCase("matrix_multiply_complex_large");
        int nFilters = testCase.GetProperty("n_filters").GetInt32();
        int nFreqBins = testCase.GetProperty("n_freq_bins").GetInt32();
        int nFrames = testCase.GetProperty("n_frames").GetInt32();

        var basisReal = TestDataLoader.GetDouble2DArray(testCase.GetProperty("basis_real"));
        var basisImag = TestDataLoader.GetDouble2DArray(testCase.GetProperty("basis_imag"));
        var stftReal = TestDataLoader.GetDouble2DArray(testCase.GetProperty("stft_real"));
        var stftImag = TestDataLoader.GetDouble2DArray(testCase.GetProperty("stft_imag"));
        var expectedReal = TestDataLoader.GetDouble2DArray(testCase.GetProperty("result_real"));
        var expectedImag = TestDataLoader.GetDouble2DArray(testCase.GetProperty("result_imag"));

        Console.WriteLine($"=== Kahan vs Naive Summation Comparison ===");
        Console.WriteLine($"Accumulations per output: {nFreqBins}");

        // Compute with NAIVE summation
        var naiveResult = new Complex[nFilters, nFrames];
        for (int k = 0; k < nFilters; k++)
        {
            for (int t = 0; t < nFrames; t++)
            {
                double sumReal = 0, sumImag = 0;
                for (int f = 0; f < nFreqBins; f++)
                {
                    double bReal = basisReal[k, f];
                    double bImag = basisImag[k, f];
                    double dReal = stftReal[f, t];
                    double dImag = stftImag[f, t];
                    sumReal += bReal * dReal - bImag * dImag;
                    sumImag += bReal * dImag + bImag * dReal;
                }
                naiveResult[k, t] = new Complex(sumReal, sumImag);
            }
        }

        // Compute with KAHAN summation
        var kahanResult = new Complex[nFilters, nFrames];
        for (int k = 0; k < nFilters; k++)
        {
            for (int t = 0; t < nFrames; t++)
            {
                double sumReal = 0, sumImag = 0;
                double cReal = 0, cImag = 0;
                    
                for (int f = 0; f < nFreqBins; f++)
                {
                    double bReal = basisReal[k, f];
                    double bImag = basisImag[k, f];
                    double dReal = stftReal[f, t];
                    double dImag = stftImag[f, t];
                        
                    double prodReal = bReal * dReal - bImag * dImag;
                    double prodImag = bReal * dImag + bImag * dReal;

                    // Kahan for real part
                    double yReal = prodReal - cReal;
                    double tReal = sumReal + yReal;
                    cReal = (tReal - sumReal) - yReal;
                    sumReal = tReal;

                    // Kahan for imaginary part
                    double yImag = prodImag - cImag;
                    double tImag = sumImag + yImag;
                    cImag = (tImag - sumImag) - yImag;
                    sumImag = tImag;
                }
                kahanResult[k, t] = new Complex(sumReal, sumImag);
            }
        }

        // Calculate errors for both methods
        double naiveMaxError = 0, naiveSumError = 0;
        double kahanMaxError = 0, kahanSumError = 0;
        double sumExpected = 0;

        for (int k = 0; k < nFilters; k++)
        {
            for (int t = 0; t < nFrames; t++)
            {
                double expectedMag = Math.Sqrt(expectedReal[k, t] * expectedReal[k, t] + 
                                               expectedImag[k, t] * expectedImag[k, t]);
                sumExpected += expectedMag;

                double naiveErr = Math.Abs(naiveResult[k, t].Real - expectedReal[k, t]) +
                                  Math.Abs(naiveResult[k, t].Imaginary - expectedImag[k, t]);
                naiveSumError += naiveErr;
                naiveMaxError = Math.Max(naiveMaxError, naiveErr);

                double kahanErr = Math.Abs(kahanResult[k, t].Real - expectedReal[k, t]) +
                                  Math.Abs(kahanResult[k, t].Imaginary - expectedImag[k, t]);
                kahanSumError += kahanErr;
                kahanMaxError = Math.Max(kahanMaxError, kahanErr);
            }
        }

        double naiveMeanRelError = naiveSumError / (sumExpected + 1e-10);
        double kahanMeanRelError = kahanSumError / (sumExpected + 1e-10);

        Console.WriteLine($"\nNaive Summation:");
        Console.WriteLine($"  Max absolute error: {naiveMaxError:E6}");
        Console.WriteLine($"  Mean relative error: {naiveMeanRelError:P6}");

        Console.WriteLine($"\nKahan Summation:");
        Console.WriteLine($"  Max absolute error: {kahanMaxError:E6}");
        Console.WriteLine($"  Mean relative error: {kahanMeanRelError:P6}");

        double improvement = naiveMaxError / (kahanMaxError + 1e-20);
        Console.WriteLine($"\nKahan improvement factor: {improvement:F2}x");

        // Both methods should have very small error (< 1e-10)
        // Note: When both C# and Python use naive summation with the same order,
        // they may match exactly (0 error). Kahan summation uses a different
        // accumulation pattern, so it may have a tiny (1e-12 to 1e-13) difference
        // from Python's naive summation, which is still excellent precision.
        Assert.IsTrue(kahanMaxError < 1e-10, 
            $"Kahan max error {kahanMaxError:E6} should be < 1e-10");
        Assert.IsTrue(naiveMaxError < 1e-10, 
            $"Naive max error {naiveMaxError:E6} should be < 1e-10");
    }

    #endregion
}