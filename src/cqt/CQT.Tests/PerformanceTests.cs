using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Diagnostics;

namespace CQT.Tests;

[TestClass]
public class PerformanceTests
{
    [TestMethod]
    public void FFT_Performance_WithPrecompute()
    {
        int size = 4096;
        var processor = new FFTProcessor(size);
        var data = new Complex[size];
        var random = new Random(42);

        // Warm up
        for (int i = 0; i < size; i++)
            data[i] = new Complex(random.NextDouble(), 0);
        processor.Forward(data);

        // Benchmark
        int iterations = 1000;
        var sw = Stopwatch.StartNew();
            
        for (int iter = 0; iter < iterations; iter++)
        {
            for (int i = 0; i < size; i++)
                data[i] = new Complex(random.NextDouble(), 0);
            processor.Forward(data);
        }
            
        sw.Stop();
        double msPerFFT = sw.Elapsed.TotalMilliseconds / iterations;
            
        Console.WriteLine($"FFT size {size}: {msPerFFT:F3} ms per transform");
        Assert.IsTrue(msPerFFT < 5, $"FFT too slow: {msPerFFT} ms");
    }

    [TestMethod]
    public void CQT_Performance_SingleFrame()
    {
        var cqt = new ConstantQTransform(
            sampleRate: 16000,
            fMin: 500,
            nBins: 48,
            binsPerOctave: 12,
            hopLength: 512
        );

        var frame = new double[cqt.FftSize];
        var random = new Random(42);
        for (int i = 0; i < frame.Length; i++)
            frame[i] = random.NextDouble() * 2 - 1;

        // Warm up
        cqt.ComputeFrame(frame);

        // Benchmark
        int iterations = 1000;
        var sw = Stopwatch.StartNew();
            
        for (int iter = 0; iter < iterations; iter++)
        {
            cqt.ComputeFrame(frame);
        }
            
        sw.Stop();
        double msPerFrame = sw.Elapsed.TotalMilliseconds / iterations;
            
        Console.WriteLine($"CQT frame: {msPerFrame:F3} ms per frame");
        Assert.IsTrue(msPerFrame < 2, $"CQT frame too slow: {msPerFrame} ms");
    }

    [TestMethod]
    public void CQT_Performance_OneSecondAudio()
    {
        var cqt = new ConstantQTransform(
            sampleRate: 16000,
            fMin: 500,
            nBins: 48,
            binsPerOctave: 12,
            hopLength: 512
        );

        var audio = new double[16000]; // 1 second
        var random = new Random(42);
        for (int i = 0; i < audio.Length; i++)
            audio[i] = random.NextDouble() * 2 - 1;

        // Warm up
        cqt.Compute(audio);

        // Benchmark
        int iterations = 100;
        var sw = Stopwatch.StartNew();
            
        for (int iter = 0; iter < iterations; iter++)
        {
            cqt.Compute(audio);
        }
            
        sw.Stop();
        double msPerSecond = sw.Elapsed.TotalMilliseconds / iterations;
            
        Console.WriteLine($"CQT 1s audio: {msPerSecond:F1} ms");
        Assert.IsTrue(msPerSecond < 100, $"CQT 1s too slow: {msPerSecond} ms");
    }

    [TestMethod]
    public void Cepstrum_Performance_TenSecondAudio()
    {
        var extractor = new CepstrumExtractor(
            sampleRate: 16000,
            fMin: 500,
            nBins: 48,
            binsPerOctave: 12,
            hopLength: 512,
            nCoeffs: 24
        );

        var audio = new double[160000]; // 10 seconds
        var random = new Random(42);
        for (int i = 0; i < audio.Length; i++)
            audio[i] = random.NextDouble() * 2 - 1;

        // Warm up
        extractor.Extract(audio);

        // Benchmark
        int iterations = 20;
        var sw = Stopwatch.StartNew();
            
        for (int iter = 0; iter < iterations; iter++)
        {
            extractor.Extract(audio);
        }
            
        sw.Stop();
        double msPerExtract = sw.Elapsed.TotalMilliseconds / iterations;
            
        Console.WriteLine($"Cepstrum 10s audio: {msPerExtract:F1} ms");
        // LibrosaCQT uses recursive octave processing with Bluestein FFT for
        // arbitrary sizes. Parallelized matrix multiplication and STFT for speed.
        // Release mode target: <500ms for 10s audio (was ~620ms before optimization)
        Assert.IsTrue(msPerExtract < 1000, $"Cepstrum too slow: {msPerExtract} ms");
    }

    [TestMethod]
    public void CQT_SparsityStats()
    {
        var cqt = new ConstantQTransform(
            sampleRate: 16000,
            fMin: 500,
            nBins: 48,
            binsPerOctave: 12,
            hopLength: 512,
            sparsityThreshold: 0.01
        );

        var (total, sparse, ratio) = cqt.GetSparsityStats();
            
        Console.WriteLine($"Kernel sparsity: {ratio:P1} ({sparse}/{total} non-zero entries)");
        Assert.IsTrue(ratio > 0.5, $"Expected >50% sparsity, got {ratio:P1}");
    }

    [TestMethod]
    public void CQT_Performance_ThreeMinuteAudio()
    {
        // Parameters matching config.yaml for actual inference
        var cqt = new ConstantQTransform(
            sampleRate: 16000,
            fMin: 500,
            nBins: 48,
            binsPerOctave: 12,
            hopLength: 512
        );

        // 3 minutes of audio at 16000 Hz (matching config.yaml)
        var audio = new double[16000 * 180]; // 180 seconds = 3 minutes
        var random = new Random(42);
        for (int i = 0; i < audio.Length; i++)
            audio[i] = random.NextDouble() * 2 - 1;

        // Warm up
        cqt.Compute(audio);

        // Benchmark
        int iterations = 10;
        var sw = Stopwatch.StartNew();
            
        for (int iter = 0; iter < iterations; iter++)
        {
            cqt.Compute(audio);
        }
            
        sw.Stop();
        double msPerExtract = sw.Elapsed.TotalMilliseconds / iterations;
        double realTimeRatio = 180000.0 / msPerExtract; // 180 seconds in ms / processing time
            
        Console.WriteLine($"CQT 3min audio: {msPerExtract:F0} ms ({realTimeRatio:F1}x real-time)");
        // Target: <1000ms for 3-minute audio (180x real-time)
        Assert.IsTrue(msPerExtract < 1000, $"CQT too slow: {msPerExtract} ms (only {realTimeRatio:F1}x real-time)");
    }

    [TestMethod]
    public void DCT_Performance_Comparison()
    {
        int size = 48;
        var input = new double[size];
        var random = new Random(42);
        for (int i = 0; i < size; i++)
            input[i] = random.NextDouble();

        // Direct DCT
        int iterations = 10000;
        var sw = Stopwatch.StartNew();
        for (int iter = 0; iter < iterations; iter++)
        {
            DCT.ComputeType2Direct(input);
        }
        sw.Stop();
        double directMs = sw.Elapsed.TotalMilliseconds / iterations;

        // Pre-computed DCT
        var processor = new DCTProcessor(size);
        var output = new double[size];
        sw.Restart();
        for (int iter = 0; iter < iterations; iter++)
        {
            processor.Compute(input, output);
        }
        sw.Stop();
        double precomputedMs = sw.Elapsed.TotalMilliseconds / iterations;

        Console.WriteLine($"DCT direct: {directMs:F4} ms, pre-computed: {precomputedMs:F4} ms");
        Assert.IsTrue(precomputedMs < directMs, "Pre-computed should be faster");
    }
}