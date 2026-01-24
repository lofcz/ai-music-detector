using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace CQT;

/// <summary>
/// Constant-Q Transform.
/// </summary>
public class ConstantQTransform
{
    private readonly int[] _windowLengths;
    private readonly Complex[] _fftBuffer;
    private readonly FFTProcessor _fftProcessor;

    // Flattened sparse kernel for better cache locality
    private readonly int[] _kernelIndices;
    private readonly double[] _kernelReals;
    private readonly double[] _kernelImags;
    private readonly int[] _kernelOffsets; // Start offset for each bin (length = NumBins + 1)

    public int SampleRate { get; }
    public double MinFrequency { get; }
    public int NumBins { get; }
    public int BinsPerOctave { get; }
    public int HopLength { get; }
    public int FftSize { get; }
    public double[] Frequencies { get; }

    public ConstantQTransform(
        int sampleRate,
        double fMin,
        int nBins,
        int binsPerOctave = 12,
        int hopLength = 512,
        double qFactor = 1.0,
        double sparsityThreshold = 0.005)
    {
        SampleRate = sampleRate;
        MinFrequency = fMin;
        NumBins = nBins;
        BinsPerOctave = binsPerOctave;
        HopLength = hopLength;

        double Q = qFactor / (Math.Pow(2.0, 1.0 / binsPerOctave) - 1.0);
            
        Frequencies = new double[nBins];
        for (int k = 0; k < nBins; k++)
            Frequencies[k] = fMin * Math.Pow(2.0, (double)k / binsPerOctave);
            
        _windowLengths = new int[nBins];
        int maxWindowLength = 0;
        for (int k = 0; k < nBins; k++)
        {
            _windowLengths[k] = (int)Math.Ceiling(Q * sampleRate / Frequencies[k]);
            if (_windowLengths[k] > maxWindowLength)
                maxWindowLength = _windowLengths[k];
        }

        FftSize = FFT.NextPowerOf2(maxWindowLength);

        _fftProcessor = new FFTProcessor(FftSize);
        _fftBuffer = new Complex[FftSize];
        BuildSparseKernel(Q, sparsityThreshold, out _kernelIndices, out _kernelReals, out _kernelImags, out _kernelOffsets);
    }

    private void BuildSparseKernel(double Q, double threshold, 
        out int[] indices, out double[] reals, out double[] imags, out int[] offsets)
    {
        Complex[] tempKernel = new Complex[FftSize];
        List<int> allIndices = new(NumBins * 64);
        List<double> allReals = new(NumBins * 64);
        List<double> allImags = new(NumBins * 64);
        offsets = new int[NumBins + 1];

        for (int k = 0; k < NumBins; k++)
        {
            offsets[k] = allIndices.Count;
            
            int Nk = _windowLengths[k];
            double Fk = Frequencies[k];
            double[] window = WindowFunctions.HannSymmetric(Nk);
            Array.Clear(tempKernel, 0, FftSize);
                
            int start = (FftSize - Nk) / 2;
            double coef = 2.0 * Math.PI * Fk / SampleRate;

            for (int n = 0; n < Nk; n++)
            {
                double phase = coef * n;
                double amplitude = window[n] / Nk;
                tempKernel[start + n] = new Complex(
                    amplitude * Math.Cos(phase),
                    amplitude * Math.Sin(phase)
                );
            }

            FFT.Forward(tempKernel);
                
            double maxMag = 0;
            double normFactor = 1.0 / FftSize;
            for (int i = 0; i < FftSize; i++)
            {
                double mag = tempKernel[i].Magnitude * normFactor;
                if (mag > maxMag) maxMag = mag;
            }
                
            double absThreshold = threshold * maxMag;
                
            for (int i = 0; i < FftSize; i++)
            {
                double real = tempKernel[i].Real * normFactor;
                double imag = -tempKernel[i].Imaginary * normFactor;
                double mag = Math.Sqrt(real * real + imag * imag);
                    
                if (mag > absThreshold)
                {
                    allIndices.Add(i);
                    allReals.Add(real);
                    allImags.Add(imag);
                }
            }
        }

        offsets[NumBins] = allIndices.Count;
        indices = allIndices.ToArray();
        reals = allReals.ToArray();
        imags = allImags.ToArray();
    }

    /// <summary>
    /// Compute CQT for a single frame (reuses internal buffer).
    /// </summary>
    public Complex[] ComputeFrame(double[] frame)
    {
        int copyLength = Math.Min(frame.Length, FftSize);
        for (int i = 0; i < copyLength; i++)
            _fftBuffer[i] = new Complex(frame[i], 0);
        for (int i = copyLength; i < FftSize; i++)
            _fftBuffer[i] = Complex.Zero;

        _fftProcessor.Forward(_fftBuffer);
        Complex[] result = new Complex[NumBins];
            
        for (int k = 0; k < NumBins; k++)
        {
            double sumReal = 0, sumImag = 0;
            int start = _kernelOffsets[k];
            int end = _kernelOffsets[k + 1];
                
            for (int e = start; e < end; e++)
            {
                int idx = _kernelIndices[e];
                double kReal = _kernelReals[e];
                double kImag = _kernelImags[e];
                double xReal = _fftBuffer[idx].Real;
                double xImag = _fftBuffer[idx].Imaginary;

                sumReal += kReal * xReal - kImag * xImag;
                sumImag += kReal * xImag + kImag * xReal;
            }
                
            result[k] = new Complex(sumReal, sumImag);
        }

        return result;
    }

    /// <summary>
    /// Compute CQT for a single frame into pre-allocated output.
    /// </summary>
    public void ComputeFrame(double[] frame, Complex[] output)
    {
        if (output.Length != NumBins)
            throw new ArgumentException($"Output must have {NumBins} elements");

        int copyLength = Math.Min(frame.Length, FftSize);
            
        for (int i = 0; i < copyLength; i++)
            _fftBuffer[i] = new Complex(frame[i], 0);
            
        for (int i = copyLength; i < FftSize; i++)
            _fftBuffer[i] = Complex.Zero;
            
        _fftProcessor.Forward(_fftBuffer);

        for (int k = 0; k < NumBins; k++)
        {
            double sumReal = 0, sumImag = 0;
            int start = _kernelOffsets[k];
            int end = _kernelOffsets[k + 1];
                
            for (int e = start; e < end; e++)
            {
                int idx = _kernelIndices[e];
                double kReal = _kernelReals[e];
                double kImag = _kernelImags[e];
                double xReal = _fftBuffer[idx].Real;
                double xImag = _fftBuffer[idx].Imaginary;
                    
                sumReal += kReal * xReal - kImag * xImag;
                sumImag += kReal * xImag + kImag * xReal;
            }
                
            output[k] = new Complex(sumReal, sumImag);
        }
    }

    /// <summary>
    /// Compute Short-Time CQT for an entire audio signal (parallelized).
    /// </summary>
    public double[,] Compute(double[] audio)
    {
        int nFrames = Math.Max(1, audio.Length / HopLength);
        double[,] result = new double[NumBins, nFrames];
        int audioLength = audio.Length;
        int fftSize = FftSize;
        int numBins = NumBins;
        int[] kernelIndices = _kernelIndices;
        double[] kernelReals = _kernelReals;
        double[] kernelImags = _kernelImags;
        int[] kernelOffsets = _kernelOffsets;

        // Parallel processing with thread-local FFT buffers
        Parallel.For(0, nFrames,
            () => new Complex[fftSize],
            (t, loopState, fftBuffer) =>
            {
                int start = t * HopLength;
                int copyLen = Math.Min(fftSize, audioLength - start);

                // Load frame into FFT buffer
                for (int i = 0; i < copyLen; i++)
                    fftBuffer[i] = new Complex(audio[start + i], 0);
                for (int i = copyLen; i < fftSize; i++)
                    fftBuffer[i] = Complex.Zero;

                // FFT (uses static cached plans - thread-safe)
                FFT.Forward(fftBuffer);

                // Sparse kernel multiply with flattened arrays
                for (int k = 0; k < numBins; k++)
                {
                    double sumReal = 0, sumImag = 0;
                    int kStart = kernelOffsets[k];
                    int kEnd = kernelOffsets[k + 1];

                    for (int e = kStart; e < kEnd; e++)
                    {
                        int idx = kernelIndices[e];
                        double kReal = kernelReals[e];
                        double kImag = kernelImags[e];
                        double xReal = fftBuffer[idx].Real;
                        double xImag = fftBuffer[idx].Imaginary;

                        sumReal += kReal * xReal - kImag * xImag;
                        sumImag += kReal * xImag + kImag * xReal;
                    }

                    result[k, t] = Math.Sqrt(sumReal * sumReal + sumImag * sumImag);
                }

                return fftBuffer;
            },
            _ => { }
        );

        return result;
    }

    /// <summary>
    /// Compute STCQT returning complex values (parallelized).
    /// </summary>
    public Complex[,] ComputeComplex(double[] audio)
    {
        int nFrames = Math.Max(1, audio.Length / HopLength);
        Complex[,] result = new Complex[NumBins, nFrames];
        int audioLength = audio.Length;
        int fftSize = FftSize;
        int numBins = NumBins;
        int[] kernelIndices = _kernelIndices;
        double[] kernelReals = _kernelReals;
        double[] kernelImags = _kernelImags;
        int[] kernelOffsets = _kernelOffsets;

        // Parallel processing with thread-local FFT buffers
        Parallel.For(0, nFrames,
            () => new Complex[fftSize],
            (t, loopState, fftBuffer) =>
            {
                int start = t * HopLength;
                int copyLen = Math.Min(fftSize, audioLength - start);

                // Load frame into FFT buffer
                for (int i = 0; i < copyLen; i++)
                    fftBuffer[i] = new Complex(audio[start + i], 0);
                for (int i = copyLen; i < fftSize; i++)
                    fftBuffer[i] = Complex.Zero;

                // FFT
                FFT.Forward(fftBuffer);

                // Sparse kernel multiply with flattened arrays
                for (int k = 0; k < numBins; k++)
                {
                    double sumReal = 0, sumImag = 0;
                    int kStart = kernelOffsets[k];
                    int kEnd = kernelOffsets[k + 1];

                    for (int e = kStart; e < kEnd; e++)
                    {
                        int idx = kernelIndices[e];
                        double kReal = kernelReals[e];
                        double kImag = kernelImags[e];
                        double xReal = fftBuffer[idx].Real;
                        double xImag = fftBuffer[idx].Imaginary;

                        sumReal += kReal * xReal - kImag * xImag;
                        sumImag += kReal * xImag + kImag * xReal;
                    }

                    result[k, t] = new Complex(sumReal, sumImag);
                }

                return fftBuffer;
            },
            _ => { }
        );

        return result;
    }

    /// <summary>
    /// Get sparsity statistics for the kernel.
    /// </summary>
    public (int totalEntries, int sparseEntries, double sparsityRatio) GetSparsityStats()
    {
        int total = NumBins * FftSize;
        int sparse = _kernelIndices.Length;
        return (total, sparse, 1.0 - (double)sparse / total);
    }
}