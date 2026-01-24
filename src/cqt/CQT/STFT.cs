using System;

namespace CQT;

public enum WindowType
{
    Hann,
    HannSymmetric,
    Hamming,
    Rectangular
}

/// <summary>
/// Short-Time Fourier Transform.
/// </summary>
public class STFT
{
    private readonly double[] _window;
    private readonly FFTProcessor _fftProcessor;
    private readonly Complex[] _fftBuffer;

    public int FftSize { get; }

    public int HopLength { get; }

    public STFT(int nFft, int hopLength, WindowType windowType = WindowType.Hann)
    {
        FftSize = nFft;
        HopLength = hopLength;
        _fftProcessor = new FFTProcessor(nFft);
        _fftBuffer = new Complex[nFft];

        _window = windowType switch
        {
            WindowType.Hann => WindowFunctions.Hann(nFft),
            WindowType.HannSymmetric => WindowFunctions.HannSymmetric(nFft),
            WindowType.Hamming => WindowFunctions.Hamming(nFft),
            WindowType.Rectangular => WindowFunctions.Rectangular(nFft),
            _ => WindowFunctions.Hann(nFft)
        };
    }

    /// <summary>
    /// Compute STFT of audio signal.
    /// Returns complex spectrogram [n_fft/2+1, n_frames].
    /// </summary>
    public Complex[,] Compute(double[] audio, string padMode = "constant")
    {
        // Pad the signal (center padding like librosa)
        int padLength = FftSize / 2;
        double[] padded = new double[audio.Length + 2 * padLength];
            
        // Zero padding (constant mode)
        Array.Copy(audio, 0, padded, padLength, audio.Length);

        // Calculate number of frames
        int nFrames = 1 + (padded.Length - FftSize) / HopLength;
        int nFreqs = FftSize / 2 + 1;

        Complex[,] result = new Complex[nFreqs, nFrames];

        for (int t = 0; t < nFrames; t++)
        {
            int start = t * HopLength;

            // Window the frame
            for (int i = 0; i < FftSize; i++)
            {
                int idx = start + i;
                if (idx < padded.Length)
                    _fftBuffer[i] = new Complex(padded[idx] * _window[i], 0);
                else
                    _fftBuffer[i] = Complex.Zero;
            }

            // FFT
            _fftProcessor.Forward(_fftBuffer);

            // Store only positive frequencies
            for (int f = 0; f < nFreqs; f++)
            {
                result[f, t] = _fftBuffer[f];
            }
        }

        return result;
    }

    /// <summary>
    /// Compute STFT and return magnitude only (for pseudo-CQT).
    /// </summary>
    public double[,] ComputeMagnitude(double[] audio, string padMode = "constant")
    {
        Complex[,] complex = Compute(audio, padMode);
        int nFreqs = complex.GetLength(0);
        int nFrames = complex.GetLength(1);

        double[,] magnitude = new double[nFreqs, nFrames];
        for (int f = 0; f < nFreqs; f++)
        {
            for (int t = 0; t < nFrames; t++)
            {
                magnitude[f, t] = complex[f, t].Magnitude;
            }
        }

        return magnitude;
    }
}