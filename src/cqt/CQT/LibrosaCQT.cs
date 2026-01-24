using System;
using System.Collections.Generic;

namespace CQT;

/// <summary>
/// Constant-Q Transform implementation matching librosa.cqt algorithm.
/// </summary>
public class LibrosaCQT
{
    private readonly double _filterScale;
    private readonly double _sparsity;
    private readonly bool _useFloat32Basis;
    private readonly bool _scale;

    private readonly int _nOctaves;
    private readonly double[] _alpha;      // Relative bandwidth
    private readonly double[] _lengths;   // Filter lengths at original sample rate

    public int SampleRate { get; }
    public double MinFrequency { get; }
    public int NumBins { get; }
    public int BinsPerOctave { get; }
    public int HopLength { get; }
    public double[] Frequencies { get; }

    public LibrosaCQT(
        int sampleRate,
        double fMin,
        int nBins,
        int binsPerOctave = 12,
        int hopLength = 512,
        double filterScale = 1.0,
        double sparsity = 0.01,
        bool scale = true,
        bool useFloat32Basis = true)
    {
        SampleRate = sampleRate;
        MinFrequency = fMin;
        NumBins = nBins;
        BinsPerOctave = binsPerOctave;
        HopLength = hopLength;
        _filterScale = filterScale;
        _sparsity = sparsity;
        _scale = scale;
        _useFloat32Basis = useFloat32Basis;
        
        _nOctaves = (int)Math.Ceiling((double)nBins / binsPerOctave);
        
        Frequencies = new double[nBins];
        for (int k = 0; k < nBins; k++)
        {
            Frequencies[k] = fMin * Math.Pow(2.0, (double)k / binsPerOctave);
        }
        
        _alpha = ComputeRelativeBandwidth(Frequencies, binsPerOctave);
        _lengths = new double[nBins];
        
        for (int k = 0; k < nBins; k++)
        {
            double Q = filterScale / _alpha[k];
            _lengths[k] = Q * sampleRate / Frequencies[k];
        }
    }
    
    private static double[] ComputeRelativeBandwidth(double[] frequencies, int binsPerOctave)
    {
        int n = frequencies.Length;
        double[] alpha = new double[n];
        double r = Math.Pow(2.0, 1.0 / binsPerOctave);
        double etAlpha = (r * r - 1) / (r * r + 1);

        for (int k = 0; k < n; k++)
        {
            alpha[k] = etAlpha;
        }

        return alpha;
    }

    /// <summary>
    /// Compute CQT magnitude spectrogram.
    /// </summary>
    public double[,] Compute(double[] audio)
    {
        Complex[,] complex = ComputeComplex(audio);
        int nBins = complex.GetLength(0);
        int nFrames = complex.GetLength(1);

        double[,] magnitude = new double[nBins, nFrames];
        for (int k = 0; k < nBins; k++)
        {
            for (int t = 0; t < nFrames; t++)
            {
                magnitude[k, t] = complex[k, t].Magnitude;
            }
        }

        return magnitude;
    }

    /// <summary>
    /// Compute CQT returning complex values.
    /// </summary>
    public Complex[,] ComputeComplex(double[] audio)
    {
        List<Complex[,]> cqtResponses = [];
        // No need to clone - Decimate2 always returns a new array
        // We use the original for the first octave, then decimated versions
        double[] myAudio = audio;
        double mySr = SampleRate;
        int myHop = HopLength;
        
        for (int octave = 0; octave < _nOctaves; octave++)
        {
            int nFilters;
            int startBin;

            if (octave == 0)
            {
                nFilters = Math.Min(BinsPerOctave, NumBins);
                startBin = NumBins - nFilters;
            }
            else
            {
                nFilters = Math.Min(BinsPerOctave, NumBins - BinsPerOctave * octave);
                if (nFilters <= 0) break;
                startBin = NumBins - BinsPerOctave * (octave + 1);
                if (startBin < 0) startBin = 0;
            }
            
            double[] octaveFreqs = new double[nFilters];
            double[] octaveAlpha = new double[nFilters];
            for (int i = 0; i < nFilters; i++)
            {
                octaveFreqs[i] = Frequencies[startBin + i];
                octaveAlpha[i] = _alpha[startBin + i];
            }
            
            (Complex[,] fftBasis, int nFft, int[] lengths) = BuildFilterBasis(mySr, octaveFreqs, octaveAlpha, myHop);

            double scaleFactor = Math.Sqrt(SampleRate / mySr);
            for (int r = 0; r < fftBasis.GetLength(0); r++)
            {
                for (int c = 0; c < fftBasis.GetLength(1); c++)
                {
                    fftBasis[r, c] = new Complex(
                        fftBasis[r, c].Real * scaleFactor,
                        fftBasis[r, c].Imaginary * scaleFactor);
                }
            }
            
            Complex[,] octaveResponse = ComputeCQTResponse(myAudio, nFft, myHop, fftBasis);
            cqtResponses.Add(octaveResponse);
            
            if (octave < _nOctaves - 1 && myHop % 2 == 0)
            {
                myHop /= 2;
                mySr /= 2.0;
                myAudio = Decimate2(myAudio);
            }
        }
        
        Complex[,] result = TrimStack(cqtResponses);
        
        if (_scale)
        {
            int nFrames = result.GetLength(1);
            for (int k = 0; k < NumBins; k++)
            {
                double scale = 1.0 / Math.Sqrt(_lengths[k]);
                for (int t = 0; t < nFrames; t++)
                {
                    result[k, t] = new Complex(
                        result[k, t].Real * scale,
                        result[k, t].Imaginary * scale);
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Build FFT filter basis for a set of frequencies.
    /// </summary>
    private (Complex[,] basis, int nFft, int[] lengths) BuildFilterBasis(
        double sr, double[] freqs, double[] alpha, int hopLength)
    {
        int nFilters = freqs.Length;
        int[] lengths = new int[nFilters];
        double[] lengthsFloat = new double[nFilters];
        int maxLength = 0;
        
        for (int k = 0; k < nFilters; k++)
        {
            double Q = _filterScale / alpha[k];
            lengthsFloat[k] = Q * sr / freqs[k];
            lengths[k] = (int)Math.Floor(lengthsFloat[k] / 2) + (int)Math.Ceiling(lengthsFloat[k] / 2);
            
            if (lengths[k] > maxLength)
                maxLength = lengths[k];
        }
        
        int baseFft = FFT.NextPowerOf2(maxLength);
        int nFft = baseFft;
        
        Complex[,] basis = new Complex[nFilters, nFft / 2 + 1];
        
        // Don't use ArrayPool for tempKernel - FFT.Forward uses array.Length
        // and ArrayPool may return larger arrays, causing incorrect FFT sizes
        Complex[] baseKernel = new Complex[baseFft];
        Complex[] tempKernel = new Complex[nFft];

        for (int k = 0; k < nFilters; k++)
        {
            int Nk = lengths[k];
            double Fk = freqs[k];
            double ilenFloat = lengthsFloat[k];
            double[] window = WindowFunctions.Hann(Nk);
            int startIdx = (int)Math.Floor(-ilenFloat / 2);
                
            Complex[] wavelet = new Complex[Nk];
            double coef = 2.0 * Math.PI * Fk / sr;
                
            for (int n = 0; n < Nk; n++)
            {
                int tIdx = startIdx + n;
                double phase = coef * tIdx;
                wavelet[n] = new Complex(
                    window[n] * Math.Cos(phase),
                    window[n] * Math.Sin(phase));
            }
            
            double norm = 0;
            
            for (int i = 0; i < Nk; i++)
            {
                norm += wavelet[i].Magnitude;
            }
            
            if (norm > 0)
            {
                for (int i = 0; i < Nk; i++)
                {
                    wavelet[i] = new Complex(
                        wavelet[i].Real / norm,
                        wavelet[i].Imaginary / norm);
                }
            }
            
            Array.Clear(baseKernel, 0, baseFft);
            int padStart = (baseFft - Nk) / 2;
            for (int i = 0; i < Nk; i++)
            {
                baseKernel[padStart + i] = wavelet[i];
            }
            
            if (nFft == baseFft)
            {
                Array.Copy(baseKernel, tempKernel, nFft);
            }
            else
            {
                Array.Clear(tempKernel, 0, nFft);
                Array.Copy(baseKernel, tempKernel, baseFft);
            }
            
            double fftNorm = lengthsFloat[k] / nFft;
            for (int i = 0; i < nFft; i++)
            {
                tempKernel[i] = new Complex(
                    tempKernel[i].Real * fftNorm,
                    tempKernel[i].Imaginary * fftNorm);
            }

            if (_useFloat32Basis)
            {
                QuantizeRowToFloat32(tempKernel, nFft);
            }
            
            FFT.Forward(tempKernel);

            if (_useFloat32Basis)
            {
                QuantizeRowToFloat32(tempKernel, nFft);
            }
            
            if (_sparsity > 0)
            {
                SparsifyRow(tempKernel, nFft / 2 + 1, _sparsity);
            }
            
            for (int i = 0; i <= nFft / 2; i++)
            {
                basis[k, i] = tempKernel[i];
            }
        }

        return (basis, nFft, lengths);
    }

    private static void SparsifyRow(Complex[] row, int length, double quantile)
    {
        switch (quantile)
        {
            case <= 0:
                return;
            case >= 1:
            {
                for (int i = 0; i < length; i++)
                    row[i] = Complex.Zero;
                return;
            }
        }

        double[] magnitudes = new double[length];
        double norm = 0;
        for (int i = 0; i < length; i++)
        {
            double mag = row[i].Magnitude;
            magnitudes[i] = mag;
            norm += mag;
        }

        if (norm <= 0)
            return;

        double[] sorted = (double[])magnitudes.Clone();
        Array.Sort(sorted);

        double cumulative = 0;
        int thresholdIdx = 0;
        for (int i = 0; i < length; i++)
        {
            cumulative += sorted[i] / norm;
            if (cumulative >= quantile)
            {
                thresholdIdx = i;
                break;
            }
        }

        double threshold = sorted[thresholdIdx];
        for (int i = 0; i < length; i++)
        {
            if (magnitudes[i] < threshold)
                row[i] = Complex.Zero;
        }
    }

    private static void QuantizeRowToFloat32(Complex[] row, int length)
    {
        for (int i = 0; i < length; i++)
        {
            row[i] = new Complex((float)row[i].Real, (float)row[i].Imaginary);
        }
    }
    
    private Complex[,] ComputeCQTResponse(double[] audio, int nFft, int hopLength, Complex[,] fftBasis)
    {
        int nFilters = fftBasis.GetLength(0);
        int nFreqBins = fftBasis.GetLength(1);

        STFT stft = new STFT(nFft, hopLength, WindowType.Rectangular);
        Complex[,] D = stft.Compute(audio);

        int nFrames = D.GetLength(1);
        Complex[,] result = new Complex[nFilters, nFrames];
        
        for (int k = 0; k < nFilters; k++)
        {
            for (int t = 0; t < nFrames; t++)
            {
                double sumReal = 0;
                double sumImag = 0;

                for (int f = 0; f < nFreqBins; f++)
                {
                    double bReal = fftBasis[k, f].Real;
                    double bImag = fftBasis[k, f].Imaginary;
                    double dReal = D[f, t].Real;
                    double dImag = D[f, t].Imaginary;

                    sumReal += bReal * dReal - bImag * dImag;
                    sumImag += bReal * dImag + bImag * dReal;
                }
                result[k, t] = new Complex(sumReal, sumImag);
            }
        }

        return result;
    }
    
    private static double[] Decimate2(double[] input)
    {
        return SoxrDecimate.Decimate2(input, scale: true);
    }
    
    private Complex[,] TrimStack(List<Complex[,]> responses)
    {
        if (responses.Count == 0)
            return new Complex[0, 0];
        
        int minFrames = int.MaxValue;
        foreach (Complex[,] resp in responses)
        {
            if (resp.GetLength(1) < minFrames)
                minFrames = resp.GetLength(1);
        }

        Complex[,] result = new Complex[NumBins, minFrames];
        
        int endBin = NumBins;
        foreach (Complex[,] resp in responses)
        {
            int nOct = resp.GetLength(0);

            if (endBin < nOct)
            {
                for (int k = 0; k < endBin; k++)
                {
                    int srcK = nOct - endBin + k;
                    for (int t = 0; t < minFrames; t++)
                    {
                        result[k, t] = resp[srcK, t];
                    }
                }
            }
            else
            {
                int startBin = endBin - nOct;
                for (int k = 0; k < nOct; k++)
                {
                    for (int t = 0; t < minFrames; t++)
                    {
                        result[startBin + k, t] = resp[k, t];
                    }
                }
            }

            endBin -= nOct;
        }

        return result;
    }
}