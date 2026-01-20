using System.Numerics;
using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;

namespace AiMusicDetector;

/// <summary>
/// Configuration for fakeprint extraction.
/// </summary>
public class FakeprintConfig
{
    /// <summary>Sample rate in Hz (default: 44100)</summary>
    public int SampleRate { get; init; } = 44100;
    
    /// <summary>FFT size (default: 16384 = 2^14)</summary>
    public int NfftSize { get; init; } = 16384;
    
    /// <summary>Minimum frequency for analysis in Hz (default: 5000)</summary>
    public int FreqMin { get; init; } = 5000;
    
    /// <summary>Maximum frequency for analysis in Hz (default: 16000)</summary>
    public int FreqMax { get; init; } = 16000;
    
    /// <summary>Window size for lower hull computation (default: 10)</summary>
    public int HullArea { get; init; } = 10;
    
    /// <summary>Maximum dB for normalization (default: 5)</summary>
    public float MaxDb { get; init; } = 5.0f;
    
    /// <summary>Minimum dB threshold (default: -45)</summary>
    public float MinDb { get; init; } = -45.0f;
}

/// <summary>
/// Extracts fakeprint features from audio for AI music detection.
/// 
/// Fakeprints capture spectral artifacts from deconvolution modules
/// in AI music generators, as described in:
/// "A Fourier Explanation of AI-Music Artifacts" (ISMIR 2025)
/// </summary>
public class FakeprintExtractor
{
    private readonly FakeprintConfig _config;
    private readonly float[] _window;
    private readonly float[] _freqBins;
    private readonly int _freqMinIdx;
    private readonly int _freqMaxIdx;
    private readonly int _outputSize;

    /// <summary>
    /// Creates a new FakeprintExtractor instance.
    /// </summary>
    /// <param name="config">Configuration options, or null for defaults</param>
    public FakeprintExtractor(FakeprintConfig? config = null)
    {
        _config = config ?? new FakeprintConfig();
        
        // Create Hann window
        _window = CreateHannWindow(_config.NfftSize);
        
        // Precompute frequency bins
        int numBins = _config.NfftSize / 2 + 1;
        _freqBins = new float[numBins];
        float freqStep = (float)_config.SampleRate / 2 / (numBins - 1);
        
        for (int i = 0; i < numBins; i++)
        {
            _freqBins[i] = i * freqStep;
        }
        
        // Find frequency range indices
        _freqMinIdx = Array.FindIndex(_freqBins, f => f >= _config.FreqMin);
        _freqMaxIdx = Array.FindLastIndex(_freqBins, f => f <= _config.FreqMax);
        
        if (_freqMinIdx < 0) _freqMinIdx = 0;
        if (_freqMaxIdx < 0) _freqMaxIdx = numBins - 1;
        
        _outputSize = _freqMaxIdx - _freqMinIdx + 1;
    }

    /// <summary>
    /// Gets the output feature dimension.
    /// </summary>
    public int OutputSize => _outputSize;

    /// <summary>
    /// Extract fakeprint from audio samples.
    /// </summary>
    /// <param name="samples">Audio samples (mono, any sample rate - will be processed as-is)</param>
    /// <returns>Normalized fakeprint feature vector</returns>
    public float[] Extract(float[] samples)
    {
        // Compute power spectrum using STFT
        var meanSpectrum = ComputeMeanPowerSpectrum(samples);
        
        // Convert to dB
        var spectrumDb = new float[meanSpectrum.Length];
        for (int i = 0; i < meanSpectrum.Length; i++)
        {
            spectrumDb[i] = 10 * MathF.Log10(Math.Max(meanSpectrum[i], 1e-10f));
        }
        
        // Extract frequency range
        var rangeSpectrum = new float[_outputSize];
        Array.Copy(spectrumDb, _freqMinIdx, rangeSpectrum, 0, _outputSize);
        
        // Compute lower hull
        var hull = ComputeLowerHull(rangeSpectrum);
        
        // Interpolate hull to full resolution
        var hullInterp = InterpolateHull(hull, _outputSize);
        
        // Clip to minimum dB
        for (int i = 0; i < hullInterp.Length; i++)
        {
            hullInterp[i] = Math.Max(hullInterp[i], _config.MinDb);
        }
        
        // Compute residue (spectrum - hull)
        var residue = new float[_outputSize];
        for (int i = 0; i < _outputSize; i++)
        {
            residue[i] = Math.Max(rangeSpectrum[i] - hullInterp[i], 0);
        }
        
        // Clip and normalize
        float maxVal = 0;
        for (int i = 0; i < residue.Length; i++)
        {
            residue[i] = Math.Min(residue[i], _config.MaxDb);
            maxVal = Math.Max(maxVal, residue[i]);
        }
        
        maxVal = Math.Max(maxVal, 1e-6f);
        for (int i = 0; i < residue.Length; i++)
        {
            residue[i] /= maxVal;
        }
        
        return residue;
    }

    private float[] ComputeMeanPowerSpectrum(float[] samples)
    {
        int nfft = _config.NfftSize;
        int hopSize = nfft / 2; // 50% overlap
        int numFrames = Math.Max(1, (samples.Length - nfft) / hopSize + 1);
        int numBins = nfft / 2 + 1;
        
        var meanSpectrum = new double[numBins];
        
        for (int frame = 0; frame < numFrames; frame++)
        {
            int start = frame * hopSize;
            
            // Apply window and prepare FFT input
            var fftBuffer = new Complex[nfft];
            for (int i = 0; i < nfft; i++)
            {
                int sampleIdx = start + i;
                float sample = sampleIdx < samples.Length ? samples[sampleIdx] : 0;
                fftBuffer[i] = new Complex(sample * _window[i], 0);
            }
            
            // Compute FFT
            Fourier.Forward(fftBuffer, FourierOptions.Matlab);
            
            // Accumulate power spectrum (only positive frequencies)
            for (int i = 0; i < numBins; i++)
            {
                double power = fftBuffer[i].MagnitudeSquared();
                meanSpectrum[i] += power;
            }
        }
        
        // Average
        var result = new float[numBins];
        for (int i = 0; i < numBins; i++)
        {
            result[i] = (float)(meanSpectrum[i] / numFrames);
        }
        
        return result;
    }

    private (int[] indices, float[] values) ComputeLowerHull(float[] spectrum)
    {
        int area = _config.HullArea;
        var indices = new List<int>();
        var values = new List<float>();
        
        for (int i = 0; i <= spectrum.Length - area; i++)
        {
            // Find minimum in window
            int minIdx = i;
            float minVal = spectrum[i];
            
            for (int j = i + 1; j < i + area; j++)
            {
                if (spectrum[j] < minVal)
                {
                    minVal = spectrum[j];
                    minIdx = j;
                }
            }
            
            if (!indices.Contains(minIdx))
            {
                indices.Add(minIdx);
                values.Add(minVal);
            }
        }
        
        // Ensure endpoints are included
        if (indices.Count == 0 || indices[0] != 0)
        {
            indices.Insert(0, 0);
            values.Insert(0, spectrum[0]);
        }
        
        if (indices[^1] != spectrum.Length - 1)
        {
            indices.Add(spectrum.Length - 1);
            values.Add(spectrum[^1]);
        }
        
        return (indices.ToArray(), values.ToArray());
    }

    private float[] InterpolateHull((int[] indices, float[] values) hull, int length)
    {
        var result = new float[length];
        var (indices, values) = hull;
        
        // Quadratic interpolation
        int hullIdx = 0;
        for (int i = 0; i < length; i++)
        {
            // Find surrounding hull points
            while (hullIdx < indices.Length - 1 && indices[hullIdx + 1] < i)
            {
                hullIdx++;
            }
            
            if (i <= indices[0])
            {
                result[i] = values[0];
            }
            else if (i >= indices[^1])
            {
                result[i] = values[^1];
            }
            else
            {
                // Linear interpolation between hull points
                int idx1 = Math.Max(0, hullIdx);
                int idx2 = Math.Min(indices.Length - 1, hullIdx + 1);
                
                if (idx1 == idx2 || indices[idx1] == indices[idx2])
                {
                    result[i] = values[idx1];
                }
                else
                {
                    float t = (float)(i - indices[idx1]) / (indices[idx2] - indices[idx1]);
                    result[i] = values[idx1] + t * (values[idx2] - values[idx1]);
                }
            }
        }
        
        return result;
    }

    private static float[] CreateHannWindow(int size)
    {
        var window = new float[size];
        for (int i = 0; i < size; i++)
        {
            window[i] = 0.5f * (1 - MathF.Cos(2 * MathF.PI * i / (size - 1)));
        }
        return window;
    }
}
