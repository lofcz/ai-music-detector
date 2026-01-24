namespace AiMusicDetector;

/// <summary>
/// Configuration for cepstrum extraction (CNN model features).
/// </summary>
public class CepstrumConfig
{
    /// <summary>Sample rate in Hz (default: 16000, matches CNN training)</summary>
    public int SampleRate { get; init; } = 16000;
    
    /// <summary>Minimum frequency for CQT in Hz (default: 500)</summary>
    public double FrequencyMin { get; init; } = 500.0;
    
    /// <summary>Number of CQT bins (default: 48 = 4 octaves)</summary>
    public int NumBins { get; init; } = 48;
    
    /// <summary>Bins per octave for CQT (default: 12 = semitone resolution)</summary>
    public int BinsPerOctave { get; init; } = 12;
    
    /// <summary>Hop length for CQT in samples (default: 512 = ~32ms at 16kHz)</summary>
    public int HopLength { get; init; } = 512;
    
    /// <summary>Number of cepstral coefficients to keep (default: 24)</summary>
    public int NumCoefficients { get; init; } = 24;
    
    /// <summary>Segment duration in seconds for feature extraction (default: 10.0)</summary>
    public double SegmentSeconds { get; init; } = 10.0;
    
    /// <summary>Seconds to skip at start of audio (default: 5.0)</summary>
    public double SkipStartSeconds { get; init; } = 5.0;
    
    /// <summary>Seconds to skip at end of audio (default: 5.0)</summary>
    public double SkipEndSeconds { get; init; } = 5.0;
}

/// <summary>
/// Extracts CQT-cepstrum features from audio for CNN-based AI music detection.
/// 
/// Uses the CQT library to compute Constant-Q Transform, then applies DCT
/// to get cepstral coefficients (similar to MFCCs but with CQT instead of mel filterbanks).
/// </summary>
public class CepstrumExtractor
{
    private readonly CepstrumConfig _config;
    private readonly CQT.CepstrumExtractor _cqtExtractor;

    /// <summary>
    /// Creates a new CepstrumExtractor with the specified configuration.
    /// </summary>
    /// <param name="config">Configuration options, or null for defaults</param>
    public CepstrumExtractor(CepstrumConfig? config = null)
    {
        _config = config ?? new CepstrumConfig();
        
        // Create the underlying CQT cepstrum extractor
        _cqtExtractor = new CQT.CepstrumExtractor(
            sampleRate: _config.SampleRate,
            fMin: _config.FrequencyMin,
            nBins: _config.NumBins,
            binsPerOctave: _config.BinsPerOctave,
            hopLength: _config.HopLength,
            nCoeffs: _config.NumCoefficients
        );
    }

    /// <summary>
    /// Gets the number of cepstral coefficients output.
    /// </summary>
    public int NumCoefficients => _cqtExtractor.NumCoefficients;

    /// <summary>
    /// Gets the expected sample rate for input audio.
    /// </summary>
    public int SampleRate => _config.SampleRate;

    /// <summary>
    /// Gets the segment length in samples.
    /// </summary>
    public int SegmentSamples => (int)(_config.SegmentSeconds * _config.SampleRate);

    /// <summary>
    /// Extract cepstrum features from audio samples.
    /// </summary>
    /// <param name="samples">Audio samples (mono, at SampleRate)</param>
    /// <returns>Cepstrum features [n_coeffs, n_frames]</returns>
    public float[,] Extract(float[] samples)
    {
        // Convert float[] to double[] for CQT library
        var doubleSamples = new double[samples.Length];
        for (int i = 0; i < samples.Length; i++)
        {
            doubleSamples[i] = samples[i];
        }

        // Extract cepstrum
        var result = _cqtExtractor.Extract(doubleSamples);

        // Convert double[,] to float[,]
        int nCoeffs = result.GetLength(0);
        int nFrames = result.GetLength(1);
        var floatResult = new float[nCoeffs, nFrames];
        
        for (int i = 0; i < nCoeffs; i++)
        {
            for (int j = 0; j < nFrames; j++)
            {
                floatResult[i, j] = (float)result[i, j];
            }
        }

        return floatResult;
    }

    /// <summary>
    /// Extract multiple segments from audio for CNN inference.
    /// Matches Python inference_cnn.py extract_segments() logic exactly.
    /// </summary>
    /// <param name="samples">Audio samples (mono, at SampleRate)</param>
    /// <param name="numSegments">Number of segments to extract (default: 5)</param>
    /// <returns>List of cepstrum features, one per segment</returns>
    public List<float[,]> ExtractSegments(float[] samples, int numSegments = 5)
    {
        int skipSamples = (int)(_config.SkipStartSeconds * _config.SampleRate); // 5 seconds
        int segmentSamples = SegmentSamples; // 10 seconds
        
        var segments = new List<float[,]>();
        
        int startOffset, endOffset;
        
        // Match Python: only skip intro/outro if audio is long enough (> 20 seconds)
        if (samples.Length > segmentSamples + 2 * skipSamples)
        {
            // Song is long enough - skip intro/outro
            startOffset = skipSamples;
            endOffset = samples.Length - skipSamples;
        }
        else
        {
            // Short song - use entire audio
            startOffset = 0;
            endOffset = samples.Length;
        }
        
        int usableLength = endOffset - startOffset;
        
        if (usableLength <= segmentSamples)
        {
            // Very short - pad or center
            if (samples.Length <= segmentSamples)
            {
                // Pad with zeros to exactly 10 seconds
                var padded = new float[segmentSamples];
                Array.Copy(samples, padded, samples.Length);
                segments.Add(Extract(padded));
            }
            else
            {
                // Use center of audio
                int center = samples.Length / 2;
                int start = Math.Max(0, center - segmentSamples / 2);
                var segment = new float[segmentSamples];
                Array.Copy(samples, start, segment, 0, segmentSamples);
                segments.Add(Extract(segment));
            }
        }
        else
        {
            // Spread segments across usable range
            int available = usableLength - segmentSamples;
            
            List<int> positions;
            if (numSegments == 1)
            {
                positions = new List<int> { startOffset + available / 2 };
            }
            else
            {
                // Use float division like Python: step = available / (n_segments - 1)
                double step = (double)available / (numSegments - 1);
                positions = Enumerable.Range(0, numSegments)
                    .Select(i => startOffset + (int)(i * step))
                    .ToList();
            }
            
            foreach (int start in positions)
            {
                var segment = new float[segmentSamples];
                Array.Copy(samples, start, segment, 0, segmentSamples);
                segments.Add(Extract(segment));
            }
        }
        
        return segments;
    }

    /// <summary>
    /// Extract a single segment from audio at the specified position.
    /// </summary>
    /// <param name="samples">Audio samples (mono, at SampleRate)</param>
    /// <param name="startSample">Start position in samples</param>
    /// <returns>Cepstrum features for the segment</returns>
    public float[,] ExtractSegment(float[] samples, int startSample)
    {
        int segmentLength = SegmentSamples;
        int actualLength = Math.Min(segmentLength, samples.Length - startSample);
        
        if (actualLength <= 0)
            throw new ArgumentException("Start sample is beyond audio length");
        
        var segment = new float[actualLength];
        Array.Copy(samples, startSample, segment, 0, actualLength);
        
        return Extract(segment);
    }
}
