namespace AiMusicDetector;

/// <summary>
/// Result of AI music detection analysis.
/// </summary>
public class DetectionResult
{
    /// <summary>
    /// Probability that the audio is AI-generated (0.0 to 1.0).
    /// </summary>
    public float AiProbability { get; init; }
    
    /// <summary>
    /// Whether the audio is classified as AI-generated (probability > threshold).
    /// </summary>
    public bool IsAiGenerated { get; init; }
    
    /// <summary>
    /// Confidence level of the classification.
    /// </summary>
    public float Confidence => Math.Abs(AiProbability - 0.5f) * 2;
    
    /// <summary>
    /// Human-readable classification result.
    /// </summary>
    public string Classification => IsAiGenerated ? "AI-Generated" : "Real Music";
    
    /// <summary>
    /// Duration of the analyzed audio in seconds.
    /// </summary>
    public double AudioDurationSeconds { get; init; }
    
    /// <summary>
    /// Processing time in milliseconds.
    /// </summary>
    public long ProcessingTimeMs { get; init; }
    
    /// <summary>
    /// Model type used for detection.
    /// </summary>
    public ModelType ModelType { get; init; }
    
    /// <summary>
    /// Number of segments analyzed (for CNN models).
    /// </summary>
    public int SegmentsAnalyzed { get; init; } = 1;
}

/// <summary>
/// Detailed debug information for comparing C# vs Python inference.
/// </summary>
public class DebugResult
{
    /// <summary>Standard detection result.</summary>
    public DetectionResult Result { get; init; } = null!;
    
    /// <summary>Total audio samples after loading.</summary>
    public int AudioSampleCount { get; init; }
    
    /// <summary>First 10 audio samples (for verification).</summary>
    public float[] AudioSamplesHead { get; init; } = Array.Empty<float>();
    
    /// <summary>Per-segment debug information.</summary>
    public List<SegmentDebugInfo> Segments { get; init; } = new();
    
    /// <summary>All segment probabilities before aggregation.</summary>
    public float[] SegmentProbabilities { get; init; } = Array.Empty<float>();
    
    /// <summary>Final probability after median pooling.</summary>
    public float FinalProbability { get; init; }
}

/// <summary>
/// Debug information for a single segment.
/// </summary>
public class SegmentDebugInfo
{
    /// <summary>Segment index (0-based).</summary>
    public int Index { get; init; }
    
    /// <summary>Start position in samples.</summary>
    public int StartSample { get; init; }
    
    /// <summary>Segment length in samples.</summary>
    public int LengthSamples { get; init; }
    
    /// <summary>Cepstrum shape [n_coeffs, n_frames].</summary>
    public int[] CepstrumShape { get; init; } = Array.Empty<int>();
    
    /// <summary>First 5 values of cepstrum[0, :] (first coefficient across first 5 frames).</summary>
    public float[] CepstrumCoeff0Head { get; init; } = Array.Empty<float>();
    
    /// <summary>First 5 values of cepstrum[:, 0] (all coefficients at first frame).</summary>
    public float[] CepstrumFrame0Head { get; init; } = Array.Empty<float>();
    
    /// <summary>Mean of all cepstrum values.</summary>
    public float CepstrumMean { get; init; }
    
    /// <summary>Model output probability for this segment.</summary>
    public float Probability { get; init; }
}

/// <summary>
/// Configuration options for MusicDetector.
/// </summary>
public class MusicDetectorOptions
{
    /// <summary>Target sample rate for audio processing. Auto-detected based on model type if not specified.</summary>
    public int? SampleRate { get; init; }
    
    /// <summary>
    /// Maximum audio duration to analyze in seconds.
    /// Set to 0 or less for no limit (matches Python inference behavior).
    /// </summary>
    public int MaxDurationSeconds { get; init; } = 0;
    
    /// <summary>Classification threshold (default: 0.5)</summary>
    public float Threshold { get; init; } = 0.5f;
    
    /// <summary>Whether to use GPU for ONNX inference if available</summary>
    public bool UseGpu { get; init; } = false;
    
    /// <summary>Fakeprint extraction configuration (for regression models)</summary>
    public FakeprintConfig? FakeprintConfig { get; init; }
    
    /// <summary>Cepstrum extraction configuration (for CNN models)</summary>
    public CepstrumConfig? CepstrumConfig { get; init; }
    
    /// <summary>Number of segments to analyze (for CNN models, default: 5)</summary>
    public int NumSegments { get; init; } = 5;
}

/// <summary>
/// AI Music Detector - Detects AI-generated music using fakeprint or cepstrum analysis.
/// 
/// Supports two model types:
/// - Regression: Uses fakeprint features (FFT-based spectral artifacts) at 44100 Hz
/// - CNN: Uses CQT-cepstrum features at 16000 Hz
/// 
/// Based on: "A Fourier Explanation of AI-Music Artifacts" (ISMIR 2025)
/// 
/// Example usage:
/// <code>
/// using var detector = MusicDetector.Load("ai_music_detector.onnx");
/// var result = detector.Analyze("song.mp3");
/// Console.WriteLine($"AI Probability: {result.AiProbability:P1}");
/// Console.WriteLine($"Classification: {result.Classification}");
/// Console.WriteLine($"Model Type: {result.ModelType}");
/// </code>
/// </summary>
public class MusicDetector : IDisposable
{
    // Default sample rates for each model type
    private const int RegressionSampleRate = 44100;
    private const int CnnSampleRate = 16000;
    
    private readonly AudioProcessor _audioProcessor;
    private readonly FakeprintExtractor? _fakeprintExtractor;
    private readonly CepstrumExtractor? _cepstrumExtractor;
    private readonly OnnxInference _inference;
    private readonly MusicDetectorOptions _options;
    private readonly int _sampleRate;
    private bool _disposed;

    private MusicDetector(
        OnnxInference inference,
        MusicDetectorOptions options)
    {
        _inference = inference;
        _options = options;
        
        // Determine sample rate based on model type
        _sampleRate = options.SampleRate ?? (inference.ModelType == ModelType.CNN ? CnnSampleRate : RegressionSampleRate);
        
        _audioProcessor = new AudioProcessor(
            _sampleRate,
            options.MaxDurationSeconds
        );
        
        // Initialize the appropriate feature extractor based on model type
        if (inference.ModelType == ModelType.CNN)
        {
            _cepstrumExtractor = new CepstrumExtractor(
                options.CepstrumConfig ?? new CepstrumConfig
                {
                    SampleRate = _sampleRate
                }
            );
        }
        else
        {
            _fakeprintExtractor = new FakeprintExtractor(
                options.FakeprintConfig ?? new FakeprintConfig
                {
                    SampleRate = _sampleRate
                }
            );
        }
    }
    
    /// <summary>
    /// Gets the model type (Regression or CNN).
    /// </summary>
    public ModelType ModelType => _inference.ModelType;
    
    /// <summary>
    /// Gets the sample rate used for audio processing.
    /// </summary>
    public int SampleRate => _sampleRate;

    /// <summary>
    /// Load a MusicDetector from an ONNX model file.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model file</param>
    /// <param name="options">Configuration options, or null for defaults</param>
    /// <returns>MusicDetector instance</returns>
    public static MusicDetector Load(string modelPath, MusicDetectorOptions? options = null)
    {
        options ??= new MusicDetectorOptions();
        var inference = new OnnxInference(modelPath, options.UseGpu);
        return new MusicDetector(inference, options);
    }

    /// <summary>
    /// Load a MusicDetector from ONNX model bytes.
    /// </summary>
    /// <param name="modelData">ONNX model data</param>
    /// <param name="options">Configuration options, or null for defaults</param>
    /// <returns>MusicDetector instance</returns>
    public static MusicDetector Load(byte[] modelData, MusicDetectorOptions? options = null)
    {
        options ??= new MusicDetectorOptions();
        var inference = new OnnxInference(modelData, options.UseGpu);
        return new MusicDetector(inference, options);
    }

    /// <summary>
    /// Analyze an audio file for AI-generated content.
    /// </summary>
    /// <param name="audioPath">Path to the audio file</param>
    /// <returns>Detection result with AI probability and classification</returns>
    public DetectionResult Analyze(string audioPath)
    {
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        
        // Get duration
        double duration = AudioProcessor.GetDuration(audioPath);
        
        // Load and process audio
        var samples = _audioProcessor.LoadAudio(audioPath);
        
        float probability;
        int segmentsAnalyzed = 1;
        
        if (_inference.ModelType == ModelType.CNN)
        {
            // CNN model: extract segments, predict each, median-pool
            var segments = _cepstrumExtractor!.ExtractSegments(samples, _options.NumSegments);
            segmentsAnalyzed = segments.Count;
            
            var predictions = new float[segments.Count];
            for (int i = 0; i < segments.Count; i++)
            {
                predictions[i] = _inference.PredictCNN(segments[i]);
            }
            
            // Median pooling
            probability = MedianPool(predictions);
        }
        else
        {
            // Regression model: extract fakeprint and predict
            var fakeprint = _fakeprintExtractor!.Extract(samples);
            probability = _inference.Predict(fakeprint);
        }
        
        stopwatch.Stop();
        
        return new DetectionResult
        {
            AiProbability = probability,
            IsAiGenerated = probability >= _options.Threshold,
            AudioDurationSeconds = duration,
            ProcessingTimeMs = stopwatch.ElapsedMilliseconds,
            ModelType = _inference.ModelType,
            SegmentsAnalyzed = segmentsAnalyzed
        };
    }

    /// <summary>
    /// Analyze audio samples for AI-generated content.
    /// </summary>
    /// <param name="samples">Audio samples (mono, normalized to [-1, 1])</param>
    /// <param name="sampleRate">Sample rate of the audio</param>
    /// <returns>Detection result with AI probability and classification</returns>
    public DetectionResult Analyze(float[] samples, int sampleRate)
    {
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        
        // Resample if needed
        float[] processedSamples = samples;
        if (sampleRate != _sampleRate)
        {
            processedSamples = Resample(samples, sampleRate, _sampleRate);
        }
        
        float probability;
        int segmentsAnalyzed = 1;
        
        if (_inference.ModelType == ModelType.CNN)
        {
            // CNN model: extract segments, predict each, median-pool
            var segments = _cepstrumExtractor!.ExtractSegments(processedSamples, _options.NumSegments);
            segmentsAnalyzed = segments.Count;
            
            var predictions = new float[segments.Count];
            for (int i = 0; i < segments.Count; i++)
            {
                predictions[i] = _inference.PredictCNN(segments[i]);
            }
            
            probability = MedianPool(predictions);
        }
        else
        {
            // Regression model: extract fakeprint and predict
            var fakeprint = _fakeprintExtractor!.Extract(processedSamples);
            probability = _inference.Predict(fakeprint);
        }
        
        stopwatch.Stop();
        
        double duration = (double)samples.Length / sampleRate;
        
        return new DetectionResult
        {
            AiProbability = probability,
            IsAiGenerated = probability >= _options.Threshold,
            AudioDurationSeconds = duration,
            ProcessingTimeMs = stopwatch.ElapsedMilliseconds,
            ModelType = _inference.ModelType,
            SegmentsAnalyzed = segmentsAnalyzed
        };
    }

    /// <summary>
    /// Analyze audio from a stream.
    /// </summary>
    /// <param name="audioStream">Audio stream</param>
    /// <param name="format">Audio format (e.g., "mp3", "wav")</param>
    /// <returns>Detection result</returns>
    public DetectionResult Analyze(Stream audioStream, string format = "mp3")
    {
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        
        var samples = _audioProcessor.LoadAudio(audioStream, format);
        
        float probability;
        int segmentsAnalyzed = 1;
        
        if (_inference.ModelType == ModelType.CNN)
        {
            var segments = _cepstrumExtractor!.ExtractSegments(samples, _options.NumSegments);
            segmentsAnalyzed = segments.Count;
            
            var predictions = new float[segments.Count];
            for (int i = 0; i < segments.Count; i++)
            {
                predictions[i] = _inference.PredictCNN(segments[i]);
            }
            
            probability = MedianPool(predictions);
        }
        else
        {
            var fakeprint = _fakeprintExtractor!.Extract(samples);
            probability = _inference.Predict(fakeprint);
        }
        
        stopwatch.Stop();
        
        double duration = (double)samples.Length / _sampleRate;
        
        return new DetectionResult
        {
            AiProbability = probability,
            IsAiGenerated = probability >= _options.Threshold,
            AudioDurationSeconds = duration,
            ProcessingTimeMs = stopwatch.ElapsedMilliseconds,
            ModelType = _inference.ModelType,
            SegmentsAnalyzed = segmentsAnalyzed
        };
    }

    /// <summary>
    /// Batch analyze multiple audio files.
    /// </summary>
    /// <param name="audioPaths">Paths to audio files</param>
    /// <returns>Array of detection results</returns>
    public DetectionResult[] AnalyzeBatch(string[] audioPaths)
    {
        return audioPaths.Select(Analyze).ToArray();
    }

    /// <summary>
    /// Get the raw AI probability for an audio file (0.0 to 1.0).
    /// </summary>
    /// <param name="audioPath">Path to the audio file</param>
    /// <returns>AI probability (0.0 = Real, 1.0 = AI-Generated)</returns>
    public float Predict(string audioPath)
    {
        return Analyze(audioPath).AiProbability;
    }

    /// <summary>
    /// Get the raw AI probability for audio samples.
    /// </summary>
    /// <param name="samples">Audio samples</param>
    /// <param name="sampleRate">Sample rate</param>
    /// <returns>AI probability</returns>
    public float Predict(float[] samples, int sampleRate)
    {
        return Analyze(samples, sampleRate).AiProbability;
    }

    /// <summary>
    /// Check if an audio file is AI-generated (above threshold).
    /// </summary>
    /// <param name="audioPath">Path to the audio file</param>
    /// <returns>True if AI-generated</returns>
    public bool IsAiGenerated(string audioPath)
    {
        return Analyze(audioPath).IsAiGenerated;
    }

    /// <summary>
    /// Extract fakeprint features from audio (for debugging/analysis).
    /// Only available for regression models.
    /// </summary>
    /// <param name="audioPath">Path to the audio file</param>
    /// <returns>Fakeprint feature vector</returns>
    /// <exception cref="InvalidOperationException">Thrown when using a CNN model</exception>
    public float[] ExtractFakeprint(string audioPath)
    {
        if (_fakeprintExtractor == null)
            throw new InvalidOperationException("Fakeprint extraction is only available for regression models, not CNN models.");
        
        var samples = _audioProcessor.LoadAudio(audioPath);
        return _fakeprintExtractor.Extract(samples);
    }
    
    /// <summary>
    /// Extract cepstrum features from audio (for debugging/analysis).
    /// Only available for CNN models.
    /// </summary>
    /// <param name="audioPath">Path to the audio file</param>
    /// <returns>Cepstrum features [n_coeffs, n_frames]</returns>
    /// <exception cref="InvalidOperationException">Thrown when using a regression model</exception>
    public float[,] ExtractCepstrum(string audioPath)
    {
        if (_cepstrumExtractor == null)
            throw new InvalidOperationException("Cepstrum extraction is only available for CNN models, not regression models.");
        
        var samples = _audioProcessor.LoadAudio(audioPath);
        return _cepstrumExtractor.Extract(samples);
    }

    /// <summary>
    /// Analyze with detailed debug output for comparing C# vs Python inference.
    /// </summary>
    /// <param name="audioPath">Path to the audio file</param>
    /// <returns>Debug result with detailed segment information</returns>
    public DebugResult AnalyzeWithDebug(string audioPath)
    {
        if (_cepstrumExtractor == null)
            throw new InvalidOperationException("Debug analysis is only available for CNN models.");
        
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        
        // Load audio
        var samples = _audioProcessor.LoadAudio(audioPath);
        double duration = AudioProcessor.GetDuration(audioPath);
        
        // Extract first 10 samples for verification
        var audioHead = new float[Math.Min(10, samples.Length)];
        Array.Copy(samples, audioHead, audioHead.Length);
        
        // Get segment positions by calling the extractor
        // We need to compute segments manually to get positions
        int skipSamples = (int)(5.0 * _sampleRate);
        int segmentSamples = _cepstrumExtractor.SegmentSamples;
        
        int startOffset, endOffset;
        if (samples.Length > segmentSamples + 2 * skipSamples)
        {
            startOffset = skipSamples;
            endOffset = samples.Length - skipSamples;
        }
        else
        {
            startOffset = 0;
            endOffset = samples.Length;
        }
        
        int usableLength = endOffset - startOffset;
        
        // Calculate segment positions (matching fixed ExtractSegments implementation)
        var segmentPositions = new List<int>();
        if (usableLength <= segmentSamples)
        {
            if (samples.Length <= segmentSamples)
            {
                segmentPositions.Add(0); // Padded case
            }
            else
            {
                // Center case
                int center = samples.Length / 2;
                int start = Math.Max(0, center - segmentSamples / 2);
                segmentPositions.Add(start);
            }
        }
        else
        {
            int available = usableLength - segmentSamples;
            if (_options.NumSegments == 1)
            {
                segmentPositions.Add(startOffset + available / 2);
            }
            else
            {
                // Use float division like Python
                double step = (double)available / (_options.NumSegments - 1);
                for (int i = 0; i < _options.NumSegments; i++)
                {
                    segmentPositions.Add(startOffset + (int)(i * step));
                }
            }
        }
        
        // Extract segments and compute features
        var segments = _cepstrumExtractor.ExtractSegments(samples, _options.NumSegments);
        var segmentDebugInfos = new List<SegmentDebugInfo>();
        var predictions = new float[segments.Count];
        
        for (int i = 0; i < segments.Count; i++)
        {
            var cepstrum = segments[i];
            int nCoeffs = cepstrum.GetLength(0);
            int nFrames = cepstrum.GetLength(1);
            
            // First 5 values of coefficient 0
            var coeff0Head = new float[Math.Min(5, nFrames)];
            for (int j = 0; j < coeff0Head.Length; j++)
                coeff0Head[j] = cepstrum[0, j];
            
            // First 5 coefficients at frame 0
            var frame0Head = new float[Math.Min(5, nCoeffs)];
            for (int j = 0; j < frame0Head.Length; j++)
                frame0Head[j] = cepstrum[j, 0];
            
            // Mean of all values
            float sum = 0;
            for (int c = 0; c < nCoeffs; c++)
                for (int f = 0; f < nFrames; f++)
                    sum += cepstrum[c, f];
            float mean = sum / (nCoeffs * nFrames);
            
            // Get probability
            predictions[i] = _inference.PredictCNN(cepstrum);
            
            int startPos = i < segmentPositions.Count ? segmentPositions[i] : 0;
            
            segmentDebugInfos.Add(new SegmentDebugInfo
            {
                Index = i,
                StartSample = startPos,
                LengthSamples = segmentSamples,
                CepstrumShape = new[] { nCoeffs, nFrames },
                CepstrumCoeff0Head = coeff0Head,
                CepstrumFrame0Head = frame0Head,
                CepstrumMean = mean,
                Probability = predictions[i]
            });
        }
        
        float finalProbability = MedianPool(predictions);
        
        stopwatch.Stop();
        
        var result = new DetectionResult
        {
            AiProbability = finalProbability,
            IsAiGenerated = finalProbability >= _options.Threshold,
            AudioDurationSeconds = duration,
            ProcessingTimeMs = stopwatch.ElapsedMilliseconds,
            ModelType = _inference.ModelType,
            SegmentsAnalyzed = segments.Count
        };
        
        return new DebugResult
        {
            Result = result,
            AudioSampleCount = samples.Length,
            AudioSamplesHead = audioHead,
            Segments = segmentDebugInfos,
            SegmentProbabilities = predictions,
            FinalProbability = finalProbability
        };
    }

    private static float[] Resample(float[] samples, int fromRate, int toRate)
    {
        if (fromRate == toRate)
            return samples;
        
        double ratio = (double)toRate / fromRate;
        int newLength = (int)(samples.Length * ratio);
        var result = new float[newLength];
        
        for (int i = 0; i < newLength; i++)
        {
            double srcIdx = i / ratio;
            int idx1 = (int)srcIdx;
            int idx2 = Math.Min(idx1 + 1, samples.Length - 1);
            double frac = srcIdx - idx1;
            
            result[i] = (float)(samples[idx1] * (1 - frac) + samples[idx2] * frac);
        }
        
        return result;
    }
    
    /// <summary>
    /// Compute the median of an array of values.
    /// </summary>
    private static float MedianPool(float[] values)
    {
        if (values.Length == 0)
            return 0f;
        if (values.Length == 1)
            return values[0];
        
        var sorted = values.OrderBy(x => x).ToArray();
        int mid = sorted.Length / 2;
        
        if (sorted.Length % 2 == 0)
        {
            return (sorted[mid - 1] + sorted[mid]) / 2f;
        }
        else
        {
            return sorted[mid];
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _audioProcessor?.Dispose();
            _inference?.Dispose();
            _disposed = true;
            GC.SuppressFinalize(this);
        }
    }
}
