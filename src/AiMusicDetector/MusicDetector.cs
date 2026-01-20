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
}

/// <summary>
/// Configuration options for MusicDetector.
/// </summary>
public class MusicDetectorOptions
{
    /// <summary>Target sample rate for audio processing (default: 44100 Hz)</summary>
    public int SampleRate { get; init; } = 44100;
    
    /// <summary>Maximum audio duration to analyze in seconds (default: 180s)</summary>
    public int MaxDurationSeconds { get; init; } = 180;
    
    /// <summary>Classification threshold (default: 0.5)</summary>
    public float Threshold { get; init; } = 0.5f;
    
    /// <summary>Whether to use GPU for ONNX inference if available</summary>
    public bool UseGpu { get; init; } = false;
    
    /// <summary>Fakeprint extraction configuration</summary>
    public FakeprintConfig? FakeprintConfig { get; init; }
}

/// <summary>
/// AI Music Detector - Detects AI-generated music using fakeprint analysis.
/// 
/// Based on: "A Fourier Explanation of AI-Music Artifacts" (ISMIR 2025)
/// 
/// Example usage:
/// <code>
/// using var detector = MusicDetector.Load("ai_music_detector.onnx");
/// var result = detector.Analyze("song.mp3");
/// Console.WriteLine($"AI Probability: {result.AiProbability:P1}");
/// Console.WriteLine($"Classification: {result.Classification}");
/// </code>
/// </summary>
public class MusicDetector : IDisposable
{
    private readonly AudioProcessor _audioProcessor;
    private readonly FakeprintExtractor _fakeprintExtractor;
    private readonly OnnxInference _inference;
    private readonly MusicDetectorOptions _options;
    private bool _disposed;

    private MusicDetector(
        OnnxInference inference,
        MusicDetectorOptions options)
    {
        _inference = inference;
        _options = options;
        
        _audioProcessor = new AudioProcessor(
            options.SampleRate,
            options.MaxDurationSeconds
        );
        
        _fakeprintExtractor = new FakeprintExtractor(
            options.FakeprintConfig ?? new FakeprintConfig
            {
                SampleRate = options.SampleRate
            }
        );
    }

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
        
        // Extract fakeprint
        var fakeprint = _fakeprintExtractor.Extract(samples);
        
        // Run inference
        float probability = _inference.Predict(fakeprint);
        
        stopwatch.Stop();
        
        return new DetectionResult
        {
            AiProbability = probability,
            IsAiGenerated = probability >= _options.Threshold,
            AudioDurationSeconds = duration,
            ProcessingTimeMs = stopwatch.ElapsedMilliseconds
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
        if (sampleRate != _options.SampleRate)
        {
            processedSamples = Resample(samples, sampleRate, _options.SampleRate);
        }
        
        // Extract fakeprint
        var fakeprint = _fakeprintExtractor.Extract(processedSamples);
        
        // Run inference
        float probability = _inference.Predict(fakeprint);
        
        stopwatch.Stop();
        
        double duration = (double)samples.Length / sampleRate;
        
        return new DetectionResult
        {
            AiProbability = probability,
            IsAiGenerated = probability >= _options.Threshold,
            AudioDurationSeconds = duration,
            ProcessingTimeMs = stopwatch.ElapsedMilliseconds
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
        var fakeprint = _fakeprintExtractor.Extract(samples);
        float probability = _inference.Predict(fakeprint);
        
        stopwatch.Stop();
        
        double duration = (double)samples.Length / _options.SampleRate;
        
        return new DetectionResult
        {
            AiProbability = probability,
            IsAiGenerated = probability >= _options.Threshold,
            AudioDurationSeconds = duration,
            ProcessingTimeMs = stopwatch.ElapsedMilliseconds
        };
    }

    /// <summary>
    /// Batch analyze multiple audio files.
    /// </summary>
    /// <param name="audioPaths">Paths to audio files</param>
    /// <returns>Array of detection results</returns>
    public DetectionResult[] AnalyzeBatch(string[] audioPaths)
    {
        return audioPaths.Select(path => Analyze(path)).ToArray();
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
    /// </summary>
    /// <param name="audioPath">Path to the audio file</param>
    /// <returns>Fakeprint feature vector</returns>
    public float[] ExtractFakeprint(string audioPath)
    {
        var samples = _audioProcessor.LoadAudio(audioPath);
        return _fakeprintExtractor.Extract(samples);
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
