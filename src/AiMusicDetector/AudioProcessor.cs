using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace AiMusicDetector;

/// <summary>
/// Handles audio loading, resampling, and preprocessing.
/// </summary>
public class AudioProcessor : IDisposable
{
    private readonly int _targetSampleRate;
    private readonly int _maxDurationSeconds;
    private bool _disposed;

    /// <summary>
    /// Creates a new AudioProcessor instance.
    /// </summary>
    /// <param name="targetSampleRate">Target sample rate for resampling (default: 44100 Hz)</param>
    /// <param name="maxDurationSeconds">Maximum audio duration to process in seconds (default: 180s)</param>
    public AudioProcessor(int targetSampleRate = 44100, int maxDurationSeconds = 180)
    {
        _targetSampleRate = targetSampleRate;
        _maxDurationSeconds = maxDurationSeconds;
    }

    /// <summary>
    /// Load audio from a file path.
    /// </summary>
    /// <param name="filePath">Path to the audio file (MP3, WAV, FLAC, etc.)</param>
    /// <returns>Mono audio samples normalized to [-1, 1] range</returns>
    public float[] LoadAudio(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Audio file not found: {filePath}");

        using var reader = CreateAudioReader(filePath);
        return ProcessAudio(reader);
    }

    /// <summary>
    /// Load audio from a stream.
    /// </summary>
    /// <param name="stream">Audio stream</param>
    /// <param name="format">Audio format hint (e.g., "mp3", "wav")</param>
    /// <returns>Mono audio samples normalized to [-1, 1] range</returns>
    public float[] LoadAudio(Stream stream, string format = "mp3")
    {
        using var reader = CreateAudioReader(stream, format);
        return ProcessAudio(reader);
    }

    private WaveStream CreateAudioReader(string filePath)
    {
        var extension = Path.GetExtension(filePath).ToLowerInvariant();
        
        return extension switch
        {
            ".mp3" => new Mp3FileReader(filePath),
            ".wav" => new WaveFileReader(filePath),
            ".aiff" or ".aif" => new AiffFileReader(filePath),
            _ => new AudioFileReader(filePath) // Generic reader
        };
    }

    private WaveStream CreateAudioReader(Stream stream, string format)
    {
        return format.ToLowerInvariant() switch
        {
            "mp3" => new Mp3FileReader(stream),
            "wav" => new WaveFileReader(stream),
            _ => throw new NotSupportedException($"Unsupported format: {format}")
        };
    }

    private float[] ProcessAudio(WaveStream reader)
    {
        // Convert to sample provider for easier manipulation
        ISampleProvider sampleProvider = reader.ToSampleProvider();
        
        // Convert to mono if stereo
        if (sampleProvider.WaveFormat.Channels > 1)
        {
            sampleProvider = new StereoToMonoSampleProvider(sampleProvider);
        }

        // Resample if needed
        if (sampleProvider.WaveFormat.SampleRate != _targetSampleRate)
        {
            sampleProvider = new WdlResamplingSampleProvider(sampleProvider, _targetSampleRate);
        }

        // Calculate max samples
        int maxSamples = _targetSampleRate * _maxDurationSeconds;
        
        // Read samples
        var samples = new List<float>();
        var buffer = new float[4096];
        int samplesRead;
        
        while ((samplesRead = sampleProvider.Read(buffer, 0, buffer.Length)) > 0)
        {
            for (int i = 0; i < samplesRead && samples.Count < maxSamples; i++)
            {
                samples.Add(buffer[i]);
            }
            
            if (samples.Count >= maxSamples)
                break;
        }

        return samples.ToArray();
    }

    /// <summary>
    /// Get audio duration in seconds.
    /// </summary>
    /// <param name="filePath">Path to the audio file</param>
    /// <returns>Duration in seconds</returns>
    public static double GetDuration(string filePath)
    {
        using var reader = new AudioFileReader(filePath);
        return reader.TotalTime.TotalSeconds;
    }

    /// <summary>
    /// Check if a file is a supported audio format.
    /// </summary>
    /// <param name="filePath">Path to check</param>
    /// <returns>True if the file format is supported</returns>
    public static bool IsSupportedFormat(string filePath)
    {
        var extension = Path.GetExtension(filePath).ToLowerInvariant();
        return extension is ".mp3" or ".wav" or ".flac" or ".ogg" or ".aiff" or ".aif" or ".m4a" or ".wma";
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            GC.SuppressFinalize(this);
        }
    }
}
