using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using FFMpegCore;
using FFMpegCore.Pipes;

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
    /// <param name="maxDurationSeconds">
    /// Maximum audio duration to process in seconds. Set to 0 or less for no limit.
    /// </param>
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

        // Use FFmpeg for all audio formats - it handles encoder delay correctly like torchaudio
        return LoadAudioWithFFmpeg(filePath);
    }
    
    /// <summary>
    /// Load audio using FFmpeg for torchaudio-compatible decoding.
    /// FFmpeg handles MP3 decoding, then we use SincResampler to match torchaudio's resampling.
    /// 
    /// IMPORTANT: To match torchaudio exactly, the order of operations must be:
    /// 1. Decode audio at NATIVE sample rate (FFmpeg)
    /// 2. Resample to target rate (SincResampler - matches torchaudio)
    /// 3. Convert to mono by averaging channels
    /// 
    /// torchaudio does: load -> resample (both channels) -> mean(dim=0)
    /// </summary>
    private float[] LoadAudioWithFFmpeg(string filePath)
    {
        var tempPcmFile = Path.GetTempFileName();
        try
        {
            // Step 1: Get native sample rate
            int nativeSampleRate = GetSampleRate(filePath);
            int channels = GetChannelCount(filePath);
            
            if (nativeSampleRate <= 0)
            {
                // Fallback: let FFmpeg handle everything
                return LoadAudioWithFFmpegLegacy(filePath, channels);
            }

            // Step 2: Decode to raw PCM at NATIVE sample rate (no resampling in FFmpeg)
            FFMpegArguments
                .FromFileInput(filePath)
                .OutputToFile(tempPcmFile, overwrite: true, options =>
                {
                    options
                        .WithAudioCodec("pcm_f32le")
                        .ForceFormat("f32le");

                    // Only apply truncation if explicitly requested.
                    if (_maxDurationSeconds > 0)
                        options.WithCustomArgument($"-t {_maxDurationSeconds}");
                })
                .ProcessSynchronously();
            
            // Read raw PCM samples
            var fileBytes = File.ReadAllBytes(tempPcmFile);
            int sampleCount = fileBytes.Length / 4;
            
            float[] samples;
            if (channels >= 2)
            {
                // Stereo: Split channels, resample each, then average
                int samplesPerChannel = sampleCount / channels;
                float[] left = new float[samplesPerChannel];
                float[] right = new float[samplesPerChannel];
                
                // Deinterleave stereo samples (LRLRLR... format)
                for (int i = 0; i < samplesPerChannel; i++)
                {
                    int byteOffset = i * channels * 4;
                    left[i] = BitConverter.ToSingle(fileBytes, byteOffset);
                    right[i] = BitConverter.ToSingle(fileBytes, byteOffset + 4);
                }
                
                // Step 3: Resample and convert to mono using SincResampler
                // This matches torchaudio's order: resample first, then mean(dim=0)
                if (nativeSampleRate != _targetSampleRate)
                {
                    samples = SincResampler.ResampleStereoToMono(left, right, nativeSampleRate, _targetSampleRate);
                }
                else
                {
                    // No resampling needed, just average channels
                    samples = new float[samplesPerChannel];
                    for (int i = 0; i < samplesPerChannel; i++)
                    {
                        samples[i] = (left[i] + right[i]) * 0.5f;
                    }
                }
            }
            else
            {
                // Mono: Just read samples
                samples = new float[sampleCount];
                Buffer.BlockCopy(fileBytes, 0, samples, 0, fileBytes.Length);
                
                // Step 3: Resample if needed
                if (nativeSampleRate != _targetSampleRate)
                {
                    samples = SincResampler.Resample(samples, nativeSampleRate, _targetSampleRate);
                }
            }
            
            return samples;
        }
        finally
        {
            if (File.Exists(tempPcmFile))
                File.Delete(tempPcmFile);
        }
    }

    /// <summary>
    /// Legacy FFmpeg loading (fallback when native sample rate detection fails).
    /// </summary>
    private float[] LoadAudioWithFFmpegLegacy(string filePath, int channels)
    {
        var tempPcmFile = Path.GetTempFileName();
        try
        {
            FFMpegArguments
                .FromFileInput(filePath)
                .OutputToFile(tempPcmFile, overwrite: true, options =>
                {
                    options
                        .WithAudioSamplingRate(_targetSampleRate)
                        .WithAudioCodec("pcm_f32le")
                        .ForceFormat("f32le");

                    if (channels >= 2)
                        options.WithCustomArgument("-af \"pan=mono|c0=0.5*c0+0.5*c1\"");
                    else
                        options.WithCustomArgument("-ac 1");

                    if (_maxDurationSeconds > 0)
                        options.WithCustomArgument($"-t {_maxDurationSeconds}");
                })
                .ProcessSynchronously();
            
            var fileBytes = File.ReadAllBytes(tempPcmFile);
            var samples = new float[fileBytes.Length / 4];
            Buffer.BlockCopy(fileBytes, 0, samples, 0, fileBytes.Length);
            return samples;
        }
        finally
        {
            if (File.Exists(tempPcmFile))
                File.Delete(tempPcmFile);
        }
    }
    
    /// <summary>
    /// Load audio using NAudio (fallback for when FFmpeg is not available).
    /// </summary>
    private float[] LoadAudioWithNAudio(string filePath)
    {
        using var reader = CreateAudioReader(filePath);
        return ProcessAudio(reader);
    }

    private static int GetChannelCount(string filePath)
    {
        try
        {
            var analysis = FFProbe.Analyse(filePath);
            var audio = analysis.PrimaryAudioStream;
            return audio?.Channels ?? 0;
        }
        catch
        {
            return 0;
        }
    }

    private static int GetSampleRate(string filePath)
    {
        try
        {
            var analysis = FFProbe.Analyse(filePath);
            var audio = analysis.PrimaryAudioStream;
            return audio?.SampleRateHz ?? 0;
        }
        catch
        {
            return 0;
        }
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
        
        // Use MediaFoundationReader for better MP3 decoding compatibility with Python's torchaudio
        // MediaFoundationReader uses Windows Media Foundation which handles encoder delay better
        if (extension == ".mp3")
        {
            try
            {
                return new MediaFoundationReader(filePath);
            }
            catch
            {
                // Fall back to Mp3FileReader if MF fails
                return new Mp3FileReader(filePath);
            }
        }
        
        return extension switch
        {
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
        bool needsResampling = sampleProvider.WaveFormat.SampleRate != _targetSampleRate;
        if (needsResampling)
        {
            sampleProvider = new WdlResamplingSampleProvider(sampleProvider, _targetSampleRate);
        }

        // Calculate max samples (0 or less means no limit)
        int maxSamples = _maxDurationSeconds > 0 ? _targetSampleRate * _maxDurationSeconds : int.MaxValue;
        
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
