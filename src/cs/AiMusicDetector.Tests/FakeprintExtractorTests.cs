using Xunit;

namespace AiMusicDetector.Tests;

public class FakeprintExtractorTests
{
    [Fact]
    public void Constructor_WithDefaultConfig_SetsCorrectDefaults()
    {
        var extractor = new FakeprintExtractor();
        
        Assert.True(extractor.OutputSize > 0);
    }

    [Fact]
    public void Constructor_WithCustomConfig_UsesProvidedValues()
    {
        var config = new FakeprintConfig
        {
            SampleRate = 22050,
            NfftSize = 8192,
            FreqMin = 4000,
            FreqMax = 10000
        };
        
        var extractor = new FakeprintExtractor(config);
        
        Assert.True(extractor.OutputSize > 0);
    }

    [Fact]
    public void Extract_WithSilentAudio_ReturnsValidFakeprint()
    {
        var extractor = new FakeprintExtractor();
        
        // Create 1 second of silence at 44100 Hz
        var samples = new float[44100];
        
        var fakeprint = extractor.Extract(samples);
        
        Assert.NotNull(fakeprint);
        Assert.Equal(extractor.OutputSize, fakeprint.Length);
        Assert.All(fakeprint, v => Assert.True(v >= 0 && v <= 1));
    }

    [Fact]
    public void Extract_WithSineWave_ReturnsValidFakeprint()
    {
        var extractor = new FakeprintExtractor();
        
        // Create 1 second of 440 Hz sine wave
        int sampleRate = 44100;
        var samples = new float[sampleRate];
        float frequency = 440f;
        
        for (int i = 0; i < samples.Length; i++)
        {
            samples[i] = MathF.Sin(2 * MathF.PI * frequency * i / sampleRate);
        }
        
        var fakeprint = extractor.Extract(samples);
        
        Assert.NotNull(fakeprint);
        Assert.Equal(extractor.OutputSize, fakeprint.Length);
        Assert.All(fakeprint, v => Assert.True(v >= 0 && v <= 1));
    }

    [Fact]
    public void Extract_WithNoise_ReturnsValidFakeprint()
    {
        var extractor = new FakeprintExtractor();
        var random = new Random(42);
        
        // Create 1 second of noise
        var samples = new float[44100];
        for (int i = 0; i < samples.Length; i++)
        {
            samples[i] = (float)(random.NextDouble() * 2 - 1);
        }
        
        var fakeprint = extractor.Extract(samples);
        
        Assert.NotNull(fakeprint);
        Assert.Equal(extractor.OutputSize, fakeprint.Length);
    }

    [Fact]
    public void Extract_WithShortAudio_HandlesGracefully()
    {
        var extractor = new FakeprintExtractor();
        
        // Very short audio (less than FFT size)
        var samples = new float[1000];
        
        // Should not throw, may have reduced accuracy
        var fakeprint = extractor.Extract(samples);
        
        Assert.NotNull(fakeprint);
    }

    [Fact]
    public void Extract_OutputIsNormalized()
    {
        var extractor = new FakeprintExtractor();
        var random = new Random(42);
        
        var samples = new float[44100 * 3]; // 3 seconds
        for (int i = 0; i < samples.Length; i++)
        {
            samples[i] = (float)(random.NextDouble() * 2 - 1);
        }
        
        var fakeprint = extractor.Extract(samples);
        
        // Check normalization - values should be in [0, 1]
        float min = fakeprint.Min();
        float max = fakeprint.Max();
        
        Assert.True(min >= 0, "Fakeprint minimum should be >= 0");
        Assert.True(max <= 1, "Fakeprint maximum should be <= 1");
        Assert.True(max > 0, "Fakeprint should have some non-zero values");
    }
}

public class AudioProcessorTests
{
    [Fact]
    public void IsSupportedFormat_RecognizesCommonFormats()
    {
        Assert.True(AudioProcessor.IsSupportedFormat("song.mp3"));
        Assert.True(AudioProcessor.IsSupportedFormat("song.wav"));
        Assert.True(AudioProcessor.IsSupportedFormat("song.flac"));
        Assert.True(AudioProcessor.IsSupportedFormat("SONG.MP3"));
    }

    [Fact]
    public void IsSupportedFormat_RejectsUnsupportedFormats()
    {
        Assert.False(AudioProcessor.IsSupportedFormat("song.txt"));
        Assert.False(AudioProcessor.IsSupportedFormat("song.pdf"));
        Assert.False(AudioProcessor.IsSupportedFormat("song.midi"));
    }
}

public class DetectionResultTests
{
    [Fact]
    public void Classification_ReturnsCorrectString()
    {
        var aiResult = new DetectionResult { AiProbability = 0.9f, IsAiGenerated = true };
        var realResult = new DetectionResult { AiProbability = 0.1f, IsAiGenerated = false };
        
        Assert.Equal("AI-Generated", aiResult.Classification);
        Assert.Equal("Real Music", realResult.Classification);
    }

    [Fact]
    public void Confidence_CalculatesCorrectly()
    {
        // High confidence AI
        var highAi = new DetectionResult { AiProbability = 0.95f };
        Assert.True(highAi.Confidence > 0.8f);
        
        // High confidence Real
        var highReal = new DetectionResult { AiProbability = 0.05f };
        Assert.True(highReal.Confidence > 0.8f);
        
        // Low confidence (near threshold)
        var uncertain = new DetectionResult { AiProbability = 0.5f };
        Assert.True(uncertain.Confidence < 0.1f);
    }
}
