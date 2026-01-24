using NAudio.Wave;
using AiMusicDetector;

if (args.Length < 1)
{
    Console.WriteLine("Usage: TestNAudio <mp3_file>");
    return;
}

var mp3Path = args[0];
Console.WriteLine($"Analyzing: {mp3Path}\n");

// Get gapless info using our parser
var gaplessInfo = Mp3EncoderDelay.GetGaplessInfo(mp3Path);
Console.WriteLine("Gapless Info from Mp3EncoderDelay:");
Console.WriteLine($"  HasXingFrame: {gaplessInfo.HasXingFrame}");
Console.WriteLine($"  SamplesPerFrame: {gaplessInfo.SamplesPerFrame}");
Console.WriteLine($"  StartSkipSamples: {gaplessInfo.StartSkipSamples}");
Console.WriteLine($"  EndTrimSamples: {gaplessInfo.EndTrimSamples}");
Console.WriteLine($"  TotalStartSkip: {gaplessInfo.TotalStartSkip}");

// Load with NAudio
Console.WriteLine("\n--- NAudio Mp3FileReader ---");
using var reader = new Mp3FileReader(mp3Path);
Console.WriteLine($"Sample rate: {reader.WaveFormat.SampleRate}");
Console.WriteLine($"Channels: {reader.WaveFormat.Channels}");
Console.WriteLine($"BitsPerSample: {reader.WaveFormat.BitsPerSample}");
Console.WriteLine($"Length (bytes): {reader.Length}");
Console.WriteLine($"TotalTime: {reader.TotalTime}");

// Read raw samples
var samples = new List<float>();
var buffer = new byte[4096];
int bytesRead;
int bytesPerSample = reader.WaveFormat.BitsPerSample / 8;

while ((bytesRead = reader.Read(buffer, 0, buffer.Length)) > 0)
{
    int sampleCount = bytesRead / bytesPerSample;
    for (int i = 0; i < sampleCount; i++)
    {
        float sample = BitConverter.ToInt16(buffer, i * 2) / 32768f;
        samples.Add(sample);
    }
}

int channels = reader.WaveFormat.Channels;
int nativeSampleRate = reader.WaveFormat.SampleRate;
int totalSamples = samples.Count;
int samplesPerChannel = totalSamples / channels;

Console.WriteLine($"\nRaw NAudio output:");
Console.WriteLine($"  Total samples: {totalSamples}");
Console.WriteLine($"  Samples per channel: {samplesPerChannel}");

// Calculate what we'd get after applying gapless
int skipPerChannel = gaplessInfo.TotalStartSkip;
int endTrimPerChannel = gaplessInfo.EndTrimSamples;
int afterTrimPerChannel = samplesPerChannel - skipPerChannel - endTrimPerChannel;

Console.WriteLine($"\nAfter gapless trimming:");
Console.WriteLine($"  Skip at start: {skipPerChannel}");
Console.WriteLine($"  Trim at end: {endTrimPerChannel}");
Console.WriteLine($"  Samples per channel after trim: {afterTrimPerChannel}");

// Calculate expected at 16kHz
int targetSampleRate = 16000;
int expectedAt16k = (int)((long)afterTrimPerChannel * targetSampleRate / nativeSampleRate);
Console.WriteLine($"\nExpected samples at {targetSampleRate} Hz: {expectedAt16k}");

// What Python/torchaudio returns: 2858240 for this file
int pythonExpected = 2858240;
Console.WriteLine($"\nPython (torchaudio) returns: {pythonExpected} samples at 16kHz");
Console.WriteLine($"Our expected output: {expectedAt16k} samples at 16kHz");
Console.WriteLine($"Difference: {expectedAt16k - pythonExpected} samples");
Console.WriteLine($"Difference as percentage: {100.0 * (expectedAt16k - pythonExpected) / pythonExpected:F2}%");

// Show first few samples
Console.WriteLine("\nFirst 10 samples (interleaved if stereo):");
for (int i = 0; i < Math.Min(10, samples.Count); i++)
{
    Console.WriteLine($"  [{i}] = {samples[i]:F6}");
}

// Show samples AFTER the gapless skip (the actual audio content)
Console.WriteLine($"\nSamples after skip (at position {skipPerChannel * channels}):");
int startIdx = skipPerChannel * channels;
for (int i = 0; i < Math.Min(10, samples.Count - startIdx); i++)
{
    Console.WriteLine($"  [{startIdx + i}] = {samples[startIdx + i]:F6}");
}

// Now process like AudioProcessor does - deinterleave, average channels, show result
Console.WriteLine("\nProcessed mono samples (after skip, channel-averaged):");
for (int i = 0; i < Math.Min(10, afterTrimPerChannel); i++)
{
    int srcIdx = (i + skipPerChannel) * channels;
    float left = samples[srcIdx];
    float right = samples[srcIdx + 1];
    float mono = (left + right) * 0.5f;
    Console.WriteLine($"  [{i}] = L:{left:F6} R:{right:F6} M:{mono:F6}");
}

// Let's also compute a simple test: sum of absolute values for first 1000 samples
Console.WriteLine("\nAudio energy check (sum of abs values):");
float energyFirst1000 = 0;
for (int i = 0; i < Math.Min(1000, afterTrimPerChannel); i++)
{
    int srcIdx = (i + skipPerChannel) * channels;
    float mono = (samples[srcIdx] + samples[srcIdx + 1]) * 0.5f;
    energyFirst1000 += Math.Abs(mono);
}
Console.WriteLine($"  Energy of first 1000 mono samples after skip: {energyFirst1000:F4}");

// Check if there's a lot of silence at the beginning
int silentSamples = 0;
for (int i = 0; i < Math.Min(10000, afterTrimPerChannel); i++)
{
    int srcIdx = (i + skipPerChannel) * channels;
    float mono = Math.Abs((samples[srcIdx] + samples[srcIdx + 1]) * 0.5f);
    if (mono < 0.001f) silentSamples++;
    else break;
}
Console.WriteLine($"  Silent samples at start (< 0.001): {silentSamples}");
