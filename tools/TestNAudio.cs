using System;
using System.IO;
using NAudio.Wave;
using AiMusicDetector;

class Program
{
    static void Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.WriteLine("Usage: TestNAudio <mp3_file>");
            return;
        }

        var mp3Path = args[0];
        Console.WriteLine($"Analyzing: {mp3Path}\n");

        // Get gapless info
        var gaplessInfo = Mp3EncoderDelay.GetGaplessInfo(mp3Path);
        Console.WriteLine("Gapless Info:");
        Console.WriteLine($"  HasXingFrame: {gaplessInfo.HasXingFrame}");
        Console.WriteLine($"  SamplesPerFrame: {gaplessInfo.SamplesPerFrame}");
        Console.WriteLine($"  StartSkipSamples: {gaplessInfo.StartSkipSamples}");
        Console.WriteLine($"  EndTrimSamples: {gaplessInfo.EndTrimSamples}");
        Console.WriteLine($"  TotalStartSkip: {gaplessInfo.TotalStartSkip}");

        // Load with NAudio
        using var reader = new Mp3FileReader(mp3Path);
        Console.WriteLine("\nNAudio Mp3FileReader:");
        Console.WriteLine($"  Sample rate: {reader.WaveFormat.SampleRate}");
        Console.WriteLine($"  Channels: {reader.WaveFormat.Channels}");
        Console.WriteLine($"  BitsPerSample: {reader.WaveFormat.BitsPerSample}");
        Console.WriteLine($"  Length: {reader.Length}");
        
        // Read raw samples
        int bytesPerSample = reader.WaveFormat.BitsPerSample / 8;
        int channels = reader.WaveFormat.Channels;
        
        var samples = new List<float>();
        var buffer = new byte[4096];
        int bytesRead;
        
        while ((bytesRead = reader.Read(buffer, 0, buffer.Length)) > 0)
        {
            int sampleCount = bytesRead / bytesPerSample;
            for (int i = 0; i < sampleCount; i++)
            {
                float sample = BitConverter.ToInt16(buffer, i * 2) / 32768f;
                samples.Add(sample);
            }
        }
        
        int samplesPerChannel = samples.Count / channels;
        Console.WriteLine($"\nRaw NAudio samples: {samples.Count}");
        Console.WriteLine($"Samples per channel: {samplesPerChannel}");
        
        // Calculate what we'd get after applying gapless
        int skipPerChannel = gaplessInfo.TotalStartSkip;
        int endTrim = gaplessInfo.EndTrimSamples;
        int outputSamples = samplesPerChannel - skipPerChannel - endTrim;
        
        Console.WriteLine($"\nAfter gapless adjustment:");
        Console.WriteLine($"  Skip at start: {skipPerChannel}");
        Console.WriteLine($"  Trim at end: {endTrim}");
        Console.WriteLine($"  Output samples: {outputSamples}");
        
        // At 44100 Hz, how many samples is that?
        // If we need 2858240 at 16kHz, that's 2858240 * 44100 / 16000 = 7875060 at 44100
        int expectedAt44100 = (int)(2858240L * 44100 / 16000);
        Console.WriteLine($"\nExpected samples at 44100 Hz (for 2858240 @ 16kHz): {expectedAt44100}");
        Console.WriteLine($"Difference from output: {outputSamples - expectedAt44100}");
    }
}
