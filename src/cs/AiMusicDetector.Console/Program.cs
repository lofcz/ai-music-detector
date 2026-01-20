using AiMusicDetector;

namespace AiMusicDetector.Console;

/// <summary>
/// Command-line tool for AI music detection.
/// </summary>
class Program
{
    static void Main(string[] args)
    {
        if (args.Length < 2)
        {
            PrintUsage();
            return;
        }

        string modelPath = args[0];
        string[] audioPaths = args.Skip(1).ToArray();

        if (!File.Exists(modelPath))
        {
            System.Console.Error.WriteLine($"Error: Model file not found: {modelPath}");
            return;
        }

        try
        {
            System.Console.WriteLine("Loading AI Music Detector...");
            using var detector = MusicDetector.Load(modelPath);
            
            System.Console.WriteLine($"Analyzing {audioPaths.Length} file(s)...\n");
            
            System.Console.WriteLine(new string('=', 80));
            System.Console.WriteLine($"{"File",-40} {"Probability",12} {"Result",15} {"Time",10}");
            System.Console.WriteLine(new string('=', 80));
            
            foreach (var audioPath in audioPaths)
            {
                if (!File.Exists(audioPath))
                {
                    System.Console.WriteLine($"{Path.GetFileName(audioPath),-40} {"NOT FOUND",12}");
                    continue;
                }

                try
                {
                    var result = detector.Analyze(audioPath);
                    
                    string fileName = Path.GetFileName(audioPath);
                    if (fileName.Length > 38)
                        fileName = fileName[..35] + "...";
                    
                    string probability = $"{result.AiProbability:P1}";
                    string classification = result.Classification;
                    string time = $"{result.ProcessingTimeMs}ms";
                    
                    // Color coding (if terminal supports it)
                    if (result.IsAiGenerated)
                    {
                        System.Console.ForegroundColor = ConsoleColor.Red;
                    }
                    else
                    {
                        System.Console.ForegroundColor = ConsoleColor.Green;
                    }
                    
                    System.Console.WriteLine($"{fileName,-40} {probability,12} {classification,15} {time,10}");
                    System.Console.ResetColor();
                }
                catch (Exception ex)
                {
                    System.Console.WriteLine($"{Path.GetFileName(audioPath),-40} {"ERROR: " + ex.Message}");
                }
            }
            
            System.Console.WriteLine(new string('=', 80));
        }
        catch (Exception ex)
        {
            System.Console.Error.WriteLine($"Error: {ex.Message}");
        }
    }

    static void PrintUsage()
    {
        System.Console.WriteLine("AI Music Detector - Command Line Tool");
        System.Console.WriteLine();
        System.Console.WriteLine("Usage:");
        System.Console.WriteLine("  AiMusicDetector.Console <model.onnx> <audio1.mp3> [audio2.mp3] ...");
        System.Console.WriteLine();
        System.Console.WriteLine("Arguments:");
        System.Console.WriteLine("  model.onnx    Path to the ONNX model file");
        System.Console.WriteLine("  audio*.mp3    Audio file(s) to analyze");
        System.Console.WriteLine();
        System.Console.WriteLine("Examples:");
        System.Console.WriteLine("  AiMusicDetector.Console models/ai_music_detector.onnx song.mp3");
        System.Console.WriteLine("  AiMusicDetector.Console models/ai_music_detector.onnx *.mp3");
        System.Console.WriteLine();
        System.Console.WriteLine("Output:");
        System.Console.WriteLine("  0.0% = Real music (human-created)");
        System.Console.WriteLine("  100% = AI-generated music");
    }
}
