using System.Diagnostics;
using System.Text.Json;
using AiMusicDetector;
using Spectre.Console;
using Spectre.Console.Rendering;

namespace AiMusicDetector.Console;

/// <summary>
/// Custom progress column showing elapsed/eta time.
/// </summary>
class TimeInfoColumn : ProgressColumn
{
    private readonly Stopwatch _stopwatch;
    
    public TimeInfoColumn(Stopwatch stopwatch) => _stopwatch = stopwatch;
    
    public override IRenderable Render(RenderOptions options, ProgressTask task, TimeSpan deltaTime)
    {
        var elapsed = _stopwatch.Elapsed;
        var done = task.Value;
        var total = task.MaxValue;
        
        TimeSpan eta;
        if (done > 0 && total > 0)
        {
            var remaining = total - done;
            eta = TimeSpan.FromTicks((long)(elapsed.Ticks * remaining / done));
        }
        else
        {
            eta = TimeSpan.Zero;
        }
        
        var elapsedStr = FormatTime(elapsed);
        var etaStr = FormatTime(eta);
        var pctStr = total > 0 ? $"{done * 100.0 / total:F2}%" : "0.00%";
        
        return new Markup($"[dim]{pctStr}[/] [yellow]{elapsedStr}[/]/[cyan]{etaStr}[/]");
    }
    
    private static string FormatTime(TimeSpan t)
    {
        if (t.TotalHours >= 1)
            return $"{(int)t.TotalHours}:{t.Minutes:D2}:{t.Seconds:D2}";
        return $"{t.Minutes}:{t.Seconds:D2}";
    }
}

/// <summary>
/// Configuration saved between sessions.
/// </summary>
class AppConfig
{
    public string? LastModelPath { get; set; }
    public string? LastInputPath { get; set; }
    public List<string> InputHistory { get; set; } = new();
}

/// <summary>
/// Command-line tool for AI music detection with interactive and CLI modes.
/// </summary>
class Program
{
    private static readonly string[] SupportedExtensions = { ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac" };
    private static readonly string ConfigFileName = "ai_music_detector_config.json";
    private static AppConfig _config = new();
    private static string _configPath = "";
    
    static int Main(string[] args)
    {
        try
        {
            // Load config from exe directory
            _configPath = Path.Combine(AppContext.BaseDirectory, ConfigFileName);
            LoadConfig();
            
            if (args.Length == 0)
            {
                return RunInteractiveMode();
            }
            else
            {
                return RunCliMode(args);
            }
        }
        catch (Exception ex)
        {
            AnsiConsole.WriteException(ex);
            return 1;
        }
    }

    #region Config Management

    static void LoadConfig()
    {
        try
        {
            if (File.Exists(_configPath))
            {
                var json = File.ReadAllText(_configPath);
                _config = JsonSerializer.Deserialize<AppConfig>(json) ?? new AppConfig();
            }
        }
        catch
        {
            _config = new AppConfig();
        }
    }

    static void SaveConfig()
    {
        try
        {
            var json = JsonSerializer.Serialize(_config, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(_configPath, json);
        }
        catch
        {
            // Ignore save errors
        }
    }

    #endregion

    #region Interactive Mode

    static int RunInteractiveMode()
    {
        DisplayBanner();
        
        // Find or select model
        string? modelPath = SelectModel();
        if (modelPath == null)
        {
            AnsiConsole.MarkupLine("[red]No model selected. Exiting.[/]");
            return 1;
        }
        
        // Load detector
        using var detector = LoadDetector(modelPath);
        if (detector == null)
            return 1;
        
        // Save successful model path
        _config.LastModelPath = modelPath;
        SaveConfig();
        
        DisplayModelInfo(detector, modelPath);
        
        // Main menu loop
        while (true)
        {
            var choice = AnsiConsole.Prompt(
                new SelectionPrompt<string>()
                    .Title("\n[bold]Select option:[/]")
                    .AddChoices("Analyze", "Test mode (with gold labels)", "Change model", "Quit"));
            
            switch (choice)
            {
                case "Analyze":
                    AnalyzeLoop(detector);
                    break;
                case "Test mode (with gold labels)":
                    RunTestModeLoop(detector);
                    break;
                case "Change model":
                    return RunInteractiveMode();
                case "Quit":
                    return 0;
            }
        }
    }

    static void DisplayBanner()
    {
        AnsiConsole.Write(
            new FigletText("AI Music Detector")
                .Color(Color.Cyan1));
        
        AnsiConsole.WriteLine();
    }

    static string? SelectModel()
    {
        // Look for models in common locations
        var modelPaths = new List<string>();
        
        // Check if last model path still exists and is valid
        if (!string.IsNullOrEmpty(_config.LastModelPath) && File.Exists(_config.LastModelPath))
        {
            modelPaths.Add(_config.LastModelPath);
        }
        
        // Check current directory
        var currentDir = Directory.GetCurrentDirectory();
        modelPaths.AddRange(Directory.GetFiles(currentDir, "*.onnx", SearchOption.TopDirectoryOnly));
        
        // Check src/cs/AiMusicDetector/Models
        var modelsDir = Path.Combine(currentDir, "src", "cs", "AiMusicDetector", "Models");
        if (Directory.Exists(modelsDir))
        {
            modelPaths.AddRange(Directory.GetFiles(modelsDir, "*.onnx"));
        }
        
        // Check relative Models folder
        var relativeModelsDir = Path.Combine(currentDir, "Models");
        if (Directory.Exists(relativeModelsDir))
        {
            modelPaths.AddRange(Directory.GetFiles(relativeModelsDir, "*.onnx"));
        }
        
        // Remove duplicates, keeping order (last used first)
        modelPaths = modelPaths.Distinct().ToList();
        
        if (modelPaths.Count > 0)
        {
            modelPaths.Add("Enter custom path...");
            
            var selected = AnsiConsole.Prompt(
                new SelectionPrompt<string>()
                    .Title("[bold]Select a model:[/]")
                    .AddChoices(modelPaths));
            
            if (selected != "Enter custom path...")
            {
                return selected;
            }
        }
        
        // Prompt for custom path using Console.ReadLine for proper arrow key support
        while (true)
        {
            AnsiConsole.Markup("[bold]Enter model path:[/] ");
            var path = System.Console.ReadLine()?.Trim().Trim('"');
            
            if (string.IsNullOrEmpty(path))
            {
                if (!string.IsNullOrEmpty(_config.LastModelPath) && File.Exists(_config.LastModelPath))
                    return _config.LastModelPath;
                continue;
            }
            
            if (!File.Exists(path))
            {
                AnsiConsole.MarkupLine("[red]File not found[/]");
                continue;
            }
            
            if (!path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
            {
                AnsiConsole.MarkupLine("[red]Not an ONNX file[/]");
                continue;
            }
            
            return path;
        }
    }

    static MusicDetector? LoadDetector(string modelPath, int segments = 5)
    {
        try
        {
            return AnsiConsole.Status()
                .Start("Loading model...", ctx =>
                {
                    ctx.Spinner(Spinner.Known.Dots);
                    return MusicDetector.Load(modelPath, new MusicDetectorOptions
                    {
                        NumSegments = segments
                    });
                });
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Error loading model: {ex.Message}[/]");
            return null;
        }
    }

    static void DisplayModelInfo(MusicDetector detector, string modelPath)
    {
        var table = new Table()
            .Border(TableBorder.Rounded)
            .AddColumn("Property")
            .AddColumn("Value");
        
        table.AddRow("Model", Path.GetFileName(modelPath));
        table.AddRow("Type", detector.ModelType.ToString());
        table.AddRow("Sample Rate", $"{detector.SampleRate} Hz");
        table.AddRow("Device", "CPU");
        
        AnsiConsole.Write(table);
    }

    static void AnalyzeLoop(MusicDetector detector)
    {
        AnsiConsole.MarkupLine("[dim]Enter path(s) to files or folders (comma-separated), 'q' to return to menu[/]");
        
        while (true)
        {
            AnsiConsole.Markup("[bold]Path:[/] ");
            var input = System.Console.ReadLine();
            
            if (string.IsNullOrWhiteSpace(input) || input.Trim().ToLower() == "q")
                break;
            
            // Parse paths - handle comma-separated and quoted paths
            var paths = ParsePaths(input);
            
            if (paths.Count == 0)
            {
                AnsiConsole.MarkupLine("[yellow]No valid paths provided.[/]");
                continue;
            }
            
            // Collect all files and count sources
            var allFiles = new List<string>();
            int fileCount = 0;
            int folderCount = 0;
            
            foreach (var path in paths)
            {
                if (Directory.Exists(path))
                {
                    var filesInFolder = GetAudioFiles(path);
                    allFiles.AddRange(filesInFolder);
                    folderCount++;
                }
                else if (File.Exists(path))
                {
                    if (IsAudioFile(path))
                    {
                        allFiles.Add(path);
                        fileCount++;
                    }
                    else
                    {
                        AnsiConsole.MarkupLine($"[yellow]Not an audio file: {Path.GetFileName(path)}[/]");
                    }
                }
                else
                {
                    AnsiConsole.MarkupLine($"[yellow]Not found: {path}[/]");
                }
            }
            
            if (allFiles.Count == 0)
            {
                AnsiConsole.MarkupLine("[yellow]No audio files found.[/]");
                continue;
            }
            
            // Show what we're analyzing
            var sources = new List<string>();
            if (fileCount > 0) sources.Add($"{fileCount} file{(fileCount > 1 ? "s" : "")}");
            if (folderCount > 0) sources.Add($"{folderCount} folder{(folderCount > 1 ? "s" : "")}");
            AnsiConsole.MarkupLine($"[bold]Loaded {allFiles.Count} audio files from {string.Join(" + ", sources)}[/]");
            
            // Save to history
            _config.LastInputPath = paths[0];
            if (!_config.InputHistory.Contains(paths[0]))
            {
                _config.InputHistory.Insert(0, paths[0]);
                if (_config.InputHistory.Count > 20)
                    _config.InputHistory.RemoveAt(_config.InputHistory.Count - 1);
            }
            SaveConfig();
            
            // Analyze
            if (allFiles.Count == 1)
            {
                AnalyzeSingleFile(detector, allFiles[0]);
            }
            else
            {
                AnalyzeMultipleFiles(detector, allFiles.ToArray());
            }
        }
    }

    static bool IsAudioFile(string path)
    {
        return SupportedExtensions.Contains(Path.GetExtension(path).ToLowerInvariant());
    }

    static List<string> ParsePaths(string input)
    {
        var paths = new List<string>();
        var current = "";
        var inQuotes = false;
        
        foreach (var c in input)
        {
            if (c == '"')
            {
                inQuotes = !inQuotes;
            }
            else if (c == ',' && !inQuotes)
            {
                var path = current.Trim().Trim('"');
                if (!string.IsNullOrEmpty(path))
                    paths.Add(path);
                current = "";
            }
            else
            {
                current += c;
            }
        }
        
        var lastPath = current.Trim().Trim('"');
        if (!string.IsNullOrEmpty(lastPath))
            paths.Add(lastPath);
        
        return paths;
    }

    static void AnalyzeSingleFile(MusicDetector detector, string path)
    {
        try
        {
            var result = AnsiConsole.Status()
                .Start($"Analyzing {Path.GetFileName(path)}...", ctx =>
                {
                    ctx.Spinner(Spinner.Known.Dots);
                    return detector.Analyze(path);
                });
            
            DisplayResult(Path.GetFileName(path), result);
        }
        catch (Exception ex)
        {
            AnsiConsole.MarkupLine($"[red]Error: {ex.Message}[/]");
        }
    }

    static void AnalyzeMultipleFiles(MusicDetector detector, string[] files)
    {
        var results = new List<(string file, DetectionResult result)>();
        
        AnsiConsole.Progress()
            .AutoClear(false)
            .HideCompleted(false)
            .Columns(
                new TaskDescriptionColumn(),
                new ProgressBarColumn(),
                new PercentageColumn(),
                new SpinnerColumn())
            .Start(ctx =>
            {
                var task = ctx.AddTask("[green]Analyzing[/]", maxValue: files.Length);
                
                foreach (var file in files)
                {
                    task.Description = $"[green]{TruncateFileName(Path.GetFileName(file), 30)}[/]";
                    try
                    {
                        var result = detector.Analyze(file);
                        results.Add((file, result));
                    }
                    catch (Exception ex)
                    {
                        AnsiConsole.MarkupLine($"[red]Error: {Path.GetFileName(file)}: {ex.Message}[/]");
                    }
                    task.Increment(1);
                }
            });
        
        DisplaySummary(results);
    }
    
    /// <summary>
    /// Batch mode: parallel processing with streaming line-by-line output.
    /// Output format per file: filename line, then "  icon (AI: XX.X%)" line
    /// </summary>
    static void AnalyzeBatch(MusicDetector detector, string[] files, int workers)
    {
        var total = files.Length;
        var processed = 0;
        var outputLock = new object();
        var stopwatch = Stopwatch.StartNew();
        
        System.Console.Error.WriteLine($"Processing {total} files with {workers} workers...");
        System.Console.Error.Flush();
        
        var options = new ParallelOptions { MaxDegreeOfParallelism = workers };
        
        Parallel.ForEach(files.Select((f, i) => (file: f, index: i)), options, item =>
        {
            var (file, index) = item;
            try
            {
                var result = detector.Analyze(file);
                var p = Interlocked.Increment(ref processed);
                
                // Thread-safe output with flush
                lock (outputLock)
                {
                    // Output filename on its own line, then result line
                    System.Console.WriteLine(Path.GetFileName(file));
                    var icon = result.IsAiGenerated ? "🤖 AI-Generated" : "🎵 Real Music";
                    System.Console.WriteLine($"  {icon} (AI: {result.AiProbability * 100:F1}%)");
                    System.Console.Out.Flush();
                    
                    // Progress to stderr
                    System.Console.Error.Write($"\r[{p}/{total}] ");
                    System.Console.Error.Flush();
                }
            }
            catch (Exception ex)
            {
                var p = Interlocked.Increment(ref processed);
                lock (outputLock)
                {
                    System.Console.Error.WriteLine($"ERROR: {Path.GetFileName(file)}: {ex.Message}");
                    System.Console.Error.Flush();
                }
            }
        });
        
        stopwatch.Stop();
        System.Console.Error.WriteLine($"\nCompleted {processed} files in {stopwatch.Elapsed.TotalSeconds:F1}s");
        System.Console.Error.Flush();
    }

    static string TruncateFileName(string name, int maxLen)
    {
        if (name.Length <= maxLen) return name;
        return name[..(maxLen - 3)] + "...";
    }

    static void DisplayResult(string fileName, DetectionResult result)
    {
        var color = result.IsAiGenerated ? "red" : "green";
        var icon = result.IsAiGenerated ? "🤖" : "🎵";
        var aiPercent = result.AiProbability * 100;
        
        AnsiConsole.WriteLine();
        AnsiConsole.MarkupLine($"[bold]{fileName}[/]");
        AnsiConsole.MarkupLine($"  {icon} [{color}]{result.Classification}[/] (AI: {aiPercent:F1}%)");
        AnsiConsole.MarkupLine($"  Duration: {result.AudioDurationSeconds:F1}s | Time: {result.ProcessingTimeMs}ms");
        if (result.SegmentsAnalyzed > 1)
        {
            AnsiConsole.MarkupLine($"  Segments: {result.SegmentsAnalyzed}");
        }
    }

    static void DisplaySummary(List<(string file, DetectionResult result)> results)
    {
        if (results.Count == 0)
            return;
        
        AnsiConsole.WriteLine();
        
        var table = new Table()
            .Border(TableBorder.Rounded)
            .AddColumn("File")
            .AddColumn("AI %", c => c.RightAligned())
            .AddColumn("Result")
            .AddColumn("Time", c => c.RightAligned());
        
        foreach (var (file, result) in results)
        {
            var color = result.IsAiGenerated ? "red" : "green";
            var aiPercent = result.AiProbability * 100;
            var fileName = TruncateFileName(Path.GetFileName(file), 40);
            
            table.AddRow(
                fileName,
                $"[{color}]{aiPercent:F1}%[/]",
                $"[{color}]{result.Classification}[/]",
                $"{result.ProcessingTimeMs}ms");
        }
        
        AnsiConsole.Write(table);
        
        var aiCount = results.Count(r => r.result.IsAiGenerated);
        var realCount = results.Count - aiCount;
        var avgTime = results.Average(r => r.result.ProcessingTimeMs);
        
        AnsiConsole.MarkupLine($"\n[bold]Summary:[/] {realCount} Real, {aiCount} AI-Generated ({results.Count} total, avg {avgTime:F0}ms)");
    }

    static void RunDebugMode(MusicDetector detector, string audioPath)
    {
        AnsiConsole.MarkupLine($"\n[bold cyan]Debug Analysis: {Path.GetFileName(audioPath)}[/]\n");
        
        try
        {
            var debug = detector.AnalyzeWithDebug(audioPath);
            
            // Display summary
            AnsiConsole.MarkupLine($"[bold]Audio:[/]");
            AnsiConsole.MarkupLine($"  Samples: {debug.AudioSampleCount:N0} ({debug.Result.AudioDurationSeconds:F2}s)");
            // Use WriteLine for data with special characters that could be parsed as markup
            AnsiConsole.WriteLine($"  First 10 samples: [{string.Join(", ", debug.AudioSamplesHead.Select(s => s.ToString("F6")))}]");
            
            AnsiConsole.MarkupLine($"\n[bold]Segments ({debug.Segments.Count}):[/]");
            
            var table = new Table()
                .Border(TableBorder.Rounded)
                .AddColumn("Seg")
                .AddColumn("Start")
                .AddColumn("Shape")
                .AddColumn("Cep Mean")
                .AddColumn("Prob");
            
            foreach (var seg in debug.Segments)
            {
                table.AddRow(
                    seg.Index.ToString(),
                    $"{seg.StartSample:N0}",
                    $"({seg.CepstrumShape[0]}, {seg.CepstrumShape[1]})",  // Use () instead of []
                    $"{seg.CepstrumMean:F4}",
                    $"{seg.Probability * 100:F2}%"
                );
            }
            AnsiConsole.Write(table);
            
            AnsiConsole.Write(new Markup("\n[bold]Segment Probabilities:[/] "));
            AnsiConsole.WriteLine($"[{string.Join(", ", debug.SegmentProbabilities.Select(p => $"{p * 100:F2}%"))}]");
            AnsiConsole.MarkupLine($"[bold]Final (median):[/] [cyan]{debug.FinalProbability * 100:F2}%[/]");
            AnsiConsole.MarkupLine($"[bold]Classification:[/] {debug.Result.Classification}");
            
            // Output JSON for comparison with Python
            AnsiConsole.MarkupLine($"\n[bold]JSON for Python comparison:[/]");
            
            var jsonOutput = new
            {
                file = Path.GetFileName(audioPath),
                audio_samples = debug.AudioSampleCount,
                audio_head = debug.AudioSamplesHead,
                segments = debug.Segments.Select(s => new
                {
                    index = s.Index,
                    start_sample = s.StartSample,
                    length_samples = s.LengthSamples,
                    cepstrum_shape = s.CepstrumShape,
                    cepstrum_coeff0_head = s.CepstrumCoeff0Head,
                    cepstrum_frame0_head = s.CepstrumFrame0Head,
                    cepstrum_mean = s.CepstrumMean,
                    probability = s.Probability
                }).ToArray(),
                segment_probabilities = debug.SegmentProbabilities,
                final_probability = debug.FinalProbability,
                classification = debug.Result.Classification
            };
            
            var jsonOptions = new JsonSerializerOptions 
            { 
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
            };
            var json = JsonSerializer.Serialize(jsonOutput, jsonOptions);
            
            AnsiConsole.WriteLine(json);
            
            // Also save to file
            var jsonPath = Path.ChangeExtension(audioPath, ".debug.json");
            File.WriteAllText(jsonPath, json);
            AnsiConsole.MarkupLine($"\n[dim]Debug output saved to: {jsonPath}[/]");
        }
        catch (Exception ex)
        {
            AnsiConsole.WriteException(ex);
        }
    }

    static void RunTestModeLoop(MusicDetector detector)
    {
        AnsiConsole.MarkupLine("[dim]Enter test folder path, 'q' to return to menu[/]");
        
        while (true)
        {
            AnsiConsole.Markup("[bold]Test folder:[/] ");
            var folderPath = System.Console.ReadLine()?.Trim().Trim('"');
            
            if (string.IsNullOrWhiteSpace(folderPath) || folderPath.ToLower() == "q")
                break;
            
            if (!Directory.Exists(folderPath))
            {
                AnsiConsole.MarkupLine("[red]Folder not found.[/]");
                continue;
            }
            
            var goldLabel = AnsiConsole.Prompt(
                new SelectionPrompt<string>()
                    .Title("[bold]Sample type (gold label):[/]")
                    .AddChoices("real", "fake", "mixed"));
            
            RunTestModeInternal(detector, folderPath, goldLabel);
        }
    }

    static void RunTestModeInternal(MusicDetector detector, string folderPath, string goldLabel, int workers = 4)
    {
        var files = GetAudioFiles(folderPath);
        if (files.Length == 0)
        {
            AnsiConsole.MarkupLine("[yellow]No audio files found in folder.[/]");
            return;
        }
        
        var falsePositives = new System.Collections.Concurrent.ConcurrentBag<(string file, float prob)>();
        var falseNegatives = new System.Collections.Concurrent.ConcurrentBag<(string file, float prob)>();
        var recentResults = new System.Collections.Concurrent.ConcurrentQueue<(string file, bool isAi, bool correct, float prob)>();
        var currentFiles = new System.Collections.Concurrent.ConcurrentDictionary<int, string>();
        
        int correct = 0;
        int processed = 0;
        int errors = 0;
        int total = files.Length;
        var stopwatch = Stopwatch.StartNew();
        var updateLock = new object();
        
        AnsiConsole.Live(CreateTestDisplay(0, total, 0, 0, 0, 0, TimeSpan.Zero, TimeSpan.Zero, currentFiles, recentResults, goldLabel))
            .AutoClear(false)
            .Overflow(VerticalOverflow.Ellipsis)
            .Start(ctx =>
            {
                var options = new ParallelOptions { MaxDegreeOfParallelism = workers };
                var lastUpdate = DateTime.MinValue;
                var updateInterval = TimeSpan.FromMilliseconds(100);
                
                Parallel.ForEach(files.Select((f, i) => (f, i)), options, item =>
                {
                    var (file, index) = item;
                    var threadId = Environment.CurrentManagedThreadId;
                    
                    currentFiles[threadId] = Path.GetFileName(file);
                    
                    try
                    {
                        var result = detector.Analyze(file);
                        var p = Interlocked.Increment(ref processed);
                        
                        bool expectedAi = GetExpectedLabel(file, goldLabel);
                        bool predictedAi = result.IsAiGenerated;
                        bool isCorrect = expectedAi == predictedAi;
                        
                        if (isCorrect)
                        {
                            Interlocked.Increment(ref correct);
                        }
                        else if (predictedAi && !expectedAi)
                        {
                            falsePositives.Add((Path.GetFileName(file), result.AiProbability));
                        }
                        else
                        {
                            falseNegatives.Add((Path.GetFileName(file), result.AiProbability));
                        }
                        
                        // Add to recent results (keep last 8)
                        recentResults.Enqueue((Path.GetFileName(file), predictedAi, isCorrect, result.AiProbability));
                        while (recentResults.Count > 8) recentResults.TryDequeue(out _);
                        
                        // Update display with throttling
                        var now = DateTime.UtcNow;
                        if (now - lastUpdate >= updateInterval || p == total)
                        {
                            lock (updateLock)
                            {
                                if (now - lastUpdate >= updateInterval || p == total)
                                {
                                    lastUpdate = now;
                                    var elapsed = stopwatch.Elapsed;
                                    var eta = p > 0 ? TimeSpan.FromTicks(elapsed.Ticks * (total - p) / p) : TimeSpan.Zero;
                                    var c = Volatile.Read(ref correct);
                                    
                                    ctx.UpdateTarget(CreateTestDisplay(p, total, c, falsePositives.Count, falseNegatives.Count, errors, elapsed, eta, currentFiles, recentResults, goldLabel));
                                }
                            }
                        }
                    }
                    catch
                    {
                        Interlocked.Increment(ref errors);
                    }
                    finally
                    {
                        currentFiles.TryRemove(threadId, out _);
                    }
                });
                
                // Final update
                var finalElapsed = stopwatch.Elapsed;
                ctx.UpdateTarget(CreateTestDisplay(processed, total, correct, falsePositives.Count, falseNegatives.Count, errors, finalElapsed, TimeSpan.Zero, currentFiles, recentResults, goldLabel));
            });
        
        stopwatch.Stop();
        
        if (errors > 0)
        {
            AnsiConsole.MarkupLine($"[yellow]{errors} files failed to process[/]");
        }
        
        DisplayTestResults(processed, correct, falsePositives.ToList(), falseNegatives.ToList(), stopwatch.Elapsed);
    }

    static IRenderable CreateTestDisplay(
        int processed, int total, int correct, int fp, int fn, int errors,
        TimeSpan elapsed, TimeSpan eta,
        System.Collections.Concurrent.ConcurrentDictionary<int, string> currentFiles,
        System.Collections.Concurrent.ConcurrentQueue<(string file, bool isAi, bool correct, float prob)> recentResults,
        string goldLabel)
    {
        var rows = new List<IRenderable>();
        
        // Stats panel
        double pct = total > 0 ? (double)processed / total * 100 : 0;
        double accuracy = processed > 0 ? (double)correct / processed * 100 : 0;
        double filesPerSec = elapsed.TotalSeconds > 0 ? processed / elapsed.TotalSeconds : 0;
        
        var elapsedStr = FormatTimeSpan(elapsed);
        var etaStr = FormatTimeSpan(eta);
        
        // Progress bar
        int barWidth = 50;
        int filled = (int)(barWidth * pct / 100);
        var progressBar = new string('━', filled) + new string('─', barWidth - filled);
        
        var statsText = new Markup(
            $"[bold cyan]{processed:N0}[/]/[dim]{total:N0}[/] " +
            $"[green]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]\n" +
            $"[dim]{progressBar}[/] [bold]{pct:F2}%[/]\n\n" +
            $"[bold]Accuracy:[/] [green]{accuracy:F2}%[/]  " +
            $"[bold]FP:[/] [red]{fp}[/]  " +
            $"[bold]FN:[/] [yellow]{fn}[/]  " +
            $"[bold]Errors:[/] [dim]{errors}[/]\n" +
            $"[bold]Speed:[/] [cyan]{filesPerSec:F1}[/] files/sec  " +
            $"[bold]Time:[/] [yellow]{elapsedStr}[/] / [dim]{etaStr}[/]");
        
        var statsPanel = new Panel(statsText)
            .Header($"[bold] Testing {total:N0} files (gold: {goldLabel}) [/]")
            .BorderColor(Color.Blue)
            .Padding(1, 0);
        rows.Add(statsPanel);
        
        // Currently processing files
        var currentList = currentFiles.Values.ToArray();
        if (currentList.Length > 0)
        {
            var currentTable = new Table()
                .Border(TableBorder.None)
                .HideHeaders()
                .AddColumn("File");
            
            foreach (var file in currentList.Take(4))
            {
                var truncated = file.Length > 60 ? file[..57] + "..." : file;
                currentTable.AddRow($"[yellow]⟳[/] [dim]{truncated}[/]");
            }
            
            var currentPanel = new Panel(currentTable)
                .Header("[bold yellow] Processing [/]")
                .BorderColor(Color.Yellow)
                .Padding(1, 0);
            rows.Add(currentPanel);
        }
        
        // Recent results
        var recentList = recentResults.ToArray();
        if (recentList.Length > 0)
        {
            var recentTable = new Table()
                .Border(TableBorder.None)
                .HideHeaders()
                .AddColumn("Icon")
                .AddColumn("File")
                .AddColumn("Result", c => c.RightAligned());
            
            foreach (var (file, isAi, isCorrect, prob) in recentList.Reverse().Take(6))
            {
                var icon = isCorrect ? "[green]✓[/]" : "[red]✗[/]";
                var truncated = file.Length > 45 ? file[..42] + "..." : file;
                var resultColor = isAi ? "red" : "green";
                var resultText = isAi ? "AI" : "Real";
                recentTable.AddRow(icon, $"[dim]{truncated}[/]", $"[{resultColor}]{resultText}[/] [dim]{prob * 100:F0}%[/]");
            }
            
            var recentPanel = new Panel(recentTable)
                .Header("[bold green] Recent [/]")
                .BorderColor(Color.Green)
                .Padding(1, 0);
            rows.Add(recentPanel);
        }
        
        return new Rows(rows);
    }
    
    static string FormatTimeSpan(TimeSpan t)
    {
        if (t.TotalHours >= 1)
            return $"{(int)t.TotalHours}:{t.Minutes:D2}:{t.Seconds:D2}";
        return $"{t.Minutes}:{t.Seconds:D2}";
    }

    static bool GetExpectedLabel(string filePath, string goldLabel)
    {
        return goldLabel switch
        {
            "fake" => true,
            "real" => false,
            "mixed" => DetermineFromFilename(filePath),
            _ => false
        };
    }

    static bool DetermineFromFilename(string filePath)
    {
        var fileName = Path.GetFileName(filePath).ToLowerInvariant();
        if (fileName.Contains("fake") || fileName.Contains("ai_") || fileName.Contains("generated") || fileName.Contains("suno") || fileName.Contains("udio"))
            return true;
        if (fileName.Contains("real") || fileName.Contains("human") || fileName.Contains("original"))
            return false;
        return false;
    }

    static void DisplayTestResults(int total, int correct, List<(string file, float prob)> falsePositives, List<(string file, float prob)> falseNegatives, TimeSpan? elapsed = null)
    {
        AnsiConsole.WriteLine();
        
        var rule = new Rule("[bold]RESULTS[/]").RuleStyle("blue");
        AnsiConsole.Write(rule);
        
        double accuracy = total > 0 ? (double)correct / total * 100 : 0;
        
        var table = new Table()
            .Border(TableBorder.Rounded)
            .AddColumn("Metric")
            .AddColumn("Value");
        
        table.AddRow("Total files", total.ToString());
        table.AddRow("Correct", correct.ToString());
        table.AddRow("Accuracy", $"[bold]{accuracy:F2}%[/]");
        table.AddRow("False Positives", $"[red]{falsePositives.Count}[/] (Real → AI)");
        table.AddRow("False Negatives", $"[yellow]{falseNegatives.Count}[/] (AI → Real)");
        
        if (elapsed.HasValue)
        {
            var e = elapsed.Value;
            var elapsedStr = e.TotalHours >= 1 
                ? $"{(int)e.TotalHours}h {e.Minutes}m {e.Seconds}s"
                : e.TotalMinutes >= 1
                    ? $"{(int)e.TotalMinutes}m {e.Seconds}s"
                    : $"{e.TotalSeconds:F1}s";
            var avgMs = total > 0 ? e.TotalMilliseconds / total : 0;
            table.AddRow("Total time", elapsedStr);
            table.AddRow("Avg per file", $"{avgMs:F0}ms");
        }
        
        AnsiConsole.Write(table);
        
        if (falsePositives.Count > 0)
        {
            AnsiConsole.MarkupLine("\n[bold red]False Positives (Real classified as AI):[/]");
            foreach (var (fp, prob) in falsePositives.OrderByDescending(x => x.prob).Take(10))
            {
                AnsiConsole.MarkupLine($"  - {fp} (AI: {prob * 100:F1}%)");
            }
            if (falsePositives.Count > 10)
            {
                AnsiConsole.MarkupLine($"  ... and {falsePositives.Count - 10} more");
            }
        }
        
        if (falseNegatives.Count > 0)
        {
            AnsiConsole.MarkupLine("\n[bold yellow]False Negatives (AI classified as Real):[/]");
            foreach (var (fn, prob) in falseNegatives.OrderBy(x => x.prob).Take(10))
            {
                AnsiConsole.MarkupLine($"  - {fn} (AI: {prob * 100:F1}%)");
            }
            if (falseNegatives.Count > 10)
            {
                AnsiConsole.MarkupLine($"  ... and {falseNegatives.Count - 10} more");
            }
        }
    }

    #endregion

    #region CLI Mode

    static int RunCliMode(string[] args)
    {
        string? modelPath = null;
        var inputPaths = new List<string>();
        string? testPath = null;
        string? testMode = null;
        string? debugPath = null;
        int segments = 5;
        int workers = 4;
        bool batchMode = false;
        
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--model":
                case "-m":
                    if (i + 1 < args.Length) modelPath = args[++i];
                    break;
                case "--input":
                case "-i":
                    if (i + 1 < args.Length) inputPaths.Add(args[++i]);
                    break;
                case "--test":
                case "-t":
                    if (i + 1 < args.Length) testPath = args[++i];
                    break;
                case "--mode":
                    if (i + 1 < args.Length) testMode = args[++i];
                    break;
                case "--debug":
                case "-d":
                    if (i + 1 < args.Length) debugPath = args[++i];
                    break;
                case "--segments":
                case "-s":
                    if (i + 1 < args.Length && int.TryParse(args[++i], out var s)) segments = s;
                    break;
                case "--workers":
                case "-w":
                    if (i + 1 < args.Length && int.TryParse(args[++i], out var w)) workers = Math.Max(1, w);
                    break;
                case "--batch":
                case "-b":
                    batchMode = true;
                    break;
                case "--help":
                case "-h":
                    PrintUsage();
                    return 0;
                default:
                    if (modelPath == null && args[i].EndsWith(".onnx", StringComparison.OrdinalIgnoreCase) && File.Exists(args[i]))
                    {
                        modelPath = args[i];
                    }
                    else if (File.Exists(args[i]) || Directory.Exists(args[i]))
                    {
                        inputPaths.Add(args[i]);
                    }
                    break;
            }
        }
        
        modelPath ??= _config.LastModelPath;
        if (modelPath == null || !File.Exists(modelPath))
        {
            modelPath = AutoDiscoverModel();
        }
        
        if (modelPath == null)
        {
            AnsiConsole.MarkupLine("[red]Error: No model specified and none found automatically.[/]");
            PrintUsage();
            return 1;
        }
        
        if (!File.Exists(modelPath))
        {
            AnsiConsole.MarkupLine($"[red]Error: Model file not found: {modelPath}[/]");
            return 1;
        }
        
        MusicDetector? detector;
        if (batchMode)
        {
            // Batch mode: plain output, no Spectre.Console spinners
            System.Console.Error.WriteLine($"Loading model: {Path.GetFileName(modelPath)}");
            System.Console.Error.Flush();
            try
            {
                detector = MusicDetector.Load(modelPath, new MusicDetectorOptions { NumSegments = segments });
            }
            catch (Exception ex)
            {
                System.Console.Error.WriteLine($"Error loading model: {ex.Message}");
                return 1;
            }
        }
        else
        {
            detector = LoadDetector(modelPath, segments);
            if (detector == null)
                return 1;
        }
        
        using var _ = detector;
        
        _config.LastModelPath = modelPath;
        SaveConfig();
        
        if (!batchMode)
        {
            DisplayModelInfo(detector, modelPath);
        }
        
        if (debugPath != null)
        {
            if (!File.Exists(debugPath))
            {
                AnsiConsole.MarkupLine($"[red]Error: Debug file not found: {debugPath}[/]");
                return 1;
            }
            
            RunDebugMode(detector, debugPath);
            return 0;
        }
        
        if (testPath != null)
        {
            if (!Directory.Exists(testPath))
            {
                AnsiConsole.MarkupLine($"[red]Error: Test folder not found: {testPath}[/]");
                return 1;
            }
            
            RunTestModeInternal(detector, testPath, testMode ?? "real");
            return 0;
        }
        
        if (inputPaths.Count > 0)
        {
            var allFiles = new List<string>();
            foreach (var inputPath in inputPaths)
            {
                if (Directory.Exists(inputPath))
                {
                    allFiles.AddRange(GetAudioFiles(inputPath));
                }
                else if (File.Exists(inputPath))
                {
                    allFiles.Add(inputPath);
                }
                else
                {
                    AnsiConsole.MarkupLine($"[yellow]Warning: Path not found: {inputPath}[/]");
                }
            }
            
            if (allFiles.Count == 0)
            {
                AnsiConsole.MarkupLine("[yellow]No audio files found.[/]");
                return 1;
            }
            
            if (batchMode)
            {
                AnalyzeBatch(detector, allFiles.ToArray(), workers);
            }
            else if (allFiles.Count == 1)
            {
                AnalyzeSingleFile(detector, allFiles[0]);
            }
            else
            {
                AnalyzeMultipleFiles(detector, allFiles.ToArray());
            }
            
            return 0;
        }
        
        AnsiConsole.MarkupLine("[dim]No input specified, entering interactive mode...[/]\n");
        
        while (true)
        {
            var choice = AnsiConsole.Prompt(
                new SelectionPrompt<string>()
                    .Title("\n[bold]Select option:[/]")
                    .AddChoices("Analyze", "Test mode (with gold labels)", "Quit"));
            
            switch (choice)
            {
                case "Analyze":
                    AnalyzeLoop(detector);
                    break;
                case "Test mode (with gold labels)":
                    RunTestModeLoop(detector);
                    break;
                case "Quit":
                    return 0;
            }
        }
    }

    static string? AutoDiscoverModel()
    {
        var searchDirs = new[]
        {
            Directory.GetCurrentDirectory(),
            Path.Combine(Directory.GetCurrentDirectory(), "Models"),
            Path.Combine(Directory.GetCurrentDirectory(), "src", "cs", "AiMusicDetector", "Models"),
            Path.Combine(AppContext.BaseDirectory, "Models")
        };
        
        foreach (var dir in searchDirs)
        {
            if (Directory.Exists(dir))
            {
                var models = Directory.GetFiles(dir, "*.onnx");
                if (models.Length > 0)
                {
                    var cnnModel = models.FirstOrDefault(m => m.Contains("cnn"));
                    return cnnModel ?? models[0];
                }
            }
        }
        
        return null;
    }

    static void PrintUsage()
    {
        AnsiConsole.MarkupLine("[bold]AI Music Detector - Command Line Tool[/]\n");
        AnsiConsole.MarkupLine("[bold]Usage:[/]");
        AnsiConsole.MarkupLine("  AiMusicDetector.Console [options]");
        AnsiConsole.MarkupLine("  AiMusicDetector.Console <model.onnx> <path> [path2] ...");
        
        AnsiConsole.MarkupLine("\n[bold]Options:[/]");
        AnsiConsole.MarkupLine("  --model, -m <path>     Path to the ONNX model file");
        AnsiConsole.MarkupLine("  --input, -i <path>     Audio file or folder to analyze (can repeat)");
        AnsiConsole.MarkupLine("  --test, -t <path>      Test mode - folder with gold labels");
        AnsiConsole.MarkupLine("  --mode <type>          Gold label type: real, fake, mixed (default: real)");
        AnsiConsole.MarkupLine("  --debug, -d <file>     Debug mode - output detailed analysis for single file");
        AnsiConsole.MarkupLine("  --segments, -s <n>     Number of segments for CNN (default: 5)");
        AnsiConsole.MarkupLine("  --workers, -w <n>      Number of parallel workers (default: 4)");
        AnsiConsole.MarkupLine("  --batch, -b            Batch mode with streaming progress output");
        AnsiConsole.MarkupLine("  --help, -h             Show this help message");
        
        AnsiConsole.MarkupLine("\n[bold]Examples:[/]");
        AnsiConsole.MarkupLine("  AiMusicDetector.Console                                    # Interactive mode");
        AnsiConsole.MarkupLine("  AiMusicDetector.Console song.mp3                           # Auto-find model, analyze file");
        AnsiConsole.MarkupLine("  AiMusicDetector.Console model.onnx song1.mp3 ./music/      # Mix files and folders");
        AnsiConsole.MarkupLine("  AiMusicDetector.Console -t ./testset/ --mode fake          # Test mode");
        AnsiConsole.MarkupLine("  AiMusicDetector.Console -b -w 4 -i ./music/                # Batch mode, 4 workers");
        
        AnsiConsole.MarkupLine("\n[bold]Output:[/]");
        AnsiConsole.MarkupLine("  AI: 0.0%  = Real music (human-created)");
        AnsiConsole.MarkupLine("  AI: 100%  = AI-generated music");
    }

    #endregion

    #region Helpers

    static string[] GetAudioFiles(string folderPath)
    {
        return Directory.GetFiles(folderPath, "*.*", SearchOption.AllDirectories)
            .Where(f => SupportedExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
            .OrderBy(f => f)
            .ToArray();
    }

    #endregion
}
