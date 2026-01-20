# AI Music Detector

A C# library for detecting AI-generated music using fakeprint analysis. Based on the research paper ["A Fourier Explanation of AI-Music Artifacts"](https://arxiv.org/abs/2506.19108) (ISMIR 2025).

## Overview

This project provides:

1. **Python Training Pipeline** - GPU-accelerated feature extraction and model training
2. **C# Inference Library** - Fast, cross-platform inference with ONNX Runtime

The detector identifies spectral artifacts from deconvolution modules commonly used in AI music generators like Suno, Udio, and others.

## How It Works

AI music generators use neural decoders that produce systematic frequency artifacts ("fakeprints"). This library:

1. Extracts a spectrogram from the audio
2. Computes the average power spectrum
3. Identifies peak residues above the spectral baseline
4. Classifies using a simple logistic regression model

**Accuracy**: 99%+ on Suno/Udio detection with <1% false positive rate.

## Quick Start

### C# Usage

```csharp
using AiMusicDetector;

// Load the detector with your trained model
using var detector = MusicDetector.Load("ai_music_detector.onnx");

// Analyze an audio file
var result = detector.Analyze("song.mp3");

Console.WriteLine($"AI Probability: {result.AiProbability:P1}");
Console.WriteLine($"Classification: {result.Classification}");
Console.WriteLine($"Confidence: {result.Confidence:P0}");
```

### Installation

```bash
# NuGet (once published)
dotnet add package AiMusicDetector

# Or add project reference
dotnet add reference src/AiMusicDetector/AiMusicDetector.csproj
```

## Training Your Own Model

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- CUDA-capable GPU (optional, speeds up feature extraction)
- ~200 GB disk space for datasets

### Step 1: Setup Environment

```bash
cd python

# Windows
setup_env.bat

# Linux/macOS
chmod +x setup_env.sh
./setup_env.sh

# Or manually:
conda env create -f environment.yml
conda activate ai-music-detector
```

### Step 2: Download Datasets

```bash
# Activate environment
conda activate ai-music-detector

# Download FMA (real music) and SONICS (AI-generated)
python download_data.py --dataset all
```

**Datasets:**
- [FMA Medium](https://github.com/mdeff/fma) - 25,000 real music tracks (22 GB)
- [SONICS](https://github.com/awsaf49/sonics) - 49,000+ AI-generated songs from Suno/Udio (~150 GB)

### Step 3: Extract Fakeprints

```bash
# Extract features from real music
python extract_fakeprints.py \
    --input ./data/fma/fma_medium \
    --output ./output/fma_fakeprints.npy \
    --label real

# Extract features from AI-generated music
python extract_fakeprints.py \
    --input ./data/sonics/fake_songs \
    --output ./output/sonics_fakeprints.npy \
    --label fake
```

### Step 4: Train Model

```bash
python train_model.py \
    --real ./output/fma_fakeprints.npy \
    --fake ./output/sonics_fakeprints.npy
```

### Step 5: Export to ONNX

```bash
python export_onnx.py --model ./models
```

The trained model will be saved to `./models/ai_music_detector.onnx`.

## Project Structure

```
ai-music-detector/
├── python/                      # Training pipeline
│   ├── requirements.txt
│   ├── config.yaml
│   ├── download_data.py         # Dataset downloader
│   ├── extract_fakeprints.py    # GPU-accelerated feature extraction
│   ├── train_model.py           # Model training
│   ├── export_onnx.py           # ONNX export
│   └── test_with_samples.py     # Quick validation
│
├── src/
│   ├── AiMusicDetector/         # C# library
│   │   ├── MusicDetector.cs     # Main API
│   │   ├── AudioProcessor.cs    # Audio loading
│   │   ├── FakeprintExtractor.cs# Feature extraction
│   │   └── OnnxInference.cs     # ONNX inference
│   │
│   ├── AiMusicDetector.Console/ # CLI tool
│   └── AiMusicDetector.Tests/   # Unit tests
│
├── reference/                   # Research papers and reference code
└── AiMusicDetector.sln
```

## API Reference

### MusicDetector

```csharp
// Load model
using var detector = MusicDetector.Load("model.onnx");
using var detector = MusicDetector.Load(modelBytes);

// Analyze files
DetectionResult result = detector.Analyze("song.mp3");
float probability = detector.Predict("song.mp3");
bool isAi = detector.IsAiGenerated("song.mp3");

// Analyze samples directly
float[] samples = LoadYourAudio();
var result = detector.Analyze(samples, sampleRate: 44100);

// Batch processing
var results = detector.AnalyzeBatch(new[] { "song1.mp3", "song2.mp3" });
```

### DetectionResult

```csharp
public class DetectionResult
{
    float AiProbability;       // 0.0 (Real) to 1.0 (AI)
    bool IsAiGenerated;        // true if probability > threshold
    float Confidence;          // 0.0 (uncertain) to 1.0 (certain)
    string Classification;     // "Real Music" or "AI-Generated"
    double AudioDurationSeconds;
    long ProcessingTimeMs;
}
```

### Configuration

```csharp
var options = new MusicDetectorOptions
{
    SampleRate = 44100,
    MaxDurationSeconds = 180,
    Threshold = 0.5f,
    UseGpu = false
};

using var detector = MusicDetector.Load("model.onnx", options);
```

## Command Line Tool

```bash
# Build
dotnet build src/AiMusicDetector.Console

# Run
dotnet run --project src/AiMusicDetector.Console -- model.onnx song1.mp3 song2.mp3
```

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 99%+ |
| False Positive Rate | <1% |
| Inference Time | ~50ms per track |
| Model Size | ~100 KB |

## Limitations

- Detection may be less accurate for:
  - Heavily compressed audio
  - Very short clips (<5 seconds)
  - Hybrid content (real + AI)
- New AI generators may not have the same artifacts

## License

- Code: MIT License
- Model: CC BY-NC 4.0 (non-commercial use only)
- Based on research by Deezer (patent pending for commercial use)

## Citation

```bibtex
@inproceedings{afchar2025fourier,
  author    = {Darius Afchar, Gabriel Meseguer‑Brocal, Kamil Akesbi and Romain Hennequin},
  title     = {A Fourier Explanation of AI‑music Artifacts},
  booktitle = {Proceedings of ISMIR},
  year      = {2025}
}
```

## Acknowledgments

- Research paper: [A Fourier Explanation of AI-Music Artifacts](https://arxiv.org/abs/2506.19108)
- Datasets: [FMA](https://github.com/mdeff/fma), [SONICS](https://github.com/awsaf49/sonics)
