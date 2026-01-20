[![AiMusicDetector](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/lofcz/ai-music-detector)

# AI Music Detector

A model, training/inference scripts, and a library for detecting Suno ≤ 5 and Udio ≤ 1.5 generated music.

## Model Description

This model detects AI-generated music by analyzing spectral artifacts left by neural 
vocoders. These artifacts, called "fakeprints," appear as regularly-spaced peaks in the 
frequency spectrum and are an inherent consequence of the transposed convolution 
(deconvolution) layers used in generative audio models.

The model identifies deterministic mathematical signatures that every sample from a given generator exhibits. 
This is why the accuracy is very high with minimal false positives.

However, this also means the model does not generalize to unseen generators. It is trained 
to detect artifacts from specific versions of Suno and Udio, and will require retraining 
when new versions with different vocoder architectures are released.

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

Evaluated on a held-out test set of 17,866 samples (5,741 real, 12,125 AI-generated).

| Metric | Value |
|--------|-------|
| Accuracy | 99.88% |
| Precision | 0.9985 |
| Recall | 0.9998 |
| F1 Score | 0.9991 |
| False Positive Rate | 0.31% |
| False Negative Rate | 0.02% |

## Limitations

- **Sample Rate Dependent**: Audio must be resampled to 16000 Hz
- **Minimum Duration**: Works best with 10+ seconds of audio
- **Evolving Generators**: Needs retraining on new generations of AI music generators

## Acknowledgements

This implementation is based on the fakeprint detection method proposed by Afchar et al. [1], 
which demonstrates that neural vocoders in generative music models produce characteristic 
frequency-domain artifacts due to their deconvolution architecture.

### References

[1] D. Afchar, G. Meseguer-Brocal, K. Akesbi, and R. Hennequin, "A Fourier Explanation of 
AI-music Artifacts," in *Proc. International Society for Music Information Retrieval 
Conference (ISMIR)*, 2025. Available: https://arxiv.org/abs/2506.19108

## License

MIT License
