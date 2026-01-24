[![AiMusicDetector](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/lofcz/ai-music-detector)

# AI Music Detector

A model, training/inference scripts, and a library for detecting Suno ≤ 5 and Udio ≤ 1.5 generated music.

## Model Description

This model detects AI-generated music by exploiting spectral artifacts inherent to neural vocoders. Most audio generators rely on deconvolution layers to upsample latent representations back to audio sample rates. These layers leave predictable fingerprints in the frequency domain.

### Deconvolution Artifacts

A deconvolution (transposed convolution) with stride $k$ is equivalent to two sequential operations:
1. Zero-upsampling: Insert $k-1$ zeros between each sample
2. Convolution: Apply the learned kernel

The zero-upsampling is equivalent to oversampling a discrete signal. For a signal $s$ sampled at frequency $f_s$, the zero-upsampled version $v$ with stride $k$ satisfies:

$$v = s \cdot \amalg_{1/kf_s}$$

where $\amalg_T$ denotes a Dirac comb with period $T$. From the Fourier duality of Dirac combs:

$$\mathcal{F}[\amalg_T] = \frac{1}{T}\amalg_{1/T}$$

This means the spectrum of $v$ is read up to frequency $kf_s$ instead of $f_s$, causing periodic replication of the original spectrum. The high-energy DC component (bias from activations and layer outputs) gets cloned throughout the frequency range, creating characteristic peaks.

### Peak Locations

For a single deconvolution with stride $k$, peaks appear at frequencies $n \cdot f_s$ for all integers $n \in [0, \lfloor k/2 \rfloor]$.

For $L$ stacked deconvolution layers with strides $\{k_1, k_2, \ldots, k_L\}$, artifacts compound recursively—each layer replicates not just the DC component but all previous peaks. The total number of peaks is:

$$P = \left\lfloor \frac{\prod_{i=1}^{L} k_i}{2} \right\rfloor + 1$$

### Architecture Fingerprinting

These artifacts depend only on the stride configuration, not on training data or learned weights. This has two implications:

1. High accuracy on known architectures: The spectral fingerprint is deterministic and consistent across all outputs from a given generator
2. Requires retraining for new architectures: Different vocoder designs produce different peak patterns, so the model must be updated when generators change their architecture

We extract a _fakeprint_ by computing the average spectrum, subtracting its lower envelope (to isolate peaks from melodic content), and analyzing the 1-8 kHz band where artifacts are most prominent.

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

### In-process FFmpeg decoding (FFmpeg.AutoGen)

For the closest match to Python/torchaudio/FFmpeg decoding, the C# pipeline can decode MP3 using **in-process FFmpeg** (no `ffmpeg.exe`) via `FFmpeg.AutoGen`.

- **Override library path**:
  - Set `AIMUSICDETECTOR_FFMPEG_LIBS` to the folder containing FFmpeg shared libraries (`avcodec`, `avformat`, `swresample`, etc.)

- **Bundled binaries layout (recommended)**:
  - `FFmpeg/bin/win-x64/*.dll`
  - `FFmpeg/bin/win-arm64/*.dll`
  - `FFmpeg/bin/linux-x64/*.so`
  - `FFmpeg/bin/linux-arm64/*.so`
  - `FFmpeg/bin/osx-x64/*.dylib`
  - `FFmpeg/bin/osx-arm64/*.dylib`

The loader also supports the legacy AutoGen example layout on Windows:
- `FFmpeg/bin/x64/*.dll`

If no in-process FFmpeg libraries are found, the code will fall back to the other decoding path(s).

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

### Alternative: CNN Model (Robust to Audio Modifications)

For better robustness against pitch shifts, EQ changes, mastering, and transcoding, 
train the CNN-based model:

```bash
# Train CNN with on-the-fly augmentations
python train_cnn.py \
    --real ./data/fma/fma_medium \
    --fake ./data/sonics/fake_songs

# Export to ONNX
python export_onnx_cnn.py

# Inference
python inference_cnn.py --model ./models/cnn_detector.onnx
```

The CNN model uses CQT spectrograms (log-frequency) which provide pitch-shift 
invariance, and trains with data augmentation including pitch shifting, EQ, 
compression, and noise injection.

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
