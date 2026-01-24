"""Compare C# and Python audio loading results sample-by-sample."""
import numpy as np
import subprocess
import torchaudio
import json
import sys

audio_path = r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v4.5-all\01326459-437a-4853-960f-fcbbc22094a7.mp3"
target_sr = 16000

# Python/torchaudio reference
print("Loading with torchaudio (Python reference)...")
audio_torch, sr = torchaudio.load(audio_path)
print(f"  Original: {sr}Hz, shape {audio_torch.shape}")

if sr != target_sr:
    audio_torch = torchaudio.functional.resample(audio_torch, sr, target_sr)
    print(f"  After resample: shape {audio_torch.shape}")

if audio_torch.shape[0] > 1:
    audio_torch = audio_torch.mean(dim=0)
    print(f"  After mono: shape {audio_torch.shape}")

audio_py = audio_torch.numpy()
print(f"  Final: {len(audio_py)} samples")
print(f"  First 10 samples: {audio_py[:10]}")

# Save Python reference for C# comparison
np.savetxt("py_audio_samples.txt", audio_py[:1000], fmt="%.10f")
print(f"\nSaved first 1000 samples to py_audio_samples.txt")

# Now we need to add audio export to C# to compare
print("\n" + "="*60)
print("To compare, run C# with --export-audio flag and check samples")
