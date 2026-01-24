"""Test if C# SincResampler matches torchaudio exactly.

This creates test vectors and compares against torchaudio.
"""
import numpy as np
import torch
import torchaudio
import json

# Test parameters - 44100 -> 16000 (common case)
orig_freq = 44100
new_freq = 16000
lowpass_filter_width = 6
rolloff = 0.99

# Create test signal: Use exactly 100 samples to match C# test
np.random.seed(42)
test_input_100 = np.random.randn(100).astype(np.float32)

def resample_torch(audio, orig_freq, new_freq):
    """Resample using torchaudio (reference)."""
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # [1, samples]
    resampled = torchaudio.functional.resample(
        audio_tensor, orig_freq, new_freq,
        lowpass_filter_width=lowpass_filter_width,
        rolloff=rolloff,
        resampling_method='sinc_interp_hann'
    )
    return resampled.squeeze(0).numpy()

# Test with 100 samples
print("Testing resampling 44100 -> 16000 with 100 input samples")
print("=" * 60)

torch_out_100 = resample_torch(test_input_100, orig_freq, new_freq)
print(f"Input: {len(test_input_100)} samples")
print(f"Output: {len(torch_out_100)} samples")
print(f"\nAll output samples (torchaudio):")
for i, v in enumerate(torch_out_100):
    print(f"  [{i}] = {v:.10f}")

# Save test data for C# verification - use exactly the same input/output
test_data = {
    "orig_freq": orig_freq,
    "new_freq": new_freq,
    "lowpass_filter_width": lowpass_filter_width,
    "rolloff": rolloff,
    "input": test_input_100.tolist(),  # Exactly 100 samples
    "expected_output": torch_out_100.tolist(),  # All output samples from those 100
}

with open("resample_test_data.json", "w") as f:
    json.dump(test_data, f, indent=2)

print(f"\nSaved test data to resample_test_data.json")

# Also test with real audio file
print("\n" + "=" * 60)
print("Testing with real audio file:")
audio_path = r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v4.5-all\01326459-437a-4853-960f-fcbbc22094a7.mp3"

try:
    audio, sr = torchaudio.load(audio_path)
    print(f"Loaded: {sr}Hz, shape {audio.shape}")
    
    # Resample
    audio_resampled = torchaudio.functional.resample(
        audio, sr, new_freq,
        lowpass_filter_width=lowpass_filter_width,
        rolloff=rolloff,
        resampling_method='sinc_interp_hann'
    )
    print(f"After resample: shape {audio_resampled.shape}")
    
    # Convert to mono
    audio_mono = audio_resampled.mean(dim=0).numpy()
    print(f"After mono: {len(audio_mono)} samples")
    
    print(f"\nFirst 20 samples after full processing:")
    for i, v in enumerate(audio_mono[:20]):
        print(f"  [{i}] = {v:.10f}")
        
    # Save for C# comparison
    real_test_data = {
        "file": audio_path,
        "orig_freq": sr,
        "new_freq": new_freq,
        "first_20_samples": audio_mono[:20].tolist(),
        "first_1000_samples": audio_mono[:1000].tolist(),
    }
    with open("real_audio_test_data.json", "w") as f:
        json.dump(real_test_data, f, indent=2)
    print(f"\nSaved real audio test data to real_audio_test_data.json")
    
except Exception as e:
    print(f"Error: {e}")
