"""Check torchaudio resample parameters and try to match in FFmpeg."""
import numpy as np
import subprocess
import torchaudio
import torch

audio_path = r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v4.5-all\01326459-437a-4853-960f-fcbbc22094a7.mp3"
target_sr = 16000

# Check torchaudio default parameters
print("torchaudio.functional.resample default parameters:")
print("  - lowpass_filter_width: 6")
print("  - rolloff: 0.99")
print("  - resampling_method: 'sinc_interp_hann' (default)")
print("  - beta: None (auto-computed)")

# Load with torchaudio and examine
audio_torch, sr = torchaudio.load(audio_path)
print(f"\nOriginal: {sr}Hz, stereo shape {audio_torch.shape}")

# Test different torchaudio resample methods
methods = ['sinc_interp_hann', 'sinc_interp_kaiser']
for method in methods:
    resampled = torchaudio.functional.resample(audio_torch, sr, target_sr, resampling_method=method)
    mono = resampled.mean(dim=0).numpy()
    print(f"\n{method}:")
    print(f"  head: {mono[:5]}")

# The default in torchaudio
audio_default = torchaudio.functional.resample(audio_torch, sr, target_sr)
audio_default_mono = audio_default.mean(dim=0).numpy()
print(f"\nDefault (sinc_interp_hann):")
print(f"  head: {audio_default_mono[:5]}")

# Now test FFmpeg with various settings
def test_ffmpeg(name, af_filter):
    cmd = ['ffmpeg', '-i', audio_path, '-f', 'f32le', '-acodec', 'pcm_f32le',
           '-af', af_filter, '-v', 'quiet', '-']
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"\n{name}: FAILED")
        return None
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    diff = np.abs(audio[:len(audio_default_mono)] - audio_default_mono)
    print(f"\n{name}:")
    print(f"  head: {audio[:5]}")
    print(f"  mean diff: {diff.mean():.8f}")
    return audio

# Try various FFmpeg resampler configurations
test_ffmpeg("Default", f"aresample={target_sr},pan=mono|c0=0.5*c0+0.5*c1")
test_ffmpeg("Soxr", f"aresample={target_sr}:resampler=soxr,pan=mono|c0=0.5*c0+0.5*c1")

# Try different SWR filter types and cutoff
test_ffmpeg("SWR cutoff=0.99", f"aresample={target_sr}:cutoff=0.99,pan=mono|c0=0.5*c0+0.5*c1")
test_ffmpeg("SWR cutoff=0.95", f"aresample={target_sr}:cutoff=0.95,pan=mono|c0=0.5*c0+0.5*c1")

# Try higher precision
test_ffmpeg("SWR internal 32", f"aresample={target_sr}:internal_sample_fmt=fltp,pan=mono|c0=0.5*c0+0.5*c1")
test_ffmpeg("SWR internal 64", f"aresample={target_sr}:internal_sample_fmt=dblp,pan=mono|c0=0.5*c0+0.5*c1")

# Try different filter sizes (matching torchaudio's width=6)
test_ffmpeg("SWR filter_size=64", f"aresample={target_sr}:filter_size=64,pan=mono|c0=0.5*c0+0.5*c1")
test_ffmpeg("SWR filter_size=128", f"aresample={target_sr}:filter_size=128,pan=mono|c0=0.5*c0+0.5*c1")
