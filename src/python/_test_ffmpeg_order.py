"""Test FFmpeg with torchaudio's exact order: resample first, then mono."""
import numpy as np
import subprocess
import torchaudio

audio_path = r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v4.5-all\01326459-437a-4853-960f-fcbbc22094a7.mp3"
target_sr = 16000

# Load with torchaudio (the reference)
audio_torch, sr = torchaudio.load(audio_path)
print(f"Original: {sr}Hz, shape {audio_torch.shape}")
if sr != target_sr:
    audio_torch = torchaudio.functional.resample(audio_torch, sr, target_sr)
    print(f"After resample: shape {audio_torch.shape}")
if audio_torch.shape[0] > 1:
    audio_torch = audio_torch.mean(dim=0)
    print(f"After mono: shape {audio_torch.shape}")
audio_torch = audio_torch.numpy()
print(f"\ntorchaudio reference: {len(audio_torch)} samples")
print(f"  head: {audio_torch[:10]}")

def test_ffmpeg(name, af_filter):
    """Test an FFmpeg configuration."""
    cmd = [
        'ffmpeg', '-i', audio_path,
        '-f', 'f32le', '-acodec', 'pcm_f32le',
        '-af', af_filter,
        '-v', 'quiet', '-'
    ]
    
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"\n{name}: FAILED - {result.stderr.decode()[:200]}")
        return
    
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    min_len = min(len(audio), len(audio_torch))
    diff = np.abs(audio[:min_len] - audio_torch[:min_len])
    
    print(f"\n{name}:")
    print(f"  samples: {len(audio)}, diff from torch: {len(audio) - len(audio_torch)}")
    print(f"  mean abs diff: {diff.mean():.8f}")
    print(f"  max abs diff: {diff.max():.8f}")
    print(f"  head: {audio[:10]}")
    return diff.mean()

# Current C# approach
test_ffmpeg("Current C# (pan+resample)", f"pan=mono|c0=0.5*c0+0.5*c1,aresample={target_sr}")

# torchaudio order: resample first, then mono
test_ffmpeg("Resample first, then mono (default)", f"aresample={target_sr},pan=mono|c0=0.5*c0+0.5*c1")
test_ffmpeg("Resample first (soxr), then mono", f"aresample={target_sr}:resampler=soxr,pan=mono|c0=0.5*c0+0.5*c1")

# Try different soxr precision levels
for prec in [20, 28, 33]:
    test_ffmpeg(f"Soxr precision={prec}, then mono", 
                f"aresample={target_sr}:resampler=soxr:precision={prec},pan=mono|c0=0.5*c0+0.5*c1")

# Try with explicit kaiser window (torchaudio uses kaiser)
test_ffmpeg("SWR kaiser, then mono", f"aresample={target_sr}:filter_type=kaiser,pan=mono|c0=0.5*c0+0.5*c1")
