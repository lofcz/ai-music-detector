"""Analyze where the differences between FFmpeg methods and torchaudio occur."""
import numpy as np
import subprocess
import torchaudio

audio_path = r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v4.5-all\01326459-437a-4853-960f-fcbbc22094a7.mp3"
target_sr = 16000

# Load with torchaudio (the reference)
audio_torch, sr = torchaudio.load(audio_path)
if sr != target_sr:
    audio_torch = torchaudio.functional.resample(audio_torch, sr, target_sr)
if audio_torch.shape[0] > 1:
    audio_torch = audio_torch.mean(dim=0)
audio_torch = audio_torch.numpy()

def load_ffmpeg(af_filter):
    cmd = ['ffmpeg', '-i', audio_path, '-f', 'f32le', '-acodec', 'pcm_f32le',
           '-af', af_filter, '-v', 'quiet', '-']
    result = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(result.stdout, dtype=np.float32)

# Current C# approach
audio_current = load_ffmpeg(f"pan=mono|c0=0.5*c0+0.5*c1,aresample={target_sr}")

# Soxr approach
audio_soxr = load_ffmpeg(f"aresample={target_sr}:resampler=soxr,pan=mono|c0=0.5*c0+0.5*c1")

# Analyze differences in segments
segment_samples = 160000  # 10 seconds at 16kHz (same as model segments)
n_segments = 5
skip_samples = 80000  # 5 seconds

# Calculate segment positions (same as model)
total_len = len(audio_torch)
usable_start = skip_samples
usable_end = total_len - skip_samples
usable_len = usable_end - usable_start
available = usable_len - segment_samples
step = available / (n_segments - 1)
positions = [usable_start + int(i * step) for i in range(n_segments)]

print("Segment-by-segment analysis (matching model segments):")
print("=" * 70)
for i, start in enumerate(positions):
    end = start + segment_samples
    seg_torch = audio_torch[start:end]
    seg_current = audio_current[start:end]
    seg_soxr = audio_soxr[start:end]
    
    diff_current = np.abs(seg_torch - seg_current)
    diff_soxr = np.abs(seg_torch - seg_soxr)
    
    print(f"\nSegment {i}: samples {start:,} - {end:,}")
    print(f"  Current C#:  mean={diff_current.mean():.6f}, max={diff_current.max():.6f}")
    print(f"  Soxr:        mean={diff_soxr.mean():.6f}, max={diff_soxr.max():.6f}")
    print(f"  Winner: {'Soxr' if diff_soxr.mean() < diff_current.mean() else 'Current'}")

# Overall comparison
print("\n\nOverall comparison:")
print("=" * 70)
diff_current_all = np.abs(audio_torch - audio_current)
diff_soxr_all = np.abs(audio_torch - audio_soxr)
print(f"Current C#:  mean={diff_current_all.mean():.6f}, max={diff_current_all.max():.6f}")
print(f"Soxr:        mean={diff_soxr_all.mean():.6f}, max={diff_soxr_all.max():.6f}")
