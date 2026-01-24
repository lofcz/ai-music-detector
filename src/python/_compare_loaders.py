"""Compare audio loading between torchaudio and ffmpeg."""
import numpy as np
import subprocess

audio_path = r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v4.5-all\01326459-437a-4853-960f-fcbbc22094a7.mp3"
target_sr = 16000

# Load with torchaudio
import torchaudio
audio_torch, sr = torchaudio.load(audio_path)
if sr != target_sr:
    audio_torch = torchaudio.functional.resample(audio_torch, sr, target_sr)
if audio_torch.shape[0] > 1:
    audio_torch = audio_torch.mean(dim=0)
audio_torch = audio_torch.numpy()

print(f"torchaudio: {len(audio_torch)} samples")
print(f"  head: {audio_torch[:10]}")

# Load with ffmpeg (matching C# approach)
cmd = [
    'ffmpeg', '-i', audio_path,
    '-f', 'f32le', '-acodec', 'pcm_f32le',
    '-ar', str(target_sr),
    '-af', 'pan=mono|c0=0.5*c0+0.5*c1',
    '-v', 'quiet', '-'
]
result = subprocess.run(cmd, capture_output=True)
audio_ffmpeg = np.frombuffer(result.stdout, dtype=np.float32)

print(f"\nffmpeg: {len(audio_ffmpeg)} samples")
print(f"  head: {audio_ffmpeg[:10]}")

# Compare
print(f"\nDifference:")
print(f"  sample count diff: {len(audio_ffmpeg) - len(audio_torch)}")
min_len = min(len(audio_torch), len(audio_ffmpeg))
if min_len > 0:
    diff = np.abs(audio_torch[:min_len] - audio_ffmpeg[:min_len])
    print(f"  mean abs diff: {diff.mean():.6f}")
    print(f"  max abs diff: {diff.max():.6f}")
    print(f"  first 10 diffs: {(audio_torch[:10] - audio_ffmpeg[:10])}")
