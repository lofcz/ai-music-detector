"""Find the best FFmpeg configuration to match torchaudio."""
import numpy as np
import subprocess
import torchaudio

audio_path = r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v4.5-all\01326459-437a-4853-960f-fcbbc22094a7.mp3"
target_sr = 16000

# Reference: torchaudio
audio_torch, sr = torchaudio.load(audio_path)
audio_torch = torchaudio.functional.resample(audio_torch, sr, target_sr)
audio_torch = audio_torch.mean(dim=0).numpy()
print(f"torchaudio reference head: {audio_torch[:5]}")

def test_ffmpeg(name, af_filter):
    cmd = ['ffmpeg', '-i', audio_path, '-f', 'f32le', '-acodec', 'pcm_f32le',
           '-af', af_filter, '-v', 'quiet', '-']
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"{name}: FAILED")
        return float('inf')
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    min_len = min(len(audio), len(audio_torch))
    diff = np.abs(audio[:min_len] - audio_torch[:min_len])
    print(f"{name}: mean={diff.mean():.8f}, max={diff.max():.6f}, head[0]={audio[0]:.8f}")
    return diff.mean()

print("\nTesting FFmpeg configurations:")
print("=" * 80)

# Current best and variations
configs = [
    ("Current C#", f"pan=mono|c0=0.5*c0+0.5*c1,aresample={target_sr}"),
    ("cutoff=0.99", f"aresample={target_sr}:cutoff=0.99,pan=mono|c0=0.5*c0+0.5*c1"),
    ("cutoff=0.99 mono first", f"pan=mono|c0=0.5*c0+0.5*c1,aresample={target_sr}:cutoff=0.99"),
    ("cutoff=0.995", f"aresample={target_sr}:cutoff=0.995,pan=mono|c0=0.5*c0+0.5*c1"),
    ("cutoff=0.98", f"aresample={target_sr}:cutoff=0.98,pan=mono|c0=0.5*c0+0.5*c1"),
    ("cutoff=1.0", f"aresample={target_sr}:cutoff=1.0,pan=mono|c0=0.5*c0+0.5*c1"),
    ("phase=linear cutoff=0.99", f"aresample={target_sr}:cutoff=0.99:phase_shift=0,pan=mono|c0=0.5*c0+0.5*c1"),
    ("soxr cutoff=0.99", f"aresample={target_sr}:resampler=soxr:cutoff=0.99,pan=mono|c0=0.5*c0+0.5*c1"),
]

results = []
for name, cfg in configs:
    diff = test_ffmpeg(name, cfg)
    results.append((name, diff))

print("\n" + "=" * 80)
print("Ranked results (best first):")
for name, diff in sorted(results, key=lambda x: x[1]):
    print(f"  {diff:.8f}: {name}")
