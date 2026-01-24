"""Compare audio decoding: torchaudio vs ffmpeg at native sample rate."""
import numpy as np
import subprocess
import torchaudio

audio_path = r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v4.5-all\01326459-437a-4853-960f-fcbbc22094a7.mp3"

# Load with torchaudio at native rate (no resampling)
audio_torch, sr = torchaudio.load(audio_path)
print(f"torchaudio: {sr}Hz, shape {audio_torch.shape}")
print(f"  samples: {audio_torch.shape[1]}")

# Load with ffmpeg at native rate (matching C# approach)
cmd = [
    'ffmpeg', '-i', audio_path,
    '-f', 'f32le', '-acodec', 'pcm_f32le',
    '-v', 'quiet', '-'
]
result = subprocess.run(cmd, capture_output=True)
audio_ffmpeg = np.frombuffer(result.stdout, dtype=np.float32)

# FFmpeg output is interleaved stereo
channels = audio_torch.shape[0]
if channels == 2:
    ffmpeg_samples = len(audio_ffmpeg) // 2
    left = audio_ffmpeg[0::2]
    right = audio_ffmpeg[1::2]
    print(f"\nffmpeg: {sr}Hz (assumed), stereo")
    print(f"  samples per channel: {ffmpeg_samples}")
else:
    ffmpeg_samples = len(audio_ffmpeg)
    print(f"\nffmpeg: {sr}Hz (assumed), mono")
    print(f"  samples: {ffmpeg_samples}")

# Compare sample counts
torch_samples = audio_torch.shape[1]
print(f"\nSample count difference: {ffmpeg_samples - torch_samples}")

# Compare actual sample values (left channel)
if channels == 2:
    torch_left = audio_torch[0].numpy()
    min_len = min(len(torch_left), len(left))
    
    # Check alignment by finding offset
    print(f"\nComparing first {min_len} samples of left channel...")
    
    # Direct comparison
    diff = np.abs(torch_left[:min_len] - left[:min_len])
    print(f"  Mean diff: {diff.mean():.8f}")
    print(f"  Max diff: {diff.max():.8f}")
    
    # Check if there's an offset
    print("\nChecking for sample offset (torchaudio may skip encoder delay)...")
    best_offset = 0
    best_diff = float('inf')
    for offset in range(-2000, 2000, 100):
        if offset < 0:
            t_start, f_start = -offset, 0
        else:
            t_start, f_start = 0, offset
        
        compare_len = min(10000, min_len - abs(offset))
        if compare_len < 1000:
            continue
            
        d = np.abs(torch_left[t_start:t_start+compare_len] - left[f_start:f_start+compare_len]).mean()
        if d < best_diff:
            best_diff = d
            best_offset = offset
    
    print(f"  Best offset: {best_offset} samples (ffmpeg ahead by this much)")
    print(f"  Mean diff at best offset: {best_diff:.8f}")
    
    # Show first few samples
    print("\nFirst 10 samples comparison:")
    print("  torchaudio:", torch_left[:10])
    print("  ffmpeg:    ", left[:10])
