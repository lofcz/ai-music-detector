"""Test different FFmpeg resampling options to match torchaudio."""
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
print(f"torchaudio reference: {len(audio_torch)} samples")
print(f"  head: {audio_torch[:5]}")

def test_ffmpeg(name, extra_args):
    """Test an FFmpeg configuration."""
    cmd = [
        'ffmpeg', '-i', audio_path,
        '-f', 'f32le', '-acodec', 'pcm_f32le',
        '-ar', str(target_sr),
    ] + extra_args + ['-v', 'quiet', '-']
    
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"\n{name}: FAILED")
        return
    
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    min_len = min(len(audio), len(audio_torch))
    diff = np.abs(audio[:min_len] - audio_torch[:min_len])
    
    print(f"\n{name}:")
    print(f"  samples: {len(audio)}, diff from torch: {len(audio) - len(audio_torch)}")
    print(f"  mean abs diff: {diff.mean():.8f}")
    print(f"  max abs diff: {diff.max():.8f}")
    print(f"  head: {audio[:5]}")

# Test various configurations
test_ffmpeg("Default -ac 1", ['-ac', '1'])
test_ffmpeg("Pan filter (C# current)", ['-af', 'pan=mono|c0=0.5*c0+0.5*c1'])
test_ffmpeg("Soxr resampler", ['-af', 'aresample=resampler=soxr', '-ac', '1'])
test_ffmpeg("Soxr + pan", ['-af', 'pan=mono|c0=0.5*c0+0.5*c1,aresample=resampler=soxr'])
test_ffmpeg("Soxr VHQ", ['-af', 'aresample=resampler=soxr:precision=33', '-ac', '1'])
test_ffmpeg("SWR linear", ['-af', 'aresample=filter_type=linear', '-ac', '1'])
test_ffmpeg("SWR cubic", ['-af', 'aresample=filter_type=cubic', '-ac', '1'])

# Try loading at native rate first, then resampling
print("\n\n--- Alternative: Load native then resample with different order ---")
# First convert to mono at native rate, then resample
test_ffmpeg("Mono first, then resample", ['-ac', '1', '-af', f'aresample={target_sr}'])
test_ffmpeg("Mono first, soxr resample", ['-ac', '1', '-af', f'aresample={target_sr}:resampler=soxr'])
