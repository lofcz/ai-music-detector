"""Compare sample counts and values between Python and C# for MP3."""
import subprocess
import numpy as np
import torchaudio
import sys
from pathlib import Path

def get_python_samples(mp3_path: str, target_sr: int = 16000, native: bool = False) -> tuple:
    """Load audio with torchaudio (matches FFmpeg behavior)."""
    audio, sr = torchaudio.load(mp3_path)
    orig_sr = sr
    if not native and sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
        sr = target_sr
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0)
    return audio.numpy(), sr, orig_sr

def get_ffmpeg_samples(mp3_path: str, target_sr: int = 16000, native: bool = False) -> tuple:
    """Load audio with FFmpeg command line."""
    if native:
        # Get native sample rate first
        probe = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', 
                               '-show_streams', mp3_path], capture_output=True, text=True)
        import json
        info = json.loads(probe.stdout)
        sr = int(info['streams'][0]['sample_rate'])
        cmd = [
            'ffmpeg', '-i', mp3_path,
            '-f', 'f32le', '-acodec', 'pcm_f32le',
            '-af', 'pan=mono|c0=0.5*c0+0.5*c1',
            '-v', 'quiet', '-'
        ]
    else:
        sr = target_sr
        cmd = [
            'ffmpeg', '-i', mp3_path,
            '-f', 'f32le', '-acodec', 'pcm_f32le',
            '-ar', str(target_sr),
            '-af', 'pan=mono|c0=0.5*c0+0.5*c1',
            '-v', 'quiet', '-'
        ]
    result = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(result.stdout, dtype=np.float32), sr

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python debug_samples.py <mp3_file>")
        sys.exit(1)
    
    mp3_path = sys.argv[1]
    print(f"Analyzing: {mp3_path}\n")
    
    # Get sample counts
    py_samples = get_python_samples(mp3_path)
    ff_samples = get_ffmpeg_samples(mp3_path)
    
    print(f"torchaudio samples: {len(py_samples)}")
    print(f"FFmpeg samples:     {len(ff_samples)}")
    print(f"Difference:         {len(ff_samples) - len(py_samples)}")
    
    # Compare values
    min_len = min(len(py_samples), len(ff_samples))
    if min_len > 0:
        diff = np.abs(py_samples[:min_len] - ff_samples[:min_len])
        print(f"\nSample comparison (first {min_len} samples):")
        print(f"  Mean diff: {diff.mean():.8f}")
        print(f"  Max diff:  {diff.max():.8f}")
        
        # Check if there's an offset
        print("\nChecking for alignment offset...")
        best_offset = 0
        best_corr = -1
        
        for offset in range(-2000, 2000):
            if offset < 0:
                py_start, ff_start = -offset, 0
            else:
                py_start, ff_start = 0, offset
            
            compare_len = min(5000, min_len - abs(offset))
            if compare_len < 1000:
                continue
            
            # Use correlation to find best alignment
            py_seg = py_samples[py_start:py_start+compare_len]
            ff_seg = ff_samples[ff_start:ff_start+compare_len]
            
            # Normalize
            py_norm = (py_seg - py_seg.mean()) / (py_seg.std() + 1e-10)
            ff_norm = (ff_seg - ff_seg.mean()) / (ff_seg.std() + 1e-10)
            
            corr = np.mean(py_norm * ff_norm)
            if corr > best_corr:
                best_corr = corr
                best_offset = offset
        
        print(f"  Best offset: {best_offset} (FFmpeg ahead by this many samples)")
        print(f"  Correlation at best offset: {best_corr:.4f}")
        
        # Show comparison at best offset
        if best_offset < 0:
            py_start, ff_start = -best_offset, 0
        else:
            py_start, ff_start = 0, best_offset
        compare_len = min(10000, min_len - abs(best_offset))
        diff_aligned = np.abs(py_samples[py_start:py_start+compare_len] - ff_samples[ff_start:ff_start+compare_len])
        print(f"  Mean diff at best offset: {diff_aligned.mean():.8f}")
        print(f"  Max diff at best offset:  {diff_aligned.max():.8f}")
    
    # Show first samples
    print("\nFirst 5 samples:")
    print(f"  torchaudio: {py_samples[:5]}")
    print(f"  FFmpeg:     {ff_samples[:5]}")
