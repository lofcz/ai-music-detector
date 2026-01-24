"""
Extract CQT-Cepstrum features for AI Music Detection.

Computes pitch-shift invariant features by:
1. CQT spectrogram (log-frequency, covers 500Hz-8kHz)
2. Log magnitude
3. DCT along frequency axis -> cepstral coefficients

Each segment produces a [n_coeffs x n_frames] cepstrum spectrogram.

Usage:
    python extract_cqt_features.py --input ./data/fma/fma_medium --output ./output/fma_cqt.pt --label real
    python extract_cqt_features.py --input ./data/sonics/fake_songs --output ./output/sonics_cqt.pt --label fake
"""

import os
import sys
import signal
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from multiprocessing import Pool, cpu_count
import numpy as np
import torch
from scipy.fft import dct
from tqdm import tqdm
import yaml

# Attempt to import librosa for CQT
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("WARNING: librosa not found. Install with: pip install librosa")


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def find_ffmpeg() -> Optional[str]:
    """Find ffmpeg executable."""
    for cmd in ['ffmpeg', 'ffmpeg.exe']:
        try:
            result = subprocess.run([cmd, '-version'], capture_output=True, timeout=5)
            if result.returncode == 0:
                return cmd
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    
    for path in [
        r'C:\ffmpeg\bin\ffmpeg.exe',
        r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
        os.path.expanduser('~/ffmpeg/bin/ffmpeg'),
    ]:
        if os.path.isfile(path):
            return path
    
    return None


def load_audio_ffmpeg(file_path: str, target_sr: int, ffmpeg_path: str) -> Optional[np.ndarray]:
    """Load audio using ffmpeg - silent and robust."""
    try:
        cmd = [
            ffmpeg_path,
            '-i', file_path,
            '-f', 'f32le',
            '-acodec', 'pcm_f32le',
            '-ar', str(target_sr),
            '-ac', '1',
            '-v', 'quiet',
            '-'
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        
        if result.returncode != 0:
            return None
        
        audio = np.frombuffer(result.stdout, dtype=np.float32)
        return audio if len(audio) > 0 else None
        
    except Exception:
        return None


# Global config for multiprocessing
_WORKER_CONFIG = {}


def init_worker(config: dict):
    """Initialize worker with config."""
    global _WORKER_CONFIG
    _WORKER_CONFIG = config
    # Ignore SIGINT in workers - let main process handle it
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def extract_cqt_cepstrum(
    audio: np.ndarray,
    sr: int,
    fmin: float,
    n_bins: int,
    bins_per_octave: int,
    hop_length: int,
    n_coeffs: int
) -> np.ndarray:
    """
    Extract CQT-cepstrum features from audio segment.
    
    Returns:
        cepstrum: [n_coeffs, n_frames] array
    """
    # Compute CQT
    cqt = librosa.cqt(
        audio,
        sr=sr,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        hop_length=hop_length
    )
    
    # Log magnitude
    log_cqt = np.log(np.abs(cqt) + 1e-6)
    
    # DCT along frequency axis for each time frame
    # This gives cepstral coefficients that capture periodicity in log-frequency
    cepstrum = dct(log_cqt, type=2, axis=0, norm='ortho')
    
    # Keep first n_coeffs (like MFCCs)
    cepstrum = cepstrum[:n_coeffs, :]
    
    return cepstrum.astype(np.float32)


def extract_segments(audio: np.ndarray, segment_samples: int, n_segments: int = 3) -> List[np.ndarray]:
    """
    Extract n_segments evenly spread across the audio.
    Each segment is returned as a separate sample for training.
    """
    segments = []
    
    if len(audio) <= segment_samples:
        # Pad short audio
        padded = np.zeros(segment_samples, dtype=np.float32)
        padded[:len(audio)] = audio
        segments.append(padded)
    else:
        # Spread segments evenly
        available = len(audio) - segment_samples
        
        if n_segments == 1:
            positions = [available // 2]
        else:
            step = available / (n_segments - 1)
            positions = [int(i * step) for i in range(n_segments)]
        
        for start in positions:
            start = max(0, min(start, len(audio) - segment_samples))
            segments.append(audio[start:start + segment_samples])
    
    return segments


def process_file(file_path: str) -> Optional[Tuple[str, List[np.ndarray]]]:
    """
    Process a single audio file.
    Returns (filename, list_of_cepstrum_features) or None on failure.
    """
    global _WORKER_CONFIG
    
    sr = _WORKER_CONFIG['sample_rate']
    segment_samples = _WORKER_CONFIG['segment_samples']
    n_segments = _WORKER_CONFIG['n_segments']
    ffmpeg_path = _WORKER_CONFIG['ffmpeg_path']
    fmin = _WORKER_CONFIG['fmin']
    n_bins = _WORKER_CONFIG['n_bins']
    bins_per_octave = _WORKER_CONFIG['bins_per_octave']
    hop_length = _WORKER_CONFIG['hop_length']
    n_coeffs = _WORKER_CONFIG['n_coeffs']
    
    # Load audio
    audio = load_audio_ffmpeg(file_path, sr, ffmpeg_path)
    if audio is None or len(audio) < sr:  # At least 1 second
        return None
    
    # Extract segments
    segments = extract_segments(audio, segment_samples, n_segments)
    if not segments:
        return None
    
    # Extract cepstrum for each segment
    cepstrum_features = []
    for segment in segments:
        try:
            cepstrum = extract_cqt_cepstrum(
                segment, sr, fmin, n_bins, bins_per_octave, hop_length, n_coeffs
            )
            cepstrum_features.append(cepstrum)
        except Exception as e:
            continue
    
    if not cepstrum_features:
        return None
    
    return (Path(file_path).name, cepstrum_features)


def find_audio_files(directory: Path) -> List[str]:
    """Find all audio files in directory."""
    extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
    files = []
    print(f"Scanning for audio files in {directory}...", flush=True)
    
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in extensions:
                files.append(os.path.join(root, filename))
    
    print(f"Found {len(files)} audio files", flush=True)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Extract CQT-cepstrum features")
    parser.add_argument("--input", type=str, required=True, help="Input directory")
    parser.add_argument("--output", type=str, required=True, help="Output .pt file")
    parser.add_argument("--label", type=str, choices=["real", "fake"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="Limit files for testing")
    parser.add_argument("--workers", type=int, default=None, help="CPU workers")
    parser.add_argument("--n-segments", type=int, default=3, help="Segments per file")
    args = parser.parse_args()
    
    if not HAS_LIBROSA:
        print("ERROR: librosa is required. Install with: pip install librosa")
        sys.exit(1)
    
    # Find ffmpeg
    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path is None:
        print("ERROR: ffmpeg not found!")
        sys.exit(1)
    print(f"Using ffmpeg: {ffmpeg_path}", flush=True)
    
    # Load config
    config = load_config()
    audio_cfg = config["audio"]
    cqt_cfg = config["cqt"]
    cepstrum_cfg = config["cepstrum"]
    
    sample_rate = audio_cfg["sample_rate"]
    segment_seconds = cepstrum_cfg["segment_seconds"]
    segment_samples = int(segment_seconds * sample_rate)
    
    # Worker config - all parameters that affect feature extraction
    worker_config = {
        'sample_rate': sample_rate,
        'segment_samples': segment_samples,
        'n_segments': args.n_segments,
        'ffmpeg_path': ffmpeg_path,
        'fmin': cqt_cfg['fmin'],
        'n_bins': cqt_cfg['n_bins'],
        'bins_per_octave': cqt_cfg['bins_per_octave'],
        'hop_length': cqt_cfg['hop_length'],
        'n_coeffs': cepstrum_cfg['n_coeffs'],
    }
    
    print(f"\nFeature extraction config:")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  CQT: fmin={cqt_cfg['fmin']}Hz, n_bins={cqt_cfg['n_bins']}, bins/oct={cqt_cfg['bins_per_octave']}")
    print(f"  Frequency range: {cqt_cfg['fmin']}Hz - {cqt_cfg['fmin'] * 2**(cqt_cfg['n_bins']/cqt_cfg['bins_per_octave']):.0f}Hz")
    print(f"  Cepstrum coefficients: {cepstrum_cfg['n_coeffs']}")
    print(f"  Segment: {segment_seconds}s ({segment_samples} samples)")
    print(f"  Segments per file: {args.n_segments}")
    
    # Find files
    input_dir = Path(args.input)
    audio_files = find_audio_files(input_dir)
    
    if args.limit:
        audio_files = audio_files[:args.limit]
        print(f"Limited to {len(audio_files)} files", flush=True)
    
    if not audio_files:
        print("No audio files found!")
        return
    
    # Output path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check for checkpoint
    checkpoint_path = output_path.with_suffix('.checkpoint.pt')
    all_features: List[Tuple[str, torch.Tensor]] = []
    processed_files = set()
    
    if checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}", flush=True)
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        all_features = checkpoint.get("features", [])
        processed_files = set(checkpoint.get("processed_files", []))
        print(f"Loaded {len(all_features)} features from {len(processed_files)} files", flush=True)
    
    # Filter already processed
    files_to_process = [f for f in audio_files if Path(f).name not in processed_files]
    print(f"Files to process: {len(files_to_process)}", flush=True)
    
    if files_to_process:
        num_workers = args.workers or 4
        print(f"Using {num_workers} workers", flush=True)
        
        # Process files
        failed = 0
        chunk_size = 2000
        
        interrupted = False
        try:
            for chunk_start in range(0, len(files_to_process), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(files_to_process))
                chunk_files = files_to_process[chunk_start:chunk_end]
                
                print(f"\nProcessing files {chunk_start + 1}-{chunk_end} of {len(files_to_process)}...", flush=True)
                
                with Pool(num_workers, initializer=init_worker, initargs=(worker_config,)) as pool:
                    results = list(tqdm(
                        pool.imap(process_file, chunk_files, chunksize=16),
                        total=len(chunk_files),
                        desc="Extracting",
                        unit="file",
                        ncols=100
                    ))
                
                # Collect results
                for result in results:
                    if result is not None:
                        filename, cepstrum_list = result
                        for seg_idx, cepstrum in enumerate(cepstrum_list):
                            sample_id = f"{filename}__seg{seg_idx}"
                            all_features.append((sample_id, torch.from_numpy(cepstrum)))
                        processed_files.add(filename)
                    else:
                        failed += 1
                
                print(f"Total: {len(all_features)} segments, {failed} files failed", flush=True)
                
                # Save checkpoint
                torch.save({
                    "features": all_features,
                    "processed_files": list(processed_files),
                    "label": args.label
                }, checkpoint_path)
        
        except KeyboardInterrupt:
            interrupted = True
            print("\n\nInterrupted! Saving checkpoint...", flush=True)
            # Save checkpoint with current progress
            torch.save({
                "features": all_features,
                "processed_files": list(processed_files),
                "label": args.label
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}", flush=True)
            print(f"Progress: {len(all_features)} segments from {len(processed_files)} files", flush=True)
            print("Run the same command to resume.", flush=True)
            sys.exit(1)
    
    print(f"\n{'='*60}", flush=True)
    print(f"Total: {len(all_features)} segment features from {len(processed_files)} files", flush=True)
    
    if not all_features:
        print("No features extracted!")
        return
    
    # Get shape info
    shape = all_features[0][1].shape
    print(f"Feature shape: {list(shape)} (n_coeffs x n_frames)", flush=True)
    
    # Convert to dict
    features_dict = {sample_id: feat for sample_id, feat in all_features}
    
    # Save
    result = {
        "features": features_dict,
        "label": args.label,
        "n_segments_per_file": args.n_segments,
        "n_files": len(processed_files),
        "n_samples": len(all_features),
        "config": {
            "sample_rate": sample_rate,
            "fmin": cqt_cfg['fmin'],
            "n_bins": cqt_cfg['n_bins'],
            "bins_per_octave": cqt_cfg['bins_per_octave'],
            "hop_length": cqt_cfg['hop_length'],
            "n_coeffs": cepstrum_cfg['n_coeffs'],
            "segment_seconds": segment_seconds,
            "feature_shape": list(shape)
        }
    }
    
    torch.save(result, output_path)
    print(f"Saved to: {output_path}", flush=True)
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB", flush=True)
    
    # Remove checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint removed", flush=True)


if __name__ == "__main__":
    main()
