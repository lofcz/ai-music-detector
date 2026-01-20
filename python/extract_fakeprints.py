"""
GPU-accelerated fakeprint extraction for AI Music Detector.

Optimized for high GPU utilization with:
- Batched STFT computation
- GPU-based resampling
- Threaded audio prefetching
- Vectorized post-processing

Based on: "A Fourier Explanation of AI-Music Artifacts" (ISMIR 2025)
"""

import os
import argparse
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from scipy import interpolate
from scipy.ndimage import minimum_filter1d
import yaml


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class FakeprintExtractor:
    """
    High-performance GPU-accelerated fakeprint extraction.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 8192,
        freq_min: int = 1000,
        freq_max: int = 8000,
        hull_area: int = 10,
        max_db: float = 5.0,
        min_db: float = -45.0,
        max_duration: int = 300,
        batch_size: int = 128,
        num_workers: int = 6,
        device: torch.device = DEVICE
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.hull_area = hull_area
        self.max_db = max_db
        self.min_db = min_db
        self.max_duration = max_duration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        
        # STFT on GPU
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            power=2,
            normalized=False
        ).to(device)
        
        # Frequency bins and mask
        self.freq_bins = np.linspace(0, sample_rate / 2, num=(n_fft // 2) + 1)
        self.freq_mask = (self.freq_bins >= freq_min) & (self.freq_bins <= freq_max)
        self.freq_range = self.freq_bins[self.freq_mask]
        self.output_dim = len(self.freq_range)
        self.mask_indices = torch.from_numpy(np.where(self.freq_mask)[0]).to(device)
        
        # Cache resamplers for common sample rates
        self.resamplers = {}
        
    def get_resampler(self, orig_sr: int) -> Optional[torchaudio.transforms.Resample]:
        """Get or create a GPU resampler for the given sample rate."""
        if orig_sr == self.sample_rate:
            return None
        if orig_sr not in self.resamplers:
            self.resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_sr, self.sample_rate
            ).to(self.device)
        return self.resamplers[orig_sr]
    
    def load_single_audio(self, file_path: str) -> Optional[tuple]:
        """Load a single audio file (runs in thread pool)."""
        try:
            audio, sr = torchaudio.load(file_path)
            # Convert to mono
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            return (file_path, audio, sr)
        except Exception as e:
            return None
    
    def process_batch_gpu(self, batch: List[tuple]) -> List[tuple]:
        """Process a batch of audio on GPU."""
        results = []
        max_samples = self.max_duration * self.sample_rate
        
        # Resample and pad to same length
        processed = []
        lengths = []
        
        for file_path, audio, sr in batch:
            # Move to GPU and resample
            audio = audio.to(self.device)
            resampler = self.get_resampler(sr)
            if resampler is not None:
                audio = resampler(audio)
            
            # Truncate if needed
            if audio.shape[1] > max_samples:
                audio = audio[:, :max_samples]
            
            processed.append(audio)
            lengths.append(audio.shape[1])
        
        # Pad to max length in batch
        max_len = max(lengths)
        padded = torch.zeros(len(batch), 1, max_len, device=self.device)
        for i, audio in enumerate(processed):
            padded[i, :, :audio.shape[1]] = audio
        
        # Batch STFT - shape: [batch, 1, freq, time]
        with torch.no_grad():
            specs = self.stft(padded.squeeze(1))  # [batch, freq, time]
        
        # Convert to dB and average across time
        specs_db = 10 * torch.log10(torch.clamp(specs, min=1e-10, max=1e6))
        
        # Create time masks for proper averaging (exclude padding)
        for i, (file_path, _, _) in enumerate(batch):
            # Get number of time frames for this audio
            n_frames = (lengths[i] // (self.n_fft // 4)) + 1  # approximate
            n_frames = min(n_frames, specs_db.shape[2])
            
            # Average only over valid frames
            if n_frames > 0:
                mean_spectrum = specs_db[i, :, :n_frames].mean(dim=1)
            else:
                mean_spectrum = specs_db[i, :, :].mean(dim=1)
            
            # Apply frequency mask on GPU
            freq_spectrum = mean_spectrum[self.mask_indices].cpu().numpy()
            
            # Compute fakeprint (CPU - vectorized)
            fakeprint = self.compute_fakeprint_from_spectrum(freq_spectrum)
            if fakeprint is not None:
                results.append((Path(file_path).name, fakeprint))
        
        return results
    
    def compute_fakeprint_from_spectrum(self, freq_spectrum: np.ndarray) -> Optional[np.ndarray]:
        """Compute fakeprint from frequency spectrum (vectorized)."""
        try:
            # Fast lower hull using minimum filter
            hull = minimum_filter1d(freq_spectrum, size=self.hull_area, mode='nearest')
            
            # Clip hull
            hull = np.clip(hull, self.min_db, None)
            
            # Compute residue
            residue = np.clip(freq_spectrum - hull, 0, None)
            
            # Normalize
            residue = np.clip(residue, 0, self.max_db)
            max_val = np.max(residue) + 1e-6
            fakeprint = residue / max_val
            
            return fakeprint.astype(np.float32)
        except Exception:
            return None
    
    def extract_batch(self, file_paths: List[str], show_progress: bool = True) -> dict:
        """Extract fakeprints with high GPU utilization."""
        fakeprints = {}
        
        # Create progress bar
        pbar = tqdm(total=len(file_paths), desc="Extracting fakeprints") if show_progress else None
        
        # Threaded audio loading queue
        audio_queue = Queue(maxsize=self.batch_size * 2)
        load_done = threading.Event()
        
        def audio_loader():
            """Background thread to load audio files."""
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for result in executor.map(self.load_single_audio, file_paths):
                    if result is not None:
                        audio_queue.put(result)
            load_done.set()
        
        # Start loader thread
        loader_thread = threading.Thread(target=audio_loader, daemon=True)
        loader_thread.start()
        
        # Process batches
        batch = []
        processed = 0
        
        while True:
            # Collect batch
            while len(batch) < self.batch_size:
                if audio_queue.empty() and load_done.is_set():
                    break
                try:
                    item = audio_queue.get(timeout=0.1)
                    batch.append(item)
                except:
                    if load_done.is_set():
                        break
            
            if not batch:
                break
            
            # Process batch on GPU
            try:
                results = self.process_batch_gpu(batch)
                for name, fakeprint in results:
                    fakeprints[name] = fakeprint
            except Exception as e:
                print(f"Batch processing error: {e}")
            
            if pbar:
                pbar.update(len(batch))
            processed += len(batch)
            batch = []
        
        if pbar:
            pbar.close()
        
        loader_thread.join(timeout=1.0)
        
        return fakeprints


def find_audio_files(directory: Path) -> list:
    """Find all audio files in a directory."""
    extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
    files = []
    print(f"Scanning for audio files in {directory}...")
    
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in extensions:
                files.append(os.path.join(root, filename))
    
    print(f"Found {len(files)} audio files")
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Extract fakeprints from audio files")
    parser.add_argument("--input", type=str, required=True, help="Input directory")
    parser.add_argument("--output", type=str, required=True, help="Output .npy file")
    parser.add_argument("--label", type=str, choices=["real", "fake"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="Limit files for testing")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for GPU processing")
    parser.add_argument("--workers", type=int, default=4, help="Number of audio loading threads")
    args = parser.parse_args()
    
    config = load_config()
    audio_cfg = config["audio"]
    fp_cfg = config["fakeprint"]
    
    extractor = FakeprintExtractor(
        sample_rate=audio_cfg["sample_rate"],
        n_fft=audio_cfg["n_fft"],
        freq_min=fp_cfg["freq_min"],
        freq_max=fp_cfg["freq_max"],
        hull_area=fp_cfg["hull_area"],
        max_db=fp_cfg["max_db"],
        min_db=fp_cfg["min_db"],
        max_duration=audio_cfg["max_duration"],
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    
    # Find files
    input_dir = Path(args.input)
    audio_files = find_audio_files(input_dir)
    
    if args.limit:
        audio_files = audio_files[:args.limit]
        print(f"Limited to {len(audio_files)} files")
    
    if not audio_files:
        print("No audio files found!")
        return
    
    # Extract
    fakeprints = extractor.extract_batch(audio_files)
    print(f"\nSuccessfully extracted {len(fakeprints)} fakeprints")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    result = {
        "fakeprints": fakeprints,
        "label": args.label,
        "config": {
            "sample_rate": extractor.sample_rate,
            "n_fft": extractor.n_fft,
            "freq_min": extractor.freq_min,
            "freq_max": extractor.freq_max,
            "output_bins": extractor.output_dim
        }
    }
    
    np.save(output_path, result, allow_pickle=True)
    print(f"Saved to: {output_path}")
    print(f"Fakeprint dimension: {extractor.output_dim}")


if __name__ == "__main__":
    main()
