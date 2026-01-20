"""
Visualize fakeprints from audio files.

Compare the spectral patterns of real vs AI-generated audio to see
the characteristic peaks that indicate AI generation.

Usage:
    python visualize_fakeprint.py audio1.mp3 audio2.mp3
    python visualize_fakeprint.py --input folder/ --output comparison.png
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter1d
import yaml
import warnings

warnings.filterwarnings("ignore", message=".*mpg123.*")
warnings.filterwarnings("ignore", message=".*id3.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Set audio backend for Windows compatibility
if hasattr(torchaudio, 'set_audio_backend'):
    try:
        torchaudio.set_audio_backend("soundfile")
    except:
        pass


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class FakeprintVisualizer:
    def __init__(self, config: dict = None):
        if config is None:
            config = load_config()
        
        audio_cfg = config["audio"]
        fp_cfg = config["fakeprint"]
        
        self.sample_rate = audio_cfg["sample_rate"]
        self.n_fft = audio_cfg["n_fft"]
        self.max_duration = audio_cfg["max_duration"]
        self.freq_min = fp_cfg["freq_min"]
        self.freq_max = fp_cfg["freq_max"]
        self.hull_area = fp_cfg["hull_area"]
        self.max_db = fp_cfg["max_db"]
        self.min_db = fp_cfg["min_db"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            power=2,
            normalized=False
        ).to(self.device)
        
        # Frequency bins
        self.freq_bins = np.linspace(0, self.sample_rate / 2, num=(self.n_fft // 2) + 1)
        self.freq_mask = (self.freq_bins >= self.freq_min) & (self.freq_bins <= self.freq_max)
        self.freq_range = self.freq_bins[self.freq_mask]
        
    def extract_components(self, file_path: str) -> dict:
        """Extract spectrum, hull, and fakeprint from audio."""
        # Normalize path for Windows
        file_path = os.path.normpath(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            audio, sr = torchaudio.load(file_path)
        except Exception as e:
            # Try with different backend
            try:
                import soundfile as sf
                data, sr = sf.read(file_path)
                audio = torch.from_numpy(data.T if data.ndim > 1 else data[np.newaxis, :]).float()
            except Exception as e2:
                raise RuntimeError(f"Failed to load audio: {e}. Fallback error: {e2}")
        
        # Resample
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # Mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Truncate
        max_samples = self.max_duration * self.sample_rate
        if audio.shape[1] > max_samples:
            audio = audio[:, :max_samples]
        
        # STFT
        audio = audio.to(self.device)
        with torch.no_grad():
            spec = self.stft(audio)
        
        spec_db = 10 * torch.log10(torch.clamp(spec, min=1e-10, max=1e6))
        mean_spectrum = spec_db.mean(dim=(0, 2)).cpu().numpy()
        
        # Apply frequency mask
        freq_spectrum = mean_spectrum[self.freq_mask]
        
        # Lower hull
        hull = minimum_filter1d(freq_spectrum, size=self.hull_area, mode='nearest')
        hull = np.clip(hull, self.min_db, None)
        
        # Residue (fakeprint before normalization)
        residue = np.clip(freq_spectrum - hull, 0, None)
        
        # Normalized fakeprint
        residue_clipped = np.clip(residue, 0, self.max_db)
        max_val = np.max(residue_clipped) + 1e-6
        fakeprint = residue_clipped / max_val
        
        return {
            "spectrum": freq_spectrum,
            "hull": hull,
            "residue": residue,
            "fakeprint": fakeprint,
            "freq_khz": self.freq_range / 1000
        }
    
    def plot_single(self, file_path: str, ax: plt.Axes = None, label: str = None):
        """Plot fakeprint for a single file."""
        components = self.extract_components(file_path)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        
        freq_khz = components["freq_khz"]
        fakeprint = components["fakeprint"]
        
        if label is None:
            label = Path(file_path).name
        
        ax.plot(freq_khz, fakeprint, linewidth=0.8, label=label)
        ax.fill_between(freq_khz, 0, fakeprint, alpha=0.3)
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Normalized Amplitude')
        ax.set_xlim(freq_khz[0], freq_khz[-1])
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_comparison(self, file_paths: list, output_path: str = None, 
                        labels: list = None, title: str = None):
        """Plot multiple fakeprints for comparison."""
        n_files = len(file_paths)
        
        if n_files == 1:
            fig, axes = plt.subplots(2, 1, figsize=(14, 8))
            axes = [axes[0], axes[1]]
        else:
            # Single column layout for cleaner comparison
            fig, axes = plt.subplots(n_files, 1, figsize=(14, 2.5*n_files))
            if n_files == 1:
                axes = [axes]
        
        for i, file_path in enumerate(file_paths):
            try:
                components = self.extract_components(file_path)
                freq_khz = components["freq_khz"]
                
                ax = axes[i] if n_files > 1 else axes[0]
                
                # Determine color based on filename
                filename = Path(file_path).name.lower()
                if 'suno' in filename or 'udio' in filename or 'fake' in filename or 'ai' in filename:
                    color = '#d62728'  # Red for AI
                    label_prefix = "[AI] "
                elif 'real' in filename or 'human' in filename or 'fma' in filename:
                    color = '#2ca02c'  # Green for real
                    label_prefix = "[Real] "
                else:
                    color = '#1f77b4'  # Blue for unknown
                    label_prefix = ""
                
                # Plot fakeprint
                label = labels[i] if labels else label_prefix + Path(file_path).name
                ax.plot(freq_khz, components["fakeprint"], 
                       color=color, linewidth=0.6, alpha=0.9)
                ax.fill_between(freq_khz, 0, components["fakeprint"], 
                               alpha=0.3, color=color)
                ax.set_title(label, fontsize=11, fontweight='bold')
                ax.set_ylabel('Amplitude')
                ax.set_xlim(freq_khz[0], freq_khz[-1])
                ax.set_ylim(0, 1.1)
                ax.grid(True, alpha=0.3)
                
                # Only show x-label on bottom plot
                if i == n_files - 1:
                    ax.set_xlabel('Frequency (kHz)')
                
                # For single file, also show spectrum vs hull
                if n_files == 1:
                    ax2 = axes[1]
                    ax2.plot(freq_khz, components["spectrum"], 
                            label='Spectrum', color='blue', alpha=0.7, linewidth=0.8)
                    ax2.plot(freq_khz, components["hull"], 
                            label='Lower Hull', color='red', alpha=0.9, linewidth=1.5)
                    ax2.set_title('Spectrum and Lower Hull')
                    ax2.set_xlabel('Frequency (kHz)')
                    ax2.set_ylabel('Power (dB)')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                axes[i].text(0.5, 0.5, f"Error: {e}", 
                            ha='center', va='center', transform=axes[i].transAxes)
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved to: {output_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize audio fakeprints")
    parser.add_argument("files", nargs="*", help="Audio files to visualize")
    parser.add_argument("--input", "-i", type=str, help="Input folder")
    parser.add_argument("--output", "-o", type=str, help="Output image file")
    parser.add_argument("--limit", type=int, default=20, help="Max files to show")
    parser.add_argument("--title", type=str, default=None, help="Plot title")
    args = parser.parse_args()
    
    visualizer = FakeprintVisualizer()
    
    # Collect files
    files = []
    if args.files:
        files.extend(args.files)
    if args.input:
        input_path = Path(args.input)
        if input_path.is_dir():
            for ext in ['.mp3', '.wav', '.flac', '.ogg', '.m4a']:
                files.extend([str(f) for f in input_path.glob(f'*{ext}')])
        else:
            files.append(str(input_path))
    
    if not files:
        print("No audio files specified.")
        print("Usage: python visualize_fakeprint.py audio1.mp3 audio2.mp3")
        return
    
    # Limit files
    files = files[:args.limit]
    print(f"Visualizing {len(files)} file(s)...")
    
    visualizer.plot_comparison(files, output_path=args.output, title=args.title)


if __name__ == "__main__":
    main()
