"""
AI Music Detector - Inference Script

Usage:
    python inference.py                         # Interactive mode
    python inference.py --input ./folder        # Batch mode
    python inference.py --input ./song.mp3      # Single file
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torchaudio
from scipy.ndimage import minimum_filter1d
import onnxruntime as ort
import yaml

warnings.filterwarnings("ignore", message=".*mpg123.*")
warnings.filterwarnings("ignore", message=".*id3.*")


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class AIDetector:
    """AI Music Detector using ONNX model."""
    
    def __init__(self, model_path: str = None, config: dict = None):
        if config is None:
            config = load_config()
        
        self.config = config
        audio_cfg = config["audio"]
        fp_cfg = config["fakeprint"]
        
        # Audio settings
        self.sample_rate = audio_cfg["sample_rate"]
        self.n_fft = audio_cfg["n_fft"]
        self.max_duration = audio_cfg["max_duration"]
        
        # Fakeprint settings
        self.freq_min = fp_cfg["freq_min"]
        self.freq_max = fp_cfg["freq_max"]
        self.hull_area = fp_cfg["hull_area"]
        self.max_db = fp_cfg["max_db"]
        self.min_db = fp_cfg["min_db"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # STFT transformer
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            power=2,
            normalized=False
        ).to(self.device)
        
        # Frequency bins
        self.freq_bins = np.linspace(0, self.sample_rate / 2, num=(self.n_fft // 2) + 1)
        self.freq_mask = (self.freq_bins >= self.freq_min) & (self.freq_bins <= self.freq_max)
        self.freq_range = self.freq_bins[self.freq_mask]
        
        # Load ONNX model
        if model_path is None:
            model_path = Path(__file__).parent / "models" / "ai_music_detector.onnx"
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.session = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name
        self.expected_features = self.session.get_inputs()[0].shape[1]
        
        print(f"Loaded model: {model_path}")
        print(f"Device: {self.device}")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Frequency range: {self.freq_min}-{self.freq_max} Hz")
        print(f"Feature dimension: {self.expected_features}")
    
    def load_audio(self, file_path: str) -> torch.Tensor:
        """Load and resample audio to target sample rate."""
        audio, sr = torchaudio.load(file_path)
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        max_samples = self.max_duration * self.sample_rate
        if audio.shape[1] > max_samples:
            audio = audio[:, :max_samples]
        
        return audio
    
    def compute_fakeprint(self, audio: torch.Tensor) -> np.ndarray:
        """Extract fakeprint from audio (must match training exactly)."""
        audio = audio.to(self.device)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        with torch.no_grad():
            spec = self.stft(audio)
        
        spec_db = 10 * torch.log10(torch.clamp(spec, min=1e-10, max=1e6))
        mean_spectrum = spec_db.mean(dim=(0, 2)).cpu().numpy()
        
        # Apply frequency mask
        freq_spectrum = mean_spectrum[self.freq_mask]
        
        # Fast lower hull using minimum filter (same as training)
        hull = minimum_filter1d(freq_spectrum, size=self.hull_area, mode='nearest')
        hull = np.clip(hull, self.min_db, None)
        
        # Compute residue
        residue = np.clip(freq_spectrum - hull, 0, None)
        residue = np.clip(residue, 0, self.max_db)
        max_val = np.max(residue) + 1e-6
        
        return (residue / max_val).astype(np.float32)
    
    def predict(self, file_path: str) -> dict:
        """Predict if audio file is AI-generated."""
        try:
            audio = self.load_audio(file_path)
            fakeprint = self.compute_fakeprint(audio)
            
            # Resize if needed
            if len(fakeprint) != self.expected_features:
                old_x = np.linspace(0, 1, len(fakeprint))
                new_x = np.linspace(0, 1, self.expected_features)
                fakeprint = np.interp(new_x, old_x, fakeprint).astype(np.float32)
            
            input_data = fakeprint.reshape(1, -1)
            outputs = self.session.run(None, {self.input_name: input_data})
            probability = float(outputs[0][0, 0])
            
            is_ai = probability >= 0.5
            
            return {
                "probability": probability,
                "is_ai": is_ai,
                "label": "AI-Generated" if is_ai else "Real Music",
                "confidence": abs(probability - 0.5) * 2
            }
        except Exception as e:
            return {"error": str(e), "probability": None, "is_ai": None, "label": "Error"}


def find_audio_files(directory: Path) -> list:
    """Find all audio files in directory."""
    extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
    files = []
    
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in extensions:
                files.append(os.path.join(root, filename))
    
    return sorted(files)


def run_interactive(detector: AIDetector):
    """Run in interactive mode."""
    print("\n" + "="*60)
    print("AI Music Detector - Interactive Mode")
    print("="*60)
    print("Enter file path to analyze (or 'quit' to exit)\n")
    
    while True:
        try:
            user_input = input(">>> ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Handle quoted paths
            if (user_input.startswith('"') and user_input.endswith('"')) or \
               (user_input.startswith("'") and user_input.endswith("'")):
                user_input = user_input[1:-1]
            
            file_path = Path(user_input)
            
            if not file_path.exists():
                print(f"File not found: {file_path}")
                continue
            
            if file_path.is_dir():
                files = find_audio_files(file_path)
                print(f"Found {len(files)} audio files")
                for f in files:
                    result = detector.predict(f)
                    name = Path(f).name
                    if result.get("error"):
                        print(f"  {name}: ERROR - {result['error']}")
                    else:
                        print(f"  {name}: {result['probability']:.1%} - {result['label']}")
            else:
                result = detector.predict(str(file_path))
                if result.get("error"):
                    print(f"Error: {result['error']}")
                else:
                    print(f"\nFile: {file_path.name}")
                    print(f"AI Probability: {result['probability']:.1%}")
                    print(f"Classification: {result['label']}")
                    print(f"Confidence: {result['confidence']:.1%}\n")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_batch(detector: AIDetector, input_path: Path, output_csv: Optional[Path] = None):
    """Run in batch mode."""
    files = [str(input_path)] if input_path.is_file() else find_audio_files(input_path)
    
    if not files:
        print("No audio files found!")
        return
    
    print(f"\nProcessing {len(files)} files...")
    print("="*80)
    print(f"{'File':<50} {'Probability':>12} {'Result':>15}")
    print("="*80)
    
    results = []
    for file_path in files:
        result = detector.predict(file_path)
        name = Path(file_path).name
        if len(name) > 48:
            name = name[:45] + "..."
        
        if result.get("error"):
            print(f"{name:<50} {'ERROR':>12} {result['error'][:15]:>15}")
        else:
            print(f"{name:<50} {result['probability']:>11.1%} {result['label']:>15}")
        
        results.append({
            "file": file_path,
            "probability": result.get("probability"),
            "is_ai": result.get("is_ai"),
            "label": result.get("label"),
            "error": result.get("error")
        })
    
    print("="*80)
    
    valid = [r for r in results if r["probability"] is not None]
    if valid:
        ai_count = sum(1 for r in valid if r["is_ai"])
        print(f"\nSummary: {ai_count} AI-generated, {len(valid) - ai_count} Real, {len(results) - len(valid)} errors")
    
    if output_csv:
        import csv
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["file", "probability", "is_ai", "label", "error"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="AI Music Detector Inference")
    parser.add_argument("--input", "-i", type=str, help="Input file or folder (batch mode)")
    parser.add_argument("--model", "-m", type=str, help="Path to ONNX model file")
    parser.add_argument("--output", "-o", type=str, help="Output CSV file for batch results")
    args = parser.parse_args()
    
    try:
        detector = AIDetector(model_path=args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you have trained and exported the model:")
        print("  python train_model.py --real ... --fake ...")
        print("  python export_onnx.py")
        sys.exit(1)
    
    if args.input:
        run_batch(detector, Path(args.input), Path(args.output) if args.output else None)
    else:
        run_interactive(detector)


if __name__ == "__main__":
    main()
