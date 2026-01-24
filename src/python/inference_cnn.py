"""
CNN-based AI Music Detection inference using CQT-Cepstrum features.

Pitch-shift invariant detection:
- CQT uses log-frequency (pitch shift = translation)
- DCT captures patterns (translation-invariant)
- Median-pooling across segments for robust classification

Usage:
    python inference_cnn.py                      # Interactive mode
    python inference_cnn.py --input song.mp3     # Single file
    python inference_cnn.py --input ./folder     # Batch mode
"""

import os
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from scipy.fft import dct
import yaml

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

try:
    from nnAudio.features import CQT as nnCQT
    HAS_NNAUDIO = True
except ImportError:
    HAS_NNAUDIO = False


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
        except:
            pass
    
    for path in [r'C:\ffmpeg\bin\ffmpeg.exe', r'C:\Program Files\ffmpeg\bin\ffmpeg.exe']:
        if os.path.isfile(path):
            return path
    
    return None


class CepstrumCNNDetector:
    """
    AI Music Detector using CNN on CQT-Cepstrum features.
    
    Uses max-pooling across segments: if ANY segment shows AI -> flag file.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        if not HAS_LIBROSA:
            raise RuntimeError("librosa is required. Install with: pip install librosa")
        
        self.config = load_config()
        audio_cfg = self.config["audio"]
        cqt_cfg = self.config["cqt"]
        cepstrum_cfg = self.config["cepstrum"]
        
        # Audio parameters - MUST match extraction
        self.sample_rate = audio_cfg["sample_rate"]
        self.segment_seconds = cepstrum_cfg["segment_seconds"]
        self.segment_samples = int(self.segment_seconds * self.sample_rate)
        
        # CQT parameters - MUST match extraction
        self.fmin = cqt_cfg["fmin"]
        self.n_bins = cqt_cfg["n_bins"]
        self.bins_per_octave = cqt_cfg["bins_per_octave"]
        self.hop_length = cqt_cfg["hop_length"]
        
        # Cepstrum parameters - MUST match extraction
        self.n_coeffs = cepstrum_cfg["n_coeffs"]
        
        # Device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # FFmpeg for audio loading
        self.ffmpeg_path = find_ffmpeg()
        if self.ffmpeg_path is None:
            print("WARNING: ffmpeg not found - audio loading may fail")
            self.ffmpeg_path = "ffmpeg"
        
        # Load model
        if model_path is None:
            model_path = Path(__file__).parent / "models" / "cnn_best.pt"
        
        self.model_path = Path(model_path)
        self.use_onnx = self.model_path.suffix == ".onnx"
        
        if self.use_onnx:
            self._load_onnx_model()
        else:
            self._load_pytorch_model()
        
        # GPU CQT disabled - nnAudio produces different output than librosa
        # which causes accuracy loss since model was trained with librosa features
        # To enable (with accuracy tradeoff): set use_gpu_cqt=True
        self.gpu_cqt = None
        # if HAS_NNAUDIO and self.device.type == 'cuda':
        #     try:
        #         self.gpu_cqt = nnCQT(
        #             sr=self.sample_rate,
        #             fmin=self.fmin,
        #             n_bins=self.n_bins,
        #             bins_per_octave=self.bins_per_octave,
        #             hop_length=self.hop_length,
        #             output_format='Magnitude',
        #             verbose=False
        #         ).to(self.device)
        #         print(f"CQT: GPU-accelerated (nnAudio)")
        #     except Exception as e:
        #         print(f"nnAudio init failed, using CPU: {e}")
        #         self.gpu_cqt = None
        
        print(f"Model: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"CQT: fmin={self.fmin}Hz, n_bins={self.n_bins}, range=[{self.fmin}-{self.fmin * 2**(self.n_bins/self.bins_per_octave):.0f}]Hz")
        print(f"Cepstrum coefficients: {self.n_coeffs}")
        if self.gpu_cqt is None:
            print(f"CQT backend: librosa (CPU)")
    
    def _load_pytorch_model(self):
        """Load PyTorch model."""
        from models.cnn_detector import get_model
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        model_config = checkpoint.get('config', {})
        
        model_type = model_config.get('model_type', 'standard')
        n_coeffs = model_config.get('n_coeffs', self.n_coeffs)
        
        self.model = get_model(model_type, n_coeffs=n_coeffs).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def _load_onnx_model(self):
        """Load ONNX model."""
        import onnxruntime as ort
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.model = None
    
    def load_audio(self, file_path: str) -> Optional[np.ndarray]:
        """Load audio - uses torchaudio if available (faster), else ffmpeg."""
        try:
            file_path = os.path.normpath(file_path)
            
            # Try torchaudio first (faster, no subprocess)
            if HAS_TORCHAUDIO:
                try:
                    audio, sr = torchaudio.load(file_path)
                    # Resample if needed
                    if sr != self.sample_rate:
                        audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
                    # Convert to mono
                    if audio.shape[0] > 1:
                        audio = audio.mean(dim=0)
                    else:
                        audio = audio[0]
                    return audio.numpy()
                except Exception:
                    pass  # Fall back to ffmpeg
            
            # Fallback to ffmpeg
            cmd = [
                self.ffmpeg_path,
                '-i', file_path,
                '-f', 'f32le',
                '-acodec', 'pcm_f32le',
                '-ar', str(self.sample_rate),
                '-ac', '1',
                '-v', 'quiet',
                '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            
            if result.returncode != 0:
                return None
            
            audio = np.frombuffer(result.stdout, dtype=np.float32)
            return audio if len(audio) > 0 else None
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def _extract_cepstra_batch_gpu(self, segments: List[np.ndarray]) -> List[np.ndarray]:
        """Batch CQT computation on GPU for all segments at once."""
        with torch.no_grad():
            # Stack segments into batch
            batch = np.stack(segments, axis=0)  # [N, samples]
            batch_tensor = torch.from_numpy(batch).float().to(self.device)
            
            # Batch CQT
            cqt_batch = self.gpu_cqt(batch_tensor)  # [N, n_bins, n_frames]
            
            # Log magnitude
            log_cqt = torch.log(cqt_batch + 1e-6)
            
            # Move to CPU for DCT (scipy)
            log_cqt_np = log_cqt.cpu().numpy()
            
            # DCT and truncate for each
            cepstra = []
            for i in range(len(segments)):
                cep = dct(log_cqt_np[i], type=2, axis=0, norm='ortho')
                cepstra.append(cep[:self.n_coeffs, :].astype(np.float32))
            
            return cepstra
    
    def extract_cqt_cepstrum(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract CQT-cepstrum features - MUST match extraction script exactly.
        
        Returns: [n_coeffs, n_frames] cepstrum spectrogram
        """
        # Use GPU CQT if available (much faster)
        if self.gpu_cqt is not None:
            with torch.no_grad():
                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
                cqt_mag = self.gpu_cqt(audio_tensor)  # [1, n_bins, n_frames]
                cqt_mag = cqt_mag.squeeze(0).cpu().numpy()  # [n_bins, n_frames]
        else:
            # Fallback to librosa (CPU)
            cqt = librosa.cqt(
                audio,
                sr=self.sample_rate,
                fmin=self.fmin,
                n_bins=self.n_bins,
                bins_per_octave=self.bins_per_octave,
                hop_length=self.hop_length
            )
            cqt_mag = np.abs(cqt)
        
        # Log magnitude
        log_cqt = np.log(cqt_mag + 1e-6)
        
        # DCT along frequency axis
        cepstrum = dct(log_cqt, type=2, axis=0, norm='ortho')
        
        # Keep first n_coeffs
        cepstrum = cepstrum[:self.n_coeffs, :]
        
        return cepstrum.astype(np.float32)
    
    def extract_segments(self, audio: np.ndarray, n_segments: int) -> List[np.ndarray]:
        """
        Extract n_segments evenly spread across audio, skipping first/last 5s if possible.
        Intros/outros often cause false positives.
        """
        segments = []
        skip_samples = 5 * self.sample_rate  # 5 seconds
        
        # Determine usable audio range (skip first/last 5s if long enough)
        if len(audio) > self.segment_samples + 2 * skip_samples:
            # Song is long enough - skip intro/outro
            start_offset = skip_samples
            end_offset = len(audio) - skip_samples
        else:
            # Short song - use all audio
            start_offset = 0
            end_offset = len(audio)
        
        usable_length = end_offset - start_offset
        
        if usable_length <= self.segment_samples:
            # Very short - just use center or pad
            if len(audio) <= self.segment_samples:
                padded = np.zeros(self.segment_samples, dtype=np.float32)
                padded[:len(audio)] = audio
                segments.append(padded)
            else:
                center = len(audio) // 2
                start = max(0, center - self.segment_samples // 2)
                segments.append(audio[start:start + self.segment_samples])
        else:
            # Spread segments across usable range
            available = usable_length - self.segment_samples
            
            if n_segments == 1:
                positions = [start_offset + available // 2]
            else:
                step = available / (n_segments - 1)
                positions = [start_offset + int(i * step) for i in range(n_segments)]
            
            for start in positions:
                segments.append(audio[start:start + self.segment_samples])
        
        return segments
    
    @torch.no_grad()
    def predict_cepstrum_batch(self, cepstra: List[np.ndarray]) -> List[float]:
        """Run model on batch of cepstra, return list of P(AI)."""
        # Stack into batch: [N, 1, n_coeffs, n_frames]
        batch = np.stack([c[np.newaxis, :, :] for c in cepstra], axis=0)
        
        if self.use_onnx:
            inputs = {self.ort_session.get_inputs()[0].name: batch.astype(np.float32)}
            output = self.ort_session.run(None, inputs)[0]
            probs = 1 / (1 + np.exp(-output[:, 0]))  # Sigmoid
            return probs.tolist()
        else:
            batch_tensor = torch.from_numpy(batch).float().to(self.device)
            output = self.model(batch_tensor)
            probs = torch.sigmoid(output).cpu().numpy()[:, 0]
            return probs.tolist()
    
    def predict(self, file_path: str, n_segments: int = 5) -> dict:
        """
        Analyze audio file for AI-generated content.
        
        Strategy:
        1. Extract N segments spread across song
        2. Compute cepstrum for each segment (parallel-ready)
        3. Batch predict P(AI) for all segments (single forward pass)
        4. MEDIAN-pooling for robust classification
        """
        audio = self.load_audio(file_path)
        if audio is None:
            return {"error": f"Could not load {file_path}"}
        
        # Extract segments
        segments = self.extract_segments(audio, n_segments)
        
        # Compute cepstra for all segments
        # Use batched GPU CQT if available, otherwise sequential CPU
        if self.gpu_cqt is not None and len(segments) > 1:
            cepstra = self._extract_cepstra_batch_gpu(segments)
        else:
            cepstra = [self.extract_cqt_cepstrum(seg) for seg in segments]
        
        # Batch inference - single forward pass for all segments
        probabilities = self.predict_cepstrum_batch(cepstra)
        
        # MEDIAN-pooling: more robust to outlier segments (intro/outro)
        max_prob = max(probabilities)
        median_prob = float(np.median(probabilities))
        final_prob = median_prob
        
        # Classification - use median for robustness
        is_ai = final_prob > 0.5
        classification = "AI-Generated" if is_ai else "Real Music"
        confidence = abs(final_prob - 0.5) * 2
        
        return {
            "ai_probability": final_prob,
            "classification": classification,
            "confidence": confidence,
            "n_segments": len(segments),
            "segment_probs": probabilities,
            "max_prob": max_prob,
            "median_prob": median_prob,
            "min_prob": min(probabilities),
            "avg_prob": float(np.mean(probabilities))
        }
    
    def analyze_batch(self, file_paths: List[str], n_segments: int = 5) -> List[dict]:
        """Analyze multiple files."""
        results = []
        for path in file_paths:
            result = self.predict(path, n_segments)
            result["file"] = Path(path).name
            results.append(result)
        return results

    def predict_with_debug(self, file_path: str, n_segments: int = 5) -> dict:
        """
        Analyze with detailed debug output for comparing Python vs C# inference.
        """
        audio = self.load_audio(file_path)
        if audio is None:
            return {"error": f"Could not load {file_path}"}
        
        # Audio head
        audio_head = audio[:10].tolist() if len(audio) >= 10 else audio.tolist()
        
        # Extract segments with position tracking
        skip_samples = 5 * self.sample_rate
        
        if len(audio) > self.segment_samples + 2 * skip_samples:
            start_offset = skip_samples
            end_offset = len(audio) - skip_samples
        else:
            start_offset = 0
            end_offset = len(audio)
        
        usable_length = end_offset - start_offset
        
        # Calculate positions
        if usable_length <= self.segment_samples:
            if len(audio) <= self.segment_samples:
                positions = [0]
            else:
                center = len(audio) // 2
                start = max(0, center - self.segment_samples // 2)
                positions = [start]
        else:
            available = usable_length - self.segment_samples
            if n_segments == 1:
                positions = [start_offset + available // 2]
            else:
                step = available / (n_segments - 1)
                positions = [start_offset + int(i * step) for i in range(n_segments)]
        
        # Extract segments
        segments = self.extract_segments(audio, n_segments)
        
        # Compute cepstra and collect debug info
        segment_debug = []
        cepstra = []
        
        for i, seg in enumerate(segments):
            cepstrum = self.extract_cqt_cepstrum(seg)
            cepstra.append(cepstrum)
            
            segment_debug.append({
                "index": i,
                "start_sample": positions[i] if i < len(positions) else 0,
                "length_samples": self.segment_samples,
                "cepstrum_shape": list(cepstrum.shape),
                "cepstrum_coeff0_head": cepstrum[0, :5].tolist() if cepstrum.shape[1] >= 5 else cepstrum[0, :].tolist(),
                "cepstrum_frame0_head": cepstrum[:5, 0].tolist() if cepstrum.shape[0] >= 5 else cepstrum[:, 0].tolist(),
                "cepstrum_mean": float(cepstrum.mean()),
            })
        
        # Batch inference
        probabilities = self.predict_cepstrum_batch(cepstra)
        
        # Add probabilities to segment debug
        for i, prob in enumerate(probabilities):
            segment_debug[i]["probability"] = prob
        
        # Final probability
        final_prob = float(np.median(probabilities))
        
        return {
            "file": Path(file_path).name,
            "audio_samples": len(audio),
            "audio_head": audio_head,
            "segments": segment_debug,
            "segment_probabilities": probabilities,
            "final_probability": final_prob,
            "classification": "AI-Generated" if final_prob > 0.5 else "Real Music"
        }


def find_audio_files(directory: Path) -> List[str]:
    """Find audio files in directory."""
    extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
    files = []
    
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in extensions:
                files.append(os.path.join(root, filename))
    
    return sorted(files)


def print_result(result: dict, filename: str = None, verbose: bool = False):
    """Pretty print detection result."""
    if filename:
        print(f"\nFile: {filename}")
    
    if "error" in result:
        print(f"  Error: {result['error']}")
        return
    
    print(f"  Classification: {result['classification']}")
    print(f"  AI Probability: {result['ai_probability']*100:.1f}%")
    print(f"  Confidence: {result['confidence']*100:.1f}%")
    
    if verbose and "segment_probs" in result:
        probs = result["segment_probs"]
        print(f"  Segments: {[f'{p*100:.0f}%' for p in probs]}")
        print(f"  Median: {result['median_prob']*100:.1f}%, Max: {result['max_prob']*100:.1f}%, Avg: {result['avg_prob']*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="AI Music Detection (CQT-Cepstrum CNN)")
    parser.add_argument("--input", type=str, help="Input file or directory")
    parser.add_argument("--model", type=str, default=None, help="Model path (.pt or .onnx)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--segments", type=int, default=5, help="Segments to analyze")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show segment details")
    parser.add_argument("--debug", type=str, default=None, help="Debug mode - output detailed JSON for single file")
    args = parser.parse_args()
    
    detector = CepstrumCNNDetector(model_path=args.model, device=args.device)
    
    # Debug mode - output JSON for comparison with C#
    if args.debug:
        import json
        
        debug_result = detector.predict_with_debug(args.debug, n_segments=args.segments)
        
        print("\n" + "=" * 60)
        print(f"Debug Analysis: {debug_result.get('file', args.debug)}")
        print("=" * 60)
        
        if "error" in debug_result:
            print(f"Error: {debug_result['error']}")
            return
        
        print(f"\nAudio: {debug_result['audio_samples']:,} samples")
        print(f"Audio head: {debug_result['audio_head']}")
        print(f"\nSegments ({len(debug_result['segments'])}):")
        
        for seg in debug_result['segments']:
            print(f"  Seg {seg['index']}: start={seg['start_sample']:,}, shape={seg['cepstrum_shape']}, "
                  f"mean={seg['cepstrum_mean']:.4f}, prob={seg['probability']*100:.2f}%")
        
        print(f"\nSegment probs: {[f'{p*100:.2f}%' for p in debug_result['segment_probabilities']]}")
        print(f"Final (median): {debug_result['final_probability']*100:.2f}%")
        print(f"Classification: {debug_result['classification']}")
        
        print("\nJSON for C# comparison:")
        print(json.dumps(debug_result, indent=2))
        
        # Save to file
        json_path = Path(args.debug).with_suffix('.debug.py.json')
        with open(json_path, 'w') as f:
            json.dump(debug_result, f, indent=2)
        print(f"\nDebug output saved to: {json_path}")
        return
    
    if args.input:
        input_path = Path(args.input)
        
        if input_path.is_dir():
            print(f"\nScanning {input_path}...")
            files = find_audio_files(input_path)
            print(f"Found {len(files)} audio files")
            
            for file_path in files:
                result = detector.predict(file_path, n_segments=args.segments)
                print_result(result, Path(file_path).name, verbose=args.verbose)
        else:
            result = detector.predict(str(input_path), n_segments=args.segments)
            print_result(result, input_path.name, verbose=args.verbose)
    else:
        # Interactive mode
        print("\n" + "=" * 60)
        print("AI Music Detector (CQT-Cepstrum CNN) - Interactive Mode")
        print("=" * 60)
        print("Enter file path to analyze (or 'quit' to exit)")
        
        while True:
            try:
                user_input = input("\n>>> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                file_path = user_input.strip('"').strip("'")
                
                if not os.path.exists(file_path):
                    print(f"File not found: {file_path}")
                    continue
                
                result = detector.predict(file_path, n_segments=args.segments)
                print_result(result, Path(file_path).name, verbose=True)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
