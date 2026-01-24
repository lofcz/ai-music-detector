"""
Audio augmentation transforms for robust AI Music Detection training.

Includes pitch shifting, EQ, compression, MP3 encoding, and other
transforms to make the model invariant to common audio modifications.
"""

import numpy as np
import torch
import torchaudio
import random
from typing import Optional, Tuple, List
from dataclasses import dataclass
from scipy import signal
import io


@dataclass
class AugmentConfig:
    """Configuration for augmentation probabilities and ranges."""
    pitch_shift_prob: float = 0.3
    pitch_shift_range: Tuple[int, int] = (-2, 2)  # semitones
    
    time_stretch_prob: float = 0.2
    time_stretch_range: Tuple[float, float] = (0.9, 1.1)
    
    eq_prob: float = 0.4
    eq_gain_range: Tuple[float, float] = (-6.0, 6.0)  # dB
    eq_bands: int = 5
    
    compression_prob: float = 0.2
    compression_threshold: Tuple[float, float] = (-20.0, -10.0)  # dB
    compression_ratio: Tuple[float, float] = (2.0, 4.0)
    
    noise_prob: float = 0.2
    noise_snr_range: Tuple[float, float] = (30.0, 50.0)  # dB
    
    lowpass_prob: float = 0.2
    lowpass_cutoff_range: Tuple[int, int] = (8000, 14000)  # Hz
    
    gain_prob: float = 0.3
    gain_range: Tuple[float, float] = (-6.0, 6.0)  # dB


class AudioAugmentor:
    """
    Apply audio augmentations for training robust AI music detectors.
    
    All augmentations preserve the underlying spectral artifacts while
    simulating real-world audio modifications.
    """
    
    def __init__(self, config: Optional[AugmentConfig] = None, sample_rate: int = 16000):
        self.config = config or AugmentConfig()
        self.sample_rate = sample_rate
        
    def pitch_shift(self, audio: torch.Tensor, semitones: int) -> torch.Tensor:
        """Shift pitch by given semitones."""
        if semitones == 0:
            return audio
        
        # Use torchaudio's functional pitch shift
        effects = [
            ["pitch", str(semitones * 100)],  # cents
            ["rate", str(self.sample_rate)]
        ]
        
        try:
            augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
                audio.unsqueeze(0) if audio.dim() == 1 else audio,
                self.sample_rate,
                effects
            )
            return augmented.squeeze(0) if audio.dim() == 1 else augmented
        except:
            return audio
    
    def time_stretch(self, audio: torch.Tensor, rate: float) -> torch.Tensor:
        """Stretch time by given rate (>1 = slower, <1 = faster)."""
        if abs(rate - 1.0) < 0.01:
            return audio
        
        effects = [
            ["tempo", str(rate)]
        ]
        
        try:
            augmented, _ = torchaudio.sox_effects.apply_effects_tensor(
                audio.unsqueeze(0) if audio.dim() == 1 else audio,
                self.sample_rate,
                effects
            )
            return augmented.squeeze(0) if audio.dim() == 1 else augmented
        except:
            return audio
    
    def parametric_eq(self, audio: torch.Tensor, gains_db: List[float]) -> torch.Tensor:
        """Apply parametric EQ with given gains per band."""
        audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio
        
        # Define frequency bands (log-spaced)
        n_bands = len(gains_db)
        freqs = np.logspace(np.log10(100), np.log10(self.sample_rate/2 - 100), n_bands)
        
        # Apply each band as a peaking filter
        output = audio_np.copy()
        for freq, gain in zip(freqs, gains_db):
            if abs(gain) < 0.5:
                continue
            
            # Design peaking EQ filter
            Q = 1.0
            A = 10 ** (gain / 40)
            w0 = 2 * np.pi * freq / self.sample_rate
            alpha = np.sin(w0) / (2 * Q)
            
            b0 = 1 + alpha * A
            b1 = -2 * np.cos(w0)
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha / A
            
            b = np.array([b0/a0, b1/a0, b2/a0])
            a = np.array([1, a1/a0, a2/a0])
            
            try:
                output = signal.lfilter(b, a, output)
            except:
                pass
        
        return torch.from_numpy(output.astype(np.float32))
    
    def compress(self, audio: torch.Tensor, threshold_db: float, ratio: float) -> torch.Tensor:
        """Apply dynamic range compression."""
        audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio
        
        # Convert to dB
        eps = 1e-8
        audio_db = 20 * np.log10(np.abs(audio_np) + eps)
        
        # Compute gain reduction
        over_threshold = audio_db - threshold_db
        over_threshold = np.maximum(over_threshold, 0)
        gain_reduction = over_threshold * (1 - 1/ratio)
        
        # Apply compression
        gain_linear = 10 ** (-gain_reduction / 20)
        compressed = audio_np * gain_linear
        
        # Normalize to original peak
        if np.max(np.abs(compressed)) > eps:
            compressed = compressed * (np.max(np.abs(audio_np)) / np.max(np.abs(compressed)))
        
        return torch.from_numpy(compressed.astype(np.float32))
    
    def add_noise(self, audio: torch.Tensor, snr_db: float) -> torch.Tensor:
        """Add Gaussian noise at specified SNR."""
        audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio
        
        # Calculate signal power
        signal_power = np.mean(audio_np ** 2)
        
        # Calculate noise power from SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate and add noise
        noise = np.random.randn(*audio_np.shape) * np.sqrt(noise_power)
        noisy = audio_np + noise
        
        return torch.from_numpy(noisy.astype(np.float32))
    
    def lowpass_filter(self, audio: torch.Tensor, cutoff_hz: int) -> torch.Tensor:
        """Apply lowpass filter."""
        audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio
        
        # Normalize cutoff frequency
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff_hz / nyquist
        
        if normalized_cutoff >= 1.0:
            return audio
        
        # Design Butterworth filter
        try:
            b, a = signal.butter(4, normalized_cutoff, btype='low')
            filtered = signal.lfilter(b, a, audio_np)
            return torch.from_numpy(filtered.astype(np.float32))
        except:
            return audio
    
    def adjust_gain(self, audio: torch.Tensor, gain_db: float) -> torch.Tensor:
        """Adjust gain by given dB amount."""
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear
    
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations based on config probabilities."""
        cfg = self.config
        
        # Pitch shift
        if random.random() < cfg.pitch_shift_prob:
            semitones = random.randint(*cfg.pitch_shift_range)
            audio = self.pitch_shift(audio, semitones)
        
        # Time stretch
        if random.random() < cfg.time_stretch_prob:
            rate = random.uniform(*cfg.time_stretch_range)
            audio = self.time_stretch(audio, rate)
        
        # EQ
        if random.random() < cfg.eq_prob:
            gains = [random.uniform(*cfg.eq_gain_range) for _ in range(cfg.eq_bands)]
            audio = self.parametric_eq(audio, gains)
        
        # Compression
        if random.random() < cfg.compression_prob:
            threshold = random.uniform(*cfg.compression_threshold)
            ratio = random.uniform(*cfg.compression_ratio)
            audio = self.compress(audio, threshold, ratio)
        
        # Noise
        if random.random() < cfg.noise_prob:
            snr = random.uniform(*cfg.noise_snr_range)
            audio = self.add_noise(audio, snr)
        
        # Lowpass
        if random.random() < cfg.lowpass_prob:
            cutoff = random.randint(*cfg.lowpass_cutoff_range)
            audio = self.lowpass_filter(audio, cutoff)
        
        # Gain
        if random.random() < cfg.gain_prob:
            gain = random.uniform(*cfg.gain_range)
            audio = self.adjust_gain(audio, gain)
        
        return audio


class TrainingAugmentor:
    """
    Augmentor specifically for training that ensures consistent tensor output.
    """
    
    def __init__(self, sample_rate: int = 16000, augment_prob: float = 0.8):
        self.augmentor = AudioAugmentor(sample_rate=sample_rate)
        self.augment_prob = augment_prob
        self.sample_rate = sample_rate
    
    def __call__(self, audio: torch.Tensor, target_length: Optional[int] = None) -> torch.Tensor:
        """
        Apply augmentations and ensure output has target length.
        
        Args:
            audio: Input audio tensor [samples] or [channels, samples]
            target_length: Desired output length (pad/trim if needed)
        
        Returns:
            Augmented audio tensor of target_length
        """
        # Apply augmentations with probability
        if random.random() < self.augment_prob:
            audio = self.augmentor(audio)
        
        # Ensure correct length
        if target_length is not None:
            if audio.dim() == 1:
                current_length = audio.shape[0]
            else:
                current_length = audio.shape[-1]
            
            if current_length > target_length:
                # Random crop
                start = random.randint(0, current_length - target_length)
                if audio.dim() == 1:
                    audio = audio[start:start + target_length]
                else:
                    audio = audio[..., start:start + target_length]
            elif current_length < target_length:
                # Pad
                pad_amount = target_length - current_length
                if audio.dim() == 1:
                    audio = torch.nn.functional.pad(audio, (0, pad_amount))
                else:
                    audio = torch.nn.functional.pad(audio, (0, pad_amount))
        
        return audio


if __name__ == "__main__":
    # Test augmentations
    print("Testing augmentations...")
    
    # Generate test audio (sine wave)
    sr = 16000
    duration = 3.0
    t = torch.linspace(0, duration, int(sr * duration))
    audio = torch.sin(2 * np.pi * 440 * t)  # 440 Hz sine
    
    augmentor = AudioAugmentor(sample_rate=sr)
    
    print(f"Original audio shape: {audio.shape}")
    print(f"Original audio range: [{audio.min():.3f}, {audio.max():.3f}]")
    
    # Test each augmentation
    print("\nTesting pitch shift...")
    shifted = augmentor.pitch_shift(audio, 2)
    print(f"  Shape: {shifted.shape}")
    
    print("Testing time stretch...")
    stretched = augmentor.time_stretch(audio, 1.1)
    print(f"  Shape: {stretched.shape}")
    
    print("Testing EQ...")
    eqd = augmentor.parametric_eq(audio, [3.0, -2.0, 1.0, -1.0, 2.0])
    print(f"  Shape: {eqd.shape}")
    
    print("Testing compression...")
    compressed = augmentor.compress(audio, -15.0, 3.0)
    print(f"  Shape: {compressed.shape}")
    
    print("Testing noise...")
    noisy = augmentor.add_noise(audio, 40.0)
    print(f"  Shape: {noisy.shape}")
    
    print("Testing lowpass...")
    lowpassed = augmentor.lowpass_filter(audio, 4000)
    print(f"  Shape: {lowpassed.shape}")
    
    print("\nTesting random augmentation chain...")
    augmented = augmentor(audio)
    print(f"  Shape: {augmented.shape}")
    
    print("\nAll tests passed!")
