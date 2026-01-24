"""
CQT Reference Implementation for testing C# port.

This provides a simple, clear CQT implementation that can be used to:
1. Generate test vectors for C# unit tests
2. Verify C# implementation correctness

Based on the algorithm from:
- Brown, Judith C. "Calculation of a constant Q spectral transform."
  The Journal of the Acoustical Society of America 89.1 (1991): 425-434.
"""

import numpy as np
from scipy.fft import fft, dct
from typing import Tuple


def hann_window(length: int) -> np.ndarray:
    """Generate Hann window."""
    n = np.arange(length)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (length - 1)))


def next_power_of_2(n: int) -> int:
    """Get next power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


class CQTReference:
    """
    Reference CQT implementation for testing.
    
    This implementation prioritizes clarity and correctness over efficiency.
    """
    
    def __init__(
        self,
        sample_rate: int,
        f_min: float,
        n_bins: int,
        bins_per_octave: int = 12,
        hop_length: int = 512,
        q_factor: float = 1.0
    ):
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.hop_length = hop_length
        
        # Calculate Q factor
        Q = q_factor / (2.0 ** (1.0 / bins_per_octave) - 1.0)
        self.Q = Q
        
        # Calculate center frequencies
        self.frequencies = np.array([
            f_min * (2.0 ** (k / bins_per_octave))
            for k in range(n_bins)
        ])
        
        # Calculate window lengths
        self.window_lengths = np.ceil(Q * sample_rate / self.frequencies).astype(int)
        
        # FFT size
        self.n_fft = next_power_of_2(self.window_lengths[0])
        
        # Build spectral kernel
        self.spectral_kernel = self._build_kernel()
    
    def _build_kernel(self) -> np.ndarray:
        """Build the spectral kernel matrix."""
        kernel = np.zeros((self.n_bins, self.n_fft), dtype=np.complex128)
        
        for k in range(self.n_bins):
            Nk = self.window_lengths[k]
            Fk = self.frequencies[k]
            window = hann_window(Nk)
            
            # Temporal kernel: windowed complex sinusoid centered in FFT buffer
            temp_kernel = np.zeros(self.n_fft, dtype=np.complex128)
            start = (self.n_fft - Nk) // 2
            
            n = np.arange(Nk)
            phase = 2.0 * np.pi * Fk / self.sample_rate * n
            temp_kernel[start:start + Nk] = (window / Nk) * np.exp(1j * phase)
            
            # FFT and store conjugate normalized
            kernel[k] = np.conj(fft(temp_kernel)) / self.n_fft
        
        return kernel
    
    def compute_frame(self, frame: np.ndarray) -> np.ndarray:
        """Compute CQT for a single frame."""
        # Prepare input (zero-pad if needed)
        x = np.zeros(self.n_fft)
        copy_len = min(len(frame), self.n_fft)
        x[:copy_len] = frame[:copy_len]
        
        # FFT of input
        X = fft(x)
        
        # Multiply with spectral kernel
        result = np.dot(self.spectral_kernel, X)
        
        return result
    
    def compute(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute Short-Time CQT.
        
        Args:
            audio: Audio samples
            
        Returns:
            CQT magnitude matrix [n_bins, n_frames]
        """
        n_frames = max(1, len(audio) // self.hop_length)
        result = np.zeros((self.n_bins, n_frames))
        
        for t in range(n_frames):
            start = t * self.hop_length
            end = start + self.n_fft
            
            if end <= len(audio):
                frame = audio[start:end]
            else:
                frame = np.zeros(self.n_fft)
                copy_len = min(len(audio) - start, self.n_fft)
                if copy_len > 0:
                    frame[:copy_len] = audio[start:start + copy_len]
            
            cqt_frame = self.compute_frame(frame)
            result[:, t] = np.abs(cqt_frame)
        
        return result
    
    def compute_complex(self, audio: np.ndarray) -> np.ndarray:
        """Compute STCQT returning complex values."""
        n_frames = max(1, len(audio) // self.hop_length)
        result = np.zeros((self.n_bins, n_frames), dtype=np.complex128)
        
        for t in range(n_frames):
            start = t * self.hop_length
            frame = np.zeros(self.n_fft)
            end = min(start + self.n_fft, len(audio))
            if end > start:
                frame[:end - start] = audio[start:end]
            
            result[:, t] = self.compute_frame(frame)
        
        return result


def compute_cepstrum(
    audio: np.ndarray,
    sample_rate: int,
    f_min: float,
    n_bins: int,
    bins_per_octave: int = 12,
    hop_length: int = 512,
    n_coeffs: int = 24
) -> np.ndarray:
    """
    Compute CQT-Cepstrum features.
    
    Args:
        audio: Audio samples
        sample_rate: Sample rate (Hz)
        f_min: Minimum frequency (Hz)
        n_bins: Number of CQT bins
        bins_per_octave: Bins per octave
        hop_length: Hop size in samples
        n_coeffs: Number of cepstral coefficients to keep
        
    Returns:
        Cepstrum matrix [n_coeffs, n_frames]
    """
    # Compute CQT
    cqt = CQTReference(sample_rate, f_min, n_bins, bins_per_octave, hop_length)
    cqt_mag = cqt.compute(audio)
    
    # Log magnitude
    log_cqt = np.log(cqt_mag + 1e-6)
    
    # DCT along frequency axis (axis=0)
    cepstrum = dct(log_cqt, type=2, axis=0, norm='ortho')
    
    # Keep first n_coeffs
    return cepstrum[:n_coeffs, :]


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt
    
    # Generate test signal: 440 Hz sine wave
    sr = 16000
    duration = 1.0
    t = np.arange(int(sr * duration)) / sr
    signal = np.sin(2 * np.pi * 440 * t)
    
    # Compute CQT
    cqt = CQTReference(sr, f_min=500, n_bins=48, bins_per_octave=12, hop_length=512)
    cqt_result = cqt.compute(signal)
    
    print(f"CQT shape: {cqt_result.shape}")
    print(f"FFT size: {cqt.n_fft}")
    print(f"Frequencies: {cqt.frequencies[0]:.1f} Hz - {cqt.frequencies[-1]:.1f} Hz")
    
    # Compute cepstrum
    cepstrum = compute_cepstrum(signal, sr, 500, 48, 12, 512, 24)
    print(f"Cepstrum shape: {cepstrum.shape}")
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].imshow(cqt_result, aspect='auto', origin='lower')
    axes[0].set_title('CQT Magnitude')
    axes[1].imshow(cepstrum, aspect='auto', origin='lower')
    axes[1].set_title('Cepstrum')
    plt.tight_layout()
    plt.savefig('cqt_test.png')
    print("Saved cqt_test.png")
