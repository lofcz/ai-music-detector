"""
Debug exactly how librosa centers and creates the wavelet.
"""
import numpy as np
import librosa
from scipy.signal import get_window

sr = 16000
fmin = 500.0
n_bins = 48
bins_per_octave = 12

# Get parameters
freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
alpha = librosa.filters._relative_bandwidth(freqs=freqs)
lengths, _ = librosa.filters.wavelet_lengths(freqs=freqs, sr=sr, filter_scale=1.0, window='hann', alpha=alpha)

# Get librosa's wavelet
librosa_basis, _ = librosa.filters.wavelet(
    freqs=freqs, sr=sr, filter_scale=1.0, norm=1, pad_fft=True,
    window='hann', alpha=alpha
)
n_fft = librosa_basis.shape[1]

print("Testing different wavelet construction methods...\n")

k = 0  # Test first bin
freq = freqs[k]
length = int(lengths[k])
print(f"Bin {k}: freq={freq:.1f} Hz, length={length}, n_fft={n_fft}")

# Find where librosa's signal is
librosa_sig = librosa_basis[k]
nonzero = np.where(np.abs(librosa_sig) > 1e-15)[0]
librosa_start = nonzero[0]
librosa_end = nonzero[-1]
print(f"Librosa non-zero: {librosa_start} to {librosa_end}")
librosa_actual = librosa_sig[librosa_start:librosa_end+1]
print(f"Librosa actual length: {len(librosa_actual)}")

# Get window
window = get_window('hann', length, fftbins=True)

# Method 1: Zero-based time, left-aligned
t = np.arange(length) / sr
sig1 = window * np.exp(2j * np.pi * freq * t)
sig1 = sig1 / np.sum(np.abs(sig1))

# Method 2: Zero-based time, centered in buffer
sig2_full = np.zeros(n_fft, dtype=complex)
start2 = (n_fft - length) // 2
sig2_full[start2:start2+length] = sig1
sig2_actual = sig2_full[start2:start2+length]

# Method 3: Centered time, left-aligned
center = length // 2
t_centered = (np.arange(length) - center) / sr
sig3 = window * np.exp(2j * np.pi * freq * t_centered)
sig3 = sig3 / np.sum(np.abs(sig3))

# Method 4: Centered time, centered in buffer
sig4_full = np.zeros(n_fft, dtype=complex)
sig4_full[start2:start2+length] = sig3

# Compare each method
print(f"\nMethod 1 (zero time, left): diff = {np.abs(sig1 - librosa_actual).max():.10f}")
print(f"Method 2 (zero time, centered): diff = {np.abs(sig2_actual - librosa_actual).max():.10f}")
print(f"Method 3 (centered time, left): diff = {np.abs(sig3 - librosa_actual).max():.10f}")
print(f"Method 4 (centered time, centered): diff = {np.abs(sig3 - librosa_actual).max():.10f}")

# The start position
print(f"\nStart position comparison:")
print(f"  Librosa: {librosa_start}")
print(f"  Centered formula: {start2}")

# Check actual values at specific positions
print(f"\nValue comparison at position 0 of actual signal:")
print(f"  Librosa: {librosa_actual[0]}")
print(f"  Method 1: {sig1[0]}")
print(f"  Method 3: {sig3[0]}")

print(f"\nValue at center of actual signal:")
mid = length // 2
print(f"  Librosa: {librosa_actual[mid]}")
print(f"  Method 1: {sig1[mid]}")
print(f"  Method 3: {sig3[mid]}")

# Phase check - at what time does the phase = 0?
# For zero-based: phase=0 at t=0 (first sample)
# For centered: phase=0 at center
print(f"\nPhase analysis:")
print(f"  Method 1 phase at t=0: {np.angle(sig1[0]):.6f}")
print(f"  Method 1 phase at center: {np.angle(sig1[mid]):.6f}")
print(f"  Method 3 phase at t=0: {np.angle(sig3[0]):.6f}")
print(f"  Method 3 phase at center: {np.angle(sig3[mid]):.6f}")
print(f"  Librosa phase at start: {np.angle(librosa_actual[0]):.6f}")
print(f"  Librosa phase at center: {np.angle(librosa_actual[mid]):.6f}")

# Now check if librosa's wavelet matches method 1 when correctly placed
sig1_full = np.zeros(n_fft, dtype=complex)
sig1_full[librosa_start:librosa_start+length] = sig1
diff_placed = np.abs(sig1_full - librosa_sig).max()
print(f"\nMethod 1 placed at librosa_start: diff = {diff_placed:.10f}")

# Maybe librosa uses a different centering formula
librosa_center_offset = librosa_start
expected_center = (n_fft - length) // 2
print(f"\nCentering:")
print(f"  Librosa start: {librosa_start}")
print(f"  Expected (n_fft-length)//2: {expected_center}")
print(f"  n_fft // 2 - length // 2: {n_fft // 2 - length // 2}")
