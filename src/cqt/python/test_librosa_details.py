"""
Extract exact librosa intermediate values to understand the algorithm precisely.
"""
import numpy as np
import librosa
from scipy.fft import fft

# Match our detector parameters
sr = 16000
fmin = 500.0
n_bins = 48
bins_per_octave = 12
hop_length = 512

# Generate test signal
np.random.seed(42)
signal = np.random.randn(8000).astype(np.float32)

# Get librosa CQT
cqt_result = librosa.cqt(
    signal, sr=sr, fmin=fmin, n_bins=n_bins,
    bins_per_octave=bins_per_octave, hop_length=hop_length
)
print(f"CQT shape: {cqt_result.shape}")
print(f"CQT magnitude range: {np.abs(cqt_result).min():.6f} to {np.abs(cqt_result).max():.6f}")

# Now let's trace through the internal functions
# First, get the frequencies
freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
print(f"\nFrequencies: {freqs[:5]}... to {freqs[-5:]}")

# Get alpha (relative bandwidth)
alpha = librosa.filters._relative_bandwidth(freqs=freqs)
print(f"Alpha: {alpha[:5]}...")

# Get wavelet lengths
lengths, _ = librosa.filters.wavelet_lengths(
    freqs=freqs, sr=sr, filter_scale=1.0, window='hann', alpha=alpha
)
print(f"Lengths: {lengths[:5]}... to {lengths[-5:]}")

# Now let's look at the wavelet basis construction
# This is what filters.wavelet does
print("\n--- Wavelet Construction Details ---")
k = 24  # Middle bin (around 2 kHz)
freq = freqs[k]
length = int(lengths[k])
print(f"Bin {k}: freq={freq:.1f} Hz, length={length}")

# Create wavelet the librosa way
from scipy.signal import get_window
window = get_window('hann', length, fftbins=True)

# Center the time axis
center = length // 2
t = (np.arange(length) - center) / sr

# Complex exponential
sig = window * np.exp(2j * np.pi * freq * t)

print(f"Before normalization:")
print(f"  sum(abs(sig)) = {np.sum(np.abs(sig)):.6f}")
print(f"  sig[:5] = {sig[:5]}")

# L1 normalize (norm=1)
sig_normalized = sig / np.sum(np.abs(sig))
print(f"After L1 normalization:")
print(f"  sum(abs(sig)) = {np.sum(np.abs(sig_normalized)):.6f}")
print(f"  sig[:5] = {sig_normalized[:5]}")

# Pad to power of 2
n_fft_pad = int(2 ** np.ceil(np.log2(max(lengths))))
n_fft = max(n_fft_pad, 2 * hop_length)
print(f"\nFFT size: {n_fft}")

# Now look at __vqt_filter_fft normalization
# basis *= lengths / n_fft
sig_scaled = sig_normalized * (length / n_fft)
print(f"After length/n_fft scaling:")
print(f"  sig[:5] = {sig_scaled[:5]}")

# Pad and FFT
padded = np.zeros(n_fft, dtype=complex)
padded[:length] = sig_scaled
fft_result = fft(padded)[:n_fft//2 + 1]
print(f"\nFFT result (positive freqs):")
print(f"  shape: {fft_result.shape}")
print(f"  magnitude range: {np.abs(fft_result).min():.6f} to {np.abs(fft_result).max():.6f}")

# Now let's verify by using librosa's internal functions
print("\n--- Using librosa internal functions ---")
basis, actual_lengths = librosa.filters.wavelet(
    freqs=freqs[:1],  # Just first frequency
    sr=sr,
    filter_scale=1.0,
    norm=1,
    pad_fft=True,
    window='hann',
    alpha=alpha[:1]
)
print(f"Wavelet basis shape: {basis.shape}")
print(f"First 5 values: {basis[0, :5]}")
print(f"Sum of abs: {np.sum(np.abs(basis[0])):.6f}")

# Check what happens with full basis
full_basis, full_lengths = librosa.filters.wavelet(
    freqs=freqs,
    sr=sr,
    filter_scale=1.0,
    norm=1,
    pad_fft=True,
    window='hann',
    alpha=alpha
)
print(f"\nFull basis shape: {full_basis.shape}")
print(f"n_fft from basis: {full_basis.shape[1]}")

# The __vqt_filter_fft does additional processing
print("\n--- FFT basis construction ---")
n_fft_basis = full_basis.shape[1]
if n_fft_basis < 2 * hop_length:
    n_fft_final = int(2 ** (1 + np.ceil(np.log2(hop_length))))
else:
    n_fft_final = n_fft_basis
print(f"Final n_fft: {n_fft_final}")

# Re-normalize for FFT length
basis_scaled = full_basis * (full_lengths[:, np.newaxis] / n_fft_final)
print(f"After scaling by lengths/n_fft:")
print(f"  Row 0 sum of abs: {np.sum(np.abs(basis_scaled[0])):.6f}")
print(f"  Row 0 first 5: {basis_scaled[0, :5]}")

# FFT and keep positive freqs
fft_basis = fft(basis_scaled, n=n_fft_final, axis=1)[:, :n_fft_final//2 + 1]
print(f"FFT basis shape: {fft_basis.shape}")
print(f"FFT basis row 0 mag range: {np.abs(fft_basis[0]).min():.6f} to {np.abs(fft_basis[0]).max():.6f}")

# Save exact values for C# comparison
import json
from pathlib import Path

output_dir = Path(__file__).parent.parent / 'testdata'
output_dir.mkdir(exist_ok=True)

test_data = {
    'sample_rate': sr,
    'fmin': fmin,
    'n_bins': n_bins,
    'bins_per_octave': bins_per_octave,
    'hop_length': hop_length,
    'n_fft': n_fft_final,
    'frequencies': freqs.tolist(),
    'alpha': alpha.tolist(),
    'lengths': full_lengths.tolist(),
    'wavelet_basis_real': basis_scaled.real.tolist(),
    'wavelet_basis_imag': basis_scaled.imag.tolist(),
    'fft_basis_real': fft_basis.real.tolist(),
    'fft_basis_imag': fft_basis.imag.tolist(),
}

with open(output_dir / 'librosa_internals.json', 'w') as f:
    json.dump(test_data, f, indent=2)
print(f"\nSaved internals to {output_dir / 'librosa_internals.json'}")

# Also verify STFT behavior
print("\n--- STFT Details ---")
stft_result = librosa.stft(signal, n_fft=n_fft_final, hop_length=hop_length, window='ones', center=True)
print(f"STFT shape: {stft_result.shape}")
print(f"STFT window: ones (rectangular)")
print(f"STFT center: True (zero-pad at start and end)")

# Compute CQT response manually
cqt_manual = fft_basis @ stft_result
print(f"\nManual CQT shape: {cqt_manual.shape}")
print(f"Manual CQT vs librosa shape: {cqt_result.shape}")

# Compare
min_frames = min(cqt_manual.shape[1], cqt_result.shape[1])
diff = np.abs(cqt_manual[:, :min_frames]) - np.abs(cqt_result[:, :min_frames])
print(f"Difference (manual vs librosa):")
print(f"  Max abs diff: {np.abs(diff).max():.6f}")
print(f"  Mean abs diff: {np.abs(diff).mean():.6f}")
