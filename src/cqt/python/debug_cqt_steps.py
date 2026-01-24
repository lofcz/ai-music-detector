"""
Debug CQT step by step to find where C# diverges from librosa.
"""
import numpy as np
import librosa
from scipy.fft import fft
from scipy.signal import get_window
import json
from pathlib import Path

sr = 16000
fmin = 500.0
n_bins = 48
bins_per_octave = 12
hop_length = 512

# Get all intermediate values for a single bin
np.random.seed(42)
signal = np.random.randn(8000).astype(np.float32)

# Get frequencies
freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
alpha = librosa.filters._relative_bandwidth(freqs=freqs)

# Get wavelet lengths (these are FLOATS!)
lengths, _ = librosa.filters.wavelet_lengths(
    freqs=freqs, sr=sr, filter_scale=1.0, window='hann', alpha=alpha
)
print(f"Filter lengths (floats): {lengths[:5]}... (first 5)")
print(f"Integer lengths: {[int(np.ceil(x)) for x in lengths[:5]]}... (first 5)")

# Let's trace bin 0 (500 Hz) in detail
k = 0
freq = freqs[k]
ilen = lengths[k]
print(f"\n=== Bin {k}: {freq:.1f} Hz ===")
print(f"Filter length (ilen): {ilen}")

# How np.arange works with floats
half = ilen // 2  # This is floor division
print(f"ilen // 2 = {half}")
print(f"-ilen // 2 = {-ilen // 2}")

time_indices = np.arange(-ilen // 2, ilen // 2, dtype=float)
print(f"np.arange(-ilen//2, ilen//2) length: {len(time_indices)}")
print(f"  range: {time_indices[0]} to {time_indices[-1]}")

# The phasor
phasor_angles = time_indices * 2 * np.pi * freq / sr
phasor = np.exp(1j * phasor_angles)
print(f"Phasor length: {len(phasor)}")

# The window (using __float_window logic)
n_min = int(np.floor(len(phasor)))
n_max = int(np.ceil(len(phasor)))
window = get_window('hann', n_min, fftbins=True)
if len(window) < n_max:
    window = np.pad(window, [(0, n_max - len(window))], mode='constant')
window[n_min:] = 0.0
print(f"Window length: {len(window)}")
print(f"Window[0:3]: {window[:3]}")
print(f"Window[-3:]: {window[-3:]}")

# Modulate
sig = phasor * window
print(f"sig length before norm: {len(sig)}")
print(f"sig[0]: {sig[0]}")

# L1 normalize
sig_norm = sig / np.sum(np.abs(sig))
print(f"sum(abs(sig_norm)): {np.sum(np.abs(sig_norm))}")
print(f"sig_norm[0]: {sig_norm[0]}")
print(f"sig_norm[len//2]: {sig_norm[len(sig_norm)//2]}")

# Now let's get the full basis from librosa
basis, returned_lengths = librosa.filters.wavelet(
    freqs=freqs, sr=sr, filter_scale=1.0, norm=1, pad_fft=True,
    window='hann', alpha=alpha
)
print(f"\nLibrosa wavelet basis shape: {basis.shape}")
n_fft_basis = basis.shape[1]

# Check if n_fft needs adjustment for hop_length
if n_fft_basis < 2 * hop_length:
    n_fft = int(2 ** (1 + np.ceil(np.log2(hop_length))))
else:
    n_fft = n_fft_basis
print(f"n_fft from basis: {n_fft_basis}, final n_fft: {n_fft}")

# Check where the wavelet is placed in the basis
nonzero = np.where(np.abs(basis[0]) > 1e-10)[0]
print(f"\nBin 0 non-zero indices: {nonzero[0]} to {nonzero[-1]} (length {len(nonzero)})")
center = (nonzero[0] + nonzero[-1]) // 2
print(f"Wavelet center in buffer: {center}, buffer center: {n_fft_basis // 2}")

# Compare first few values
print(f"\nBin 0 first 3 non-zero values: {basis[0, nonzero[0]:nonzero[0]+3]}")
print(f"Bin 0 at center: {basis[0, center]}")

# Do the __vqt_filter_fft scaling
basis_scaled = basis * (returned_lengths[:, np.newaxis] / n_fft)
print(f"\nAfter length/n_fft scaling:")
print(f"  Bin 0 first 3 non-zero: {basis_scaled[0, nonzero[0]:nonzero[0]+3]}")

# FFT
fft_basis = fft(basis_scaled, n=n_fft, axis=1)[:, :n_fft//2 + 1]
print(f"\nFFT basis shape: {fft_basis.shape}")
print(f"FFT basis bin 0, freq 0: {fft_basis[0, 0]}")
print(f"FFT basis bin 0, freq 1: {fft_basis[0, 1]}")

# Save the exact filter bank for C# comparison
output_dir = Path(__file__).parent.parent / 'testdata'
output_dir.mkdir(exist_ok=True)

filter_data = {
    'sample_rate': int(sr),
    'fmin': float(fmin),
    'n_bins': int(n_bins),
    'bins_per_octave': int(bins_per_octave),
    'hop_length': int(hop_length),
    'n_fft': int(n_fft),
    'frequencies': freqs.tolist(),
    'alpha': alpha.tolist(),
    'lengths': returned_lengths.tolist(),  # The raw float lengths
    'fft_basis_real': fft_basis.real.tolist(),
    'fft_basis_imag': fft_basis.imag.tolist(),
}

with open(output_dir / 'librosa_filter_basis.json', 'w') as f:
    json.dump(filter_data, f)
print(f"\nSaved filter basis to {output_dir / 'librosa_filter_basis.json'}")

# Now compute STFT with ones window
from librosa import stft
stft_result = stft(signal, n_fft=n_fft, hop_length=hop_length, window='ones', center=True)
print(f"\nSTFT shape: {stft_result.shape}")
print(f"STFT[0, 0]: {stft_result[0, 0]}")
print(f"STFT[1, 0]: {stft_result[1, 0]}")

# Compute CQT response
cqt_response = fft_basis @ stft_result
print(f"\nCQT response shape: {cqt_response.shape}")
print(f"CQT response bin 0, frame 0: {cqt_response[0, 0]}")
print(f"CQT response bin 0, frame 0 magnitude: {np.abs(cqt_response[0, 0])}")

# Compare to librosa.cqt
cqt_librosa = librosa.cqt(
    signal, sr=sr, fmin=fmin, n_bins=n_bins,
    bins_per_octave=bins_per_octave, hop_length=hop_length,
    scale=True  # default
)
print(f"\nlibrosa.cqt shape: {cqt_librosa.shape}")
print(f"librosa.cqt bin 0, frame 0: {cqt_librosa[0, 0]}")
print(f"librosa.cqt bin 0, frame 0 magnitude: {np.abs(cqt_librosa[0, 0])}")

# The difference - is scaling different?
# librosa applies sqrt(lengths) scaling when scale=True
# and sqrt(sr/my_sr) for downsampling compensation

# Manual single-octave CQT
# For our frequencies (500-8000 Hz at 16 kHz), we have 4 octaves
# The top octave processes bins 36-47 at original sample rate
# Then resample and process lower octaves

print("\n=== Checking scale differences ===")
print(f"sqrt(n_fft): {np.sqrt(n_fft)}")
print(f"sqrt(lengths[0]): {np.sqrt(returned_lengths[0])}")
print(f"Ratio: {np.sqrt(n_fft) / np.sqrt(returned_lengths[0])}")
