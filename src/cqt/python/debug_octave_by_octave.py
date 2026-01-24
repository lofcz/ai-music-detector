"""
Debug CQT octave by octave to find where the difference comes from.
"""
import numpy as np
import librosa
from scipy.fft import fft

SR = 16000
FMIN = 500.0
N_BINS = 48
BINS_PER_OCTAVE = 12
HOP_LENGTH = 512

np.random.seed(42)
signal = np.random.randn(16000).astype(np.float64)

freqs = librosa.cqt_frequencies(n_bins=N_BINS, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE)
print(f"Frequencies: {freqs[0]:.0f} Hz to {freqs[-1]:.0f} Hz")
print(f"Bins 0-11: {freqs[0]:.0f}-{freqs[11]:.0f} Hz (lowest octave)")
print(f"Bins 36-47: {freqs[36]:.0f}-{freqs[47]:.0f} Hz (highest octave)")

# Test with just the highest octave (no resampling needed)
print("\n=== Testing ONLY highest octave (bins 36-47, no resampling) ===")

# Get librosa's CQT for just the top octave
cqt_top12 = librosa.cqt(
    signal, sr=SR, fmin=freqs[36], n_bins=12,  # Just top 12 bins
    bins_per_octave=BINS_PER_OCTAVE, hop_length=HOP_LENGTH, scale=True,
    res_type='scipy'
)
print(f"librosa CQT (top 12 bins) shape: {cqt_top12.shape}")

# Manual implementation for just the top octave
top_freqs = freqs[36:48]
alpha = librosa.filters._relative_bandwidth(freqs=top_freqs)
lengths, _ = librosa.filters.wavelet_lengths(
    freqs=top_freqs, sr=SR, filter_scale=1.0, window='hann', alpha=alpha
)

basis, _ = librosa.filters.wavelet(
    freqs=top_freqs, sr=SR, filter_scale=1.0, norm=1, pad_fft=True,
    window='hann', alpha=alpha
)
n_fft = basis.shape[1]

if n_fft < 2 * HOP_LENGTH:
    n_fft = int(2 ** (1 + np.ceil(np.log2(HOP_LENGTH))))

print(f"Filter basis shape: {basis.shape}")
print(f"n_fft: {n_fft}")

# Scale by lengths/n_fft then FFT
basis_scaled = basis * (lengths[:, np.newaxis] / n_fft)
fft_basis = fft(basis_scaled, n=n_fft, axis=1)[:, :n_fft//2 + 1]

# STFT with ones window (as librosa's __cqt_response does)
stft_result = librosa.stft(signal, n_fft=n_fft, hop_length=HOP_LENGTH, window='ones', center=True)
print(f"STFT shape: {stft_result.shape}")

# CQT response
response = fft_basis @ stft_result

# Apply sqrt(length) scaling
result_scaled = response / np.sqrt(lengths[:, np.newaxis])
result_mag = np.abs(result_scaled)
librosa_mag = np.abs(cqt_top12)

print(f"Manual result shape: {result_mag.shape}")
print(f"librosa result shape: {librosa_mag.shape}")

# Compare
min_f = min(result_mag.shape[1], librosa_mag.shape[1])
diff = np.abs(result_mag[:, :min_f] - librosa_mag[:, :min_f])
mean_diff = diff.mean()
rel_diff = mean_diff / (librosa_mag[:, :min_f].mean() + 1e-10)

print(f"\nMean absolute difference: {mean_diff:.8f}")
print(f"Mean relative difference: {100*rel_diff:.4f}%")

if rel_diff < 0.001:
    print("✓ Top octave matches within 0.1%!")
else:
    print(f"✗ Top octave differs by {100*rel_diff:.4f}%")
    
    # Debug: compare sample values
    print("\n--- Debug: First few values comparison ---")
    print("Bin | librosa[0] | manual[0] | diff")
    for i in range(min(5, 12)):
        lib_val = librosa_mag[i, 0]
        man_val = result_mag[i, 0]
        print(f" {i:2d} | {lib_val:10.6f} | {man_val:10.6f} | {abs(lib_val-man_val):10.6f}")

# Now test: what if we use librosa's exact filter basis?
print("\n=== Testing with librosa's filter basis ===")

# Get librosa's internal filter basis
fft_basis_librosa, n_fft_lib, _ = librosa.core.spectrum.__vqt_filter_fft(
    sr=SR,
    freqs=top_freqs,
    filter_scale=1.0,
    norm=1,
    sparsity=0.01,
    hop_length=HOP_LENGTH,
    window='hann',
    dtype=np.complex128,
    alpha=alpha,
)

print(f"librosa fft_basis shape: {fft_basis_librosa.shape}")
print(f"librosa n_fft: {n_fft_lib}")

# Use librosa's basis but our STFT
stft_lib = librosa.stft(signal, n_fft=n_fft_lib, hop_length=HOP_LENGTH, window='ones', center=True)

# Convert sparse to dense if needed
if hasattr(fft_basis_librosa, 'toarray'):
    fft_basis_dense = fft_basis_librosa.toarray()
else:
    fft_basis_dense = np.asarray(fft_basis_librosa)

response_with_lib_basis = fft_basis_dense @ stft_lib
response_with_lib_basis_scaled = response_with_lib_basis / np.sqrt(lengths[:, np.newaxis])
mag_with_lib_basis = np.abs(response_with_lib_basis_scaled)

min_f = min(mag_with_lib_basis.shape[1], librosa_mag.shape[1])
diff2 = np.abs(mag_with_lib_basis[:, :min_f] - librosa_mag[:, :min_f])
rel_diff2 = diff2.mean() / (librosa_mag[:, :min_f].mean() + 1e-10)

print(f"Using librosa's filter basis:")
print(f"  Mean relative difference: {100*rel_diff2:.6f}%")

if rel_diff2 < 0.0001:
    print("  ✓ Using librosa's basis gives exact match!")
    print("  → Problem is in filter basis construction, not STFT or stacking")
else:
    print(f"  ✗ Still differs by {100*rel_diff2:.4f}%")
    print("  → Problem is in STFT or final processing, not filter basis")
