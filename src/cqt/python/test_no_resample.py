"""
Test if we can compute CQT without resampling by processing all bins at original sample rate.
This avoids the soxr_hq vs FFT decimation mismatch entirely.
"""
import numpy as np
import librosa
from scipy.fft import fft
import time

# Match our detector's exact parameters
SR = 16000
FMIN = 500.0
N_BINS = 48
BINS_PER_OCTAVE = 12
HOP_LENGTH = 512

np.random.seed(42)
signal = np.random.randn(16000).astype(np.float64)  # 1 second

print("=== Direct CQT (no resampling) vs librosa.cqt ===\n")

# Get librosa's CQT (uses resampling internally)
t0 = time.time()
cqt_librosa = librosa.cqt(
    signal, sr=SR, fmin=FMIN, n_bins=N_BINS,
    bins_per_octave=BINS_PER_OCTAVE, hop_length=HOP_LENGTH, scale=True
)
librosa_time = time.time() - t0
cqt_mag_librosa = np.abs(cqt_librosa)

print(f"librosa.cqt shape: {cqt_mag_librosa.shape}, time: {librosa_time*1000:.1f}ms")

# Now compute directly at full sample rate for ALL bins (no octave decomposition)
freqs = librosa.cqt_frequencies(n_bins=N_BINS, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE)
alpha = librosa.filters._relative_bandwidth(freqs=freqs)

# Get filter lengths at ORIGINAL sample rate
lengths, _ = librosa.filters.wavelet_lengths(
    freqs=freqs, sr=SR, filter_scale=1.0, window='hann', alpha=alpha
)

print(f"\nFilter lengths at {SR} Hz (no downsampling):")
print(f"  Bin 0 ({freqs[0]:.0f} Hz): {lengths[0]:.0f} samples")
print(f"  Bin 47 ({freqs[47]:.0f} Hz): {lengths[47]:.0f} samples")
print(f"  Max filter length: {max(lengths):.0f}")

# Build single filter basis at original sample rate
t0 = time.time()
basis, _ = librosa.filters.wavelet(
    freqs=freqs, sr=SR, filter_scale=1.0, norm=1, pad_fft=True,
    window='hann', alpha=alpha
)
n_fft = basis.shape[1]

# Ensure n_fft >= 2*hop
if n_fft < 2 * HOP_LENGTH:
    n_fft = int(2 ** (1 + np.ceil(np.log2(HOP_LENGTH))))

print(f"\nSingle FFT size for all bins: {n_fft}")
print(f"Filter basis shape: {basis.shape}")

# Scale and FFT the basis
basis_scaled = basis * (lengths[:, np.newaxis] / n_fft)
fft_basis = fft(basis_scaled, n=n_fft, axis=1)[:, :n_fft//2 + 1]

# No downsampling compensation needed since we're at original SR
# (scale_factor = sqrt(SR/SR) = 1)

# Compute STFT with ones window
stft_result = librosa.stft(signal, n_fft=n_fft, hop_length=HOP_LENGTH, window='ones', center=True)
print(f"STFT shape: {stft_result.shape}")

# CQT response
cqt_direct = fft_basis @ stft_result

# Apply final scaling (divide by sqrt(lengths)) - same as librosa scale=True
cqt_direct_scaled = cqt_direct / np.sqrt(lengths[:, np.newaxis])
cqt_direct_mag = np.abs(cqt_direct_scaled)
direct_time = time.time() - t0

print(f"Direct CQT shape: {cqt_direct_mag.shape}, time: {direct_time*1000:.1f}ms")

# Compare
min_frames = min(cqt_mag_librosa.shape[1], cqt_direct_mag.shape[1])
diff = np.abs(cqt_direct_mag[:, :min_frames] - cqt_mag_librosa[:, :min_frames])
mean_diff = diff.mean()
max_diff = diff.max()
rel_diff = mean_diff / (cqt_mag_librosa[:, :min_frames].mean() + 1e-10)

print(f"\n=== Comparison ===")
print(f"Mean absolute difference: {mean_diff:.10f}")
print(f"Max absolute difference:  {max_diff:.10f}")
print(f"Mean relative difference: {100*rel_diff:.6f}%")

# Per-octave comparison
print(f"\nPer-octave relative differences:")
for octave in range(4):
    start_bin = octave * 12
    end_bin = (octave + 1) * 12
    octave_diff = np.abs(cqt_direct_mag[start_bin:end_bin, :min_frames] - cqt_mag_librosa[start_bin:end_bin, :min_frames])
    octave_ref = np.abs(cqt_mag_librosa[start_bin:end_bin, :min_frames])
    octave_rel = octave_diff.mean() / (octave_ref.mean() + 1e-10)
    print(f"  Octave {octave} (bins {start_bin}-{end_bin-1}, {freqs[start_bin]:.0f}-{freqs[end_bin-1]:.0f} Hz): {100*octave_rel:.6f}%")

if rel_diff < 0.0001:  # 0.01%
    print(f"\n✓ Direct CQT matches librosa within 0.01%!")
    print("  This approach avoids resampling entirely - no soxr needed!")
elif rel_diff < 0.001:  # 0.1%
    print(f"\n✓ Direct CQT matches librosa within 0.1% - acceptable!")
else:
    print(f"\n✗ Direct CQT differs by {100*rel_diff:.4f}% - too much")

# Check if shapes match
if cqt_direct_mag.shape != cqt_mag_librosa.shape:
    print(f"\nWARNING: Shape mismatch!")
    print(f"  Direct: {cqt_direct_mag.shape}")
    print(f"  librosa: {cqt_mag_librosa.shape}")
