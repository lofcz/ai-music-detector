"""
Verify that librosa res_type='scipy' matches our FFT-based decimation EXACTLY.
If it does, we can re-train with scipy and the C# will match perfectly.
"""
import numpy as np
import librosa
from scipy.fft import fft
from scipy.signal import resample

SR = 16000
FMIN = 500.0
N_BINS = 48
BINS_PER_OCTAVE = 12
HOP_LENGTH = 512

np.random.seed(42)
signal = np.random.randn(16000).astype(np.float64)

def fft_decimate2(y):
    """Our FFT-based decimation (what C# does)."""
    n = len(y)
    m = n // 2
    
    X = np.fft.rfft(y)
    m2 = m // 2 + 1
    X_trunc = X[:m2].copy()
    
    if m % 2 == 0:
        X_trunc[m // 2] *= 2
    
    X_trunc /= 2.0
    result = np.fft.irfft(X_trunc, n=m)
    result *= np.sqrt(2.0)  # librosa scale=True
    
    return result

def scipy_resample_decimate2(y):
    """scipy.signal.resample (what librosa res_type='scipy' uses internally)."""
    result = resample(y, len(y) // 2)
    result *= np.sqrt(2.0)  # librosa scale=True
    return result

# Test decimation match
print("=== Testing decimation implementations ===")
test_sig = signal[:2048].copy()

fft_dec = fft_decimate2(test_sig)
scipy_dec = scipy_resample_decimate2(test_sig)

diff = np.abs(fft_dec - scipy_dec).max()
print(f"FFT decimation vs scipy.signal.resample:")
print(f"  Max difference: {diff:.2e}")

if diff < 1e-10:
    print("  ✓ IDENTICAL - scipy.signal.resample uses FFT internally!")
else:
    print(f"  ✗ Different by {diff}")

# Now test full CQT with res_type='scipy' against manual implementation
print("\n=== Testing full CQT with res_type='scipy' ===")

# Get librosa CQT with scipy resampling
cqt_scipy = np.abs(librosa.cqt(
    signal, sr=SR, fmin=FMIN, n_bins=N_BINS,
    bins_per_octave=BINS_PER_OCTAVE, hop_length=HOP_LENGTH, scale=True,
    res_type='scipy'
))

print(f"librosa CQT (res_type='scipy') shape: {cqt_scipy.shape}")

# Manual implementation with FFT decimation (matching C#)
freqs = librosa.cqt_frequencies(n_bins=N_BINS, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE)
alpha = librosa.filters._relative_bandwidth(freqs=freqs)
n_octaves = int(np.ceil(N_BINS / BINS_PER_OCTAVE))
lengths_orig, _ = librosa.filters.wavelet_lengths(
    freqs=freqs, sr=SR, filter_scale=1.0, window='hann', alpha=alpha
)

my_y = signal.copy()
my_sr = float(SR)
my_hop = HOP_LENGTH
n_filters = BINS_PER_OCTAVE

cqt_responses = []

for octave in range(n_octaves):
    if octave == 0:
        sl = slice(-n_filters, None)
    else:
        sl = slice(-n_filters * (octave + 1), -n_filters * octave)
    
    octave_freqs = freqs[sl]
    octave_alpha = alpha[sl]
    
    octave_lengths, _ = librosa.filters.wavelet_lengths(
        freqs=octave_freqs, sr=my_sr, filter_scale=1.0, window='hann', alpha=octave_alpha
    )
    
    basis, _ = librosa.filters.wavelet(
        freqs=octave_freqs, sr=my_sr, filter_scale=1.0, norm=1, pad_fft=True,
        window='hann', alpha=octave_alpha
    )
    n_fft = basis.shape[1]
    
    if n_fft < 2 * my_hop:
        n_fft = int(2 ** (1 + np.ceil(np.log2(my_hop))))
    
    basis_scaled = basis * (octave_lengths[:, np.newaxis] / n_fft)
    fft_basis = fft(basis_scaled, n=n_fft, axis=1)[:, :n_fft//2 + 1]
    scale_factor = np.sqrt(SR / my_sr)
    fft_basis *= scale_factor
    
    stft_result = librosa.stft(my_y, n_fft=n_fft, hop_length=int(my_hop), window='ones', center=True)
    response = fft_basis @ stft_result
    cqt_responses.append(response)
    
    if octave < n_octaves - 1 and my_hop % 2 == 0:
        my_hop //= 2
        my_sr /= 2
        my_y = fft_decimate2(my_y)  # Use FFT decimation (matches C#)

# Stack
min_frames = min(r.shape[1] for r in cqt_responses)
result = np.zeros((N_BINS, min_frames), dtype=complex)
end_bin = N_BINS
for resp in cqt_responses:
    n_oct = resp.shape[0]
    if end_bin < n_oct:
        result[:end_bin, :] = resp[-end_bin:, :min_frames]
    else:
        result[end_bin - n_oct:end_bin, :] = resp[:, :min_frames]
    end_bin -= n_oct

result_scaled = result / np.sqrt(lengths_orig[:, np.newaxis])
result_mag = np.abs(result_scaled)

print(f"Manual CQT (FFT decimation) shape: {result_mag.shape}")

# Compare
min_f = min(cqt_scipy.shape[1], result_mag.shape[1])
diff = np.abs(result_mag[:, :min_f] - cqt_scipy[:, :min_f])
mean_diff = diff.mean()
max_diff = diff.max()
rel_diff = mean_diff / (cqt_scipy[:, :min_f].mean() + 1e-10)

print(f"\n=== Comparison: Manual FFT vs librosa res_type='scipy' ===")
print(f"Mean absolute difference: {mean_diff:.10f}")
print(f"Max absolute difference:  {max_diff:.10f}")
print(f"Mean relative difference: {100*rel_diff:.6f}%")

if rel_diff < 0.0001:
    print(f"\n✓ PERFECT MATCH within 0.01%!")
    print("\nCONCLUSION: If you re-train with res_type='scipy',")
    print("            the C# FFT-based implementation will match EXACTLY.")
elif rel_diff < 0.001:
    print(f"\n✓ Very close match within 0.1%!")
    print("\nCONCLUSION: Re-training with res_type='scipy' should work well.")
else:
    print(f"\n✗ Still differs by {100*rel_diff:.4f}%")
    print("   There may be other algorithmic differences to investigate.")

# Per-octave breakdown
print("\nPer-octave differences:")
for octave in range(4):
    start_bin = octave * 12
    end_bin = (octave + 1) * 12
    oct_diff = np.abs(result_mag[start_bin:end_bin, :min_f] - cqt_scipy[start_bin:end_bin, :min_f])
    oct_ref = cqt_scipy[start_bin:end_bin, :min_f]
    oct_rel = oct_diff.mean() / (oct_ref.mean() + 1e-10)
    print(f"  Octave {octave} (bins {start_bin}-{end_bin-1}): {100*oct_rel:.6f}%")
