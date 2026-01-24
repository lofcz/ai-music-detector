"""
Exact step-by-step comparison of librosa.cqt to identify where C# diverges.
"""
import numpy as np
import librosa
from scipy.fft import fft, ifft
from scipy.fft import dct
from scipy.signal import resample
import json
from pathlib import Path
import yaml

# Match our detector's exact parameters
SR = 16000
FMIN = 500.0
N_BINS = 48
BINS_PER_OCTAVE = 12
HOP_LENGTH = 512

def load_config():
    config_path = Path(__file__).parents[2] / "python" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

np.random.seed(42)
signal = np.random.randn(8000).astype(np.float64)  # Use float64 for precision

def librosa_decimate2(y):
    """Exact librosa downsampling by 2 with scale=True."""
    return librosa.resample(y, orig_sr=2, target_sr=1, res_type='soxr_hq', scale=True)

def scipy_decimate2(y):
    """scipy.signal.resample-based decimation (what C# tries to match)."""
    n = len(y)
    m = n // 2
    # scipy.signal.resample with Fourier method
    result = resample(y, m)
    # librosa scale=True applies sqrt(2) factor for energy preservation
    result *= np.sqrt(2.0)
    return result

def fft_decimate2(y):
    """Pure FFT-based decimation matching C# implementation."""
    n = len(y)
    m = n // 2
    
    # rfft
    X = np.fft.rfft(y)  # N/2+1 complex values
    
    # Truncate to m/2+1 bins
    m2 = m // 2 + 1
    X_trunc = X[:m2].copy()
    
    # Nyquist correction for downsampling (if m is even)
    if m % 2 == 0:
        X_trunc[m // 2] *= 2
    
    # Scale by 1/s_fac (s_fac = 2 for 2x decimation)
    X_trunc /= 2.0
    
    # irfft
    result = np.fft.irfft(X_trunc, n=m)
    
    # librosa scale=True: multiply by sqrt(2)
    result *= np.sqrt(2.0)
    
    return result

print("=== Testing decimation methods ===")
test_sig = signal[:1024].copy()

lib_dec = librosa_decimate2(test_sig)
scipy_dec = scipy_decimate2(test_sig)
fft_dec = fft_decimate2(test_sig)

print(f"Input length: {len(test_sig)}")
print(f"librosa output length: {len(lib_dec)}")
print(f"scipy output length: {len(scipy_dec)}")
print(f"fft output length: {len(fft_dec)}")

lib_scipy_diff = np.abs(lib_dec - scipy_dec).mean()
lib_fft_diff = np.abs(lib_dec - fft_dec).mean()
scipy_fft_diff = np.abs(scipy_dec - fft_dec).mean()

print(f"\nMean absolute differences:")
print(f"  librosa vs scipy: {lib_scipy_diff:.8f}")
print(f"  librosa vs fft:   {lib_fft_diff:.8f}")
print(f"  scipy vs fft:     {scipy_fft_diff:.8f}")

rel_lib_scipy = lib_scipy_diff / (np.abs(lib_dec).mean() + 1e-10)
rel_lib_fft = lib_fft_diff / (np.abs(lib_dec).mean() + 1e-10)
print(f"\nRelative differences vs librosa:")
print(f"  scipy: {100*rel_lib_scipy:.4f}%")
print(f"  fft:   {100*rel_lib_fft:.4f}%")

print("\n" + "="*60)
print("=== Testing full CQT pipeline ===")

# Get librosa's CQT
cqt_librosa = librosa.cqt(
    signal, sr=SR, fmin=FMIN, n_bins=N_BINS,
    bins_per_octave=BINS_PER_OCTAVE, hop_length=HOP_LENGTH, scale=True,
    pad_mode="constant"
)
cqt_mag_librosa = np.abs(cqt_librosa)

print(f"librosa CQT shape: {cqt_mag_librosa.shape}")
print(f"librosa CQT mean magnitude: {cqt_mag_librosa.mean():.6f}")

# Now trace through librosa's algorithm manually
freqs = librosa.cqt_frequencies(n_bins=N_BINS, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE)
alpha = librosa.filters._relative_bandwidth(freqs=freqs)
n_octaves = int(np.ceil(N_BINS / BINS_PER_OCTAVE))

print(f"\nFrequencies: {freqs[0]:.1f} Hz to {freqs[-1]:.1f} Hz")
print(f"Alpha (relative bandwidth): {alpha[0]:.6f}")
print(f"Number of octaves: {n_octaves}")

# Get filter lengths at original sample rate (for final scaling)
lengths_orig, _ = librosa.filters.wavelet_lengths(
    freqs=freqs, sr=SR, filter_scale=1.0, window='hann', alpha=alpha
)
print(f"\nFilter lengths at {SR} Hz:")
print(f"  Bin 0: {lengths_orig[0]:.1f}")
print(f"  Bin 47: {lengths_orig[47]:.1f}")

# Manual CQT computation
my_y = signal.copy()
my_sr = float(SR)
my_hop = HOP_LENGTH
n_filters = BINS_PER_OCTAVE

cqt_responses = []

print("\n=== Per-octave processing ===")
for octave in range(n_octaves):
    if octave == 0:
        sl = slice(-n_filters, None)
    else:
        sl = slice(-n_filters * (octave + 1), -n_filters * octave)
    
    octave_freqs = freqs[sl]
    octave_alpha = alpha[sl]
    
    # Get filter lengths at current sample rate
    octave_lengths, _ = librosa.filters.wavelet_lengths(
        freqs=octave_freqs, sr=my_sr, filter_scale=1.0, window='hann', alpha=octave_alpha
    )
    
    # Build filter basis
    basis, _ = librosa.filters.wavelet(
        freqs=octave_freqs, sr=my_sr, filter_scale=1.0, norm=1, pad_fft=True,
        window='hann', alpha=octave_alpha
    )
    n_fft = basis.shape[1]
    
    if n_fft < 2 * my_hop:
        n_fft = int(2 ** (1 + np.ceil(np.log2(my_hop))))
    
    # Scale basis by lengths/n_fft then FFT
    basis_scaled = basis * (octave_lengths[:, np.newaxis] / n_fft)
    fft_basis = fft(basis_scaled, n=n_fft, axis=1)[:, :n_fft//2 + 1]
    
    # Apply downsampling compensation
    scale_factor = np.sqrt(SR / my_sr)
    fft_basis *= scale_factor
    
    # STFT with ones window
    stft_result = librosa.stft(
        my_y, n_fft=n_fft, hop_length=int(my_hop), window='ones',
        center=True, pad_mode="constant"
    )
    
    # CQT response
    response = fft_basis @ stft_result
    cqt_responses.append(response)
    
    print(f"Octave {octave}: sr={my_sr:.0f}, hop={my_hop}, n_fft={n_fft}, shape={response.shape}")
    
    # Downsample for next octave
    if octave < n_octaves - 1 and my_hop % 2 == 0:
        my_hop //= 2
        my_sr /= 2
        my_y = librosa_decimate2(my_y)

# Stack responses
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

# Apply final scaling (divide by sqrt(lengths))
result_scaled = result / np.sqrt(lengths_orig[:, np.newaxis])
result_mag = np.abs(result_scaled)

print(f"\nManual CQT shape: {result_mag.shape}")
print(f"Manual CQT mean magnitude: {result_mag.mean():.6f}")

# Compare
diff = np.abs(result_mag - cqt_mag_librosa[:, :min_frames])
mean_diff = diff.mean()
rel_diff = mean_diff / (cqt_mag_librosa[:, :min_frames].mean() + 1e-10)

print(f"\n=== Comparison: Manual vs librosa.cqt ===")
print(f"Mean absolute difference: {mean_diff:.8f}")
print(f"Mean relative difference: {100*rel_diff:.4f}%")
print(f"Max absolute difference: {diff.max():.8f}")

if rel_diff < 0.0001:
    print("✓ Manual implementation matches librosa within 0.01%!")
else:
    print(f"✗ Difference too large: {100*rel_diff:.4f}%")

# Now test with FFT-based decimation instead of librosa's resampler
print("\n" + "="*60)
print("=== Testing with FFT-based decimation ===")

my_y = signal.copy()
my_sr = float(SR)
my_hop = HOP_LENGTH

cqt_responses_fft = []

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
    
    stft_result = librosa.stft(
        my_y, n_fft=n_fft, hop_length=int(my_hop), window='ones',
        center=True, pad_mode="constant"
    )
    response = fft_basis @ stft_result
    cqt_responses_fft.append(response)
    
    if octave < n_octaves - 1 and my_hop % 2 == 0:
        my_hop //= 2
        my_sr /= 2
        my_y = fft_decimate2(my_y)  # Use FFT decimation instead

# Stack
min_frames = min(r.shape[1] for r in cqt_responses_fft)
result_fft = np.zeros((N_BINS, min_frames), dtype=complex)
end_bin = N_BINS
for resp in cqt_responses_fft:
    n_oct = resp.shape[0]
    if end_bin < n_oct:
        result_fft[:end_bin, :] = resp[-end_bin:, :min_frames]
    else:
        result_fft[end_bin - n_oct:end_bin, :] = resp[:, :min_frames]
    end_bin -= n_oct

result_fft_scaled = result_fft / np.sqrt(lengths_orig[:, np.newaxis])
result_fft_mag = np.abs(result_fft_scaled)

diff_fft = np.abs(result_fft_mag - cqt_mag_librosa[:, :min_frames])
mean_diff_fft = diff_fft.mean()
rel_diff_fft = mean_diff_fft / (cqt_mag_librosa[:, :min_frames].mean() + 1e-10)

print(f"Mean absolute difference: {mean_diff_fft:.8f}")
print(f"Mean relative difference: {100*rel_diff_fft:.4f}%")

if rel_diff_fft < 0.001:
    print("✓ FFT decimation matches librosa within 0.1%!")
else:
    print(f"✗ FFT decimation causes {100*rel_diff_fft:.4f}% difference")
    print("  → The C# implementation needs better resampling")

# Save comparison data for C# testing
output_dir = Path(__file__).parent.parent / 'testdata'
output_dir.mkdir(exist_ok=True)

config = load_config()
n_coeffs = int(config["cepstrum"]["n_coeffs"])
log_cqt = np.log(cqt_mag_librosa + 1e-6)
cepstrum = dct(log_cqt, type=2, axis=0, norm='ortho')[:n_coeffs, :]

comparison_data = {
    'sample_rate': SR,
    'f_min': FMIN,
    'n_bins': N_BINS,
    'bins_per_octave': BINS_PER_OCTAVE,
    'hop_length': HOP_LENGTH,
    'n_coeffs': n_coeffs,
    'input': signal.tolist(),
    'expected_magnitude': cqt_mag_librosa.tolist(),
    'expected_cepstrum': cepstrum.tolist(),
    'librosa_decimate_vs_fft_decimate_rel_diff': float(rel_diff_fft),
}

with open(output_dir / 'librosa_cqt_exact.json', 'w') as f:
    json.dump(comparison_data, f)

print(f"\nSaved comparison data to {output_dir / 'librosa_cqt_exact.json'}")
