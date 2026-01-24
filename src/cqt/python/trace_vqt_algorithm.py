"""
Trace the VQT/CQT algorithm to understand the exact scaling applied.
"""
import numpy as np
import librosa
from scipy.fft import fft
import json
from pathlib import Path

sr = 16000
fmin = 500.0
n_bins = 48
bins_per_octave = 12
hop_length = 512

np.random.seed(42)
signal = np.random.randn(8000).astype(np.float32)

# Get all frequencies
freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
print(f"All frequencies: {freqs[0]:.1f} Hz to {freqs[-1]:.1f} Hz")

# Compute the number of octaves
n_octaves = int(np.ceil(n_bins / bins_per_octave))
print(f"Number of octaves: {n_octaves}")

# librosa VQT processes from highest to lowest octave
print("\n=== Octave-by-octave analysis ===")

# For CQT (gamma=0), each octave is processed at a different sample rate
# Starting with the highest octave at the original sample rate

# Get alpha (relative bandwidth)
alpha = librosa.filters._relative_bandwidth(freqs=freqs)
print(f"Alpha (constant for equal temperament): {alpha[0]:.6f}")

# Get all filter lengths at original sample rate
lengths, _ = librosa.filters.wavelet_lengths(
    freqs=freqs, sr=sr, filter_scale=1.0, window='hann', alpha=alpha
)
print(f"Filter lengths at {sr} Hz sample rate:")
print(f"  Bin 0 (500 Hz): {lengths[0]:.1f}")
print(f"  Bin 12 (1000 Hz): {lengths[12]:.1f}")
print(f"  Bin 24 (2000 Hz): {lengths[24]:.1f}")
print(f"  Bin 36 (4000 Hz): {lengths[36]:.1f}")
print(f"  Bin 47 (highest): {lengths[47]:.1f}")

# Now let's manually trace through each octave like VQT does
my_y = signal.copy()
my_sr = sr
my_hop = hop_length
n_filters = bins_per_octave

cqt_responses = []

for octave in range(n_octaves):
    if octave == 0:
        sl = slice(-n_filters, None)  # Highest octave: bins 36-47
    else:
        sl = slice(-n_filters * (octave + 1), -n_filters * octave)
    
    octave_freqs = freqs[sl]
    octave_alpha = alpha[sl]
    
    print(f"\n--- Octave {octave} ---")
    print(f"  Bins: {sl}")
    print(f"  Frequencies: {octave_freqs[0]:.1f} Hz to {octave_freqs[-1]:.1f} Hz")
    print(f"  Current sample rate: {my_sr}")
    print(f"  Current hop length: {my_hop}")
    
    # Get filter lengths at CURRENT sample rate
    octave_lengths, _ = librosa.filters.wavelet_lengths(
        freqs=octave_freqs, sr=my_sr, filter_scale=1.0, window='hann', alpha=octave_alpha
    )
    print(f"  Filter lengths at this SR: {octave_lengths[0]:.1f} to {octave_lengths[-1]:.1f}")
    
    # Build filter basis at current sample rate
    basis, _ = librosa.filters.wavelet(
        freqs=octave_freqs, sr=my_sr, filter_scale=1.0, norm=1, pad_fft=True,
        window='hann', alpha=octave_alpha
    )
    n_fft = basis.shape[1]
    if n_fft < 2 * my_hop:
        n_fft = int(2 ** (1 + np.ceil(np.log2(my_hop))))
    print(f"  n_fft: {n_fft}")
    
    # Scale by lengths/n_fft
    basis_scaled = basis * (octave_lengths[:, np.newaxis] / n_fft)
    
    # FFT
    fft_basis = fft(basis_scaled, n=n_fft, axis=1)[:, :n_fft//2 + 1]
    
    # Apply downsampling compensation: sqrt(sr / my_sr)
    scale_factor = np.sqrt(sr / my_sr)
    fft_basis *= scale_factor
    print(f"  Downsampling compensation: sqrt({sr}/{my_sr}) = {scale_factor:.4f}")
    
    # Compute STFT with ones window
    stft_result = librosa.stft(my_y, n_fft=n_fft, hop_length=my_hop, window='ones', center=True)
    print(f"  STFT shape: {stft_result.shape}")
    
    # CQT response
    response = fft_basis @ stft_result
    cqt_responses.append(response)
    print(f"  CQT response shape: {response.shape}")
    print(f"  Response[0,0] magnitude: {np.abs(response[0,0]):.4f}")
    
    # Downsample for next octave (if hop is even)
    if octave < n_octaves - 1 and my_hop % 2 == 0:
        my_hop //= 2
        my_sr /= 2
        # Resample by 2 with energy preservation
        my_y = librosa.resample(my_y, orig_sr=2, target_sr=1, res_type='soxr_hq', scale=True)
        print(f"  After resampling: signal length = {len(my_y)}")

# Now stack and apply final scaling (scale=True means divide by sqrt(lengths))
print("\n=== Stacking and final scaling ===")

# Get min frames
min_frames = min(r.shape[1] for r in cqt_responses)
print(f"Min frames: {min_frames}")

# Stack from highest to lowest
result = np.zeros((n_bins, min_frames), dtype=complex)
end_bin = n_bins
for i, resp in enumerate(cqt_responses):
    n_oct = resp.shape[0]
    if end_bin < n_oct:
        result[:end_bin, :] = resp[-end_bin:, :min_frames]
    else:
        result[end_bin - n_oct:end_bin, :] = resp[:, :min_frames]
    end_bin -= n_oct

print(f"Stacked result shape: {result.shape}")
print(f"Before scale, bin 0 frame 0 magnitude: {np.abs(result[0, 0]):.4f}")

# Apply sqrt(lengths) scaling (scale=True)
# Recompute lengths at ORIGINAL sample rate for this scaling
lengths_for_scale, _ = librosa.filters.wavelet_lengths(
    freqs=freqs, sr=sr, filter_scale=1.0, window='hann', alpha=alpha
)
result_scaled = result / np.sqrt(lengths_for_scale[:, np.newaxis])
print(f"After scale (divide by sqrt(lengths)), bin 0 frame 0 magnitude: {np.abs(result_scaled[0, 0]):.4f}")

# Compare to librosa
cqt_librosa = librosa.cqt(
    signal, sr=sr, fmin=fmin, n_bins=n_bins,
    bins_per_octave=bins_per_octave, hop_length=hop_length, scale=True
)
print(f"\nlibrosa.cqt bin 0 frame 0 magnitude: {np.abs(cqt_librosa[0, 0]):.4f}")

# Difference
diff = np.abs(result_scaled[:, :min_frames]) - np.abs(cqt_librosa[:, :min_frames])
mean_abs_diff = np.abs(diff).mean()
mean_rel_diff = np.abs(diff).mean() / (np.abs(cqt_librosa[:, :min_frames]).mean() + 1e-10)
print(f"\nMean absolute difference: {mean_abs_diff:.6f}")
print(f"Mean relative difference: {100*mean_rel_diff:.2f}%")

# Save intermediate values for C# comparison
output_dir = Path(__file__).parent.parent / 'testdata'

# For each octave, save the filter basis
for octave in range(n_octaves):
    if octave == 0:
        sl = slice(-n_filters, None)
    else:
        sl = slice(-n_filters * (octave + 1), -n_filters * octave)
    
    octave_freqs = freqs[sl]
    octave_alpha = alpha[sl]
    octave_sr = sr / (2 ** octave)
    octave_hop = hop_length // (2 ** octave)
    
    octave_lengths, _ = librosa.filters.wavelet_lengths(
        freqs=octave_freqs, sr=octave_sr, filter_scale=1.0, window='hann', alpha=octave_alpha
    )
    
    basis, _ = librosa.filters.wavelet(
        freqs=octave_freqs, sr=octave_sr, filter_scale=1.0, norm=1, pad_fft=True,
        window='hann', alpha=octave_alpha
    )
    n_fft = basis.shape[1]
    if n_fft < 2 * octave_hop:
        n_fft = int(2 ** (1 + np.ceil(np.log2(octave_hop))))
    
    basis_scaled = basis * (octave_lengths[:, np.newaxis] / n_fft)
    fft_basis = fft(basis_scaled, n=n_fft, axis=1)[:, :n_fft//2 + 1]
    scale_factor = np.sqrt(sr / octave_sr)
    fft_basis *= scale_factor
    
    octave_data = {
        'octave': octave,
        'sample_rate': float(octave_sr),
        'hop_length': int(octave_hop),
        'n_fft': int(n_fft),
        'frequencies': octave_freqs.tolist(),
        'lengths': octave_lengths.tolist(),
        'scale_factor': float(scale_factor),
        'fft_basis_real': fft_basis.real.tolist(),
        'fft_basis_imag': fft_basis.imag.tolist(),
    }
    
    with open(output_dir / f'librosa_octave_{octave}.json', 'w') as f:
        json.dump(octave_data, f)
    print(f"Saved octave {octave} data")
