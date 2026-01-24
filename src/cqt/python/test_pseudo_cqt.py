"""
Test pseudo_cqt which uses a simpler single-pass approach.
This should be easier to match in C#.
"""
import numpy as np
import librosa
from scipy.fft import fft
import json
from pathlib import Path

# Match our detector parameters
sr = 16000
fmin = 500.0
n_bins = 48
bins_per_octave = 12
hop_length = 512

# Generate test signal
np.random.seed(42)
signal = np.random.randn(8000).astype(np.float32)

# Use pseudo_cqt instead of cqt (single FFT size, no recursive downsampling)
pseudo_result = librosa.pseudo_cqt(
    signal, sr=sr, fmin=fmin, n_bins=n_bins,
    bins_per_octave=bins_per_octave, hop_length=hop_length,
    scale=True
)
print(f"Pseudo-CQT shape: {pseudo_result.shape}")
print(f"Pseudo-CQT range: {pseudo_result.min():.6f} to {pseudo_result.max():.6f}")

# Compare to full CQT
cqt_result = np.abs(librosa.cqt(
    signal, sr=sr, fmin=fmin, n_bins=n_bins,
    bins_per_octave=bins_per_octave, hop_length=hop_length
))
print(f"\nFull CQT shape: {cqt_result.shape}")
print(f"Full CQT range: {cqt_result.min():.6f} to {cqt_result.max():.6f}")

# Difference
min_frames = min(pseudo_result.shape[1], cqt_result.shape[1])
diff = pseudo_result[:, :min_frames] - cqt_result[:, :min_frames]
print(f"\nDiff pseudo vs full CQT:")
print(f"  Max abs diff: {np.abs(diff).max():.6f}")
print(f"  Mean abs diff: {np.abs(diff).mean():.6f}")
print(f"  Relative diff: {100 * np.abs(diff).mean() / (np.abs(cqt_result[:, :min_frames]).mean() + 1e-10):.2f}%")

# Now let's manually reconstruct pseudo_cqt
print("\n--- Manual Pseudo-CQT reconstruction ---")

# Get frequencies
freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
alpha = librosa.filters._relative_bandwidth(freqs=freqs)

# Get wavelet lengths
lengths, _ = librosa.filters.wavelet_lengths(
    freqs=freqs, sr=sr, filter_scale=1.0, window='hann', alpha=alpha
)
print(f"Lengths: {lengths[:3]}... max={lengths.max():.1f}")

# Build filter bank (as in __vqt_filter_fft)
basis, _ = librosa.filters.wavelet(
    freqs=freqs, sr=sr, filter_scale=1.0, norm=1, pad_fft=True, 
    window='hann', alpha=alpha
)
n_fft = basis.shape[1]

# Ensure n_fft >= 2 * hop_length
if n_fft < 2 * hop_length:
    n_fft = int(2 ** (1 + np.ceil(np.log2(hop_length))))
print(f"n_fft: {n_fft}")

# Re-normalize for FFT length
basis_scaled = basis * (lengths[:, np.newaxis] / n_fft)

# FFT and keep positive frequencies
fft_basis = fft(basis_scaled, n=n_fft, axis=1)[:, :n_fft//2 + 1]

# For pseudo_cqt, we use abs of fft_basis
fft_basis_mag = np.abs(fft_basis)
print(f"FFT basis mag shape: {fft_basis_mag.shape}")

# Compute STFT with hann window (NOT ones!)
# Looking at pseudo_cqt source: it uses window='hann' in __cqt_response with phase=False
stft_result = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, window='hann', center=True)
stft_mag = np.abs(stft_result)
print(f"STFT shape: {stft_result.shape}")

# For pseudo_cqt, it multiplies abs(fft_basis) @ abs(stft)
manual_pseudo = fft_basis_mag @ stft_mag
print(f"Manual pseudo shape: {manual_pseudo.shape}")

# Apply scaling: scale=True means divide by sqrt(n_fft)
manual_pseudo_scaled = manual_pseudo / np.sqrt(n_fft)
print(f"After sqrt(n_fft) scaling: range {manual_pseudo_scaled.min():.6f} to {manual_pseudo_scaled.max():.6f}")

# Compare
min_frames = min(manual_pseudo_scaled.shape[1], pseudo_result.shape[1])
diff = manual_pseudo_scaled[:, :min_frames] - pseudo_result[:, :min_frames]
print(f"\nDiff manual vs librosa pseudo_cqt:")
print(f"  Max abs diff: {np.abs(diff).max():.10f}")
print(f"  Mean abs diff: {np.abs(diff).mean():.10f}")

# This should be very close (numerical precision only)
if np.abs(diff).max() < 1e-5:
    print("  SUCCESS: Manual matches librosa pseudo_cqt!")
else:
    print("  Still some difference - investigating...")
    # Check per-bin
    for k in [0, 12, 24, 36, 47]:
        bin_diff = np.abs(manual_pseudo_scaled[k, :min_frames] - pseudo_result[k, :min_frames])
        print(f"  Bin {k} max diff: {bin_diff.max():.10f}")

# Save the exact filter bank for C# testing
output_dir = Path(__file__).parent.parent / 'testdata'
output_dir.mkdir(exist_ok=True)

# Save complete test case
test_data = {
    'sample_rate': int(sr),
    'fmin': float(fmin),
    'n_bins': int(n_bins),
    'bins_per_octave': int(bins_per_octave),
    'hop_length': int(hop_length),
    'n_fft': int(n_fft),
    'input': signal.tolist(),
    'frequencies': freqs.tolist(),
    'alpha': alpha.tolist(),
    'lengths': lengths.tolist(),
    'fft_basis_magnitude': fft_basis_mag.tolist(),
    'expected_pseudo_cqt': pseudo_result.tolist(),
    'stft_magnitude': stft_mag.tolist(),
}

with open(output_dir / 'pseudo_cqt_test.json', 'w') as f:
    json.dump(test_data, f)
print(f"\nSaved to {output_dir / 'pseudo_cqt_test.json'}")

# Also test a 1kHz sine wave
print("\n--- 1kHz Sine Wave Test ---")
duration = 0.5
t = np.arange(int(sr * duration)) / sr
sine_1khz = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

pseudo_sine = librosa.pseudo_cqt(
    sine_1khz, sr=sr, fmin=fmin, n_bins=n_bins,
    bins_per_octave=bins_per_octave, hop_length=hop_length, scale=True
)
print(f"Sine pseudo-CQT shape: {pseudo_sine.shape}")

# Find peak bin
total_energy = pseudo_sine.sum(axis=1)
peak_bin = np.argmax(total_energy)
peak_freq = freqs[peak_bin]
print(f"Peak bin: {peak_bin} ({peak_freq:.1f} Hz)")
print(f"Expected: bin 12 (1000 Hz)")

# Top 5 bins
top_bins = np.argsort(total_energy)[-5:][::-1]
print(f"Top 5 bins: {top_bins} with freqs {freqs[top_bins]}")

sine_test = {
    'sample_rate': int(sr),
    'fmin': float(fmin),
    'n_bins': int(n_bins),
    'bins_per_octave': int(bins_per_octave),
    'hop_length': int(hop_length),
    'input': sine_1khz.tolist(),
    'expected_pseudo_cqt': pseudo_sine.tolist(),
}

with open(output_dir / 'pseudo_cqt_sine.json', 'w') as f:
    json.dump(sine_test, f)
print(f"Saved sine test to {output_dir / 'pseudo_cqt_sine.json'}")
