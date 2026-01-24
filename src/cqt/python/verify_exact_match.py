"""
Verify the C# LibrosaCQT implementation matches librosa.cqt exactly.
This script generates test data and saves it to compare with C# output.
"""
import numpy as np
import librosa
import json
from pathlib import Path

# Match our detector parameters from config.yaml
sr = 16000
fmin = 500.0
n_bins = 48
bins_per_octave = 12
hop_length = 512

# Generate test signals
np.random.seed(42)
random_signal = np.random.randn(8000).astype(np.float32)

# Generate 1kHz sine wave
t = np.arange(8000) / sr
sine_1khz = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

print("=" * 60)
print("librosa.cqt Reference Values")
print("=" * 60)

# Compute CQT for random signal
cqt_random = librosa.cqt(
    random_signal, sr=sr, fmin=fmin, n_bins=n_bins,
    bins_per_octave=bins_per_octave, hop_length=hop_length
)
cqt_random_mag = np.abs(cqt_random)

print(f"\nRandom signal CQT:")
print(f"  Shape: {cqt_random_mag.shape}")
print(f"  Range: {cqt_random_mag.min():.6f} to {cqt_random_mag.max():.6f}")
print(f"  Mean: {cqt_random_mag.mean():.6f}")
print(f"  First 5 values [0, :5]: {cqt_random_mag[0, :5]}")

# Compute CQT for sine wave
cqt_sine = librosa.cqt(
    sine_1khz, sr=sr, fmin=fmin, n_bins=n_bins,
    bins_per_octave=bins_per_octave, hop_length=hop_length
)
cqt_sine_mag = np.abs(cqt_sine)

print(f"\n1kHz Sine wave CQT:")
print(f"  Shape: {cqt_sine_mag.shape}")
print(f"  Range: {cqt_sine_mag.min():.6f} to {cqt_sine_mag.max():.6f}")

# Find peak bin
total_energy = cqt_sine_mag.sum(axis=1)
peak_bin = np.argmax(total_energy)
freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
print(f"  Peak bin: {peak_bin} ({freqs[peak_bin]:.1f} Hz)")
print(f"  Expected: bin 12 (1000 Hz)")

# Save test data
output_dir = Path(__file__).parent.parent / 'testdata'
output_dir.mkdir(exist_ok=True)

# Random signal test case
random_test = {
    'sample_rate': int(sr),
    'f_min': float(fmin),
    'n_bins': int(n_bins),
    'bins_per_octave': int(bins_per_octave),
    'hop_length': int(hop_length),
    'input': random_signal.tolist(),
    'expected_magnitude': cqt_random_mag.tolist(),
}

with open(output_dir / 'librosa_cqt_random.json', 'w') as f:
    json.dump(random_test, f)
print(f"\nSaved: {output_dir / 'librosa_cqt_random.json'}")

# Sine wave test case
sine_test = {
    'sample_rate': int(sr),
    'f_min': float(fmin),
    'n_bins': int(n_bins),
    'bins_per_octave': int(bins_per_octave),
    'hop_length': int(hop_length),
    'input': sine_1khz.tolist(),
    'expected_magnitude': cqt_sine_mag.tolist(),
}

with open(output_dir / 'librosa_cqt_sine_1khz.json', 'w') as f:
    json.dump(sine_test, f)
print(f"Saved: {output_dir / 'librosa_cqt_sine_1khz.json'}")

# Also compute cepstrum for full pipeline test
cqt_for_cepstrum = librosa.cqt(
    random_signal, sr=sr, fmin=fmin, n_bins=n_bins,
    bins_per_octave=bins_per_octave, hop_length=hop_length
)
log_cqt = np.log(np.abs(cqt_for_cepstrum) + 1e-10)
n_coeffs = 24
from scipy.fft import dct
cepstrum = dct(log_cqt, type=2, axis=0, norm='ortho')[:n_coeffs, :]

cepstrum_test = {
    'sample_rate': int(sr),
    'f_min': float(fmin),
    'n_bins': int(n_bins),
    'bins_per_octave': int(bins_per_octave),
    'hop_length': int(hop_length),
    'n_coeffs': int(n_coeffs),
    'input': random_signal.tolist(),
    'expected': cepstrum.tolist(),
}

with open(output_dir / 'librosa_cepstrum_full.json', 'w') as f:
    json.dump(cepstrum_test, f)
print(f"Saved: {output_dir / 'librosa_cepstrum_full.json'}")

print("\n" + "=" * 60)
print("Test data regenerated! Run C# tests to verify match.")
print("=" * 60)
