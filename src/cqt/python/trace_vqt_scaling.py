"""Trace VQT algorithm to understand scaling behavior."""

import numpy as np
import librosa
from scipy.signal import resample

# Parameters matching config.yaml
sr = 16000
fmin = 500.0
n_bins = 48
bins_per_octave = 12
hop_length = 512

# Generate test signal
np.random.seed(42)
signal = np.random.randn(8000).astype(np.float64)

print("=" * 70)
print("VQT Algorithm Trace - Scaling Analysis")
print("=" * 70)

# Simulate the VQT algorithm step by step
my_y = signal.copy()
my_sr = sr
my_hop = hop_length
n_octaves = n_bins // bins_per_octave

for octave in range(n_octaves):
    print(f"\n--- Octave {octave} ---")
    print(f"  my_sr = {my_sr}")
    print(f"  my_hop = {my_hop}")
    print(f"  signal RMS before STFT = {np.sqrt(np.mean(my_y**2)):.6f}")
    
    # The filter scaling factor applied in VQT
    filter_scale = np.sqrt(sr / my_sr)
    print(f"  Filter scale factor = sqrt({sr}/{my_sr}) = {filter_scale:.4f}")
    
    # Downsample for next octave (if not last)
    if octave < n_octaves - 1 and my_hop % 2 == 0:
        # Option 1: scipy.signal.resample with no scale
        y_scipy_noscale = resample(my_y, len(my_y) // 2)
        print(f"  After scipy resample (no scale): RMS = {np.sqrt(np.mean(y_scipy_noscale**2)):.6f}")
        
        # Option 2: scipy.signal.resample with librosa scale=True
        y_scipy_scale = y_scipy_noscale / np.sqrt(2)
        print(f"  After scipy resample + scale=True: RMS = {np.sqrt(np.mean(y_scipy_scale**2)):.6f}")
        
        # Option 3: librosa.resample with scale=True (what VQT uses)
        y_librosa = librosa.resample(my_y, orig_sr=2, target_sr=1, res_type='scipy', scale=True)
        print(f"  After librosa.resample(scale=True): RMS = {np.sqrt(np.mean(y_librosa**2)):.6f}")
        
        # Verify they match
        diff = np.max(np.abs(y_scipy_scale - y_librosa))
        print(f"  Diff between manual scale and librosa: {diff:.2e}")
        
        # Update for next iteration (using librosa's output)
        my_y = y_librosa
        my_hop //= 2
        my_sr /= 2.0

print("\n" + "=" * 70)
print("Compare CQT with and without scale=True in resample")
print("=" * 70)

# CQT with scipy + scale=True (what librosa.vqt does)
cqt_with_scale = librosa.cqt(
    signal, sr=sr, fmin=fmin, n_bins=n_bins, 
    bins_per_octave=bins_per_octave, hop_length=hop_length,
    res_type='scipy', scale=True
)

# Now let's manually compute what happens if we DON'T apply scale=True in resample
# but we still have the filter scaling sqrt(sr/my_sr)

# We can't easily do this with librosa.cqt, so let's just compare the magnitudes

print(f"\nCQT with scale=True shape: {cqt_with_scale.shape}")
print("\nPer-octave mean magnitudes:")
for octave in range(4):
    start = octave * 12
    end = start + 12
    mean_mag = np.mean(np.abs(cqt_with_scale[start:end, 0]))
    print(f"  Octave {octave} (bins {start}-{end-1}): {mean_mag:.4f}")
