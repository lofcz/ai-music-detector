"""
Compare librosa CQT output per-octave to identify where C# diverges.
"""
import numpy as np
import librosa

sr = 16000
fmin = 500.0
n_bins = 48
bins_per_octave = 12
hop_length = 512

np.random.seed(42)
signal = np.random.randn(8000).astype(np.float32)

# Get librosa CQT
cqt_librosa = librosa.cqt(
    signal, sr=sr, fmin=fmin, n_bins=n_bins,
    bins_per_octave=bins_per_octave, hop_length=hop_length, scale=True
)
cqt_mag = np.abs(cqt_librosa)

print("=== Per-octave energy distribution ===")
print("\nOctave | Bin Range | Freq Range (Hz) | Mean Magnitude | Max Magnitude")
print("-" * 70)

for octave in range(4):
    start_bin = octave * 12
    end_bin = (octave + 1) * 12
    freqs = fmin * (2 ** (np.arange(start_bin, end_bin) / 12.0))
    
    octave_mag = cqt_mag[start_bin:end_bin, :]
    mean_mag = octave_mag.mean()
    max_mag = octave_mag.max()
    
    print(f"   {octave}   | {start_bin:2d} - {end_bin-1:2d}   | {freqs[0]:7.1f} - {freqs[-1]:7.1f} | {mean_mag:12.4f}  | {max_mag:12.4f}")

print("\n=== Individual bin analysis ===")
print("\nBin | Frequency (Hz) | Mean Mag | Frame 0 Mag | Frame 7 Mag")
print("-" * 65)

for k in [0, 6, 11, 12, 24, 36, 47]:
    freq = fmin * (2 ** (k / 12.0))
    mean_mag = cqt_mag[k, :].mean()
    print(f" {k:2d} |    {freq:7.1f}     | {mean_mag:8.4f} | {cqt_mag[k, 0]:11.4f} | {cqt_mag[k, 7]:11.4f}")

# Output exact values for the first frame
print("\n=== First frame magnitudes (for C# comparison) ===")
print("Bin 0:", cqt_mag[0, 0])
print("Bin 11:", cqt_mag[11, 0])
print("Bin 12:", cqt_mag[12, 0])
print("Bin 23:", cqt_mag[23, 0])
print("Bin 24:", cqt_mag[24, 0])
print("Bin 35:", cqt_mag[35, 0])
print("Bin 36:", cqt_mag[36, 0])
print("Bin 47:", cqt_mag[47, 0])
