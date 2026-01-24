"""
Debug exactly how librosa.filters.wavelet creates the basis.
"""
import numpy as np
import librosa
from scipy.fft import fft
from scipy.signal import get_window

sr = 16000
fmin = 500.0
n_bins = 48
bins_per_octave = 12

# Get frequencies
freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
alpha = librosa.filters._relative_bandwidth(freqs=freqs)
lengths, _ = librosa.filters.wavelet_lengths(freqs=freqs, sr=sr, filter_scale=1.0, window='hann', alpha=alpha)

# Get librosa's wavelet
librosa_basis, _ = librosa.filters.wavelet(
    freqs=freqs, sr=sr, filter_scale=1.0, norm=1, pad_fft=True,
    window='hann', alpha=alpha
)
print(f"Librosa basis shape: {librosa_basis.shape}")

# Now manually create the same
n_fft = librosa_basis.shape[1]
manual_basis = np.zeros((n_bins, n_fft), dtype=complex)

for k in range(n_bins):
    freq = freqs[k]
    length = int(lengths[k])
    
    # Get window
    window = get_window('hann', length, fftbins=True)
    
    # Create centered time axis (this is key!)
    # librosa centers the wavelet
    center = length // 2
    t = (np.arange(length) - center) / sr
    
    # Complex exponential modulated by window
    sig = window * np.exp(2j * np.pi * freq * t)
    
    # L1 normalize
    sig = sig / np.sum(np.abs(sig))
    
    # Place in padded array - centered or left-aligned?
    # Let's check librosa's placement
    manual_basis[k, :length] = sig

# Compare
diff = manual_basis - librosa_basis
max_diff = np.abs(diff).max()
print(f"Max difference (left-aligned): {max_diff:.10f}")

# Try centered placement
manual_basis_centered = np.zeros((n_bins, n_fft), dtype=complex)
for k in range(n_bins):
    freq = freqs[k]
    length = int(lengths[k])
    
    window = get_window('hann', length, fftbins=True)
    center = length // 2
    t = (np.arange(length) - center) / sr
    sig = window * np.exp(2j * np.pi * freq * t)
    sig = sig / np.sum(np.abs(sig))
    
    # Center the signal in the FFT buffer
    start = (n_fft - length) // 2
    manual_basis_centered[k, start:start+length] = sig

diff_centered = manual_basis_centered - librosa_basis
max_diff_centered = np.abs(diff_centered).max()
print(f"Max difference (centered): {max_diff_centered:.10f}")

# Check where librosa puts the signal
print(f"\nLibrosa basis row 0:")
nonzero = np.where(np.abs(librosa_basis[0]) > 1e-10)[0]
print(f"  Non-zero indices: {nonzero[0]} to {nonzero[-1]} (length {len(nonzero)})")
print(f"  Expected length: {int(lengths[0])}")

# Check row 24 (middle)
print(f"\nLibrosa basis row 24:")
nonzero24 = np.where(np.abs(librosa_basis[24]) > 1e-10)[0]
print(f"  Non-zero indices: {nonzero24[0]} to {nonzero24[-1]} (length {len(nonzero24)})")
print(f"  Expected length: {int(lengths[24])}")

# So librosa left-aligns. Let's check the complex exponential
print("\n--- Complex exponential check ---")
k = 0
freq = freqs[k]
length = int(lengths[k])
window = get_window('hann', length, fftbins=True)

# My version: centered time axis
center = length // 2
t_centered = (np.arange(length) - center) / sr
sig_centered = window * np.exp(2j * np.pi * freq * t_centered)
sig_centered = sig_centered / np.sum(np.abs(sig_centered))

# Alternative: time from 0
t_zero = np.arange(length) / sr
sig_zero = window * np.exp(2j * np.pi * freq * t_zero)
sig_zero = sig_zero / np.sum(np.abs(sig_zero))

librosa_sig = librosa_basis[0, :length]

print(f"Librosa first 3:  {librosa_sig[:3]}")
print(f"Centered first 3: {sig_centered[:3]}")
print(f"Zero-based first 3: {sig_zero[:3]}")

# Check difference
diff_centered = np.abs(sig_centered - librosa_sig).max()
diff_zero = np.abs(sig_zero - librosa_sig).max()
print(f"\nMax diff (centered time): {diff_centered:.10f}")
print(f"Max diff (zero-based time): {diff_zero:.10f}")

# The answer should be clear now!
if diff_centered < diff_zero:
    print("\n=> Librosa uses CENTERED time axis")
else:
    print("\n=> Librosa uses ZERO-BASED time axis")

# Now check if librosa uses center=True in the FFT shift
# (i.e., does it shift the phase for centering?)
print("\n--- Phase at center check ---")
print(f"Librosa sig at center ({length//2}): {librosa_sig[length//2]}")
print(f"Centered at center: {sig_centered[length//2]}")
print(f"Zero at center: {sig_zero[length//2]}")
