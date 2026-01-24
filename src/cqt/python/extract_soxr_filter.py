"""
Extract the effective impulse response of soxr for 2:1 decimation.
This tells us exactly what filter we need to implement.
"""
import numpy as np

try:
    import soxr
except ImportError:
    print("Install soxr: pip install soxr")
    exit(1)

print("=== Extracting soxr_hq filter for 2:1 decimation ===\n")

# Create impulse signal (delta function at various positions)
# The filter response tells us exactly what soxr does
N = 4096
impulse = np.zeros(N, dtype=np.float64)
impulse[N//2] = 1.0

# Apply soxr 2:1 decimation
# io_ratio = 2 (in_rate=2, out_rate=1)
response = soxr.resample(impulse, 2, 1, quality='soxr_hq')

print(f"Input length: {len(impulse)}")
print(f"Output length: {len(response)}")
print(f"Expected output length: {N//2} = {N//2}")

# Find the peak (center of impulse response)
peak_idx = np.argmax(np.abs(response))
print(f"\nImpulse response peak at index: {peak_idx}")
print(f"Peak value: {response[peak_idx]:.6f}")

# Find effective filter length (where response drops below threshold)
threshold = 1e-10
nonzero = np.where(np.abs(response) > threshold)[0]
if len(nonzero) > 0:
    filter_start = nonzero[0]
    filter_end = nonzero[-1]
    effective_length = filter_end - filter_start + 1
    print(f"\nEffective filter support: indices {filter_start} to {filter_end}")
    print(f"Effective filter length: {effective_length} samples")
else:
    filter_start = 0
    filter_end = len(response) - 1
    effective_length = len(response)

# Extract the filter coefficients
# Center around the peak
half_len = 256  # Should be enough for soxr_hq
start = max(0, peak_idx - half_len)
end = min(len(response), peak_idx + half_len + 1)
filter_coeffs = response[start:end]

print(f"\nExtracted filter around peak: {len(filter_coeffs)} coefficients")

# Check symmetry (linear phase filters are symmetric)
center = len(filter_coeffs) // 2
left_half = filter_coeffs[:center]
right_half = filter_coeffs[center+1:][::-1]
if len(left_half) == len(right_half):
    symmetry_error = np.abs(left_half - right_half).max()
    print(f"Symmetry error: {symmetry_error:.2e}")
    if symmetry_error < 1e-10:
        print("✓ Filter is symmetric (linear phase)")
    else:
        print("✗ Filter is NOT symmetric")

# Find number of significant taps
sig_threshold = 1e-8
significant = np.abs(filter_coeffs) > sig_threshold * np.abs(filter_coeffs).max()
first_sig = np.where(significant)[0][0] if significant.any() else 0
last_sig = np.where(significant)[0][-1] if significant.any() else len(filter_coeffs)-1
num_significant = last_sig - first_sig + 1
print(f"Number of significant taps: {num_significant}")

# Trim to significant portion
trimmed = filter_coeffs[first_sig:last_sig+1]
print(f"Trimmed filter length: {len(trimmed)}")

# Sum of coefficients (should be ~1 for unity gain at DC)
dc_gain = np.sum(trimmed)
print(f"\nDC gain (sum of coeffs): {dc_gain:.6f}")

# Verify by convolving test signal
print("\n=== Verification ===")
np.random.seed(42)
test_signal = np.random.randn(1024).astype(np.float64)

# soxr output
soxr_out = soxr.resample(test_signal, 2, 1, quality='soxr_hq')

# Our convolution + decimation
from scipy.signal import convolve
convolved = convolve(test_signal, trimmed, mode='same')
our_out = convolved[::2]  # Decimate by 2

# Compare (accounting for delay differences)
# Find best alignment
best_offset = 0
best_corr = 0
for offset in range(-50, 50):
    if offset >= 0:
        corr = np.corrcoef(soxr_out[offset:offset+400], our_out[:400])[0,1]
    else:
        corr = np.corrcoef(soxr_out[:400], our_out[-offset:-offset+400])[0,1]
    if abs(corr) > abs(best_corr):
        best_corr = corr
        best_offset = offset

print(f"Best correlation: {best_corr:.6f} at offset {best_offset}")

# Calculate actual difference at best alignment
if best_offset >= 0:
    s1 = soxr_out[best_offset:best_offset+400]
    s2 = our_out[:400]
else:
    s1 = soxr_out[:400]
    s2 = our_out[-best_offset:-best_offset+400]

diff = np.abs(s1 - s2)
print(f"Max difference: {diff.max():.6f}")
print(f"Mean difference: {diff.mean():.6f}")
rel_diff = diff.mean() / (np.abs(s1).mean() + 1e-10)
print(f"Relative difference: {100*rel_diff:.4f}%")

# Save the filter coefficients
print("\n=== Filter coefficients ===")
print(f"Filter length: {len(trimmed)}")

# For C# implementation, we need the coefficients
# Save to a format that can be easily copied
np.set_printoptions(precision=17, threshold=10000)
print("\nCoefficients (first 20 and last 5):")
print(trimmed[:20])
print("...")
print(trimmed[-5:])

# Save to file
np.save('soxr_hq_decimation_filter.npy', trimmed)
print(f"\nSaved filter to soxr_hq_decimation_filter.npy")

# Also check what librosa's scale=True does
import librosa
librosa_out_unscaled = librosa.resample(test_signal, orig_sr=2, target_sr=1, res_type='soxr_hq', scale=False)
librosa_out_scaled = librosa.resample(test_signal, orig_sr=2, target_sr=1, res_type='soxr_hq', scale=True)

scale_factor = librosa_out_scaled[100] / (librosa_out_unscaled[100] + 1e-10)
print(f"\nlibrosa scale factor: {scale_factor:.6f}")
print(f"Expected sqrt(2): {np.sqrt(2):.6f}")
