"""
Test soxr_hq 2:1 decimation and verify the half-band FIR algorithm.
"""
import numpy as np

# soxr half_fir_coefs_7 from half-coefs.h (used for SOXR_HQ = 20-bit quality)
HALF_FIR_COEFS_7 = np.array([
    3.1062656496657370e-01, -8.4998810699955796e-02,  3.4007044621123500e-02,
   -1.2839903789829387e-02,  3.9899380181723145e-03, -8.9355202017945374e-04,
    1.0918292424806546e-04,
])

def soxr_halfband_decimate(signal):
    """
    Decimate by 2 using soxr's half-band FIR filter (SOXR_HQ quality).
    
    Algorithm from half-fir.h:
      sum = input[0] * 0.5
      for j = 0 to 6:
          sum += (input[-(2*j+1)] + input[(2*j+1)]) * COEFS[j]
      output = sum
    
    The filter is symmetric, so it can be computed as:
      center_tap = 0.5
      for each odd offset (1, 3, 5, 7, 9, 11, 13): apply coefs
    """
    coefs = HALF_FIR_COEFS_7
    num_coefs = len(coefs)
    
    # Pad input: need num_coefs samples before and after
    # (actually 2*num_coefs-1 = 13 samples on each side for the symmetric filter)
    pad_len = 2 * num_coefs
    padded = np.pad(signal, (pad_len, pad_len), mode='constant', constant_values=0)
    
    # Output length
    n_out = len(signal) // 2
    output = np.zeros(n_out)
    
    # Process each output sample
    for i in range(n_out):
        # Input index (center of filter)
        idx = pad_len + 2 * i
        
        # Center tap
        s = padded[idx] * 0.5
        
        # Symmetric taps at odd offsets
        for j in range(num_coefs):
            offset = 2 * j + 1  # 1, 3, 5, 7, 9, 11, 13
            s += (padded[idx - offset] + padded[idx + offset]) * coefs[j]
        
        output[i] = s
    
    return output

# Test with some signal
np.random.seed(42)
test_signal = np.random.randn(1024).astype(np.float64)

# Our implementation
our_result = soxr_halfband_decimate(test_signal)

# Compare with actual soxr
try:
    import soxr
    
    # soxr.resample(y, in_rate, out_rate, quality)
    # For 2:1 decimation: in_rate=2, out_rate=1
    soxr_result = soxr.resample(test_signal, 2, 1, quality='soxr_hq')
    
    print(f"Input length: {len(test_signal)}")
    print(f"Our output length: {len(our_result)}")
    print(f"soxr output length: {len(soxr_result)}")
    
    # soxr might have slightly different length due to edge handling
    min_len = min(len(our_result), len(soxr_result))
    
    diff = np.abs(our_result[:min_len] - soxr_result[:min_len])
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    print(f"\nComparison (first {min_len} samples):")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    
    # Print first few values
    print(f"\nFirst 5 values:")
    print(f"  Our:  {our_result[:5]}")
    print(f"  soxr: {soxr_result[:5]}")
    
    if max_diff < 1e-10:
        print("\n✓ EXACT MATCH!")
    elif max_diff < 1e-6:
        print("\n✓ Match within 1e-6")
    else:
        print(f"\n✗ Differs by {max_diff:.2e}")
        
        # Check if there's a scaling difference
        ratio = our_result[:min_len] / (soxr_result[:min_len] + 1e-10)
        mean_ratio = np.mean(ratio[np.abs(soxr_result[:min_len]) > 0.1])
        print(f"\n  Mean ratio (our/soxr): {mean_ratio:.6f}")
        
        if abs(mean_ratio - 1.0) < 0.01:
            print("  → Results are proportional (scaling matches)")
        else:
            print(f"  → Possible scaling factor: {mean_ratio:.6f}")

except ImportError:
    print("soxr not installed. Install with: pip install soxr")
    print("\nOur implementation output:")
    print(f"  Input length: {len(test_signal)}")
    print(f"  Output length: {len(our_result)}")
    print(f"  First 5 values: {our_result[:5]}")

# Now test with librosa's resample
print("\n" + "="*60)
print("Testing with librosa.resample (soxr_hq)")
print("="*60)

try:
    import librosa
    
    # librosa.resample with soxr_hq
    librosa_result = librosa.resample(test_signal, orig_sr=2, target_sr=1, res_type='soxr_hq', scale=False)
    
    print(f"librosa output length: {len(librosa_result)}")
    
    min_len = min(len(our_result), len(librosa_result))
    diff = np.abs(our_result[:min_len] - librosa_result[:min_len])
    
    print(f"\nComparison vs librosa (scale=False):")
    print(f"  Max difference: {diff.max():.2e}")
    print(f"  Mean difference: {diff.mean():.2e}")
    
    # Also test with scale=True (what CQT uses)
    librosa_scaled = librosa.resample(test_signal, orig_sr=2, target_sr=1, res_type='soxr_hq', scale=True)
    
    print(f"\nWith scale=True (energy preservation):")
    print(f"  librosa applies: y /= sqrt(ratio) = y *= sqrt(2) for 2:1 decimation")
    
    our_scaled = our_result * np.sqrt(2.0)
    diff_scaled = np.abs(our_scaled[:min_len] - librosa_scaled[:min_len])
    
    print(f"  Max diff (our*sqrt(2) vs librosa): {diff_scaled.max():.2e}")
    
    if diff_scaled.max() < 1e-6:
        print("\n✓ Scaling matches!")

except ImportError:
    print("librosa not installed")
