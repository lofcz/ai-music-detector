"""
Test polyphase decimation to match soxr exactly.
"""
import numpy as np
import soxr

print("=== Understanding soxr's polyphase decimation ===\n")

# Test with various signal lengths and positions
def test_soxr_behavior():
    """Understand soxr's exact behavior."""
    
    # Test 1: Impulse at different positions
    print("Test 1: Impulse response at different positions")
    for pos in [100, 101, 200, 201]:
        impulse = np.zeros(512, dtype=np.float64)
        impulse[pos] = 1.0
        out = soxr.resample(impulse, 2, 1, quality='soxr_hq')
        peak_idx = np.argmax(np.abs(out))
        print(f"  Impulse at {pos} -> output peak at {peak_idx}, ratio: {pos/peak_idx:.4f}")
    
    # Test 2: Check if output[i] = f(input[2i], input[2i+1], ...)
    print("\nTest 2: Output sample dependencies")
    
    # Create signals that are 0 except at specific positions
    np.random.seed(42)
    
    # Full random signal
    full = np.random.randn(1024).astype(np.float64)
    full_out = soxr.resample(full, 2, 1, quality='soxr_hq')
    
    # Same signal but zero out different regions
    # This tells us which input samples affect which output samples
    
    # Zero out first half of input
    first_half_zero = full.copy()
    first_half_zero[:512] = 0
    first_half_zero_out = soxr.resample(first_half_zero, 2, 1, quality='soxr_hq')
    
    # Find where outputs start to differ
    diff = np.abs(full_out - first_half_zero_out)
    first_diff_idx = np.where(diff > 1e-10)[0][0] if (diff > 1e-10).any() else len(diff)
    print(f"  Zeroing input[:512] affects output from index {first_diff_idx}")
    
    # Test 3: Linear combination test
    print("\nTest 3: Linearity verification")
    a = np.random.randn(512).astype(np.float64)
    b = np.random.randn(512).astype(np.float64)
    
    out_a = soxr.resample(a, 2, 1, quality='soxr_hq')
    out_b = soxr.resample(b, 2, 1, quality='soxr_hq')
    out_ab = soxr.resample(a + b, 2, 1, quality='soxr_hq')
    
    linearity_error = np.abs(out_ab - (out_a + out_b)).max()
    print(f"  Linearity error: {linearity_error:.2e}")
    
    # Test 4: Scale test
    print("\nTest 4: DC response")
    dc = np.ones(1024, dtype=np.float64)
    dc_out = soxr.resample(dc, 2, 1, quality='soxr_hq')
    print(f"  Input DC = 1.0")
    print(f"  Output DC (middle) = {dc_out[256]:.6f}")
    print(f"  Expected for 2:1 decimation = 0.5")

test_soxr_behavior()

print("\n" + "="*60)
print("=== Matching soxr with polyphase FIR ===\n")

# The key insight: soxr uses overlap-save convolution
# For 2:1 decimation, we can use polyphase decomposition

# Extract the filter more carefully
N = 8192
impulse = np.zeros(N, dtype=np.float64)
impulse[N//2] = 1.0
h_response = soxr.resample(impulse, 2, 1, quality='soxr_hq')

# The filter is centered around N//4 in output (because input impulse at N//2, 2:1 decimation)
peak_out = np.argmax(np.abs(h_response))
print(f"Output impulse peak at: {peak_out}")

# Find the full support of the filter
threshold = 1e-12
active = np.abs(h_response) > threshold
first_active = np.where(active)[0][0]
last_active = np.where(active)[0][-1]
print(f"Active filter support: {first_active} to {last_active} ({last_active - first_active + 1} samples)")

# Extract the FULL filter
full_filter = h_response[first_active:last_active+1]
print(f"Full filter length: {len(full_filter)}")

# For polyphase 2:1 decimation:
# h[n] = sum_k filter[k] * x[2n-k]
# 
# This can be written as two polyphase components:
# h0[k] = filter[2k]   (applied to x[2n], x[2n-2], x[2n-4], ...)
# h1[k] = filter[2k+1] (applied to x[2n-1], x[2n-3], x[2n-5], ...)
#
# output[n] = sum_k h0[k]*x[2n-2k] + sum_k h1[k]*x[2n-2k-1]

# But wait - the filter we extracted is already for the OUTPUT samples
# We need to think about this differently.

# The impulse response tells us: what output samples are generated when input[N//2] = 1
# For 2:1 decimation, output[i] corresponds to input samples around input[2i]

# Let's verify our understanding by implementing a direct convolution that matches

def soxr_like_decimate(x, h, peak_position):
    """
    Attempt to match soxr's decimation.
    
    x: input signal
    h: filter impulse response (from soxr)
    peak_position: where the filter peak is in h
    """
    out_len = len(x) // 2
    output = np.zeros(out_len)
    
    h_len = len(h)
    
    for i in range(out_len):
        # Output sample i corresponds to input sample 2i
        # The filter h tells us the relationship
        
        # We need to figure out the exact mapping
        # If h[peak] is the response at delay 0, then:
        # output[i] = sum_k h[k] * x[2i - (k - peak)]
        
        acc = 0.0
        for k in range(h_len):
            input_idx = 2 * i - (k - peak_position)
            if 0 <= input_idx < len(x):
                acc += h[k] * x[input_idx]
        output[i] = acc
    
    return output

# Test this
np.random.seed(42)
test_signal = np.random.randn(1024).astype(np.float64)

# soxr reference
soxr_out = soxr.resample(test_signal, 2, 1, quality='soxr_hq')

# Our implementation
peak_in_filter = peak_out - first_active
our_out = soxr_like_decimate(test_signal, full_filter, peak_in_filter)

# Compare with various offsets to handle edge effects
print(f"\nsoxr output length: {len(soxr_out)}")
print(f"Our output length: {len(our_out)}")

# Skip edges and compare middle portion
margin = 100
s1 = soxr_out[margin:-margin]
s2 = our_out[margin:-margin]

if len(s1) != len(s2):
    min_len = min(len(s1), len(s2))
    s1 = s1[:min_len]
    s2 = s2[:min_len]

diff = np.abs(s1 - s2)
print(f"\nComparison (excluding {margin} samples from edges):")
print(f"  Max difference: {diff.max():.6e}")
print(f"  Mean difference: {diff.mean():.6e}")

rel_diff = diff.mean() / (np.abs(s1).mean() + 1e-10)
print(f"  Relative difference: {100*rel_diff:.6f}%")

if rel_diff < 0.001:
    print("\n✓ Match within 0.1%!")
elif rel_diff < 0.01:
    print("\n✓ Match within 1%")
else:
    print(f"\n✗ Still differs by {100*rel_diff:.4f}%")
    
    # Debug: check correlation
    corr = np.corrcoef(s1, s2)[0,1]
    print(f"  Correlation: {corr:.6f}")
    
    # Check if there's a scale factor
    scale = np.dot(s1, s2) / (np.dot(s2, s2) + 1e-10)
    print(f"  Best scale factor: {scale:.6f}")
    
    s2_scaled = s2 * scale
    diff_scaled = np.abs(s1 - s2_scaled)
    print(f"  After scaling - max diff: {diff_scaled.max():.6e}")
