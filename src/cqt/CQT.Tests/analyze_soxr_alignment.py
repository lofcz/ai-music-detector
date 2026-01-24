"""
Deep analysis of soxr's sample alignment and resampling behavior.
"""

import numpy as np
import soxr

def analyze_alignment():
    """Find exact relationship between input and output samples."""
    sr_in = 16000
    sr_out = 8000
    
    print("=" * 60)
    print("Analyzing soxr sample alignment for 2:1 decimation")
    print("=" * 60)
    
    # Test 1: Single impulse at various positions
    print("\n1. Single impulse response at different positions:")
    
    for imp_pos in [50, 100, 200, 500, 1000]:
        test = np.zeros(2000, dtype=np.float64)
        test[imp_pos] = 1.0
        
        out = soxr.resample(test, sr_in, sr_out, quality='HQ')
        
        # Find peak
        peak_idx = np.argmax(np.abs(out))
        peak_val = out[peak_idx]
        
        # Find significant energy spread
        threshold = np.max(np.abs(out)) * 0.001
        significant = np.where(np.abs(out) > threshold)[0]
        spread_start = significant[0] if len(significant) > 0 else 0
        spread_end = significant[-1] if len(significant) > 0 else 0
        
        print(f"  Input impulse at {imp_pos}: output peak at {peak_idx} "
              f"(ratio: {peak_idx/imp_pos:.3f}), spread [{spread_start}-{spread_end}]")
    
    # Test 2: Verify linearity and shift-invariance
    print("\n2. Verifying linearity:")
    
    np.random.seed(42)
    x1 = np.random.randn(1000).astype(np.float64)
    x2 = np.random.randn(1000).astype(np.float64)
    a, b = 2.5, -1.3
    
    y1 = soxr.resample(x1, sr_in, sr_out, quality='HQ')
    y2 = soxr.resample(x2, sr_in, sr_out, quality='HQ')
    y_combined = soxr.resample(a * x1 + b * x2, sr_in, sr_out, quality='HQ')
    
    diff = np.max(np.abs(y_combined - (a * y1 + b * y2)))
    print(f"  Max diff between soxr(a*x1 + b*x2) and a*soxr(x1) + b*soxr(x2): {diff:.2e}")
    
    # Test 3: Exact sample correspondence
    print("\n3. Testing exact sample correspondence:")
    
    # Create a signal where we know exactly what each sample should map to
    # Use very slow sinusoid
    test_len = 4000
    t = np.arange(test_len) / sr_in
    # Very slow signal - essentially linear ramp
    x = np.arange(test_len, dtype=np.float64) / test_len
    
    y = soxr.resample(x, sr_in, sr_out, quality='HQ')
    
    print(f"  Input: linear ramp 0 to 1, length {test_len}")
    print(f"  Output length: {len(y)}")
    
    # Check output values at specific points
    for i in [0, 10, 50, 100, 500, 999]:
        if i < len(y):
            expected = (2 * i) / test_len  # If output[i] = input[2*i]
            print(f"  y[{i}] = {y[i]:.6f}, expected from x[{2*i}] = {expected:.6f}, diff = {abs(y[i] - expected):.6f}")
    
    # Test 4: Use scipy for comparison
    print("\n4. Comparing with scipy.signal.resample:")
    
    from scipy import signal
    
    np.random.seed(42)
    x = np.random.randn(1000).astype(np.float64)
    
    y_soxr = soxr.resample(x, sr_in, sr_out, quality='HQ')
    y_scipy = signal.resample(x, len(x) // 2)
    
    # Scale scipy output (it preserves amplitude differently)
    # scipy.signal.resample preserves total energy, so for 2:1 decimation
    # it divides by sqrt(2)
    
    diff = np.abs(y_soxr[10:-10] - y_scipy[10:-10])
    print(f"  soxr vs scipy.signal.resample:")
    print(f"    Max diff: {np.max(diff):.6f}")
    print(f"    Mean diff: {np.mean(diff):.6f}")
    print(f"    Correlation: {np.corrcoef(y_soxr[10:-10], y_scipy[10:-10])[0,1]:.6f}")
    
    # Test 5: Polyphase decomposition attempt
    print("\n5. Testing polyphase interpretation:")
    
    # For 2:1 decimation, polyphase would be:
    # y[n] = sum_k h[k] * x[2n - k]
    # which can be split into even and odd phases
    
    # Get impulse response
    impulse = np.zeros(4096, dtype=np.float64)
    impulse[2048] = 1.0
    h = soxr.resample(impulse, sr_in, sr_out, quality='HQ')
    
    # Find the filter
    peak = np.argmax(np.abs(h))
    # Take a window around peak
    win = 300
    h_filter = h[peak-win:peak+win+1]
    
    # For polyphase: h_even = h[::2], h_odd = h[1::2]
    # But this is in output domain...
    
    print(f"  Filter length: {len(h_filter)}")
    print(f"  Filter sum: {np.sum(h_filter):.6f}")
    print(f"  Filter center (peak): at index {win}")
    
    # The key insight: for decimation by 2 using polyphase,
    # output[n] = sum_{k} h[2k] * input[2n - 2k] + h[2k+1] * input[2n - 2k - 1]
    # This is equivalent to filtering with h, then taking every other sample
    
    # Let's verify by direct computation
    np.random.seed(42)
    x = np.random.randn(2000).astype(np.float64)
    y_soxr = soxr.resample(x, sr_in, sr_out, quality='HQ')
    
    # Convolve and decimate
    y_conv = np.convolve(x, h_filter, mode='same')
    y_dec = y_conv[::2]  # Take every other sample
    
    # Align
    # y_soxr corresponds to x at positions 0, 2, 4, ...
    # y_dec also takes x at 0, 2, 4, ... after filtering
    
    min_len = min(len(y_soxr), len(y_dec))
    
    # Try different alignments
    best_corr = 0
    best_offset = 0
    for offset in range(-50, 50):
        if offset >= 0:
            a = y_soxr[offset:min_len-50]
            b = y_dec[:min_len-50-offset]
        else:
            a = y_soxr[:min_len-50+offset]
            b = y_dec[-offset:min_len-50]
        
        if len(a) > 100 and len(b) > 100:
            m = min(len(a), len(b))
            corr = np.corrcoef(a[:m], b[:m])[0,1]
            if corr > best_corr:
                best_corr = corr
                best_offset = offset
    
    print(f"  Best correlation with convolve+decimate: {best_corr:.6f} at offset {best_offset}")
    
    # Now try with the actual h
    # Use full h centered properly
    full_h_len = len(h)
    h_full = np.zeros(full_h_len)
    h_full = h.copy()
    
    # The filter h represents output values, but we need input-domain filter
    # For decimation: we filter input at original rate, then decimate
    # The impulse response h IS the filter in the output domain
    
    # Alternative interpretation: h tells us how input impulse spreads to output
    # So to reconstruct: y[n] = sum_k h[n - k/2] * x[k] for k even that maps to n
    # This is confusing... let me think differently
    
    print("\n6. Direct decimation approach:")
    
    # What if soxr is doing:
    # 1. Upsample by L=1 (no change for decimation)
    # 2. Filter with lowpass
    # 3. Downsample by M=2
    
    # For L=1, M=2: standard decimation
    # y[n] = sum_k h[k] * x[M*n - k] = sum_k h[k] * x[2n - k]
    
    # This is linear convolution followed by keeping every M-th sample
    # The filter h should be the lowpass filter
    
    # Extract lowpass filter by looking at frequency response
    from scipy import fft
    
    H = fft.fft(h, 4096)
    freqs = fft.fftfreq(4096, d=1/sr_out)
    
    # Lowpass should have flat response below cutoff
    pos_freqs = freqs[:2048]
    pos_H = np.abs(H[:2048])
    
    # Find -3dB point
    max_H = np.max(pos_H)
    cutoff_idx = np.where(pos_H < max_H * 0.707)[0]
    if len(cutoff_idx) > 0:
        cutoff = pos_freqs[cutoff_idx[0]]
        print(f"  -3dB cutoff: {cutoff:.1f} Hz")
    
    # The filter's DC gain
    print(f"  DC gain (|H[0]|): {np.abs(H[0]):.6f}")
    
    return h, h_filter


def try_exact_replication():
    """
    Try to exactly replicate soxr output using understood principles.
    """
    print("\n" + "=" * 60)
    print("Attempting exact soxr replication")
    print("=" * 60)
    
    sr_in = 16000
    sr_out = 8000
    
    # Get the impulse response (this IS the decimation filter)
    impulse = np.zeros(8192, dtype=np.float64)
    center = 4096
    impulse[center] = 1.0
    
    h_raw = soxr.resample(impulse, sr_in, sr_out, quality='HQ')
    
    # Find the actual filter
    # The output is half the length, so the peak should be around center/2
    peak = np.argmax(np.abs(h_raw))
    print(f"Peak position in output: {peak} (expected ~{center//2})")
    
    # The filter coefficients in h_raw represent:
    # h_raw[n] = response at output sample n due to input impulse at sample center
    # 
    # For the general case: output[n] relates to input[2n] (approximately)
    # The filter tells us: output[n] = sum_k input[k] * filter_response(k -> n)
    #
    # Since input only has 1 at position 'center', we get:
    # h_raw[n] = filter_response(center -> n)
    #
    # For a time-invariant system:
    # output[n] = sum_k input[k] * h[n - k/2] where k/2 is the output position k maps to
    #
    # This is getting complicated. Let me try a different approach.
    
    # For any linear system, we can characterize it by its response matrix
    # output = A @ input where A[i,j] = response at output i due to input at j
    
    # For decimation by 2: output has half the samples
    # output[i] depends on input samples around position 2*i
    
    # Let's extract the effective "convolution kernel" that soxr uses
    # by testing multiple impulses
    
    filter_len = 600  # Approximate based on earlier analysis
    
    # Test: can we predict output from input using local filtering?
    np.random.seed(42)
    test_len = 4000
    x = np.random.randn(test_len).astype(np.float64)
    y_soxr = soxr.resample(x, sr_in, sr_out, quality='HQ')
    
    # Build the impulse response matrix for a few output positions
    # This will tell us exactly which input samples contribute to each output
    
    print("\nBuilding local impulse response:")
    
    # For output sample at position out_pos, find which input samples contribute
    out_pos = 500  # Test position
    
    # Create input with 1 at each position, see effect on output[out_pos]
    h_local = []
    check_range = 300  # Check input positions around 2*out_pos
    input_center = 2 * out_pos
    
    for k in range(input_center - check_range, input_center + check_range):
        if 0 <= k < test_len:
            impulse_test = np.zeros(test_len, dtype=np.float64)
            impulse_test[k] = 1.0
            out_test = soxr.resample(impulse_test, sr_in, sr_out, quality='HQ')
            if out_pos < len(out_test):
                h_local.append((k, out_test[out_pos]))
    
    # Convert to filter centered at input_center
    h_extracted = np.zeros(2 * check_range)
    for k, val in h_local:
        idx = k - (input_center - check_range)
        h_extracted[idx] = val
    
    # This h_extracted should be the filter that produces output[out_pos]
    # from input samples around 2*out_pos
    
    # Trim to significant coefficients
    threshold = np.max(np.abs(h_extracted)) * 1e-10
    nonzero = np.where(np.abs(h_extracted) > threshold)[0]
    if len(nonzero) > 0:
        h_trimmed = h_extracted[nonzero[0]:nonzero[-1]+1]
        offset = nonzero[0] - check_range  # Offset from center
        print(f"  Extracted filter: {len(h_trimmed)} taps, offset from center: {offset}")
        print(f"  Filter sum: {np.sum(h_trimmed):.6f}")
        print(f"  First 5 coeffs: {h_trimmed[:5]}")
        print(f"  Center coeffs: {h_trimmed[len(h_trimmed)//2-2:len(h_trimmed)//2+3]}")
    
    # Now verify: convolve x with h_trimmed, properly aligned
    # output[n] = sum_k h[k] * x[2n - k + offset]
    
    # Actually, the standard way:
    # y = conv(x, h)[::2] with proper alignment
    
    # Let's try
    y_conv = np.convolve(x, h_trimmed, mode='full')
    
    # The output length from convolution is len(x) + len(h) - 1
    # We want to decimate and align with soxr output
    
    # soxr output[0] corresponds to input around position 0
    # conv output[k] corresponds to input at position k - (len(h)-1)/2 approximately
    
    # For decimation: we want output[n] from input[2n]
    # After convolution, this means conv output at position 2n + (filter_delay)
    
    filter_delay = len(h_trimmed) // 2 - offset
    
    y_our = []
    for n in range(len(y_soxr)):
        conv_idx = 2 * n + filter_delay
        if 0 <= conv_idx < len(y_conv):
            y_our.append(y_conv[conv_idx])
        else:
            y_our.append(0)
    
    y_our = np.array(y_our)
    
    # Compare
    diff = np.abs(y_soxr[50:-50] - y_our[50:-50])
    rel_diff = np.sum(diff) / (np.sum(np.abs(y_soxr[50:-50])) + 1e-10)
    
    print(f"\n  Verification against soxr:")
    print(f"    Max diff: {np.max(diff):.6f}")
    print(f"    Relative diff: {rel_diff:.4%}")
    
    if rel_diff < 0.01:
        print("\n  ✓ SUCCESS! Found the correct filter and alignment")
        return h_trimmed, offset
    else:
        print("\n  ✗ Still not matching, trying alternatives...")
        
        # Try different decimation phases
        for phase in [0, 1]:
            y_test = y_conv[phase::2]
            if len(y_test) >= len(y_soxr):
                diff = np.abs(y_soxr[50:-50] - y_test[50:len(y_soxr)-50])
                rel = np.sum(diff) / (np.sum(np.abs(y_soxr[50:-50])) + 1e-10)
                print(f"    Phase {phase}: rel diff = {rel:.4%}")
        
        return h_trimmed, offset


if __name__ == '__main__':
    h, h_filter = analyze_alignment()
    h_exact, offset = try_exact_replication()
