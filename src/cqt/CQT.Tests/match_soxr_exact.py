"""
Find exact algorithm to match soxr 2:1 decimation.
"""

import numpy as np
import soxr
from scipy import signal
import json
import os

TESTDATA_DIR = os.path.join(os.path.dirname(__file__), 'testdata')


def test_scipy_decimate():
    """Test if scipy.signal.decimate matches soxr."""
    print("=" * 60)
    print("Testing scipy.signal.decimate vs soxr")
    print("=" * 60)
    
    np.random.seed(42)
    sr_in = 16000
    sr_out = 8000
    
    x = np.random.randn(4000).astype(np.float64)
    
    y_soxr = soxr.resample(x, sr_in, sr_out, quality='HQ')
    
    # scipy.signal.decimate with different filters
    for ftype in ['iir', 'fir']:
        try:
            y_scipy = signal.decimate(x, 2, ftype=ftype)
            
            min_len = min(len(y_soxr), len(y_scipy))
            diff = np.abs(y_soxr[50:min_len-50] - y_scipy[50:min_len-50])
            rel = np.sum(diff) / (np.sum(np.abs(y_soxr[50:min_len-50])) + 1e-10)
            
            print(f"  {ftype}: rel diff = {rel:.4%}, max diff = {np.max(diff):.6f}")
        except Exception as e:
            print(f"  {ftype}: FAILED - {e}")


def test_resample_poly():
    """Test scipy.signal.resample_poly which uses polyphase filtering."""
    print("\n" + "=" * 60)
    print("Testing scipy.signal.resample_poly vs soxr")
    print("=" * 60)
    
    np.random.seed(42)
    sr_in = 16000
    sr_out = 8000
    
    x = np.random.randn(4000).astype(np.float64)
    
    y_soxr = soxr.resample(x, sr_in, sr_out, quality='HQ')
    
    # resample_poly with up=1, down=2 for decimation
    y_poly = signal.resample_poly(x, 1, 2)
    
    min_len = min(len(y_soxr), len(y_poly))
    diff = np.abs(y_soxr[50:min_len-50] - y_poly[50:min_len-50])
    rel = np.sum(diff) / (np.sum(np.abs(y_soxr[50:min_len-50])) + 1e-10)
    
    print(f"  resample_poly(up=1, down=2): rel diff = {rel:.4%}")
    print(f"    max diff = {np.max(diff):.6f}")
    print(f"    correlation = {np.corrcoef(y_soxr[50:min_len-50], y_poly[50:min_len-50])[0,1]:.6f}")


def test_manual_fir_decimate():
    """Test manual FIR decimation with various filters."""
    print("\n" + "=" * 60)
    print("Testing manual FIR decimation")
    print("=" * 60)
    
    np.random.seed(42)
    sr_in = 16000
    sr_out = 8000
    
    x = np.random.randn(4000).astype(np.float64)
    y_soxr = soxr.resample(x, sr_in, sr_out, quality='HQ')
    
    # Design various lowpass filters and test
    nyq = sr_out / 2  # Output Nyquist = 4000 Hz
    cutoff = 0.9 * nyq  # 90% of Nyquist = 3600 Hz
    
    for num_taps in [101, 201, 301, 401, 501]:
        # Design Kaiser window FIR filter
        # Cutoff relative to input Nyquist (8000 Hz)
        normalized_cutoff = cutoff / (sr_in / 2)  # = 3600/8000 = 0.45
        
        h = signal.firwin(num_taps, normalized_cutoff, window=('kaiser', 10), pass_zero=True)
        
        # Convolve and decimate
        y_filtered = np.convolve(x, h, mode='same')
        y_dec = y_filtered[::2]
        
        min_len = min(len(y_soxr), len(y_dec))
        diff = np.abs(y_soxr[50:min_len-50] - y_dec[50:min_len-50])
        rel = np.sum(diff) / (np.sum(np.abs(y_soxr[50:min_len-50])) + 1e-10)
        
        print(f"  Kaiser FIR {num_taps} taps: rel diff = {rel:.4%}")


def extract_soxr_effective_filter():
    """
    Extract the effective FIR filter that soxr uses by deconvolution.
    """
    print("\n" + "=" * 60)
    print("Extracting soxr effective filter via deconvolution")
    print("=" * 60)
    
    sr_in = 16000
    sr_out = 8000
    
    # Method: Create input signal that reveals the filter
    # Use a long impulse train
    
    test_len = 8192
    
    # Put impulse at center
    x = np.zeros(test_len, dtype=np.float64)
    x[test_len // 2] = 1.0
    
    y = soxr.resample(x, sr_in, sr_out, quality='HQ')
    
    # The output is the impulse response, but it's in the OUTPUT sample rate
    # For decimation, the filter is applied BEFORE decimation, so:
    # y[n] = sum_k h[k] * x[2n - k]
    # 
    # With x[test_len//2] = 1, we get:
    # y[n] = h[2n - test_len//2]
    # 
    # So h[j] = y[(j + test_len//2) / 2] for j + test_len//2 even
    
    # But this only gives us every other tap of h...
    # That's the problem with decimation - we lose half the information!
    
    # Alternative: use two impulses at adjacent positions
    x1 = np.zeros(test_len, dtype=np.float64)
    x1[test_len // 2] = 1.0
    y1 = soxr.resample(x1, sr_in, sr_out, quality='HQ')
    
    x2 = np.zeros(test_len, dtype=np.float64)
    x2[test_len // 2 + 1] = 1.0
    y2 = soxr.resample(x2, sr_in, sr_out, quality='HQ')
    
    # y1[n] = h[2n - test_len//2]  (even taps of h shifted)
    # y2[n] = h[2n - test_len//2 - 1] = h[2n - test_len//2 - 1]  (odd taps of h shifted)
    
    # From y1, we get h at positions: test_len//2 - 2n for n = output indices
    # For n = test_len//4 (center of output), h position = test_len//2 - 2*(test_len//4) = 0
    
    # Let's reconstruct h
    out_len = len(y1)
    center_out = out_len // 2
    
    # h[2*(center_out - n)] = y1[center_out - n] for the even phases
    # Wait, this is getting confusing. Let me think differently.
    
    # For decimation by 2:
    # y[n] = sum_k h_effective[k] * x[2n - k]
    # 
    # The effective filter h_effective can have both even and odd indexed values.
    # y1 (impulse at even position test_len//2) gives us certain combinations
    # y2 (impulse at odd position test_len//2+1) gives us other combinations
    
    # Actually, the cleanest way is to use the OUTPUT impulse response directly.
    # y1 tells us: for an impulse at input position test_len//2,
    # what is the output at each output position?
    
    # For output position n, input position test_len//2 contributes y1[n]
    # For output position n, input position 2n is the "aligned" position
    # So the filter coefficient h[2n - test_len//2] = y1[n]
    
    # Rearranging: y1[n] = h[2n - test_len//2]
    # So h[k] = y1[(k + test_len//2) / 2] when k + test_len//2 is even
    
    # This gives us filter taps at even positions relative to test_len//2
    
    # From y2 (impulse at test_len//2 + 1):
    # y2[n] = h[2n - (test_len//2 + 1)] = h[2n - test_len//2 - 1]
    # So h[k] = y2[(k + test_len//2 + 1) / 2] when k + test_len//2 + 1 is even
    #                                          i.e., when k + test_len//2 is odd
    
    # So we can reconstruct the full filter h!
    
    # Range of k values:
    # From y1: k = 2n - test_len//2 for n in [0, out_len)
    #         k ranges from -test_len//2 to 2*(out_len-1) - test_len//2 = 2*out_len - 2 - test_len//2
    
    # For test_len=8192, out_len≈4096:
    # k ranges from -4096 to 4094
    
    # Build h
    h_len = test_len  # Filter could be up to this long
    h = np.zeros(h_len, dtype=np.float64)
    h_center = h_len // 2  # Center h around position h_len//2
    
    for n in range(len(y1)):
        # k = 2n - test_len//2
        k = 2 * n - test_len // 2
        h_idx = k + h_center
        if 0 <= h_idx < h_len:
            h[h_idx] = y1[n]
    
    for n in range(len(y2)):
        # k = 2n - test_len//2 - 1
        k = 2 * n - test_len // 2 - 1
        h_idx = k + h_center
        if 0 <= h_idx < h_len:
            h[h_idx] = y2[n]
    
    # Trim h to significant values
    threshold = np.max(np.abs(h)) * 1e-10
    nonzero = np.where(np.abs(h) > threshold)[0]
    if len(nonzero) > 0:
        h_trimmed = h[nonzero[0]:nonzero[-1]+1]
        h_offset = nonzero[0] - h_center  # Offset from center
        
        print(f"  Extracted filter: {len(h_trimmed)} taps")
        print(f"  Offset from center: {h_offset}")
        print(f"  Sum: {np.sum(h_trimmed):.6f}")
        print(f"  Peak position: {np.argmax(np.abs(h_trimmed))}")
        print(f"  Peak value: {np.max(np.abs(h_trimmed)):.6f}")
        
        return h_trimmed, h_offset
    
    return h, 0


def verify_extracted_filter(h, h_offset):
    """Verify the extracted filter matches soxr."""
    print("\n" + "=" * 60)
    print("Verifying extracted filter")
    print("=" * 60)
    
    np.random.seed(42)
    sr_in = 16000
    sr_out = 8000
    
    x = np.random.randn(4000).astype(np.float64)
    y_soxr = soxr.resample(x, sr_in, sr_out, quality='HQ')
    
    # Apply the filter: y[n] = sum_k h[k] * x[2n - k]
    # This is convolution at half rate
    
    # Equivalent to: convolve x with h at full rate, then take every other sample
    # But we need to handle alignment carefully
    
    # h was extracted with h_offset from center
    # The filter's center corresponds to x[0] mapping to y[0]
    
    # Full convolution gives output of length len(x) + len(h) - 1
    y_conv = np.convolve(x, h, mode='full')
    
    # The convolution result y_conv[k] = sum_j h[j] * x[k-j]
    # For our decimation: y[n] = sum_j h[j] * x[2n - j]
    # This is y_conv at position 2n
    
    # But we need to account for h_offset
    # The filter center is at position len(h)//2 + h_offset from the start of h
    # After convolution, position k in y_conv corresponds to:
    #   x contributions centered at k - (len(h)//2 + h_offset)
    
    # For output[n] to come from input[2n], we need:
    # conv_index = 2n + (len(h)//2 + h_offset)
    
    filter_delay = len(h) // 2 + h_offset
    
    y_our = []
    for n in range(len(y_soxr)):
        conv_idx = 2 * n + filter_delay
        if 0 <= conv_idx < len(y_conv):
            y_our.append(y_conv[conv_idx])
        else:
            y_our.append(0)
    
    y_our = np.array(y_our)
    
    # Compare
    edge = 50
    diff = np.abs(y_soxr[edge:-edge] - y_our[edge:-edge])
    rel_diff = np.sum(diff) / (np.sum(np.abs(y_soxr[edge:-edge])) + 1e-10)
    
    print(f"  Max absolute diff: {np.max(diff):.6f}")
    print(f"  Mean relative diff: {rel_diff:.4%}")
    print(f"  Correlation: {np.corrcoef(y_soxr[edge:-edge], y_our[edge:-edge])[0,1]:.6f}")
    
    if rel_diff < 0.01:
        print("\n  ✓ SUCCESS! Filter matches soxr")
        return True
    else:
        # Try different alignments
        print("\n  Trying different alignments...")
        
        best_rel = rel_diff
        best_delay = filter_delay
        
        for delay_adj in range(-20, 21):
            test_delay = filter_delay + delay_adj
            
            y_test = []
            for n in range(len(y_soxr)):
                conv_idx = 2 * n + test_delay
                if 0 <= conv_idx < len(y_conv):
                    y_test.append(y_conv[conv_idx])
                else:
                    y_test.append(0)
            
            y_test = np.array(y_test)
            diff = np.abs(y_soxr[edge:-edge] - y_test[edge:-edge])
            rel = np.sum(diff) / (np.sum(np.abs(y_soxr[edge:-edge])) + 1e-10)
            
            if rel < best_rel:
                best_rel = rel
                best_delay = test_delay
        
        print(f"  Best alignment: delay={best_delay}, rel_diff={best_rel:.4%}")
        
        return best_rel < 0.01


def save_working_filter(h, delay):
    """Save the filter that works."""
    path = os.path.join(TESTDATA_DIR, 'soxr_hq_2x_decimation_filter.json')
    
    data = {
        'description': 'soxr HQ 2:1 decimation filter - apply via convolve then decimate',
        'num_taps': len(h),
        'delay': delay,
        'coefficients': h.tolist(),
        'usage': 'y_conv = convolve(x, h, "full"); y_out = y_conv[delay::2][:len(x)//2]'
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved working filter to: {path}")


if __name__ == '__main__':
    test_scipy_decimate()
    test_resample_poly()
    test_manual_fir_decimate()
    
    h, h_offset = extract_soxr_effective_filter()
    
    if h is not None:
        if verify_extracted_filter(h, h_offset):
            # Find exact delay that works
            filter_delay = len(h) // 2 + h_offset
            save_working_filter(h, filter_delay)
