"""
Extract exact soxr filter coefficients for 2:1 decimation at HQ quality.
This captures the exact impulse response that soxr uses internally.
"""

import numpy as np
import json
import os
import soxr

TESTDATA_DIR = os.path.join(os.path.dirname(__file__), 'testdata')
os.makedirs(TESTDATA_DIR, exist_ok=True)


def extract_soxr_impulse_response():
    """
    Extract soxr's effective impulse response by passing a unit impulse.
    For 2:1 decimation, we need to be careful about the output length.
    """
    sr_in = 48000  # Use higher rate for better resolution
    sr_out = 24000
    
    # Create impulse at different positions to understand the filter
    # The filter length can be determined by finding where the response becomes negligible
    
    # Start with a long signal to capture full impulse response
    test_len = 4096
    
    # Create unit impulse at center
    impulse = np.zeros(test_len, dtype=np.float64)
    impulse_pos = test_len // 2
    impulse[impulse_pos] = 1.0
    
    # Resample with soxr HQ
    response = soxr.resample(impulse, sr_in, sr_out, quality='HQ')
    
    print(f"Input length: {test_len}")
    print(f"Output length: {len(response)}")
    print(f"Impulse position (input): {impulse_pos}")
    
    # Find the peak in the response
    peak_idx = np.argmax(np.abs(response))
    print(f"Peak position (output): {peak_idx}")
    print(f"Peak value: {response[peak_idx]}")
    
    # Find significant filter taps (above threshold)
    threshold = np.max(np.abs(response)) * 1e-10
    significant = np.abs(response) > threshold
    first_sig = np.argmax(significant)
    last_sig = len(response) - 1 - np.argmax(significant[::-1])
    
    print(f"First significant tap: {first_sig}")
    print(f"Last significant tap: {last_sig}")
    print(f"Filter length (significant): {last_sig - first_sig + 1}")
    
    # Extract the filter coefficients
    # Center around the peak
    margin = 500  # Should be enough for HQ filter
    start_idx = max(0, peak_idx - margin)
    end_idx = min(len(response), peak_idx + margin + 1)
    
    filter_coeffs = response[start_idx:end_idx]
    peak_offset = peak_idx - start_idx
    
    # Trim leading/trailing zeros
    nonzero = np.abs(filter_coeffs) > threshold
    first_nz = np.argmax(nonzero)
    last_nz = len(filter_coeffs) - 1 - np.argmax(nonzero[::-1])
    
    filter_coeffs = filter_coeffs[first_nz:last_nz+1]
    peak_offset = peak_offset - first_nz
    
    print(f"\nExtracted filter:")
    print(f"  Length: {len(filter_coeffs)}")
    print(f"  Peak offset: {peak_offset}")
    print(f"  Sum: {np.sum(filter_coeffs)}")
    print(f"  DC gain: {np.sum(filter_coeffs)}")
    
    return filter_coeffs, peak_offset


def verify_filter(filter_coeffs, peak_offset):
    """
    Verify that convolving with the extracted filter matches soxr output.
    """
    np.random.seed(42)
    
    sr_in = 48000
    sr_out = 24000
    
    # Test signal
    test_len = 8000
    test_signal = np.random.randn(test_len).astype(np.float64)
    
    # soxr reference
    soxr_output = soxr.resample(test_signal, sr_in, sr_out, quality='HQ')
    
    # Our convolution + decimation
    # Full convolution
    convolved = np.convolve(test_signal, filter_coeffs, mode='full')
    
    # Decimate by 2, accounting for filter delay
    # The peak_offset tells us where the center of the filter is
    # For output sample i, we want input sample 2*i
    # After convolution, the output at index k corresponds to input k - peak_offset
    # So for output i, we want convolved[2*i + peak_offset]
    
    output_len = len(test_signal) // 2
    our_output = np.zeros(output_len)
    
    for i in range(output_len):
        conv_idx = 2 * i + peak_offset
        if 0 <= conv_idx < len(convolved):
            our_output[i] = convolved[conv_idx]
    
    # Compare
    min_len = min(len(soxr_output), len(our_output))
    
    # Skip edges where boundary effects matter
    edge = 50
    soxr_mid = soxr_output[edge:min_len-edge]
    our_mid = our_output[edge:min_len-edge]
    
    diff = np.abs(soxr_mid - our_mid)
    rel_diff = np.sum(diff) / (np.sum(np.abs(soxr_mid)) + 1e-10)
    
    print(f"\nVerification:")
    print(f"  soxr output length: {len(soxr_output)}")
    print(f"  Our output length: {len(our_output)}")
    print(f"  Max absolute diff (mid): {np.max(diff)}")
    print(f"  Mean relative diff (mid): {rel_diff:.4%}")
    
    return rel_diff < 0.01  # Should be < 1% for good match


def extract_with_varying_lengths():
    """
    Extract filter for different input lengths to ensure consistency.
    """
    sr_in = 48000
    sr_out = 24000
    
    print("Testing filter consistency across input lengths...")
    
    filters = []
    for test_len in [2048, 4096, 8192, 16384]:
        impulse = np.zeros(test_len, dtype=np.float64)
        impulse_pos = test_len // 2
        impulse[impulse_pos] = 1.0
        
        response = soxr.resample(impulse, sr_in, sr_out, quality='HQ')
        
        # Extract around peak
        peak_idx = np.argmax(np.abs(response))
        margin = 300
        start = max(0, peak_idx - margin)
        end = min(len(response), peak_idx + margin + 1)
        
        filt = response[start:end]
        
        # Trim
        threshold = np.max(np.abs(filt)) * 1e-12
        nonzero = np.abs(filt) > threshold
        if np.any(nonzero):
            first = np.argmax(nonzero)
            last = len(filt) - 1 - np.argmax(nonzero[::-1])
            filt = filt[first:last+1]
        
        filters.append(filt)
        print(f"  Length {test_len}: filter has {len(filt)} taps, sum={np.sum(filt):.6f}")
    
    # Check consistency
    ref_len = len(filters[-1])  # Use longest test
    print(f"\nReference filter length: {ref_len}")
    
    return filters[-1]


def save_filter_coefficients(filter_coeffs):
    """Save the filter coefficients for C# to load."""
    path = os.path.join(TESTDATA_DIR, 'soxr_hq_2x_filter.json')
    
    data = {
        'description': 'soxr HQ quality 2:1 decimation filter coefficients',
        'sample_rate_ratio': '2:1 (e.g., 48000->24000)',
        'quality': 'HQ',
        'num_taps': len(filter_coeffs),
        'coefficients': filter_coeffs.tolist()
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved filter to: {path}")
    print(f"  {len(filter_coeffs)} coefficients")


def analyze_soxr_behavior():
    """
    Detailed analysis of soxr's resampling behavior.
    """
    print("=" * 60)
    print("Analyzing soxr 2:1 decimation behavior")
    print("=" * 60)
    
    sr_in = 16000  # Match our detector's sample rate
    sr_out = 8000
    
    # Test 1: DC response (constant signal)
    dc_signal = np.ones(1000, dtype=np.float64)
    dc_out = soxr.resample(dc_signal, sr_in, sr_out, quality='HQ')
    print(f"\nDC test:")
    print(f"  Input: constant 1.0, length {len(dc_signal)}")
    print(f"  Output: mean={np.mean(dc_out[10:-10]):.6f}, length {len(dc_out)}")
    
    # Test 2: Sine wave at various frequencies
    print(f"\nSine wave tests (input sr={sr_in}, output sr={sr_out}):")
    for freq in [100, 1000, 2000, 3000, 3500, 3900]:
        t = np.arange(sr_in) / sr_in  # 1 second
        sine = np.sin(2 * np.pi * freq * t)
        out = soxr.resample(sine, sr_in, sr_out, quality='HQ')
        
        # Measure output energy (skip edges)
        in_energy = np.mean(sine[100:-100]**2)
        out_energy = np.mean(out[50:-50]**2)
        
        ratio = out_energy / in_energy if in_energy > 0 else 0
        print(f"  {freq:4d} Hz: energy ratio = {ratio:.4f} ({'preserved' if ratio > 0.9 else 'attenuated'})")
    
    # Test 3: Sample alignment
    print(f"\nSample alignment test:")
    # Create signal with markers at known positions
    test = np.zeros(1000, dtype=np.float64)
    test[100] = 1.0  # Marker at sample 100
    test[200] = 1.0  # Marker at sample 200
    test[500] = 1.0  # Marker at sample 500
    
    out = soxr.resample(test, sr_in, sr_out, quality='HQ')
    
    # Find peaks in output
    peaks = []
    for i in range(10, len(out)-10):
        if out[i] > 0.5 and out[i] > out[i-1] and out[i] > out[i+1]:
            peaks.append(i)
    
    print(f"  Input markers at: [100, 200, 500]")
    print(f"  Output peaks near: {peaks[:6]}")
    print(f"  Expected (input/2): [50, 100, 250]")


if __name__ == '__main__':
    # First analyze behavior
    analyze_soxr_behavior()
    
    print("\n" + "=" * 60)
    print("Extracting soxr filter coefficients")
    print("=" * 60)
    
    # Extract with consistency check
    filter_coeffs = extract_with_varying_lengths()
    
    # Get precise extraction
    filter_coeffs, peak_offset = extract_soxr_impulse_response()
    
    # Verify
    if verify_filter(filter_coeffs, peak_offset):
        print("\n✓ Filter verification PASSED")
        save_filter_coefficients(filter_coeffs)
    else:
        print("\n✗ Filter verification FAILED - need different approach")
        
        # Try alternative: extract at native 16kHz
        print("\nTrying at 16kHz...")
        
        sr_in = 16000
        sr_out = 8000
        test_len = 4096
        
        impulse = np.zeros(test_len, dtype=np.float64)
        impulse_pos = test_len // 2
        impulse[impulse_pos] = 1.0
        
        response = soxr.resample(impulse, sr_in, sr_out, quality='HQ')
        
        peak_idx = np.argmax(np.abs(response))
        margin = 300
        start = max(0, peak_idx - margin)
        end = min(len(response), peak_idx + margin + 1)
        
        filter_coeffs = response[start:end]
        
        # Trim
        threshold = np.max(np.abs(filter_coeffs)) * 1e-12
        nonzero = np.abs(filter_coeffs) > threshold
        first = np.argmax(nonzero)
        last = len(filter_coeffs) - 1 - np.argmax(nonzero[::-1])
        filter_coeffs = filter_coeffs[first:last+1]
        peak_offset = peak_idx - start - first
        
        print(f"16kHz filter: {len(filter_coeffs)} taps, peak at {peak_offset}")
        
        if verify_filter(filter_coeffs, peak_offset):
            print("✓ 16kHz filter verification PASSED")
            save_filter_coefficients(filter_coeffs)
        else:
            print("✗ Still failing - saving anyway for debugging")
            save_filter_coefficients(filter_coeffs)
