"""Test script to verify exact scipy.signal.resample behavior for 2x decimation."""

import numpy as np
from scipy.signal import resample
from scipy.fft import rfft, irfft
import json

def test_resample_algorithm():
    """Trace the exact algorithm for 2x decimation."""
    
    # Test with a simple signal
    np.random.seed(42)
    N = 16  # input length
    x = np.random.randn(N).astype(np.float64)
    
    print(f"Input length N = {N}")
    print(f"Input: {x}")
    
    # Target: 2x decimation
    num = N // 2  # output length
    print(f"Output length num = {num}")
    
    # scipy.signal.resample algorithm for real input, no window
    n_x = N
    s_fac = n_x / num  # = 2.0
    m = min(num, n_x)  # = num = N/2
    m2 = m // 2 + 1    # = N/4 + 1
    
    print(f"\ns_fac = {s_fac}")
    print(f"m = {m}")
    print(f"m2 = {m2}")
    
    # Step 1: rfft
    X = rfft(x)
    print(f"\nrfft(x) shape: {X.shape}")
    print(f"rfft(x): {X}")
    
    # Step 2: truncate to m2 bins
    X_trunc = X[:m2].copy()
    print(f"\nAfter truncation to {m2} bins:")
    print(f"X_trunc: {X_trunc}")
    
    # Step 3: Handle unpaired Nyquist bin
    # For downsampling: if m is even and num != n_x, multiply bin[m//2] by 2
    if m % 2 == 0 and num != n_x:
        print(f"\nm is even ({m}), applying Nyquist correction at bin {m//2}")
        X_trunc[m//2] *= 2
        print(f"After Nyquist correction: {X_trunc}")
    
    # Step 4: Scale by 1/s_fac
    X_scaled = X_trunc / s_fac
    print(f"\nAfter scaling by 1/{s_fac}:")
    print(f"X_scaled: {X_scaled}")
    
    # Step 5: irfft with n=num
    x_r = irfft(X_scaled, n=num)
    print(f"\nirfft(X_scaled, n={num}):")
    print(f"x_r: {x_r}")
    
    # Compare with scipy.signal.resample
    x_scipy = resample(x, num)
    print(f"\nscipy.signal.resample result:")
    print(f"x_scipy: {x_scipy}")
    
    # Check match
    diff = np.abs(x_r - x_scipy)
    print(f"\nDifference: {diff}")
    print(f"Max diff: {np.max(diff)}")
    
    return np.allclose(x_r, x_scipy)

def generate_test_data():
    """Generate test data for C# verification."""
    
    np.random.seed(42)
    
    test_cases = []
    
    # Test various input lengths (all even for 2x decimation)
    for N in [8, 16, 32, 64, 128, 100, 200]:
        x = np.random.randn(N).astype(np.float64)
        num = N // 2
        y = resample(x, num)
        
        test_cases.append({
            'input': x.tolist(),
            'output': y.tolist(),
            'input_length': N,
            'output_length': num
        })
    
    # Save to JSON
    with open('resample_test_data.json', 'w') as f:
        json.dump(test_cases, f, indent=2)
    
    print(f"Generated {len(test_cases)} test cases")
    return test_cases

if __name__ == '__main__':
    print("=" * 60)
    print("Testing resample algorithm step by step")
    print("=" * 60)
    
    success = test_resample_algorithm()
    print(f"\nAlgorithm match: {success}")
    
    print("\n" + "=" * 60)
    print("Generating test data")
    print("=" * 60)
    
    test_cases = generate_test_data()
    
    # Also trace with a power-of-2 length to understand the pattern
    print("\n" + "=" * 60)
    print("Testing with non-power-of-2 length (100)")
    print("=" * 60)
    
    np.random.seed(123)
    x = np.random.randn(100)
    y = resample(x, 50)
    
    print(f"Input length: 100")
    print(f"Output length: 50")
    print(f"Output[0:5]: {y[:5]}")
