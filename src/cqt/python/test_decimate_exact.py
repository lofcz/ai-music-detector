"""Test exact decimation match between scipy.signal.resample and expected algorithm."""

import numpy as np
from scipy.signal import resample
from scipy.fft import rfft, irfft
import json

def scipy_resample_2x(x):
    """2x decimation using scipy.signal.resample"""
    return resample(x, len(x) // 2)

def manual_resample_2x(x):
    """Manual implementation matching scipy.signal.resample for 2x decimation."""
    N = len(x)
    M = N // 2
    s_fac = N / M  # = 2.0
    m = min(M, N)  # = M for downsampling
    m2 = m // 2 + 1  # = M/2 + 1 = N/4 + 1
    
    # Step 1: rfft
    X = rfft(x)
    print(f"rfft shape: {X.shape}, expected: {N//2+1}")
    
    # Step 2: truncate to m2 bins
    X_trunc = X[:m2].copy()
    print(f"X_trunc shape: {X_trunc.shape}, expected: {m2}")
    
    # Step 3: Nyquist correction
    if m % 2 == 0 and M != N:
        print(f"Applying Nyquist correction at bin {m//2}")
        X_trunc[m//2] *= 2
    
    # Step 4: Scale by 1/s_fac
    X_scaled = X_trunc / s_fac
    
    # Step 5: irfft with n=M
    y = irfft(X_scaled, n=M)
    
    return y

def test_match():
    """Test that manual implementation matches scipy exactly."""
    print("=" * 60)
    print("Testing 2x decimation match")
    print("=" * 60)
    
    np.random.seed(42)
    
    for N in [16, 32, 64, 100, 128, 200, 256]:
        x = np.random.randn(N).astype(np.float64)
        
        y_scipy = scipy_resample_2x(x)
        y_manual = manual_resample_2x(x)
        
        max_diff = np.max(np.abs(y_scipy - y_manual))
        print(f"N={N:3d}: max diff = {max_diff:.2e}")
        
        if max_diff > 1e-10:
            print(f"  MISMATCH! scipy vs manual:")
            print(f"  scipy:  {y_scipy[:5]}")
            print(f"  manual: {y_manual[:5]}")

def generate_test_vectors():
    """Generate test vectors for C# verification."""
    np.random.seed(42)
    
    test_cases = []
    for N in [16, 32, 64, 100, 128, 200]:
        x = np.random.randn(N).astype(np.float64)
        y = scipy_resample_2x(x)
        
        test_cases.append({
            'input': x.tolist(),
            'output': y.tolist()
        })
    
    with open('decimate_test_vectors.json', 'w') as f:
        json.dump(test_cases, f, indent=2)
    
    print(f"\nGenerated {len(test_cases)} test vectors to decimate_test_vectors.json")

if __name__ == '__main__':
    test_match()
    generate_test_vectors()
