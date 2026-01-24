"""
Generate test data for comparing C# SoxrDecimate against Python soxr library.
Run from: conda activate ai-music-detector && python generate_soxr_test_data.py
"""

import numpy as np
import json
import os
import soxr
import librosa
from scipy import signal as scipy_signal

# Output directory (same as other test data)
TESTDATA_DIR = os.path.join(os.path.dirname(__file__), 'testdata')
os.makedirs(TESTDATA_DIR, exist_ok=True)

# Match C# test expectations
SPARSITY = 0.0


def save_test_case(name, data):
    """Save test case to JSON file."""
    path = os.path.join(TESTDATA_DIR, f'{name}.json')
    
    # Convert numpy arrays to lists
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(item) for item in obj]
        return obj
    
    with open(path, 'w') as f:
        json.dump(convert(data), f, indent=2)
    
    print(f"Saved: {path}")


def generate_soxr_decimate_test():
    """Generate test data for basic soxr 2:1 decimation."""
    np.random.seed(42)
    
    # Random signal
    sr = 16000
    duration = 0.5  # seconds
    input_signal = np.random.randn(int(sr * duration)).astype(np.float64)
    
    # Use soxr for 2:1 decimation with HQ quality
    output_signal = soxr.resample(input_signal, sr, sr // 2, quality='HQ')
    
    print(f"soxr_decimate_test:")
    print(f"  Input shape: {input_signal.shape}")
    print(f"  Output shape: {output_signal.shape}")
    
    save_test_case('soxr_decimate_test', {
        'input': input_signal,
        'expected_output': output_signal,
        'sample_rate': sr,
        'target_rate': sr // 2,
        'quality': 'HQ'
    })


def generate_soxr_decimate_chirp():
    """Generate test data for chirp signal decimation."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float64)
    
    # Chirp from 100 Hz to 3000 Hz (below new Nyquist of 4 kHz)
    f0, f1 = 100, 3000
    input_signal = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration) * t**2))
    
    output_signal = soxr.resample(input_signal, sr, sr // 2, quality='HQ')
    
    print(f"soxr_decimate_chirp:")
    print(f"  Input shape: {input_signal.shape}")
    print(f"  Output shape: {output_signal.shape}")
    
    save_test_case('soxr_decimate_chirp', {
        'input': input_signal,
        'expected_output': output_signal,
        'sample_rate': sr,
        'target_rate': sr // 2,
        'quality': 'HQ'
    })


def generate_librosa_cqt_soxr():
    """Generate test data for librosa.cqt with res_type='soxr_hq'."""
    np.random.seed(42)
    
    # Parameters matching our detector config
    sr = 16000
    f_min = 500.0
    n_bins = 48
    bins_per_octave = 12
    hop_length = 512
    
    # Random signal - 2 seconds
    duration = 2.0
    input_signal = np.random.randn(int(sr * duration)).astype(np.float64)
    
    # Compute CQT with soxr_hq resampling
    # Use sparsity=0.0 to match C# test configuration
    # Use dtype=np.complex128 for float64 precision matching C# test data
    cqt_result = librosa.cqt(
        input_signal,
        sr=sr,
        fmin=f_min,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        hop_length=hop_length,
        res_type='soxr_hq',
        sparsity=SPARSITY,
        dtype=np.complex128
    )
    
    magnitude = np.abs(cqt_result)
    
    print(f"librosa_cqt_soxr_random:")
    print(f"  Input shape: {input_signal.shape}")
    print(f"  CQT magnitude shape: {magnitude.shape}")
    
    save_test_case('librosa_cqt_soxr_random', {
        'input': input_signal,
        'expected_magnitude': magnitude,
        'sample_rate': sr,
        'f_min': f_min,
        'n_bins': n_bins,
        'bins_per_octave': bins_per_octave,
        'hop_length': hop_length,
        'res_type': 'soxr_hq'
    })


def generate_librosa_cqt_soxr_sine():
    """Generate test data for sine wave CQT with soxr_hq."""
    sr = 16000
    f_min = 500.0
    n_bins = 48
    bins_per_octave = 12
    hop_length = 512
    
    # 1 kHz sine wave
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float64)
    input_signal = np.sin(2 * np.pi * 1000 * t)
    
    # Use sparsity=0.0 and float64 to match C# test data
    cqt_result = librosa.cqt(
        input_signal,
        sr=sr,
        fmin=f_min,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        hop_length=hop_length,
        res_type='soxr_hq',
        sparsity=SPARSITY,
        dtype=np.complex128
    )
    
    magnitude = np.abs(cqt_result)
    
    # Find peak bin
    bin_energy = np.sum(magnitude, axis=1)
    peak_bin = np.argmax(bin_energy)
    peak_freq = f_min * (2 ** (peak_bin / bins_per_octave))
    
    print(f"librosa_cqt_soxr_sine:")
    print(f"  Input: 1 kHz sine wave")
    print(f"  CQT magnitude shape: {magnitude.shape}")
    print(f"  Peak bin: {peak_bin} ({peak_freq:.1f} Hz)")
    
    save_test_case('librosa_cqt_soxr_sine', {
        'input': input_signal,
        'expected_magnitude': magnitude,
        'sample_rate': sr,
        'f_min': f_min,
        'n_bins': n_bins,
        'bins_per_octave': bins_per_octave,
        'hop_length': hop_length,
        'res_type': 'soxr_hq'
    })


def generate_fft_comparison_tests():
    """Generate FFT test vectors for direct C# vs Python comparison.
    
    Tests multiple FFT sizes to identify size-dependent precision issues.
    """
    print("\nGenerating FFT comparison test data...")
    
    for size in [512, 1024, 2048, 4096]:
        np.random.seed(42)  # Same seed for reproducibility
        input_data = np.random.randn(size).astype(np.float64)
        
        # Compute FFT using numpy (same as scipy.fft for real input)
        fft_output = np.fft.fft(input_data)
        
        # Store real and imaginary parts separately for JSON
        save_test_case(f'fft_comparison_{size}', {
            'size': size,
            'input': input_data,
            'expected_real': fft_output.real,
            'expected_imag': fft_output.imag
        })
        
        print(f"  fft_comparison_{size}: input shape={input_data.shape}, output shape={fft_output.shape}")
    
    # Also generate a test with known analytical result (impulse)
    for size in [512, 1024]:
        impulse = np.zeros(size, dtype=np.float64)
        impulse[0] = 1.0
        fft_impulse = np.fft.fft(impulse)
        
        save_test_case(f'fft_impulse_{size}', {
            'size': size,
            'input': impulse,
            'expected_real': fft_impulse.real,
            'expected_imag': fft_impulse.imag
        })
        print(f"  fft_impulse_{size}: all ones expected")
    
    # Test with DC signal
    for size in [512, 1024]:
        dc = np.ones(size, dtype=np.float64)
        fft_dc = np.fft.fft(dc)
        
        save_test_case(f'fft_dc_{size}', {
            'size': size,
            'input': dc,
            'expected_real': fft_dc.real,
            'expected_imag': fft_dc.imag
        })
        print(f"  fft_dc_{size}: energy at bin 0 expected")


def generate_intermediate_cqt_values():
    """Generate intermediate CQT computation values for debugging.
    
    This captures:
    - Wavelet filter values after FFT
    - STFT output for a simple signal
    - Per-octave filter basis values
    """
    print("\nGenerating intermediate CQT values for debugging...")
    
    # Parameters matching our detector config
    sr = 16000
    f_min = 500.0
    n_bins = 48
    bins_per_octave = 12
    hop_length = 512
    
    # Use a simple impulse for predictable results
    duration = 0.5
    test_signal = np.zeros(int(sr * duration), dtype=np.float64)
    test_signal[len(test_signal) // 2] = 1.0  # Impulse at center
    
    # Compute STFT to get intermediate values
    # STFT with rectangular window (matches librosa's __cqt_response)
    n_fft = 1024
    f, t, Zxx = scipy_signal.stft(
        test_signal, 
        fs=sr, 
        window='boxcar',  # rectangular
        nperseg=n_fft, 
        noverlap=n_fft - hop_length,
        boundary=None,
        padded=False
    )
    
    # Also compute using numpy.fft directly (closer to librosa's approach)
    # Pad signal for center=True behavior
    pad_length = n_fft // 2
    padded = np.pad(test_signal, (pad_length, pad_length), mode='constant')
    
    # Compute frames
    n_frames = 1 + (len(padded) - n_fft) // hop_length
    stft_result = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    
    for t_idx in range(n_frames):
        start = t_idx * hop_length
        frame = padded[start:start + n_fft]
        # No window (rectangular) for CQT response
        spectrum = np.fft.fft(frame)
        stft_result[:, t_idx] = spectrum[:n_fft // 2 + 1]
    
    save_test_case('stft_impulse_rectangular', {
        'input': test_signal,
        'sample_rate': sr,
        'n_fft': n_fft,
        'hop_length': hop_length,
        'window': 'rectangular',
        'stft_real': stft_result.real,
        'stft_imag': stft_result.imag,
        'n_frames': n_frames
    })
    
    print(f"  stft_impulse_rectangular: shape={stft_result.shape}")
    
    # Generate wavelet filter coefficients for one frequency
    # This matches librosa.filters.wavelet
    test_freq = 1000.0  # 1 kHz
    filter_scale = 1.0
    
    # Compute Q factor and filter length
    r = 2 ** (1.0 / bins_per_octave)
    alpha = (r * r - 1) / (r * r + 1)  # Relative bandwidth
    Q = filter_scale / alpha
    ilen = Q * sr / test_freq
    
    # Create wavelet with centered time axis (like librosa)
    Nk = int(np.floor(ilen / 2)) + int(np.ceil(ilen / 2))
    start_idx = int(np.floor(-ilen / 2))
    
    # Hann window (periodic)
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(Nk) / Nk))
    
    # Create wavelet
    wavelet = np.zeros(Nk, dtype=np.complex128)
    coef = 2.0 * np.pi * test_freq / sr
    for n in range(Nk):
        t_idx = start_idx + n
        phase = coef * t_idx
        wavelet[n] = window[n] * np.exp(1j * phase)
    
    # L1 normalize
    norm = np.sum(np.abs(wavelet))
    if norm > 0:
        wavelet /= norm
    
    # FFT the wavelet (padded to power of 2)
    n_fft_wavelet = int(2 ** np.ceil(np.log2(Nk)))
    
    # Center the wavelet
    wavelet_padded = np.zeros(n_fft_wavelet, dtype=np.complex128)
    pad_start = (n_fft_wavelet - Nk) // 2
    wavelet_padded[pad_start:pad_start + Nk] = wavelet
    
    # Scale by lengths / n_fft
    wavelet_padded *= ilen / n_fft_wavelet
    
    # FFT
    wavelet_fft = np.fft.fft(wavelet_padded)
    
    save_test_case('wavelet_filter_1khz', {
        'frequency': test_freq,
        'sample_rate': sr,
        'filter_scale': filter_scale,
        'bins_per_octave': bins_per_octave,
        'filter_length': Nk,
        'ilen_float': float(ilen),
        'n_fft': n_fft_wavelet,
        'window': window.tolist(),
        'wavelet_real': wavelet.real,
        'wavelet_imag': wavelet.imag,
        'wavelet_fft_real': wavelet_fft.real,
        'wavelet_fft_imag': wavelet_fft.imag
    })
    
    print(f"  wavelet_filter_1khz: filter_length={Nk}, n_fft={n_fft_wavelet}")


def generate_matrix_multiply_test():
    """Generate test data for matrix multiply precision testing.
    
    This tests the accumulation of many complex multiplications.
    """
    print("\nGenerating matrix multiply precision test data...")
    
    np.random.seed(42)
    
    # Simulate the CQT matrix multiply scenario:
    # result = fftBasis @ stft where fftBasis is [n_filters, n_freq_bins]
    # and stft is [n_freq_bins, n_frames]
    
    n_filters = 12  # One octave
    n_freq_bins = 513  # n_fft/2+1 for n_fft=1024
    n_frames = 10
    
    # Random complex matrices
    basis_real = np.random.randn(n_filters, n_freq_bins).astype(np.float64)
    basis_imag = np.random.randn(n_filters, n_freq_bins).astype(np.float64)
    basis = basis_real + 1j * basis_imag
    
    stft_real = np.random.randn(n_freq_bins, n_frames).astype(np.float64)
    stft_imag = np.random.randn(n_freq_bins, n_frames).astype(np.float64)
    stft = stft_real + 1j * stft_imag
    
    # Compute matrix multiply
    result = basis @ stft
    
    save_test_case('matrix_multiply_complex', {
        'n_filters': n_filters,
        'n_freq_bins': n_freq_bins,
        'n_frames': n_frames,
        'basis_real': basis_real,
        'basis_imag': basis_imag,
        'stft_real': stft_real,
        'stft_imag': stft_imag,
        'result_real': result.real,
        'result_imag': result.imag
    })
    
    print(f"  matrix_multiply_complex: basis={basis.shape}, stft={stft.shape}, result={result.shape}")
    
    # Also test with larger accumulation (2049 bins for n_fft=4096)
    n_freq_bins_large = 2049
    basis_large = np.random.randn(n_filters, n_freq_bins_large).astype(np.float64) + \
                  1j * np.random.randn(n_filters, n_freq_bins_large).astype(np.float64)
    stft_large = np.random.randn(n_freq_bins_large, n_frames).astype(np.float64) + \
                 1j * np.random.randn(n_freq_bins_large, n_frames).astype(np.float64)
    result_large = basis_large @ stft_large
    
    save_test_case('matrix_multiply_complex_large', {
        'n_filters': n_filters,
        'n_freq_bins': n_freq_bins_large,
        'n_frames': n_frames,
        'basis_real': basis_large.real,
        'basis_imag': basis_large.imag,
        'stft_real': stft_large.real,
        'stft_imag': stft_large.imag,
        'result_real': result_large.real,
        'result_imag': result_large.imag
    })
    
    print(f"  matrix_multiply_complex_large: {n_freq_bins_large} accumulations per output")


def check_soxr_version():
    """Check soxr version and capabilities."""
    print("=" * 60)
    print("SOXR Library Information")
    print("=" * 60)
    
    try:
        import soxr
        print(f"soxr version: {soxr.__version__}")
        
        # Test basic functionality
        sr = 16000
        test_signal = np.random.randn(1000).astype(np.float64)
        
        for quality in ['QQ', 'LQ', 'MQ', 'HQ', 'VHQ']:
            try:
                output = soxr.resample(test_signal, sr, sr // 2, quality=quality)
                print(f"  {quality}: OK, output length = {len(output)}")
            except Exception as e:
                print(f"  {quality}: FAILED - {e}")
        
    except ImportError:
        print("ERROR: soxr not installed. Run: pip install soxr")
        return False
    
    print()
    return True


if __name__ == '__main__':
    print("Generating SOXR test data for C# comparison tests")
    print("=" * 60)
    
    if not check_soxr_version():
        exit(1)
    
    try:
        # Original soxr tests
        generate_soxr_decimate_test()
        generate_soxr_decimate_chirp()
        generate_librosa_cqt_soxr()
        generate_librosa_cqt_soxr_sine()
        
        # NEW: FFT comparison tests for precision investigation
        generate_fft_comparison_tests()
        
        # NEW: Intermediate CQT values for debugging
        generate_intermediate_cqt_values()
        
        # NEW: Matrix multiply precision tests
        generate_matrix_multiply_test()
        
        print("\n" + "=" * 60)
        print("All test data generated successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
