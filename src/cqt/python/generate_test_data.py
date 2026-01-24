"""
Generate test data for verifying C# CQT implementation.

Outputs JSON files with input signals and expected outputs
that can be loaded by C# unit tests.
"""

import json
import numpy as np
from pathlib import Path
from scipy.fft import fft, dct

from cqt_reference import CQTReference, compute_cepstrum, hann_window


def save_test_case(name: str, data: dict, output_dir: Path):
    """Save a test case to JSON."""
    # Convert numpy arrays to lists
    def convert(obj):
        if isinstance(obj, np.ndarray):
            if np.iscomplexobj(obj):
                return {
                    'real': obj.real.tolist(),
                    'imag': obj.imag.tolist()
                }
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj
    
    data = convert(data)
    
    output_path = output_dir / f"{name}.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {output_path}")


def generate_fft_tests(output_dir: Path):
    """Generate FFT test cases."""
    
    # Test 1: Simple impulse (delta function)
    impulse = np.zeros(8)
    impulse[0] = 1.0
    fft_result = fft(impulse)
    
    save_test_case('fft_impulse', {
        'input': impulse,
        'expected': fft_result,
        'description': 'FFT of unit impulse - should be all ones'
    }, output_dir)
    
    # Test 2: DC signal
    dc = np.ones(8)
    fft_result = fft(dc)
    
    save_test_case('fft_dc', {
        'input': dc,
        'expected': fft_result,
        'description': 'FFT of DC signal - energy at bin 0'
    }, output_dir)
    
    # Test 3: Sine wave at Nyquist/2
    n = 16
    sine = np.sin(2 * np.pi * 2 * np.arange(n) / n)  # 2 cycles in 16 samples
    fft_result = fft(sine)
    
    save_test_case('fft_sine', {
        'input': sine,
        'expected': fft_result,
        'description': 'FFT of sine wave - peaks at bins 2 and N-2'
    }, output_dir)
    
    # Test 4: Random signal (for numerical precision)
    np.random.seed(42)
    random_signal = np.random.randn(32)
    fft_result = fft(random_signal)
    
    save_test_case('fft_random', {
        'input': random_signal,
        'expected': fft_result,
        'description': 'FFT of random signal - numerical precision test'
    }, output_dir)


def generate_window_tests(output_dir: Path):
    """Generate window function test cases."""
    
    for length in [8, 16, 32]:
        window = hann_window(length)
        save_test_case(f'hann_window_{length}', {
            'length': length,
            'expected': window,
            'description': f'Hann window of length {length}'
        }, output_dir)


def generate_dct_tests(output_dir: Path):
    """Generate DCT test cases."""
    
    # Test 1: Simple signal
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    dct_result = dct(signal, type=2, norm='ortho')
    
    save_test_case('dct_simple', {
        'input': signal,
        'expected': dct_result,
        'description': 'DCT-II of simple ramp signal'
    }, output_dir)
    
    # Test 2: Random signal
    np.random.seed(123)
    random_signal = np.random.randn(16)
    dct_result = dct(random_signal, type=2, norm='ortho')
    
    save_test_case('dct_random', {
        'input': random_signal,
        'expected': dct_result,
        'description': 'DCT-II of random signal'
    }, output_dir)
    
    # Test 3: 2D signal (along axis 0)
    signal_2d = np.random.randn(8, 4)
    dct_result = dct(signal_2d, type=2, axis=0, norm='ortho')
    
    save_test_case('dct_2d', {
        'input': signal_2d,
        'expected': dct_result,
        'description': 'DCT-II along axis 0 of 2D array'
    }, output_dir)


def generate_cqt_tests(output_dir: Path):
    """Generate CQT test cases."""
    
    # Common parameters (matching our detector config)
    sample_rate = 16000
    f_min = 500.0
    n_bins = 48
    bins_per_octave = 12
    hop_length = 512
    
    # Test 1: Sine wave at known frequency
    duration = 0.5  # 0.5 seconds
    t = np.arange(int(sample_rate * duration)) / sample_rate
    frequency = 1000.0  # 1 kHz - should be in our range
    signal = np.sin(2 * np.pi * frequency * t)
    
    cqt = CQTReference(sample_rate, f_min, n_bins, bins_per_octave, hop_length)
    cqt_result = cqt.compute(signal)
    
    save_test_case('cqt_sine_1khz', {
        'sample_rate': sample_rate,
        'f_min': f_min,
        'n_bins': n_bins,
        'bins_per_octave': bins_per_octave,
        'hop_length': hop_length,
        'input': signal,
        'expected': cqt_result,
        'frequencies': cqt.frequencies,
        'n_fft': cqt.n_fft,
        'description': 'CQT of 1kHz sine wave'
    }, output_dir)
    
    # Test 2: Short random signal
    np.random.seed(456)
    random_signal = np.random.randn(4096)  # ~0.25 seconds
    cqt_result = cqt.compute(random_signal)
    
    save_test_case('cqt_random', {
        'sample_rate': sample_rate,
        'f_min': f_min,
        'n_bins': n_bins,
        'bins_per_octave': bins_per_octave,
        'hop_length': hop_length,
        'input': random_signal,
        'expected': cqt_result,
        'n_fft': cqt.n_fft,
        'description': 'CQT of random noise'
    }, output_dir)
    
    # Test 3: Single frame
    frame = np.random.randn(cqt.n_fft)
    frame_result = cqt.compute_frame(frame)
    
    save_test_case('cqt_single_frame', {
        'sample_rate': sample_rate,
        'f_min': f_min,
        'n_bins': n_bins,
        'bins_per_octave': bins_per_octave,
        'hop_length': hop_length,
        'input': frame,
        'expected_complex': frame_result,
        'expected_magnitude': np.abs(frame_result),
        'n_fft': cqt.n_fft,
        'description': 'Single CQT frame'
    }, output_dir)


def generate_cepstrum_tests(output_dir: Path):
    """Generate cepstrum (CQT + log + DCT) test cases."""
    
    sample_rate = 16000
    f_min = 500.0
    n_bins = 48
    bins_per_octave = 12
    hop_length = 512
    n_coeffs = 24
    
    # Test signal
    np.random.seed(789)
    signal = np.random.randn(8000)  # 0.5 seconds
    
    # Full cepstrum
    cepstrum = compute_cepstrum(
        signal, sample_rate, f_min, n_bins, bins_per_octave, hop_length, n_coeffs
    )
    
    # Also save intermediate results for debugging
    cqt = CQTReference(sample_rate, f_min, n_bins, bins_per_octave, hop_length)
    cqt_mag = cqt.compute(signal)
    log_cqt = np.log(cqt_mag + 1e-6)
    full_dct = dct(log_cqt, type=2, axis=0, norm='ortho')
    
    save_test_case('cepstrum_full', {
        'sample_rate': sample_rate,
        'f_min': f_min,
        'n_bins': n_bins,
        'bins_per_octave': bins_per_octave,
        'hop_length': hop_length,
        'n_coeffs': n_coeffs,
        'input': signal,
        'cqt_magnitude': cqt_mag,
        'log_cqt': log_cqt,
        'full_dct': full_dct,
        'expected': cepstrum,
        'n_fft': cqt.n_fft,
        'description': 'Full cepstrum extraction pipeline'
    }, output_dir)


def main():
    output_dir = Path(__file__).parent.parent / 'testdata'
    output_dir.mkdir(exist_ok=True)
    
    print("Generating test data...")
    print(f"Output directory: {output_dir}")
    print()
    
    generate_fft_tests(output_dir)
    generate_window_tests(output_dir)
    generate_dct_tests(output_dir)
    generate_cqt_tests(output_dir)
    generate_cepstrum_tests(output_dir)
    
    print()
    print("Done! Test data generated.")
    print(f"Files saved to: {output_dir}")


if __name__ == "__main__":
    main()
