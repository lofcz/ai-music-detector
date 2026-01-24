"""
Generate test data using librosa.cqt to match our actual detector pipeline.

This verifies the C# implementation matches librosa's output exactly.
"""

import json
import numpy as np
from pathlib import Path
import librosa
from scipy.fft import dct


def save_test_case(name: str, data: dict, output_dir: Path):
    """Save a test case to JSON."""
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


def generate_librosa_cqt_tests(output_dir: Path):
    """Generate CQT test cases using librosa."""
    
    # Match our detector's exact parameters
    sample_rate = 16000
    f_min = 500.0
    n_bins = 48
    bins_per_octave = 12
    hop_length = 512
    n_coeffs = 24
    
    print(f"\nGenerating librosa CQT tests with:")
    print(f"  sample_rate={sample_rate}")
    print(f"  fmin={f_min}")
    print(f"  n_bins={n_bins}")
    print(f"  bins_per_octave={bins_per_octave}")
    print(f"  hop_length={hop_length}")
    print()
    
    # Test 1: Sine wave at 1kHz
    duration = 0.5
    t = np.arange(int(sample_rate * duration)) / sample_rate
    frequency = 1000.0
    signal = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    cqt_result = librosa.cqt(
        signal, sr=sample_rate, fmin=f_min, n_bins=n_bins, 
        bins_per_octave=bins_per_octave, hop_length=hop_length
    )
    cqt_mag = np.abs(cqt_result)
    
    save_test_case('librosa_cqt_sine_1khz', {
        'sample_rate': sample_rate,
        'f_min': f_min,
        'n_bins': n_bins,
        'bins_per_octave': bins_per_octave,
        'hop_length': hop_length,
        'input': signal,
        'expected_magnitude': cqt_mag,
        'expected_complex': cqt_result,
        'description': 'librosa CQT of 1kHz sine wave'
    }, output_dir)
    
    # Test 2: Random noise
    np.random.seed(42)
    random_signal = np.random.randn(8000).astype(np.float32)
    
    cqt_result = librosa.cqt(
        random_signal, sr=sample_rate, fmin=f_min, n_bins=n_bins, 
        bins_per_octave=bins_per_octave, hop_length=hop_length
    )
    cqt_mag = np.abs(cqt_result)
    
    save_test_case('librosa_cqt_random', {
        'sample_rate': sample_rate,
        'f_min': f_min,
        'n_bins': n_bins,
        'bins_per_octave': bins_per_octave,
        'hop_length': hop_length,
        'input': random_signal,
        'expected_magnitude': cqt_mag,
        'description': 'librosa CQT of random noise'
    }, output_dir)
    
    # Test 3: Full cepstrum pipeline (matches extract_cqt_features.py)
    np.random.seed(789)
    test_signal = np.random.randn(16000).astype(np.float32)  # 1 second
    
    # Step 1: CQT
    cqt_result = librosa.cqt(
        test_signal, sr=sample_rate, fmin=f_min, n_bins=n_bins, 
        bins_per_octave=bins_per_octave, hop_length=hop_length
    )
    cqt_mag = np.abs(cqt_result)
    
    # Step 2: Log magnitude
    log_cqt = np.log(cqt_mag + 1e-6)
    
    # Step 3: DCT along frequency axis
    full_dct = dct(log_cqt, type=2, axis=0, norm='ortho')
    
    # Step 4: Keep first n_coeffs
    cepstrum = full_dct[:n_coeffs, :]
    
    save_test_case('librosa_cepstrum_full', {
        'sample_rate': sample_rate,
        'f_min': f_min,
        'n_bins': n_bins,
        'bins_per_octave': bins_per_octave,
        'hop_length': hop_length,
        'n_coeffs': n_coeffs,
        'input': test_signal,
        'cqt_magnitude': cqt_mag,
        'log_cqt': log_cqt,
        'full_dct': full_dct,
        'expected': cepstrum,
        'description': 'Full cepstrum extraction matching extract_cqt_features.py'
    }, output_dir)
    
    print(f"\nCQT output shape: {cqt_mag.shape}")
    print(f"Cepstrum output shape: {cepstrum.shape}")
    
    # Test 4: Short segment (like inference)
    np.random.seed(123)
    segment = np.random.randn(160000).astype(np.float32)  # 10 seconds at 16kHz
    
    cqt_result = librosa.cqt(
        segment, sr=sample_rate, fmin=f_min, n_bins=n_bins, 
        bins_per_octave=bins_per_octave, hop_length=hop_length
    )
    cqt_mag = np.abs(cqt_result)
    log_cqt = np.log(cqt_mag + 1e-6)
    cepstrum = dct(log_cqt, type=2, axis=0, norm='ortho')[:n_coeffs, :]
    
    save_test_case('librosa_cepstrum_10s', {
        'sample_rate': sample_rate,
        'f_min': f_min,
        'n_bins': n_bins,
        'bins_per_octave': bins_per_octave,
        'hop_length': hop_length,
        'n_coeffs': n_coeffs,
        'input': segment,
        'cqt_magnitude': cqt_mag,
        'expected': cepstrum,
        'n_frames': cepstrum.shape[1],
        'description': '10-second segment cepstrum (like inference)'
    }, output_dir)
    
    print(f"10s segment: {cepstrum.shape[1]} frames")


def compare_our_cqt_vs_librosa():
    """Show difference between our reference implementation and librosa."""
    from cqt_reference import CQTReference
    
    sample_rate = 16000
    f_min = 500.0
    n_bins = 48
    bins_per_octave = 12
    hop_length = 512
    
    np.random.seed(42)
    signal = np.random.randn(8000).astype(np.float32)
    
    # Our implementation
    our_cqt = CQTReference(sample_rate, f_min, n_bins, bins_per_octave, hop_length)
    our_result = our_cqt.compute(signal)
    
    # librosa
    librosa_result = np.abs(librosa.cqt(
        signal, sr=sample_rate, fmin=f_min, n_bins=n_bins, 
        bins_per_octave=bins_per_octave, hop_length=hop_length
    ))
    
    print("\nComparison: Our CQT Reference vs librosa.cqt")
    print(f"  Our shape: {our_result.shape}")
    print(f"  librosa shape: {librosa_result.shape}")
    
    # Shapes might differ due to different padding/framing
    min_frames = min(our_result.shape[1], librosa_result.shape[1])
    our_trimmed = our_result[:, :min_frames]
    lib_trimmed = librosa_result[:, :min_frames]
    
    abs_diff = np.abs(our_trimmed - lib_trimmed)
    rel_diff = abs_diff / (np.abs(lib_trimmed) + 1e-10)
    
    print(f"  Max absolute difference: {abs_diff.max():.6f}")
    print(f"  Mean absolute difference: {abs_diff.mean():.6f}")
    print(f"  Max relative difference: {rel_diff.max():.2%}")
    print(f"  Mean relative difference: {rel_diff.mean():.2%}")
    
    print("\n  NOTE: librosa uses a different algorithm (more accurate).")
    print("  C# implementation should match librosa for best results.")


def main():
    output_dir = Path(__file__).parent.parent / 'testdata'
    output_dir.mkdir(exist_ok=True)
    
    print("Generating librosa-based test data...")
    print(f"Output directory: {output_dir}")
    
    generate_librosa_cqt_tests(output_dir)
    
    print("\n" + "="*60)
    compare_our_cqt_vs_librosa()
    
    print("\n" + "="*60)
    print("Done! librosa test data generated.")


if __name__ == "__main__":
    main()
