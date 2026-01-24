"""
Generate test data using librosa.cqt with res_type='scipy' to match C# implementation.
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


def generate_tests():
    """Generate CQT test cases using librosa with res_type='scipy'."""
    
    output_dir = Path(__file__).parent.parent / "CQT.Tests" / "testdata"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Match our detector's exact parameters
    sample_rate = 16000
    f_min = 500.0
    n_bins = 48
    bins_per_octave = 12
    hop_length = 512
    
    print(f"\nGenerating librosa CQT tests with res_type='scipy':")
    print(f"  sample_rate={sample_rate}")
    print(f"  fmin={f_min}")
    print(f"  n_bins={n_bins}")
    print(f"  bins_per_octave={bins_per_octave}")
    print(f"  hop_length={hop_length}")
    print()
    
    # Test 1: Random noise (reproducible)
    np.random.seed(42)
    random_signal = np.random.randn(8000).astype(np.float64)
    
    # Use res_type='scipy' to match our C# implementation
    cqt_result = librosa.cqt(
        random_signal, sr=sample_rate, fmin=f_min, n_bins=n_bins, 
        bins_per_octave=bins_per_octave, hop_length=hop_length,
        res_type='scipy', scale=True
    )
    cqt_mag = np.abs(cqt_result)
    
    save_test_case('librosa_cqt_scipy_random', {
        'sample_rate': sample_rate,
        'f_min': f_min,
        'n_bins': n_bins,
        'bins_per_octave': bins_per_octave,
        'hop_length': hop_length,
        'res_type': 'scipy',
        'input': random_signal,
        'expected_magnitude': cqt_mag,
        'description': 'librosa CQT with res_type=scipy of random signal'
    }, output_dir)
    
    # Print some reference values
    print(f"\nCQT shape: {cqt_result.shape}")
    print(f"\nPer-octave frame 0 magnitudes:")
    for octave in range(4):
        start_bin = octave * 12
        end_bin = start_bin + 12
        octave_mags = cqt_mag[start_bin:end_bin, 0]
        print(f"  Octave {octave} (bins {start_bin}-{end_bin-1}): mean={np.mean(octave_mags):.4f}")
    
    print("\nFirst 5 bins, frame 0:")
    for i in range(5):
        print(f"  Bin {i}: {cqt_mag[i, 0]:.6f}")
    
    # Also compare with soxr_hq to show the difference
    cqt_soxr = librosa.cqt(
        random_signal, sr=sample_rate, fmin=f_min, n_bins=n_bins, 
        bins_per_octave=bins_per_octave, hop_length=hop_length,
        res_type='soxr_hq', scale=True
    )
    cqt_mag_soxr = np.abs(cqt_soxr)
    
    print("\n\nComparison: scipy vs soxr_hq resampling:")
    for octave in range(4):
        start_bin = octave * 12
        end_bin = start_bin + 12
        scipy_mags = cqt_mag[start_bin:end_bin, 0]
        soxr_mags = cqt_mag_soxr[start_bin:end_bin, 0]
        rel_diff = np.mean(np.abs(scipy_mags - soxr_mags) / (soxr_mags + 1e-10)) * 100
        print(f"  Octave {octave}: {rel_diff:.1f}% relative difference")


if __name__ == '__main__':
    generate_tests()
