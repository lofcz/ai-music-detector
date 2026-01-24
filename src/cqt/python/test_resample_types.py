"""
Test different res_type options in librosa to find one we can match.
"""
import numpy as np
import librosa

# Match our detector's exact parameters
SR = 16000
FMIN = 500.0
N_BINS = 48
BINS_PER_OCTAVE = 12
HOP_LENGTH = 512

np.random.seed(42)
signal = np.random.randn(16000).astype(np.float64)

# Get the default CQT (soxr_hq)
cqt_default = np.abs(librosa.cqt(
    signal, sr=SR, fmin=FMIN, n_bins=N_BINS,
    bins_per_octave=BINS_PER_OCTAVE, hop_length=HOP_LENGTH, scale=True,
    res_type='soxr_hq'
))

print("Testing different res_type options:")
print(f"Default (soxr_hq) shape: {cqt_default.shape}\n")

# Test different resampling methods
res_types = [
    'soxr_hq',      # Default, high-quality SoX resampler
    'soxr_mq',      # Medium quality SoX
    'soxr_lq',      # Low quality SoX
    'soxr_vhq',     # Very high quality SoX
    'scipy',        # scipy.signal.resample (FFT-based!)
    'polyphase',    # scipy.signal.resample_poly
    'fft',          # FFT-based (same as scipy)
    'kaiser_best',  # scipy.signal.resample with kaiser window
    'kaiser_fast',  # scipy.signal.resample with kaiser window (fast)
]

for res_type in res_types:
    try:
        cqt = np.abs(librosa.cqt(
            signal, sr=SR, fmin=FMIN, n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE, hop_length=HOP_LENGTH, scale=True,
            res_type=res_type
        ))
        
        diff = np.abs(cqt - cqt_default).mean()
        rel_diff = diff / (cqt_default.mean() + 1e-10)
        
        if rel_diff < 0.0001:
            status = "≈ identical"
        elif rel_diff < 0.01:
            status = f"~{100*rel_diff:.2f}% diff"
        else:
            status = f"{100*rel_diff:.1f}% diff"
        
        print(f"  {res_type:15s}: {status}")
        
    except Exception as e:
        print(f"  {res_type:15s}: ERROR - {e}")

print("\n" + "="*60)
print("Testing if 'scipy'/'fft' res_type matches FFT-based decimation:")

# Test with scipy res_type
cqt_scipy = np.abs(librosa.cqt(
    signal, sr=SR, fmin=FMIN, n_bins=N_BINS,
    bins_per_octave=BINS_PER_OCTAVE, hop_length=HOP_LENGTH, scale=True,
    res_type='scipy'
))

diff_scipy_default = np.abs(cqt_scipy - cqt_default).mean()
rel_diff_scipy = diff_scipy_default / (cqt_default.mean() + 1e-10)

print(f"\nCQT with res_type='scipy' vs default (soxr_hq):")
print(f"  Mean absolute diff: {diff_scipy_default:.6f}")
print(f"  Mean relative diff: {100*rel_diff_scipy:.4f}%")

if rel_diff_scipy > 0.01:
    print(f"\n⚠️  res_type='scipy' differs significantly from default!")
    print(f"   If C# uses FFT-based resampling, train with res_type='scipy'")
    print(f"   Or port soxr to C# to match default training")

# Check if scipy is available for re-training
print("\n" + "="*60)
print("RECOMMENDATION:")
print("-" * 60)

if rel_diff_scipy < 0.001:
    print("res_type='scipy' is close enough to soxr_hq")
    print("C# FFT-based decimation should work")
else:
    print("To match C# FFT-based decimation, you have two options:")
    print("")
    print("Option 1: RE-TRAIN with res_type='scipy'")
    print("  - Modify extract_cqt_features.py to use res_type='scipy'")
    print("  - Re-extract features and re-train the model")
    print("  - C# implementation will then match")
    print("")
    print("Option 2: PORT soxr_hq to C#")
    print("  - Source: https://github.com/chirlu/soxr")
    print("  - Complex polyphase resampler implementation")
    print("  - Significant engineering effort")
