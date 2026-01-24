"""
Compare scipy.signal.resample to soxr in detail.
If we can understand the difference, we can either:
1. Port soxr's approach
2. Use scipy's approach (which we can implement in C#) and retrain
"""
import numpy as np
from scipy.signal import resample, resample_poly
from scipy.fft import rfft, irfft
import soxr

np.random.seed(42)

print("=== Comparing decimation methods ===\n")

# Test signal
test_signal = np.random.randn(2048).astype(np.float64)

# 1. soxr_hq
soxr_out = soxr.resample(test_signal, 2, 1, quality='soxr_hq')

# 2. scipy.signal.resample (FFT-based)
scipy_out = resample(test_signal, len(test_signal) // 2)

# 3. scipy.signal.resample_poly (polyphase)
# This requires integer up/down factors
poly_out = resample_poly(test_signal, 1, 2)

print(f"Input length: {len(test_signal)}")
print(f"soxr output length: {len(soxr_out)}")
print(f"scipy resample output length: {len(scipy_out)}")
print(f"scipy resample_poly output length: {len(poly_out)}")

def compare(name, out, ref):
    """Compare output to reference (soxr)."""
    min_len = min(len(out), len(ref))
    diff = np.abs(out[:min_len] - ref[:min_len])
    rel_diff = diff.mean() / (np.abs(ref[:min_len]).mean() + 1e-10)
    print(f"\n{name} vs soxr_hq:")
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  Relative diff: {100*rel_diff:.4f}%")
    
    # Check correlation
    corr = np.corrcoef(out[:min_len], ref[:min_len])[0,1]
    print(f"  Correlation: {corr:.6f}")
    
    return rel_diff

compare("scipy.resample", scipy_out, soxr_out)
compare("scipy.resample_poly", poly_out, soxr_out)

# Check DC response
dc = np.ones(1024, dtype=np.float64)
print("\n=== DC Response ===")
print(f"  soxr_hq: {soxr.resample(dc, 2, 1, quality='soxr_hq')[256]:.6f}")
print(f"  scipy.resample: {resample(dc, 512)[256]:.6f}")
print(f"  scipy.resample_poly: {resample_poly(dc, 1, 2)[256]:.6f}")

# Check impulse response shape
print("\n=== Impulse Response Analysis ===")
N = 4096
impulse = np.zeros(N, dtype=np.float64)
impulse[N//2] = 1.0

soxr_ir = soxr.resample(impulse, 2, 1, quality='soxr_hq')
scipy_ir = resample(impulse, N//2)
poly_ir = resample_poly(impulse, 1, 2)

print(f"soxr_hq peak: {np.abs(soxr_ir).max():.6f} at idx {np.argmax(np.abs(soxr_ir))}")
print(f"scipy peak: {np.abs(scipy_ir).max():.6f} at idx {np.argmax(np.abs(scipy_ir))}")
print(f"poly peak: {np.abs(poly_ir).max():.6f} at idx {np.argmax(np.abs(poly_ir))}")

# Analyze frequency response
print("\n=== Frequency Response ===")
# Create a chirp signal to test frequency response
from scipy.signal import chirp
t = np.linspace(0, 1, 4096)
# Chirp from 0 to Nyquist (at sr=2, Nyquist=1)
chirp_sig = chirp(t, f0=0, f1=0.5, t1=1, method='linear')

soxr_chirp = soxr.resample(chirp_sig, 2, 1, quality='soxr_hq')
scipy_chirp = resample(chirp_sig, 2048)

# Look at where they start to differ (transition band)
diff_chirp = np.abs(soxr_chirp - scipy_chirp)
print(f"Max diff in chirp test: {diff_chirp.max():.6f}")

# Find where the difference is largest
peak_diff_idx = np.argmax(diff_chirp)
print(f"Peak difference at output sample {peak_diff_idx} (freq ~{peak_diff_idx/2048:.3f} of Nyquist)")

# The key difference might be:
# 1. Filter design (Kaiser vs. ideal rectangular cutoff)
# 2. Transition band width
# 3. Stopband attenuation

print("\n=== Summary ===")
print("scipy.resample uses ideal brick-wall filter in frequency domain")
print("soxr uses Kaiser-windowed sinc filter with specific transition band")
print("")
print("To match soxr in C#, we need either:")
print("1. Port soxr's filter design + overlap-save convolution")
print("2. Use scipy's FFT-based approach (simpler) and retrain the model")

# Check if the difference matters for CQT
print("\n=== Impact on CQT ===")
import librosa

cqt_soxr = librosa.cqt(test_signal, sr=16000, fmin=500, n_bins=48,
                       bins_per_octave=12, hop_length=512, res_type='soxr_hq')
cqt_scipy = librosa.cqt(test_signal, sr=16000, fmin=500, n_bins=48,
                        bins_per_octave=12, hop_length=512, res_type='scipy')

cqt_diff = np.abs(np.abs(cqt_soxr) - np.abs(cqt_scipy))
cqt_rel = cqt_diff.mean() / (np.abs(cqt_soxr).mean() + 1e-10)

print(f"CQT magnitude difference (soxr vs scipy): {100*cqt_rel:.4f}%")
print("This is the difference we'd see if we retrained with scipy.")
