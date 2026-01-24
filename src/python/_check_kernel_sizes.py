"""Check SincResampler kernel sizes for common sample rates."""
import math

def get_kernel_info(orig_freq, new_freq, lowpass_filter_width=6, rolloff=0.99):
    """Calculate kernel parameters matching torchaudio."""
    gcd = math.gcd(orig_freq, new_freq)
    orig_reduced = orig_freq // gcd
    new_reduced = new_freq // gcd
    
    base_freq = min(orig_reduced, new_reduced) * rolloff
    width = math.ceil(lowpass_filter_width * orig_reduced / base_freq)
    filter_len = 2 * width + orig_reduced
    
    # Total kernel bank size: new_reduced * filter_len
    kernel_size = new_reduced * filter_len
    kernel_mb = kernel_size * 8 / (1024 * 1024)  # double = 8 bytes
    
    return {
        'gcd': gcd,
        'orig_reduced': orig_reduced,
        'new_reduced': new_reduced,
        'width': width,
        'filter_len': filter_len,
        'kernel_elements': kernel_size,
        'kernel_mb': kernel_mb
    }

target_sr = 16000
sample_rates = [8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000]

print("Kernel sizes for resampling to 16000Hz:")
print("-" * 80)
print(f"{'Source':<10} {'GCD':<8} {'Ratio':<12} {'Width':<8} {'FilterLen':<12} {'Kernel MB':<10}")
print("-" * 80)

for sr in sample_rates:
    if sr == target_sr:
        print(f"{sr:<10} (no resampling needed)")
        continue
    
    info = get_kernel_info(sr, target_sr)
    ratio = f"{info['orig_reduced']}:{info['new_reduced']}"
    print(f"{sr:<10} {info['gcd']:<8} {ratio:<12} {info['width']:<8} {info['filter_len']:<12} {info['kernel_mb']:.2f}")
