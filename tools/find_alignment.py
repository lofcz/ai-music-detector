"""Find the correct alignment offset between NAudio and FFmpeg output."""
import subprocess
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python find_alignment.py <mp3_file>")
    sys.exit(1)

mp3_path = sys.argv[1]
print(f"Analyzing: {mp3_path}\n")

# Get FFmpeg output at native rate (stereo)
cmd = [
    'ffmpeg', '-i', mp3_path,
    '-f', 'f32le', '-acodec', 'pcm_f32le',
    '-v', 'quiet', '-'
]
result = subprocess.run(cmd, capture_output=True)
ff_stereo = np.frombuffer(result.stdout, dtype=np.float32)
ff_left = ff_stereo[0::2]
ff_right = ff_stereo[1::2]
ff_mono = (ff_left + ff_right) * 0.5

print(f"FFmpeg mono output: {len(ff_mono)} samples")

# Simulate NAudio raw output (we know it has 8576640 samples per channel)
# Since we can't run NAudio from Python, we'll use FFmpeg without gapless handling
# by setting start_skip_samples to 0 (decode everything)

# Actually, let's cheat - we know NAudio decodes everything including Xing frame
# The Xing frame produces 1152 samples of "garbage" at the start
# So NAudio_raw = FFmpeg + 1920 samples at the beginning

# What we need to find is the exact offset where NAudio (with our skip) would match FFmpeg

# Test different skip values
print("\nTesting different skip values (samples to skip from NAudio raw):")

best_skip = 0
best_correlation = -1

for skip in range(1800, 2100, 10):
    # Simulate: if we skip 'skip' samples from start and 0 from end
    # Then compare with FFmpeg (which is the reference)
    
    # Since NAudio raw = FFmpeg + prefix, and we skip 'skip' samples
    # The result should match FFmpeg if skip == len(prefix)
    
    # For testing, we can compare sample values at corresponding positions
    # If NAudio[skip + i] == FFmpeg[i], then skip is correct
    
    # We can't actually run NAudio here, so let's use the expected relationship:
    # NAudio raw has 1920 more samples than FFmpeg at the start
    # If skip = 1920, NAudio[1920:] should match FFmpeg[:]
    
    # But we can verify this expectation by checking FFmpeg's first samples
    pass

# Actually, let's just verify by checking sample values
# If NAudio[1920:] matches FFmpeg[:], then skip=1920 is correct

# From the previous test, NAudio first samples after skip=2257 are around -0.001114
# FFmpeg first samples are around -0.00079528

# Let's check what FFmpeg samples look like at different positions
print("\nFFmpeg mono samples at key positions:")
print(f"  Position 0: {ff_mono[0]:.8f}")
print(f"  Position 100: {ff_mono[100]:.8f}")
print(f"  Position 1000: {ff_mono[1000]:.8f}")

# From NAudio test, first mono sample after skip=2257 was -0.001114
# Let's find where in FFmpeg this value appears
target = -0.001114
print(f"\nSearching for value ~{target} in FFmpeg output...")
for i in range(min(5000, len(ff_mono))):
    if abs(ff_mono[i] - target) < 0.0001:
        print(f"  Found similar value at position {i}: {ff_mono[i]:.6f}")
        if i < 10:
            break

# Let's also calculate what the skip should be based on the formula
# FFmpeg docs say: start_skip_samples = start_pad + 528 + 1
# But we observe FFmpeg only trims 768 samples

# For this file:
# - start_pad = 576
# - Our formula: 1152 (Xing) + 576 (start_pad) + 528 + 1 = 2257
# - Observed: 1920 total

# The difference is 337 samples (2257 - 1920)
# 337 = 529 - 192 ... so actual decoder delay might be 192, not 529?

# Or: the additional skip beyond Xing frame is:
# 1920 - 1152 = 768
# 768 = 576 + 192
# So skip = Xing + start_pad + 192

# Let's check if 192 is a fixed value:
print("\n--- Analysis ---")
xing_frame = 1152
observed_total_skip = 1920
skip_after_xing = observed_total_skip - xing_frame
start_pad = 576

print(f"Xing frame samples: {xing_frame}")
print(f"Observed total skip: {observed_total_skip}")
print(f"Skip after Xing: {skip_after_xing}")
print(f"start_pad from header: {start_pad}")
print(f"Additional beyond start_pad: {skip_after_xing - start_pad}")

# Formula hypothesis: skip = Xing + start_pad + 192
hypothesis = xing_frame + start_pad + 192
print(f"\nHypothesis: skip = Xing + start_pad + 192 = {hypothesis}")
print(f"Matches observed: {hypothesis == observed_total_skip}")

# Alternative: maybe the 192 is always 576/3?
alt = xing_frame + start_pad + start_pad // 3
print(f"\nAlternative: skip = Xing + start_pad + start_pad//3 = {alt}")
print(f"Matches observed: {alt == observed_total_skip}")
