"""Compare NAudio vs FFmpeg output at native rate (no resampling)."""
import subprocess
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python compare_native_rate.py <mp3_file>")
    sys.exit(1)

mp3_path = sys.argv[1]
print(f"Analyzing: {mp3_path}\n")

# Get native sample rate
import json
probe = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', 
                       '-show_streams', mp3_path], capture_output=True, text=True)
info = json.loads(probe.stdout)
native_sr = int(info['streams'][0]['sample_rate'])
print(f"Native sample rate: {native_sr}")

# Load with FFmpeg at native rate, stereo (no resampling, no mono conversion)
cmd = [
    'ffmpeg', '-i', mp3_path,
    '-f', 'f32le', '-acodec', 'pcm_f32le',
    '-v', 'quiet', '-'
]
result = subprocess.run(cmd, capture_output=True)
ff_stereo = np.frombuffer(result.stdout, dtype=np.float32)
ff_left = ff_stereo[0::2]
ff_right = ff_stereo[1::2]

print(f"\nFFmpeg output (stereo):")
print(f"  Total samples: {len(ff_stereo)}")
print(f"  Samples per channel: {len(ff_left)}")

# Load with FFmpeg at native rate, mono
cmd_mono = [
    'ffmpeg', '-i', mp3_path,
    '-f', 'f32le', '-acodec', 'pcm_f32le',
    '-af', 'pan=mono|c0=0.5*c0+0.5*c1',
    '-v', 'quiet', '-'
]
result_mono = subprocess.run(cmd_mono, capture_output=True)
ff_mono = np.frombuffer(result_mono.stdout, dtype=np.float32)
print(f"FFmpeg output (mono): {len(ff_mono)} samples")

# What NAudio would produce:
# According to our test, NAudio produces 8,576,640 samples per channel at 48kHz
# After gapless trimming (skip 2257, end trim 1344): 8,573,039 samples
naudio_samples_per_channel = 8576640
naudio_skip = 2257
naudio_end_trim = 1344
naudio_output = naudio_samples_per_channel - naudio_skip - naudio_end_trim

print(f"\nNAudio expected (from test):")
print(f"  Raw samples per channel: {naudio_samples_per_channel}")
print(f"  Skip: {naudio_skip}")
print(f"  End trim: {naudio_end_trim}")
print(f"  Output samples: {naudio_output}")

# Compare with FFmpeg (which should have already applied gapless)
print(f"\nComparison:")
print(f"  FFmpeg samples per channel: {len(ff_left)}")
print(f"  NAudio expected output: {naudio_output}")
print(f"  Difference: {len(ff_left) - naudio_output}")

# Check if FFmpeg's sample count matches NAudio after gapless
# FFmpeg should be the reference since it matches torchaudio
ff_expected_raw = len(ff_left) + naudio_skip + naudio_end_trim
print(f"\nExpected raw NAudio samples: {ff_expected_raw}")
print(f"Actual raw NAudio samples:   {naudio_samples_per_channel}")
print(f"Difference: {naudio_samples_per_channel - ff_expected_raw}")

# Show first few samples after gapless skip position
print(f"\nFFmpeg first 5 mono samples:")
ff_mono_manual = (ff_left + ff_right) * 0.5
print(f"  {ff_mono_manual[:5]}")

print(f"\nFFmpeg mono (from filter) first 5 samples:")
print(f"  {ff_mono[:5]}")

# Let's find the actual alignment between FFmpeg and what NAudio would produce
# NAudio raw samples after skipping Xing frame (1152) and start_skip (1105)
# First, let's see what offset makes NAudio match FFmpeg

print("\n--- Finding correct offset ---")

# If NAudio has Xing frame decoded, first 1152 samples are from Xing frame (likely silence)
# Then the start_skip should bring us to where FFmpeg starts

# Let's find the offset by correlation
# We'll compare FFmpeg stereo output with "simulated" NAudio stereo

# Actually, let's just check: does FFmpeg apply end_pad trimming?
# FFmpeg output: 8,574,720 samples
# Total frames from Xing: 7445
# Audio content frames: 7444 (excluding Xing)
# Audio samples: 7444 * 1152 = 8,575,488

audio_frames = 7444
audio_samples = audio_frames * 1152
print(f"Audio frames (excluding Xing): {audio_frames}")
print(f"Audio samples at native rate: {audio_samples}")

# FFmpeg trimmed: audio_samples - ffmpeg_output
ffmpeg_trimmed = audio_samples - len(ff_left)
print(f"FFmpeg trimmed total: {ffmpeg_trimmed} samples")

# If this equals start_skip only (no end trim):
start_skip = 576 + 528 + 1  # FFmpeg formula
end_pad = 1344
print(f"Expected start_skip: {start_skip}")
print(f"Expected end_pad: {end_pad}")
print(f"Expected total trim: {start_skip + end_pad}")

# Difference
print(f"\nDifference (expected - actual): {start_skip + end_pad - ffmpeg_trimmed}")
print(f"This suggests end_pad is {'applied' if ffmpeg_trimmed > start_skip else 'NOT applied'}")

# Let's verify: maybe FFmpeg only applies start_skip, not end_pad when outputting PCM
if ffmpeg_trimmed == start_skip:
    print("FFmpeg applies ONLY start_skip (no end trim)")
elif ffmpeg_trimmed == start_skip + end_pad:
    print("FFmpeg applies BOTH start_skip AND end_pad")
else:
    # Maybe partial?
    print(f"FFmpeg trims {ffmpeg_trimmed} samples, which is {start_skip} + {ffmpeg_trimmed - start_skip}")
    if ffmpeg_trimmed < start_skip:
        print(f"  This is {start_skip - ffmpeg_trimmed} LESS than start_skip alone")
