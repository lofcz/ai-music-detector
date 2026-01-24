"""Check sample rates of test files to find edge cases."""
import os
from pathlib import Path
import subprocess
import json
from collections import Counter

# Check the v2 folder (used in 100-file test)
v2_dir = Path(r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v2")

sample_rates = Counter()
channel_counts = Counter()

files = list(v2_dir.glob("*.mp3"))[:200]  # Check first 200

print(f"Checking {len(files)} files...")

for f in files:
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', str(f)],
            capture_output=True, text=True, timeout=10
        )
        data = json.loads(result.stdout)
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'audio':
                sr = stream.get('sample_rate', 'unknown')
                ch = stream.get('channels', 'unknown')
                sample_rates[sr] += 1
                channel_counts[ch] += 1
                break
    except Exception as e:
        print(f"Error on {f.name}: {e}")

print("\nSample rates:")
for sr, count in sample_rates.most_common():
    print(f"  {sr}Hz: {count} files")

print("\nChannel counts:")
for ch, count in channel_counts.most_common():
    print(f"  {ch} channels: {count} files")
