"""Debug script to check audio sample rates and fakeprint characteristics."""

import sys
import numpy as np
import torchaudio
from pathlib import Path

def analyze_file(file_path):
    print(f"\n{'='*60}")
    print(f"File: {Path(file_path).name}")
    print('='*60)
    
    audio, sr = torchaudio.load(file_path)
    duration = audio.shape[1] / sr
    
    print(f"Sample rate: {sr} Hz")
    print(f"Channels: {audio.shape[0]}")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Samples: {audio.shape[1]}")
    
    return sr

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_audio.py <file1.mp3> [file2.mp3] ...")
        print("\nChecking SONICS sample files...")
        
        # Check a SONICS file
        sonics_path = Path("./data/sonics/fake_songs")
        if sonics_path.exists():
            for part in sonics_path.iterdir():
                if part.is_dir():
                    inner = part / "fake_songs"
                    if inner.exists():
                        for f in inner.iterdir():
                            if f.suffix == '.mp3':
                                analyze_file(str(f))
                                break
                    break
        
        # Check an FMA file
        fma_path = Path("./data/fma/fma_medium")
        if fma_path.exists():
            for folder in fma_path.iterdir():
                if folder.is_dir() and folder.name.isdigit():
                    for f in folder.iterdir():
                        if f.suffix == '.mp3':
                            analyze_file(str(f))
                            break
                    break
    else:
        for f in sys.argv[1:]:
            analyze_file(f)

if __name__ == "__main__":
    main()
