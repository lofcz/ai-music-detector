"""
Quick test script to validate the fakeprint extraction and model on sample files.

This script can be used to test the pipeline with the reference audio samples
provided in the deepfake-detector-benchmarking folder.

Usage:
    python test_with_samples.py
"""

import os
import sys
from pathlib import Path
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from extract_fakeprints import FakeprintExtractor, find_audio_files

def main():
    # Check for reference samples
    reference_dir = Path(__file__).parent.parent / "reference" / "deepfake-detector-benchmarking" / "audio_examples"
    
    if not reference_dir.exists():
        print("Reference audio samples not found.")
        print(f"Expected at: {reference_dir}")
        print("\nTo test, provide your own audio files or download the datasets.")
        return
    
    print("AI Music Detector - Test Script")
    print("=" * 60)
    
    # Create extractor
    extractor = FakeprintExtractor()
    print(f"Fakeprint extractor initialized")
    print(f"  Device: {extractor.device}")
    print(f"  FFT size: {extractor.n_fft}")
    print(f"  Frequency range: {extractor.freq_min}-{extractor.freq_max} Hz")
    print(f"  Output dimension: {len(extractor.freq_range)}")
    print()
    
    # Find sample directories
    sample_dirs = [d for d in reference_dir.iterdir() if d.is_dir()]
    
    results = {}
    
    for sample_dir in sorted(sample_dirs):
        label = sample_dir.name
        is_fake = label != "real"
        
        audio_files = find_audio_files(sample_dir)
        if not audio_files:
            continue
        
        print(f"\n{label.upper()} samples ({len(audio_files)} files):")
        print("-" * 40)
        
        fakeprints = []
        for audio_file in audio_files[:5]:  # Limit for quick testing
            try:
                fp = extractor.extract_fakeprint(audio_file)
                if fp is not None:
                    fakeprints.append(fp)
                    
                    # Analyze fakeprint characteristics
                    peak_count = np.sum(fp > 0.5)
                    mean_val = np.mean(fp)
                    max_val = np.max(fp)
                    
                    filename = Path(audio_file).name
                    print(f"  {filename}: peaks={peak_count}, mean={mean_val:.3f}, max={max_val:.3f}")
                    
            except Exception as e:
                print(f"  Error processing {audio_file}: {e}")
        
        if fakeprints:
            results[label] = {
                "fakeprints": np.stack(fakeprints, axis=0),
                "is_fake": is_fake
            }
    
    # Summary analysis
    if results:
        print("\n" + "=" * 60)
        print("SUMMARY ANALYSIS")
        print("=" * 60)
        
        for label, data in results.items():
            fps = data["fakeprints"]
            mean_fp = np.mean(fps, axis=0)
            
            # Measure artifact strength
            peak_strength = np.mean(np.max(fps, axis=1))
            artifact_score = np.mean(fps > 0.3)
            
            status = "FAKE" if data["is_fake"] else "REAL"
            print(f"\n{label} [{status}]:")
            print(f"  Peak strength: {peak_strength:.3f}")
            print(f"  Artifact ratio: {artifact_score:.3f}")
            
            # AI-generated audio should have stronger artifacts
            if data["is_fake"]:
                if artifact_score > 0.1:
                    print("  ✓ Detected: Strong artifacts suggest AI-generated")
                else:
                    print("  ? Weak artifacts - may be hard to detect")
            else:
                if artifact_score < 0.1:
                    print("  ✓ Expected: Low artifacts suggest real audio")
                else:
                    print("  ? Unexpected artifacts in real audio")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run download_data.py to get full datasets")
    print("2. Run extract_fakeprints.py on real and fake datasets")
    print("3. Run train_model.py to train the classifier")
    print("4. Run export_onnx.py to create the C# model")


if __name__ == "__main__":
    main()
