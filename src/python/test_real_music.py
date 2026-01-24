"""
Test AI Music Detector against a music corpus.

Usage:
    python test_real_music.py
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from collections import Counter

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm
from inference_cnn import CepstrumCNNDetector


def find_audio_files(directory: Path) -> dict:
    """Find all audio files in directory, grouped by extension."""
    extensions = {'.mp3', '.wav', '.flac', '.ogg'}
    files_by_ext = {ext: [] for ext in extensions}
    
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in extensions:
                files_by_ext[ext].append(os.path.join(root, filename))
    
    return files_by_ext


def get_gold_label(file_path: str, mode: str) -> Optional[bool]:
    """
    Get gold label for a file.
    
    Args:
        file_path: Path to audio file
        mode: 'real', 'fake', or 'mixed'
    
    Returns:
        True if AI/fake, False if real, None if cannot determine
    """
    if mode == "real":
        return False  # All real
    elif mode == "fake":
        return True   # All AI
    else:
        # Mixed mode: check filename and path
        path_lower = file_path.lower()
        
        has_fake = "fake" in path_lower or "ai" in path_lower or "suno" in path_lower or "udio" in path_lower
        has_real = "real" in path_lower or "human" in path_lower
        
        if has_fake and not has_real:
            return True
        elif has_real and not has_fake:
            return False
        else:
            return None  # Cannot determine


def test_music(
    test_dir: Path,
    mode: str,
    model_path: str = None,
    segments: int = 5  # 5 segments for robust median
) -> Tuple[dict, List[dict]]:
    """
    Test model against music corpus.
    
    Returns:
        (stats_dict, results_list)
    """
    # Find all audio files
    print(f"\nScanning for audio files in: {test_dir}")
    files_by_ext = find_audio_files(test_dir)
    
    # Print file counts
    total = 0
    print("\nAudio files found:")
    for ext in ['.mp3', '.wav', '.flac', '.ogg']:
        count = len(files_by_ext[ext])
        if count > 0:
            print(f"  {ext}: {count}")
        total += count
    print(f"  Total: {total}")
    
    if total == 0:
        print("ERROR: No audio files found!")
        return {}, []
    
    # Flatten file list
    all_files = []
    for ext_files in files_by_ext.values():
        all_files.extend(ext_files)
    all_files.sort()
    
    # Validate gold labels for mixed mode
    if mode == "mixed":
        unlabeled = []
        for f in all_files:
            if get_gold_label(f, mode) is None:
                unlabeled.append(f)
        
        if unlabeled:
            print(f"\nERROR: Cannot determine gold label for {len(unlabeled)} files in mixed mode.")
            print("Files must contain 'fake'/'ai'/'suno'/'udio' or 'real'/'human' in path/filename.")
            print("\nExamples of unlabeled files:")
            for f in unlabeled[:10]:
                print(f"  {Path(f).name}")
            if len(unlabeled) > 10:
                print(f"  ... and {len(unlabeled) - 10} more")
            sys.exit(1)
    
    # Load detector
    print(f"\nLoading model...")
    detector = CepstrumCNNDetector(model_path=model_path)
    
    # Test each file
    print(f"\nTesting {total} files (mode: {mode})...")
    
    correct = 0
    tested = 0
    errors = []
    false_positives = []  # Real classified as AI
    false_negatives = []  # AI classified as Real
    
    pbar = tqdm(all_files, desc="Testing", unit="file")
    interrupted = False
    
    try:
        for file_path in pbar:
            filename = Path(file_path).name
            gold_label = get_gold_label(file_path, mode)  # True = AI, False = Real
            
            result = detector.predict(file_path, n_segments=segments)
            
            if "error" in result:
                errors.append({"file": file_path, "error": result["error"]})
                continue
            
            tested += 1
            predicted_ai = result["classification"] == "AI-Generated"
            
            # Check correctness
            if predicted_ai == gold_label:
                correct += 1
            else:
                entry = {
                    "file": file_path,
                    "filename": filename,
                    "gold": "AI" if gold_label else "Real",
                    "predicted": "AI" if predicted_ai else "Real",
                    "ai_probability": result["ai_probability"],
                    "segment_probs": result["segment_probs"],
                    "max_prob": result["max_prob"],
                    "median_prob": result["median_prob"],
                    "avg_prob": result["avg_prob"]
                }
                if gold_label:  # Was AI, predicted Real
                    false_negatives.append(entry)
                else:  # Was Real, predicted AI
                    false_positives.append(entry)
            
            # Update progress bar with current accuracy
            acc = correct / tested if tested > 0 else 0
            pbar.set_postfix({
                "acc": f"{acc*100:.1f}%",
                "FP": len(false_positives),
                "FN": len(false_negatives)
            })
    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving partial results...")
        interrupted = True
    finally:
        pbar.close()
    
    # Compute final stats
    accuracy = correct / tested if tested > 0 else 0
    
    stats = {
        "total_files": total,
        "tested": tested,
        "errors": len(errors),
        "correct": correct,
        "accuracy": accuracy,
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives)
    }
    
    # Print results
    print(f"\n{'='*70}")
    print("RESULTS" + (" (PARTIAL - interrupted)" if interrupted else ""))
    print(f"{'='*70}")
    print(f"Total files:       {total}")
    print(f"Tested:            {tested}")
    print(f"Errors:            {len(errors)}")
    print(f"Correct:           {correct}")
    print(f"Accuracy:          {accuracy*100:.2f}%")
    print(f"False Positives:   {len(false_positives)} (Real -> AI)")
    print(f"False Negatives:   {len(false_negatives)} (AI -> Real)")
    print(f"{'='*70}")
    
    # Show false positives
    if false_positives:
        print(f"\nFALSE POSITIVES - Real music classified as AI ({len(false_positives)}):")
        print("-" * 70)
        for fp in false_positives[:15]:
            print(f"  {fp['ai_probability']*100:5.1f}% {fp['file']}")
        if len(false_positives) > 15:
            print(f"  ... and {len(false_positives) - 15} more")
    
    # Show false negatives
    if false_negatives:
        print(f"\nFALSE NEGATIVES - AI music classified as Real ({len(false_negatives)}):")
        print("-" * 70)
        for fn in false_negatives[:15]:
            print(f"  {fn['ai_probability']*100:5.1f}% {fn['file']}")
        if len(false_negatives) > 15:
            print(f"  ... and {len(false_negatives) - 15} more")
    
    # Write full paths to output files
    output_dir = Path(test_dir)
    
    if false_positives:
        fp_file = output_dir / "misclassified_false_positives.txt"
        with open(fp_file, 'w', encoding='utf-8') as f:
            f.write(f"# False Positives: Real music classified as AI\n")
            f.write(f"# Total: {len(false_positives)}\n\n")
            for fp in sorted(false_positives, key=lambda x: x['ai_probability'], reverse=True):
                f.write(f"{fp['ai_probability']*100:.2f}%\t{fp['file']}\n")
        print(f"\nFalse positives written to: {fp_file}")
    
    if false_negatives:
        fn_file = output_dir / "misclassified_false_negatives.txt"
        with open(fn_file, 'w', encoding='utf-8') as f:
            f.write(f"# False Negatives: AI music classified as Real\n")
            f.write(f"# Total: {len(false_negatives)}\n\n")
            for fn in sorted(false_negatives, key=lambda x: x['ai_probability']):
                f.write(f"{fn['ai_probability']*100:.2f}%\t{fn['file']}\n")
        print(f"False negatives written to: {fn_file}")
    
    # Combined misclassified file
    if false_positives or false_negatives:
        all_file = output_dir / "misclassified_all.txt"
        with open(all_file, 'w', encoding='utf-8') as f:
            f.write(f"# All misclassified files\n")
            f.write(f"# False Positives (Real->AI): {len(false_positives)}\n")
            f.write(f"# False Negatives (AI->Real): {len(false_negatives)}\n\n")
            for entry in false_positives + false_negatives:
                gold = "Real" if entry in false_positives else "AI"
                pred = "AI" if entry in false_positives else "Real"
                f.write(f"{entry['ai_probability']*100:.2f}%\t{gold}->{pred}\t{entry['file']}\n")
        print(f"All misclassified written to: {all_file}")
    
    return stats, false_positives + false_negatives


def main():
    print("=" * 70)
    print("AI Music Detector - Test Suite")
    print("=" * 70)
    
    # Get folder path interactively
    print("\nEnter path to folder containing audio files:")
    folder_input = input(">>> ").strip().strip('"').strip("'")
    
    test_dir = Path(folder_input)
    if not test_dir.exists():
        print(f"ERROR: Folder not found: {test_dir}")
        sys.exit(1)
    
    if not test_dir.is_dir():
        print(f"ERROR: Not a directory: {test_dir}")
        sys.exit(1)
    
    # Get sample type interactively
    print("\nWhat type of samples are in this folder?")
    print("  [1] real  - All songs are real (human-made)")
    print("  [2] fake  - All songs are AI-generated")
    print("  [3] mixed - Mix of real and fake (uses filename to classify)")
    
    while True:
        mode_input = input(">>> ").strip().lower()
        if mode_input in ["1", "real"]:
            mode = "real"
            break
        elif mode_input in ["2", "fake"]:
            mode = "fake"
            break
        elif mode_input in ["3", "mixed"]:
            mode = "mixed"
            break
        else:
            print("Invalid input. Enter 1/real, 2/fake, or 3/mixed:")
    
    print(f"\nMode: {mode}")
    
    # Run test
    stats, errors = test_music(test_dir, mode)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
