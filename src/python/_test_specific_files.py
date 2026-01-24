"""Test specific files and output detailed diagnostics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from inference_cnn import CepstrumCNNDetector

# Files that Python misclassified as Real (but are AI)
test_files = [
    r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v2\bc44019d-be0f-4b84-b124-b9f1d217fe3f.mp3",
    r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v3.5\3cd737c1-4294-432f-93d7-2653f2313798.mp3",
    r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v3.5\4c4c1218-e4de-468f-8d58-ad5b54b90fb9.mp3",
    r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v3\7af359af-e5e6-45d1-8c79-c94841f0aaa0.mp3",
    r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v3\8e2a8828-b8ee-46b6-8789-21bc0475fc2d.mp3",
]

def main():
    print("Loading detector...")
    detector = CepstrumCNNDetector()
    
    print("\n" + "="*80)
    print("Testing specific misclassified files")
    print("="*80)
    
    for filepath in test_files:
        if not Path(filepath).exists():
            print(f"\nFILE NOT FOUND: {filepath}")
            continue
            
        print(f"\n{'-'*80}")
        print(f"File: {Path(filepath).name}")
        print(f"Path: {filepath}")
        
        # Test with different segment counts
        for n_seg in [1, 3, 5]:
            result = detector.predict(filepath, n_segments=n_seg)
            if "error" in result:
                print(f"  [{n_seg} seg] ERROR: {result['error']}")
            else:
                probs = result.get('segment_probs', [])
                probs_str = ', '.join([f"{p*100:.1f}%" for p in probs])
                print(f"  [{n_seg} seg] AI={result['ai_probability']*100:5.1f}%  class={result['classification']:12}  probs=[{probs_str}]")

if __name__ == "__main__":
    main()
