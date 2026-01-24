"""Compare Python vs C# inference on 100 files."""
import subprocess
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))
from inference_cnn import CepstrumCNNDetector

def run_csharp_batch(filepaths: list) -> dict:
    """Run C# inference in batch and return dict of filename -> AI probability."""
    result = subprocess.run(
        ["dotnet", "run", "--", "--batch", "--workers", "4"] + filepaths,
        cwd=r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\cs\AiMusicDetector.Console",
        capture_output=True,
        text=True
    )
    
    results = {}
    lines = result.stdout.split('\n')
    current_file = None
    for line in lines:
        if line.strip().endswith('.mp3') and not line.startswith(' '):
            current_file = re.sub(r'^\[\d+/\d+\]\s*', '', line.strip())
        match = re.search(r'AI:\s*([\d.]+)%', line)
        if match and current_file:
            results[current_file] = float(match.group(1)) / 100.0
            current_file = None
    
    return results

def main():
    # Read test files
    with open("test_100_files.txt", "r", encoding="utf-8-sig") as f:
        test_files = [line.strip() for line in f if line.strip()]
    
    print(f"Testing {len(test_files)} files")
    
    print("Loading Python detector...")
    detector = CepstrumCNNDetector()
    
    print("\nRunning C# batch inference...")
    cs_results = run_csharp_batch(test_files)
    print(f"C# returned {len(cs_results)} results")
    
    print("\nRunning Python inference...")
    
    disagreements = []
    py_correct = 0
    cs_correct = 0
    max_diff = 0
    
    for i, filepath in enumerate(test_files):
        name = Path(filepath).name
        
        # Python
        result = detector.predict(filepath, n_segments=5)
        py_prob = result.get('ai_probability', -1) if 'error' not in result else -1
        
        # C#
        cs_prob = cs_results.get(name, -1)
        
        if py_prob < 0 or cs_prob < 0:
            continue
            
        diff = abs(py_prob - cs_prob)
        max_diff = max(max_diff, diff)
        
        py_ai = py_prob > 0.5
        cs_ai = cs_prob > 0.5
        
        # All files are AI (fake), so correct = classified as AI
        if py_ai:
            py_correct += 1
        if cs_ai:
            cs_correct += 1
        
        if py_ai != cs_ai:
            disagreements.append({
                'file': name,
                'python': py_prob,
                'csharp': cs_prob
            })
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(test_files)}...")
    
    total = len(test_files)
    print(f"\n{'='*60}")
    print(f"RESULTS ({total} files, all AI-generated)")
    print(f"{'='*60}")
    print(f"Python accuracy: {py_correct}/{total} = {py_correct/total*100:.2f}%")
    print(f"C# accuracy:     {cs_correct}/{total} = {cs_correct/total*100:.2f}%")
    print(f"Disagreements:   {len(disagreements)}")
    print(f"Max difference:  {max_diff*100:.3f}%")
    
    if disagreements:
        print(f"\nDisagreements (different classifications):")
        for d in disagreements:
            print(f"  {d['file']}: Python={d['python']*100:.1f}%, C#={d['csharp']*100:.1f}%")

if __name__ == "__main__":
    main()
