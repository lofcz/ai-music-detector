"""Compare Python vs C# inference on the same files."""
import subprocess
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))
from inference_cnn import CepstrumCNNDetector

# Test files
test_files = [
    r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v3.5\11e59acf-b954-46bc-ab43-5f3f097271da.mp3",
    r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v3.5\15904142-00d2-4b47-8fb7-02487f7dd4d7.mp3",
    r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v3.5\2b8c361d-2900-4af1-a09b-ea82aa5a1c2b.mp3",
    r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v3.5\509db95a-59ac-4eda-95e1-742ddd55d04a.mp3",
    r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v3.5\5378f4ce-4281-48a4-a7f4-0ee890dbcd26.mp3",
    r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v3.5\57dd32fa-2b08-43ea-8076-72e755d09fc0.mp3",
    r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v3.5\6c6f2e26-d9a1-4411-a027-de3b1732cf82.mp3",
    r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v3.5\6f519d74-4783-4038-ae74-40ed733bae10.mp3",
    r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v3.5\83011d82-11ce-4426-bfea-0c804225d5ae.mp3",
    r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v3.5\90982ee1-9920-4458-89e0-96ede40ed9d3.mp3",
    r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v3.5\9ad2d003-ed57-467a-a914-5e92e32817a9.mp3",
    r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v3.5\9d590e23-18ee-4049-9637-10a9d568cbce.mp3",
    r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v3.5\aa7f3705-0c0b-4da3-9a0d-704ba4039651.mp3",
    r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v3.5\cca82e8b-8e09-4615-90f7-916c95c53bc0.mp3",
    r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v3.5\cd137b5a-1086-407c-ba16-e92fb411142e.mp3",
]

def run_csharp_batch(filepaths: list) -> dict:
    """Run C# inference in batch and return dict of filename -> AI probability."""
    result = subprocess.run(
        ["dotnet", "run", "--", "--batch", "--workers", "4"] + filepaths,
        cwd=r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\cs\AiMusicDetector.Console",
        capture_output=True,
        text=True
    )
    
    # Parse output: "filename.mp3\n  🎵 Real Music (AI: XX.X%)"
    results = {}
    lines = result.stdout.split('\n')
    current_file = None
    for line in lines:
        # Check for filename line (ends with .mp3 and not indented)
        if line.strip().endswith('.mp3') and not line.startswith(' '):
            # Remove [X/Y] prefix if present
            current_file = re.sub(r'^\[\d+/\d+\]\s*', '', line.strip())
        # Check for AI percentage
        match = re.search(r'AI:\s*([\d.]+)%', line)
        if match and current_file:
            results[current_file] = float(match.group(1)) / 100.0
            current_file = None
    
    return results

def main():
    print("Loading Python detector...")
    detector = CepstrumCNNDetector()
    
    print("\nRunning C# batch inference...")
    cs_results = run_csharp_batch(test_files)
    
    print("\n" + "="*90)
    print(f"{'File':<45} {'Python':>10} {'C#':>10} {'Diff':>10} {'Match':>8}")
    print("="*90)
    
    disagreements = []
    max_diff = 0
    
    for filepath in test_files:
        name = Path(filepath).name
        short_name = name[:40]
        
        # Python
        result = detector.predict(filepath, n_segments=5)
        py_prob = result.get('ai_probability', -1) if 'error' not in result else -1
        
        # C#
        cs_prob = cs_results.get(name, -1)
        
        diff = abs(py_prob - cs_prob) if py_prob >= 0 and cs_prob >= 0 else -1
        max_diff = max(max_diff, diff) if diff >= 0 else max_diff
        py_class = "AI" if py_prob > 0.5 else "Real"
        cs_class = "AI" if cs_prob > 0.5 else "Real"
        match = "✓" if py_class == cs_class else "✗ DIFF"
        
        print(f"{short_name:<45} {py_prob*100:>9.1f}% {cs_prob*100:>9.1f}% {diff*100:>9.2f}% {match:>8}")
        
        if py_class != cs_class:
            disagreements.append({
                'file': filepath,
                'python': py_prob,
                'csharp': cs_prob
            })
    
    print("="*90)
    print(f"\nDisagreements: {len(disagreements)}/{len(test_files)}")
    print(f"Max difference: {max_diff*100:.3f}%")
    
    if disagreements:
        print("\nFiles with different classifications:")
        for d in disagreements:
            print(f"  {Path(d['file']).name}")
            print(f"    Python: {d['python']*100:.1f}%, C#: {d['csharp']*100:.1f}%")

if __name__ == "__main__":
    main()
