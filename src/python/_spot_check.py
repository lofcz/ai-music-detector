"""Spot check: compare C# vs Python on files near decision boundary."""
import os
import sys
import json
import inference_cnn

model_path = r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\cs\AiMusicDetector\Models\cnn_detector.onnx"

# Get a list of files to check - look for files near 50%
# For now, just check a few known files
test_files = [
    r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v4.5-all\01326459-437a-4853-960f-fcbbc22094a7.mp3",
    r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v4.5-all\004be5b4-c36b-48a3-a0be-a1bdc0fb5405.mp3",
]

# Add files from command line if provided
if len(sys.argv) > 1:
    test_files = sys.argv[1:]

d = inference_cnn.CepstrumCNNDetector(model_path, "cpu")

print("Checking files near decision boundary...")
for f in test_files:
    if not os.path.exists(f):
        print(f"File not found: {f}")
        continue
    
    r = d.predict(f, 5)
    prob = r['ai_probability']
    is_ai = prob > 0.5
    
    print(f"\n{os.path.basename(f)}")
    print(f"  Python: {prob*100:.2f}% {'AI' if is_ai else 'Real'}")
