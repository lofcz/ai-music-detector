"""Debug a single file - compare with C# intermediate values."""
import sys
import json
import inference_cnn

model_path = r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\cs\AiMusicDetector\Models\cnn_detector.onnx"
audio_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v4.5-all\01326459-437a-4853-960f-fcbbc22094a7.mp3"

d = inference_cnn.CepstrumCNNDetector(model_path=model_path, device="cpu")
result = d.predict_with_debug(audio_path, n_segments=5)

print("\n" + "=" * 60)
print(f"Python Debug: {result.get('file', audio_path)}")
print("=" * 60)

print(f"\nAudio samples: {result['audio_samples']:,}")
print(f"Audio head: {result['audio_head']}")

print(f"\nSegments ({len(result['segments'])}):")
for seg in result['segments']:
    print(f"  Seg {seg['index']}: start={seg['start_sample']:,}, shape={seg['cepstrum_shape']}, "
          f"mean={seg['cepstrum_mean']:.4f}, prob={seg['probability']*100:.2f}%")

print(f"\nSegment probs: {[f'{p*100:.2f}%' for p in result['segment_probabilities']]}")
print(f"Final (median): {result['final_probability']*100:.2f}%")
print(f"Classification: {result['classification']}")

# Save JSON
json_path = audio_path.replace('.mp3', '.debug.py.json')
with open(json_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved: {json_path}")
