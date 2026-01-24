"""Quick test for single file."""
import inference_cnn

model_path = r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\cs\AiMusicDetector\Models\cnn_detector.onnx"
audio_path = r"C:\Users\lordo\Documents\GitHub\ai-music-detector\src\scrape\downloads\v4.5-all\004be5b4-c36b-48a3-a0be-a1bdc0fb5405.mp3"

d = inference_cnn.CepstrumCNNDetector(model_path, "cpu")
r = d.predict(audio_path, 5)
print(f"AI probability: {r['ai_probability']*100:.1f}%")
