"""
Export trained model to ONNX format for C# inference.

The logistic regression model is wrapped in a simple neural network
for ONNX export, which makes it easy to load in C# with ONNX Runtime.

Usage:
    python export_onnx.py --model ./models --output ./models/ai_music_detector.onnx
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class LogisticRegressionModel(nn.Module):
    """
    PyTorch wrapper for logistic regression weights.
    
    This model performs: sigmoid(input @ weights.T + bias)
    """
    
    def __init__(self, weights: np.ndarray, bias: np.ndarray):
        super().__init__()
        
        # weights shape: (1, n_features)
        # bias shape: (1,)
        self.linear = nn.Linear(weights.shape[1], 1)
        
        # Load weights
        with torch.no_grad():
            self.linear.weight.copy_(torch.from_numpy(weights))
            self.linear.bias.copy_(torch.from_numpy(bias))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Fakeprint features of shape (batch_size, n_features)
            
        Returns:
            AI probability of shape (batch_size, 1)
        """
        logits = self.linear(x)
        probability = torch.sigmoid(logits)
        return probability


class FakeprintModel(nn.Module):
    """
    Complete model including fakeprint normalization and classification.
    
    This can optionally include preprocessing steps if needed,
    but for now we keep it simple as preprocessing is done in C#.
    """
    
    def __init__(self, weights: np.ndarray, bias: np.ndarray):
        super().__init__()
        self.classifier = LogisticRegressionModel(weights, bias)
    
    def forward(self, fakeprint: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            fakeprint: Normalized fakeprint of shape (batch_size, n_features)
            
        Returns:
            AI probability of shape (batch_size, 1)
        """
        return self.classifier(fakeprint)


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_size: int,
    opset_version: int = 14
):
    """Export PyTorch model to ONNX format."""
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, input_size, dtype=torch.float32)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["fakeprint"],
        output_names=["ai_probability"],
        dynamic_axes={
            "fakeprint": {0: "batch_size"},
            "ai_probability": {0: "batch_size"}
        }
    )
    
    print(f"Exported ONNX model: {output_path}")
    
    # Verify the model
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification: PASSED")
    
    # Print model info
    print(f"\nModel Info:")
    print(f"  Input: fakeprint ({input_size} features)")
    print(f"  Output: ai_probability (0.0 = Real, 1.0 = AI-Generated)")
    print(f"  Opset version: {opset_version}")
    
    return onnx_model


def test_onnx_inference(onnx_path: Path, input_size: int):
    """Test ONNX model with ONNX Runtime."""
    import onnxruntime as ort
    
    print("\nTesting ONNX inference...")
    
    # Create session
    session = ort.InferenceSession(str(onnx_path))
    
    # Test with random input
    test_input = np.random.randn(1, input_size).astype(np.float32)
    
    # Run inference
    outputs = session.run(None, {"fakeprint": test_input})
    probability = outputs[0]
    
    print(f"  Test input shape: {test_input.shape}")
    print(f"  Output shape: {probability.shape}")
    print(f"  Test probability: {probability[0, 0]:.4f}")
    print("ONNX Runtime test: PASSED")


def create_model_metadata(output_dir: Path, config: dict, input_size: int):
    """Create metadata file for the ONNX model."""
    
    metadata = {
        "model_name": "AI Music Detector",
        "version": "1.0.0",
        "description": "Detects AI-generated music using fakeprint analysis",
        "paper": "A Fourier Explanation of AI-Music Artifacts (ISMIR 2025)",
        "input": {
            "name": "fakeprint",
            "type": "float32",
            "shape": ["batch_size", input_size],
            "description": "Normalized fakeprint features"
        },
        "output": {
            "name": "ai_probability",
            "type": "float32",
            "shape": ["batch_size", 1],
            "description": "Probability of AI-generated content (0=Real, 1=AI)",
            "threshold": 0.5
        },
        "preprocessing": {
            "sample_rate": config.get("sample_rate", 16000),
            "n_fft": config.get("n_fft", 8192),
            "freq_min": config.get("freq_min", 1000),
            "freq_max": config.get("freq_max", 8000),
            "hull_area": 10
        }
    }
    
    metadata_path = output_dir / "model_metadata.yaml"
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    print(f"Created metadata: {metadata_path}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Export trained model to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model directory containing weights.npz"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX file path"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="ONNX opset version"
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    # Paths
    model_dir = Path(args.model) if args.model else Path(config["paths"]["model_dir"])
    weights_path = model_dir / "weights.npz"
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    
    # Load weights
    print(f"Loading weights from: {weights_path}")
    data = np.load(weights_path)
    weights = data["weights"]
    bias = data["bias"]
    
    print(f"  Weights shape: {weights.shape}")
    print(f"  Bias: {bias}")
    
    input_size = weights.shape[1]
    
    # Create PyTorch model
    model = FakeprintModel(weights, bias)
    
    # Test PyTorch model
    print("\nTesting PyTorch model...")
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, input_size)
        test_output = model(test_input)
        print(f"  Test output: {test_output.item():.4f}")
    
    # Export to ONNX
    opset_version = args.opset or config["onnx"]["opset_version"]
    onnx_filename = config["onnx"]["model_name"]
    
    output_path = Path(args.output) if args.output else model_dir / onnx_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    export_to_onnx(model, output_path, input_size, opset_version)
    
    # Test ONNX inference
    test_onnx_inference(output_path, input_size)
    
    # Create metadata
    model_config_path = model_dir / "model_config.yaml"
    if model_config_path.exists():
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
        feature_config = model_config.get("feature_config", {})
    else:
        feature_config = {}
    
    create_model_metadata(output_path.parent, feature_config, input_size)
    
    print("\n" + "="*60)
    print("ONNX Export Complete!")
    print("="*60)
    print(f"\nModel file: {output_path}")
    print(f"Model size: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"\nThis model can now be used with the C# inference library.")


if __name__ == "__main__":
    main()
