"""
Export trained CQT-Cepstrum CNN model to ONNX format.

Usage:
    python export_onnx_cnn.py
    python export_onnx_cnn.py --input models/cnn_best.pt --output models/cnn_detector.onnx
"""

import argparse
from pathlib import Path
import torch
import yaml

from models.cnn_detector import get_model


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def export_to_onnx(
    input_path: Path,
    output_path: Path,
    opset_version: int = 14
):
    """Export PyTorch model to ONNX."""
    
    print(f"Loading model from: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    model_config = checkpoint.get('config', {})
    
    # Get model parameters
    model_type = model_config.get('model_type', 'standard')
    n_coeffs = model_config.get('n_coeffs', 24)
    feature_config = model_config.get('feature_config', {})
    
    print(f"Model type: {model_type}")
    print(f"Cepstrum coefficients: {n_coeffs}")
    
    # Create model
    model = get_model(model_type, n_coeffs=n_coeffs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Create dummy input
    # Shape: [batch, 1, n_coeffs, n_frames]
    # n_frames depends on segment length: 10s * 16000Hz / 512 hop = ~313 frames
    n_frames = 313
    dummy_input = torch.randn(1, 1, n_coeffs, n_frames)
    
    print(f"\nExporting to ONNX...")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output path: {output_path}")
    
    # Export
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=['cepstrum'],
        output_names=['logit'],
        dynamic_axes={
            'cepstrum': {0: 'batch', 3: 'time'},
            'logit': {0: 'batch'}
        },
        opset_version=opset_version,
        do_constant_folding=True
    )
    
    print(f"\nONNX export successful!")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Verify
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("  ONNX model validation: PASSED")
    except ImportError:
        print("  (Install 'onnx' package to validate model)")
    except Exception as e:
        print(f"  ONNX validation warning: {e}")
    
    # Test inference
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(str(output_path), providers=['CPUExecutionProvider'])
        
        # Test with dummy input
        outputs = session.run(None, {'cepstrum': dummy_input.numpy()})
        
        # Compare with PyTorch
        with torch.no_grad():
            pt_output = model(dummy_input)
        
        diff = abs(outputs[0] - pt_output.numpy()).max()
        print(f"  PyTorch vs ONNX diff: {diff:.6f}")
        
        if diff < 1e-5:
            print("  Inference test: PASSED")
        else:
            print("  Inference test: WARNING - outputs differ slightly")
            
    except ImportError:
        print("  (Install 'onnxruntime' to test inference)")
    
    # Save metadata
    metadata = {
        'model_type': model_type,
        'n_coeffs': n_coeffs,
        'feature_config': feature_config,
        'val_accuracy': checkpoint.get('val_acc', None),
        'n_parameters': n_params,
        'opset_version': opset_version
    }
    
    metadata_path = output_path.with_suffix('.yaml')
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    print(f"\nMetadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Export CNN to ONNX")
    parser.add_argument("--input", type=str, default=None, help="Input .pt file")
    parser.add_argument("--output", type=str, default=None, help="Output .onnx file")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    args = parser.parse_args()
    
    config = load_config()
    
    input_path = Path(args.input) if args.input else Path(config["paths"]["model_dir"]) / "cnn_best.pt"
    output_path = Path(args.output) if args.output else Path(config["paths"]["model_dir"]) / "cnn_detector.onnx"
    
    if not input_path.exists():
        print(f"ERROR: Model not found: {input_path}")
        print("Train a model first with: python train_cnn.py --real ... --fake ...")
        return
    
    export_to_onnx(input_path, output_path, args.opset)
    
    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print(f"\nTo use the ONNX model:")
    print(f"  python inference_cnn.py --model {output_path}")


if __name__ == "__main__":
    main()
