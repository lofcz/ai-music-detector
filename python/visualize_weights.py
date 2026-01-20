"""
Visualize learned model weights to see which frequencies detect AI artifacts.

Creates plots similar to Figure 5 in the paper, showing the frequency-domain
patterns the model has learned to identify AI-generated music.

Usage:
    python visualize_weights.py
    python visualize_weights.py --output weights_plot.png
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yaml


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Visualize model weights")
    parser.add_argument("--model", type=str, default=None, help="Path to weights.npz")
    parser.add_argument("--output", type=str, default=None, help="Save plot to file")
    parser.add_argument("--title", type=str, default="AI Music Detector - Learned Weights", 
                        help="Plot title")
    args = parser.parse_args()
    
    config = load_config()
    audio_cfg = config["audio"]
    fp_cfg = config["fakeprint"]
    
    # Load weights
    model_dir = Path(args.model) if args.model else Path(config["paths"]["model_dir"])
    if model_dir.is_dir():
        weights_path = model_dir / "weights.npz"
    else:
        weights_path = model_dir
    
    if not weights_path.exists():
        print(f"Weights not found: {weights_path}")
        print("Train a model first with train_model.py")
        return
    
    data = np.load(weights_path)
    weights = data["weights"].flatten()  # Shape: (n_features,)
    bias = data["bias"][0]
    
    print(f"Loaded weights from: {weights_path}")
    print(f"  Feature dimension: {len(weights)}")
    print(f"  Bias: {bias:.4f}")
    print(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    
    # Compute frequency axis
    sample_rate = audio_cfg["sample_rate"]
    n_fft = audio_cfg["n_fft"]
    freq_min = fp_cfg["freq_min"]
    freq_max = fp_cfg["freq_max"]
    
    # Full frequency bins
    freq_bins = np.linspace(0, sample_rate / 2, num=(n_fft // 2) + 1)
    freq_mask = (freq_bins >= freq_min) & (freq_bins <= freq_max)
    freq_range = freq_bins[freq_mask]
    
    # Ensure dimensions match
    if len(freq_range) != len(weights):
        print(f"Warning: freq_range ({len(freq_range)}) != weights ({len(weights)})")
        # Interpolate frequency axis to match weights
        freq_range = np.linspace(freq_min, freq_max, len(weights))
    
    # Convert to kHz for display
    freq_khz = freq_range / 1000
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Full weights
    ax1 = axes[0]
    ax1.plot(freq_khz, weights, color='#1f77b4', linewidth=0.5, alpha=0.8)
    ax1.fill_between(freq_khz, 0, weights, alpha=0.3, color='#1f77b4')
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Frequency (kHz)', fontsize=12)
    ax1.set_ylabel('Weight', fontsize=12)
    ax1.set_title(args.title, fontsize=14, fontweight='bold')
    ax1.set_xlim(freq_khz[0], freq_khz[-1])
    ax1.grid(True, alpha=0.3)
    
    # Annotate top peaks
    n_peaks = 10
    peak_indices = np.argsort(np.abs(weights))[-n_peaks:]
    for idx in peak_indices:
        if abs(weights[idx]) > np.std(weights) * 2:
            ax1.annotate(f'{freq_khz[idx]:.2f}kHz', 
                        xy=(freq_khz[idx], weights[idx]),
                        fontsize=8, alpha=0.7)
    
    # Plot 2: Positive weights only (what triggers "AI detected")
    ax2 = axes[1]
    positive_weights = np.maximum(weights, 0)
    ax2.bar(freq_khz, positive_weights, width=(freq_khz[1]-freq_khz[0]), 
            color='#d62728', alpha=0.7, edgecolor='none')
    ax2.set_xlabel('Frequency (kHz)', fontsize=12)
    ax2.set_ylabel('Weight (positive only)', fontsize=12)
    ax2.set_title('Frequencies that indicate AI-generated audio', fontsize=12)
    ax2.set_xlim(freq_khz[0], freq_khz[-1])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if args.output:
        output_path = Path(args.output)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to: {output_path}")
    else:
        plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("Weight Statistics")
    print("="*60)
    
    # Find significant peaks
    threshold = np.std(weights) * 2
    significant_positive = np.where(weights > threshold)[0]
    significant_negative = np.where(weights < -threshold)[0]
    
    print(f"\nSignificant positive peaks (indicate AI):")
    if len(significant_positive) > 0:
        sorted_pos = sorted(significant_positive, key=lambda i: weights[i], reverse=True)[:10]
        for idx in sorted_pos:
            print(f"  {freq_khz[idx]:.2f} kHz: {weights[idx]:.4f}")
    else:
        print("  None found")
    
    print(f"\nSignificant negative peaks (indicate Real):")
    if len(significant_negative) > 0:
        sorted_neg = sorted(significant_negative, key=lambda i: weights[i])[:10]
        for idx in sorted_neg:
            print(f"  {freq_khz[idx]:.2f} kHz: {weights[idx]:.4f}")
    else:
        print("  None found")
    
    # Weight energy distribution
    total_energy = np.sum(weights**2)
    positive_energy = np.sum(weights[weights > 0]**2)
    negative_energy = np.sum(weights[weights < 0]**2)
    
    print(f"\nWeight energy distribution:")
    print(f"  Positive weights: {positive_energy/total_energy*100:.1f}%")
    print(f"  Negative weights: {negative_energy/total_energy*100:.1f}%")


if __name__ == "__main__":
    main()
