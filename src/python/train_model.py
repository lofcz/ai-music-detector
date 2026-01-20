"""
Train logistic regression model for AI Music Detection.

Uses extracted fakeprints to train a simple but effective classifier.

Based on: "A Fourier Explanation of AI-Music Artifacts" (ISMIR 2025)

Usage:
    python train_model.py --real ./output/fma_fakeprints.npy --fake ./output/sonics_fakeprints.npy
"""

import argparse
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import yaml
import pickle


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_fakeprints(file_path: str) -> tuple:
    """Load fakeprints from .npy file."""
    data = np.load(file_path, allow_pickle=True).item()
    fakeprints = data["fakeprints"]
    label = data["label"]
    config = data["config"]
    
    print(f"Loaded {len(fakeprints)} {label} samples from {file_path}")
    print(f"  Config: {config}")
    
    return fakeprints, label, config


def prepare_dataset(real_fakeprints: dict, fake_fakeprints: dict, test_split: float = 0.2, seed: int = 42):
    """Prepare train/test dataset from fakeprints."""
    
    # Convert to arrays
    real_keys = list(real_fakeprints.keys())
    fake_keys = list(fake_fakeprints.keys())
    
    X_real = np.stack([real_fakeprints[k] for k in real_keys], axis=0)
    X_fake = np.stack([fake_fakeprints[k] for k in fake_keys], axis=0)
    
    # Create labels (0 = real, 1 = fake/AI-generated)
    y_real = np.zeros(len(X_real), dtype=np.float32)
    y_fake = np.ones(len(X_fake), dtype=np.float32)
    
    # Combine
    X = np.concatenate([X_real, X_fake], axis=0)
    y = np.concatenate([y_real, y_fake], axis=0)
    
    print(f"\nDataset: {len(X)} samples ({len(X_real)} real, {len(X_fake)} fake)")
    print(f"Feature dimension: {X.shape[1]}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=seed, stratify=y
    )
    
    print(f"Train: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def train_classifier(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train logistic regression classifier."""
    print("\nTraining logistic regression classifier...")
    
    clf = LogisticRegression(
        class_weight="balanced",  # Handle class imbalance
        max_iter=1000,
        solver="lbfgs",
        random_state=42
    )
    
    clf.fit(X_train, y_train)
    
    print("Training complete!")
    print(f"  Coefficients shape: {clf.coef_.shape}")
    print(f"  Intercept: {clf.intercept_[0]:.4f}")
    
    return clf


def evaluate_classifier(clf: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate classifier on test set."""
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives (Real correctly identified):  {tn}")
    print(f"  False Positives (Real incorrectly as Fake): {fp}")
    print(f"  False Negatives (Fake incorrectly as Real): {fn}")
    print(f"  True Positives (Fake correctly identified):  {tp}")
    
    # Per-class metrics
    real_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    fake_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\nPer-Class Accuracy:")
    print(f"  Real music detection:     {real_acc:.4f} ({real_acc*100:.2f}%)")
    print(f"  AI-generated detection:   {fake_acc:.4f} ({fake_acc*100:.2f}%)")
    print(f"  False positive rate:      {(1-real_acc)*100:.2f}%")
    print(f"  False negative rate:      {(1-fake_acc)*100:.2f}%")
    
    # Full classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real", "AI-Generated"]))
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "real_accuracy": real_acc,
        "fake_accuracy": fake_acc
    }


def save_model(clf: LogisticRegression, output_path: Path, config: dict):
    """Save trained model and weights."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save sklearn model
    model_path = output_path / "classifier.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Saved sklearn model: {model_path}")
    
    # Save weights separately for ONNX export
    weights_path = output_path / "weights.npz"
    np.savez(
        weights_path,
        weights=clf.coef_.astype(np.float32),
        bias=clf.intercept_.astype(np.float32),
        classes=clf.classes_
    )
    print(f"Saved weights: {weights_path}")
    
    # Save config
    config_path = output_path / "model_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Saved config: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Train AI Music Detector model")
    parser.add_argument(
        "--real",
        type=str,
        required=True,
        help="Path to real music fakeprints .npy file"
    )
    parser.add_argument(
        "--fake",
        type=str,
        required=True,
        help="Path to AI-generated music fakeprints .npy file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for model files"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=None,
        help="Test split ratio (default from config)"
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    training_config = config["training"]
    
    # Load fakeprints
    real_fakeprints, _, real_config = load_fakeprints(args.real)
    fake_fakeprints, _, fake_config = load_fakeprints(args.fake)
    
    # Verify configs match
    real_bins = real_config.get("output_bins", len(list(real_fakeprints.values())[0]))
    fake_bins = fake_config.get("output_bins", len(list(fake_fakeprints.values())[0]))
    if real_bins != fake_bins:
        raise ValueError(f"Fakeprint dimensions don't match! Real: {real_bins}, Fake: {fake_bins}")
    
    # Prepare dataset
    test_split = args.test_split or training_config["test_split"]
    seed = training_config["random_seed"]
    
    X_train, X_test, y_train, y_test = prepare_dataset(
        real_fakeprints, 
        fake_fakeprints,
        test_split=test_split,
        seed=seed
    )
    
    # Train classifier
    clf = train_classifier(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_classifier(clf, X_test, y_test)
    
    # Save model
    output_dir = Path(args.output) if args.output else Path(config["paths"]["model_dir"])
    
    # Count test set composition
    n_test_real = int(np.sum(y_test == 0))
    n_test_fake = int(np.sum(y_test == 1))
    
    model_config = {
        "feature_config": real_config,
        "training_metrics": {
            **{k: float(v) if isinstance(v, (np.floating, float)) else v 
               for k, v in metrics.items() if k != "confusion_matrix"},
            "n_test": len(y_test),
            "n_test_real": n_test_real,
            "n_test_fake": n_test_fake,
            "fpr": float(1 - metrics["real_accuracy"]),
            "fnr": float(1 - metrics["fake_accuracy"])
        },
        "n_real_samples": len(real_fakeprints),
        "n_fake_samples": len(fake_fakeprints)
    }
    
    save_model(clf, output_dir, model_config)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nNext step: Run export_onnx.py to export model for C# inference")


if __name__ == "__main__":
    main()
