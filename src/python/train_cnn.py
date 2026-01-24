"""
Train CNN-based AI Music Detector on CQT-Cepstrum features.

The cepstrum features provide pitch-shift invariance because:
- CQT uses log-frequency (pitch shift = translation in log-freq)
- DCT captures patterns of differences (translation-invariant)

Each segment is an individual training sample. At inference,
max-pooling across segments catches artifacts anywhere in the song.

Usage:
    python extract_cqt_features.py --input ./data/fma/fma_medium --output ./output/fma_cqt.pt --label real
    python extract_cqt_features.py --input ./data/sonics/fake_songs --output ./output/sonics_cqt.pt --label fake
    python train_cnn.py --real ./output/fma_cqt.pt --fake ./output/sonics_cqt.pt
"""

import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml

from models.cnn_detector import get_model, count_parameters


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CepstrumDataset(Dataset):
    """
    Dataset for CQT-cepstrum features.
    Each sample is a [n_coeffs, n_frames] cepstrum spectrogram.
    
    Uses contiguous tensor storage for efficient multiprocessing (no dict copying).
    """
    
    def __init__(
        self,
        features: torch.Tensor,  # [N, n_coeffs, n_frames] contiguous
        labels: torch.Tensor,    # [N] contiguous
        augment: bool = True,
        augment_prob: float = 0.5
    ):
        self.features = features  # Contiguous tensor - efficient for workers
        self.labels = labels
        self.augment = augment
        self.augment_prob = augment_prob
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def apply_augmentation(self, cepstrum: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to cepstrum spectrogram.
        
        Since cepstrum is already pitch-shift invariant, we focus on:
        - Time masking (SpecAugment-style)
        - Coefficient masking
        - Small additive noise
        """
        if random.random() > self.augment_prob:
            return cepstrum
        
        cepstrum = cepstrum.clone()
        n_coeffs, n_frames = cepstrum.shape
        
        # Time masking
        if random.random() < 0.5:
            mask_size = random.randint(1, n_frames // 8)
            mask_start = random.randint(0, n_frames - mask_size)
            cepstrum[:, mask_start:mask_start + mask_size] = 0
        
        # Coefficient masking (don't mask too many - cepstrum is compact)
        if random.random() < 0.3:
            mask_size = random.randint(1, n_coeffs // 6)
            mask_start = random.randint(0, n_coeffs - mask_size)
            cepstrum[mask_start:mask_start + mask_size, :] = 0
        
        # Small additive noise
        if random.random() < 0.3:
            noise_level = random.uniform(0.01, 0.05)
            noise = torch.randn_like(cepstrum) * noise_level
            cepstrum = cepstrum + noise
        
        return cepstrum
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cepstrum = self.features[idx]  # Direct tensor indexing - O(1)
        label = self.labels[idx]
        
        if self.augment:
            cepstrum = self.apply_augmentation(cepstrum)
        
        # Add channel dimension [1, n_coeffs, n_frames]
        cepstrum = cepstrum.unsqueeze(0)
        
        return cepstrum, label


def load_features(
    real_path: Path,
    fake_path: Path,
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[Dataset, Dataset, dict]:
    """
    Load pre-computed cepstrum features.
    Split by FILE to prevent data leakage (segments from same file stay together).
    
    Returns contiguous tensor datasets for efficient multiprocessing.
    """
    
    print(f"Loading real features from {real_path}...")
    real_data = torch.load(real_path, weights_only=False)
    real_features = real_data["features"]
    real_config = real_data["config"]
    print(f"  {len(real_features)} segments, shape: {real_config['feature_shape']}")
    
    print(f"Loading fake features from {fake_path}...")
    fake_data = torch.load(fake_path, weights_only=False)
    fake_features = fake_data["features"]
    fake_config = fake_data["config"]
    print(f"  {len(fake_features)} segments, shape: {fake_config['feature_shape']}")
    
    # Verify configs match
    for key in ['sample_rate', 'fmin', 'n_bins', 'n_coeffs']:
        if real_config.get(key) != fake_config.get(key):
            raise ValueError(f"Config mismatch for {key}: real={real_config.get(key)}, fake={fake_config.get(key)}")
    
    # Group segments by file
    def get_filename(sample_id: str) -> str:
        if "__seg" in sample_id:
            return sample_id.rsplit("__seg", 1)[0]
        return sample_id
    
    real_files: Dict[str, list] = {}
    for sample_id in real_features.keys():
        filename = get_filename(sample_id)
        real_files.setdefault(filename, []).append(sample_id)
    
    fake_files: Dict[str, list] = {}
    for sample_id in fake_features.keys():
        filename = get_filename(sample_id)
        fake_files.setdefault(filename, []).append(sample_id)
    
    print(f"\nReal files: {len(real_files)}, Fake files: {len(fake_files)}")
    
    # Split by file
    rng = random.Random(seed)
    
    real_file_list = list(real_files.keys())
    fake_file_list = list(fake_files.keys())
    rng.shuffle(real_file_list)
    rng.shuffle(fake_file_list)
    
    real_split = int(len(real_file_list) * (1 - val_split))
    fake_split = int(len(fake_file_list) * (1 - val_split))
    
    train_real_files = set(real_file_list[:real_split])
    val_real_files = set(real_file_list[real_split:])
    train_fake_files = set(fake_file_list[:fake_split])
    val_fake_files = set(fake_file_list[fake_split:])
    
    # Collect features into lists (for stacking into contiguous tensors)
    train_features_list = []
    train_labels_list = []
    val_features_list = []
    val_labels_list = []
    
    for sample_id, feat in real_features.items():
        filename = get_filename(sample_id)
        if filename in train_real_files:
            train_features_list.append(feat)
            train_labels_list.append(0.0)  # Real
        else:
            val_features_list.append(feat)
            val_labels_list.append(0.0)
    
    for sample_id, feat in fake_features.items():
        filename = get_filename(sample_id)
        if filename in train_fake_files:
            train_features_list.append(feat)
            train_labels_list.append(1.0)  # Fake/AI
        else:
            val_features_list.append(feat)
            val_labels_list.append(1.0)
    
    print(f"\nTrain: {len(train_features_list)} segments ({len(train_real_files)} real + {len(train_fake_files)} fake files)")
    print(f"Val: {len(val_features_list)} segments ({len(val_real_files)} real + {len(val_fake_files)} fake files)")
    
    # Stack into contiguous tensors (critical for efficient multiprocessing!)
    print("Stacking into contiguous tensors...")
    train_features_tensor = torch.stack(train_features_list).contiguous()
    train_labels_tensor = torch.tensor(train_labels_list, dtype=torch.float32).contiguous()
    val_features_tensor = torch.stack(val_features_list).contiguous()
    val_labels_tensor = torch.tensor(val_labels_list, dtype=torch.float32).contiguous()
    
    # Free the lists
    del train_features_list, train_labels_list, val_features_list, val_labels_list
    del real_features, fake_features
    
    print(f"Train tensor: {train_features_tensor.shape}, Val tensor: {val_features_tensor.shape}")
    
    # Get augment_prob from config
    config = load_config()
    augment_prob = config.get("cnn", {}).get("augment_prob", 0.5)
    
    train_dataset = CepstrumDataset(train_features_tensor, train_labels_tensor, augment=True, augment_prob=augment_prob)
    val_dataset = CepstrumDataset(val_features_tensor, val_labels_tensor, augment=False)
    
    return train_dataset, val_dataset, real_config


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for data, target in pbar:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                output = model(data).squeeze(-1)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data).squeeze(-1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pred = (torch.sigmoid(output) > 0.5).float()
        correct += (pred == target).sum().item()
        total += target.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.1f}%'})
    
    return total_loss / len(train_loader), correct / total


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, dict]:
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    
    for data, target in tqdm(val_loader, desc="Validating", leave=False):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        output = model(data).squeeze(-1)
        loss = criterion(output, target)
        
        total_loss += loss.item()
        pred = (torch.sigmoid(output) > 0.5).float()
        correct += (pred == target).sum().item()
        total += target.size(0)
        
        tp += ((pred == 1) & (target == 1)).sum().item()
        tn += ((pred == 0) & (target == 0)).sum().item()
        fp += ((pred == 1) & (target == 0)).sum().item()
        fn += ((pred == 0) & (target == 1)).sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return total_loss / len(val_loader), correct / total, {
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def save_checkpoint(model, optimizer, epoch, val_acc, path, config):
    """Save training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'config': config
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train CNN on CQT-Cepstrum features")
    parser.add_argument("--real", type=str, required=True, help="Path to real features (.pt)")
    parser.add_argument("--fake", type=str, required=True, help="Path to fake features (.pt)")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--model", type=str, default="standard", choices=["standard", "large"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    config = load_config()
    cnn_cfg = config.get("cnn", {})
    cepstrum_cfg = config.get("cepstrum", {})
    
    set_seed(config["training"]["random_seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load features
    real_path = Path(args.real)
    fake_path = Path(args.fake)
    
    if not real_path.suffix == '.pt':
        print("ERROR: Please provide pre-computed .pt files.")
        return
    
    train_dataset, val_dataset, feature_config = load_features(
        real_path, fake_path,
        val_split=config["training"]["test_split"],
        seed=config["training"]["random_seed"]
    )
    
    n_coeffs = feature_config['n_coeffs']
    print(f"\nCepstrum coefficients: {n_coeffs}")
    
    batch_size = args.batch_size or cnn_cfg.get("batch_size", 64)
    
    # Use persistent_workers to avoid worker restart overhead between epochs
    num_workers = 4 if torch.cuda.is_available() else 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    
    print(f"Batch size: {batch_size}")
    
    # Create model
    model = get_model(
        args.model,
        n_coeffs=n_coeffs,
        base_channels=cnn_cfg.get("base_channels", 32),
        dropout=cnn_cfg.get("dropout", 0.3)
    ).to(device)
    
    print(f"Model: {args.model}, Parameters: {count_parameters(model):,}")
    
    # Training setup
    epochs = args.epochs or cnn_cfg.get("epochs", 50)
    lr = args.lr or cnn_cfg.get("learning_rate", 1e-4)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    start_epoch = 0
    best_val_acc = 0
    
    if args.resume:
        ckpt = torch.load(args.resume, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt['val_acc']
        print(f"Resumed from epoch {start_epoch}, best acc: {best_val_acc:.4f}")
    
    output_dir = Path(args.output) if args.output else Path(config["paths"]["model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTraining for {epochs} epochs...")
    print("=" * 60)
    
    patience = cnn_cfg.get("early_stopping_patience", 10)
    patience_counter = 0
    
    # Save model config for inference
    model_config = {
        'model_type': args.model,
        'n_coeffs': n_coeffs,
        'feature_config': feature_config,
        **config
    }
    
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc, metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_acc)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_acc, output_dir / "cnn_best.pt", model_config)
            print(f"  * New best! Val Acc: {val_acc:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping after {patience} epochs without improvement")
            break
        
        save_checkpoint(model, optimizer, epoch, val_acc, output_dir / "cnn_latest.pt", model_config)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_dir / 'cnn_best.pt'}")
    print("\nNext: python export_onnx_cnn.py")


if __name__ == "__main__":
    main()
