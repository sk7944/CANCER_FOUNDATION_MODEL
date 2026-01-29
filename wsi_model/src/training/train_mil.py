"""
MIL Model Training Script
=========================

Trains ABMIL or TransMIL models on pre-extracted WSI features
for 3-year survival prediction.

Usage:
    python train_mil.py --config config.yaml
    python train_mil.py --features_dir ./data/processed/features --labels ./labels.csv
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from wsi_model.src.models import ABMIL, TransMIL, MILModelConfig
from wsi_model.src.data import WSIFeatureDataset, WSIDataModule, collate_features

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_type: str = 'abmil'  # 'abmil' or 'transmil'
    feature_dim: int = 768
    hidden_dim: int = 256
    num_classes: int = 1
    dropout: float = 0.25

    # Training
    epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5

    # Data
    max_patches: Optional[int] = 10000  # Limit patches per slide
    num_workers: int = 4

    # Early stopping
    early_stopping_patience: int = 15
    min_delta: float = 0.001

    # Mixed precision
    use_amp: bool = True

    # Logging
    log_interval: int = 10
    save_interval: int = 5


class MILTrainer:
    """
    Trainer for MIL models.

    Args:
        model: MIL model (ABMIL or TransMIL)
        config: TrainingConfig
        device: Device to use
        output_dir: Directory to save checkpoints and logs
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: torch.device,
        output_dir: str,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Loss function (binary cross entropy with logits)
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.01,
        )

        # Mixed precision
        self.scaler = GradScaler() if config.use_amp else None

        # Tracking
        self.best_val_auc = 0.0
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_acc': [],
            'learning_rate': [],
        }

        # Save config
        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Training", leave=False)
        for batch in pbar:
            features = batch['features'].to(self.device)
            labels = batch['labels'].float().to(self.device)
            mask = batch['mask'].to(self.device)

            self.optimizer.zero_grad()

            if self.config.use_amp:
                with autocast():
                    output = self.model(features, mask)
                    logits = output['logits'].squeeze(-1)
                    loss = self.criterion(logits, labels)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(features, mask)
                logits = output['logits'].squeeze(-1)
                loss = self.criterion(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': loss.item()})

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation/test set."""
        self.model.eval()
        total_loss = 0.0
        all_probs = []
        all_labels = []
        all_slide_ids = []

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            features = batch['features'].to(self.device)
            labels = batch['labels'].float().to(self.device)
            mask = batch['mask'].to(self.device)

            output = self.model(features, mask)
            logits = output['logits'].squeeze(-1)
            loss = self.criterion(logits, labels)

            probs = torch.sigmoid(logits)

            total_loss += loss.item()
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_slide_ids.extend(batch['slide_ids'])

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_preds = (all_probs > 0.5).astype(int)

        # Calculate metrics
        metrics = {
            'loss': total_loss / len(dataloader),
            'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
        }

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Full training loop.

        Returns:
            Training history and best metrics
        """
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Model: {self.config.model_type}")
        logger.info(f"Device: {self.device}")

        for epoch in range(1, self.config.epochs + 1):
            logger.info(f"\nEpoch {epoch}/{self.config.epochs}")

            # Train
            train_loss = self.train_epoch(train_loader)

            # Evaluate
            val_metrics = self.evaluate(val_loader)

            # Update scheduler
            self.scheduler.step()

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])

            # Log
            logger.info(
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val AUC: {val_metrics['auc']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

            # Check for improvement
            if val_metrics['auc'] > self.best_val_auc + self.config.min_delta:
                self.best_val_auc = val_metrics['auc']
                self.epochs_without_improvement = 0

                # Save best model
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                logger.info(f"New best model! Val AUC: {self.best_val_auc:.4f}")
            else:
                self.epochs_without_improvement += 1

            # Save periodic checkpoint
            if epoch % self.config.save_interval == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=False)

            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Save final history
        history_path = self.output_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        return {
            'history': self.history,
            'best_val_auc': self.best_val_auc,
            'final_epoch': epoch,
        }

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': {k: v.cpu() for k, v in self.model.state_dict().items()},
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': asdict(self.config),
        }

        if is_best:
            path = self.output_dir / 'best_model.pth'
        else:
            path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'

        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


def train_mil_model(
    features_dir: str,
    labels_path: str,
    output_dir: str,
    splits_path: Optional[str] = None,
    config: Optional[TrainingConfig] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train MIL model end-to-end.

    Args:
        features_dir: Directory with HDF5 feature files
        labels_path: Path to labels CSV
        output_dir: Directory to save outputs
        splits_path: Optional path to pre-defined splits
        config: Training configuration
        device: Device to use

    Returns:
        Training results
    """
    if config is None:
        config = TrainingConfig()

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(output_dir) / f"mil_training_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Create data module
    data_module = WSIDataModule(
        features_dir=features_dir,
        labels_path=labels_path,
        splits_path=splits_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_patches=config.max_patches,
    )

    # Save splits
    data_module.save_splits(str(output_dir / 'splits.json'))

    # Create model
    model_config = MILModelConfig(
        feature_dim=config.feature_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        dropout=config.dropout,
    )

    if config.model_type.lower() == 'abmil':
        model = ABMIL(model_config)
    elif config.model_type.lower() == 'transmil':
        model = TransMIL(model_config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {config.model_type}, Parameters: {num_params:,}")

    # Create trainer
    trainer = MILTrainer(
        model=model,
        config=config,
        device=device,
        output_dir=str(output_dir),
    )

    # Train
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    results = trainer.train(train_loader, val_loader)

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    trainer.load_checkpoint(str(output_dir / 'best_model.pth'))
    test_loader = data_module.test_dataloader()
    test_metrics = trainer.evaluate(test_loader)

    logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Sensitivity: {test_metrics['sensitivity']:.4f}")
    logger.info(f"Test Specificity: {test_metrics['specificity']:.4f}")

    # Save test results
    results['test_metrics'] = test_metrics
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'best_val_auc': results['best_val_auc'],
            'final_epoch': results['final_epoch'],
            'test_metrics': test_metrics,
        }, f, indent=2)

    logger.info(f"\nTraining complete! Results saved to {output_dir}")

    return results


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description='Train MIL model for WSI classification')

    # Data arguments
    parser.add_argument('--features_dir', required=True, help='Directory with HDF5 feature files')
    parser.add_argument('--labels', required=True, help='Path to labels CSV')
    parser.add_argument('--splits', default=None, help='Path to splits JSON')
    parser.add_argument('--output_dir', default='./results', help='Output directory')

    # Model arguments
    parser.add_argument('--model', default='abmil', choices=['abmil', 'transmil'], help='Model type')
    parser.add_argument('--feature_dim', type=int, default=768, help='Feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.25, help='Dropout rate')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--max_patches', type=int, default=10000, help='Max patches per slide')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')

    # Other arguments
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader workers')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision')

    args = parser.parse_args()

    # Create config
    config = TrainingConfig(
        model_type=args.model,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_patches=args.max_patches,
        early_stopping_patience=args.patience,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
    )

    # Train
    train_mil_model(
        features_dir=args.features_dir,
        labels_path=args.labels,
        output_dir=args.output_dir,
        splits_path=args.splits,
        config=config,
        device=args.device,
    )


if __name__ == "__main__":
    main()
