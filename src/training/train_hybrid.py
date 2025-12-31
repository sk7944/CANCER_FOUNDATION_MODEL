"""
Hybrid FC-NN + TabTransformer Training Script

3-Year Overall Survival Classification with Missing Modality Learning

Patient Distribution (8,577 = Cox ∪ Methylation):
- Cox=✅ Meth=✅ (둘 다):   4,151 patients → Clinical + Cox + Methylation
- Cox=✅ Meth=❌ (Cox만):    353 patients → Clinical + Cox + [ZERO]
- Cox=❌ Meth=✅ (Meth만): 4,073 patients → Clinical + [ZERO] + Methylation

Target: 3-year survival (0=alive, 1=death)
Loss: BCEWithLogitsLoss
Metric: AUC
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import json
import argparse
from pathlib import Path
import sys
import warnings
from datetime import datetime

# Project root
current_dir = Path(__file__).resolve().parent
project_root = current_dir
while project_root.name != 'CANCER_FOUNDATION_MODEL' and project_root.parent != project_root:
    project_root = project_root.parent
if project_root.name == 'CANCER_FOUNDATION_MODEL':
    sys.path.insert(0, str(project_root))

from src.models.hybrid_fc_tabtransformer import HybridMultiModalModel
from src.data.hybrid_dataset import create_dataloaders

warnings.filterwarnings('ignore')


def create_3year_labels(event_times: torch.Tensor, event_status: torch.Tensor) -> torch.Tensor:
    """
    Create 3-year survival labels

    Args:
        event_times: (batch,) survival times in days
        event_status: (batch,) event indicator (1=death, 0=censored)

    Returns:
        labels: (batch,) 0=survived 3 years, 1=died within 3 years
    """
    three_years_days = 3 * 365.25

    # Death within 3 years: label=1
    # Survived 3+ years: label=0
    # Censored before 3 years: exclude (return -1 to filter)

    labels = torch.zeros_like(event_times)

    for i in range(len(event_times)):
        if event_status[i] == 1:  # Event occurred
            if event_times[i] <= three_years_days:
                labels[i] = 1  # Died within 3 years
            else:
                labels[i] = 0  # Died after 3 years (survived 3 years)
        else:  # Censored
            if event_times[i] >= three_years_days:
                labels[i] = 0  # Censored after 3 years (assumed alive)
            else:
                labels[i] = -1  # Censored before 3 years (unknown, exclude)

    return labels


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch in progress_bar:
        clinical_cat = batch['clinical_cat'].to(device)
        cox_omics = batch['cox_omics'].to(device)
        methylation = batch['methylation'].to(device)
        cox_mask = batch['cox_mask'].squeeze(1).to(device)  # (batch,)
        meth_mask = batch['meth_mask'].squeeze(1).to(device)  # (batch,)
        event_time = batch['event_time'].squeeze(1).to(device)
        event_status = batch['event_status'].squeeze(1).to(device)

        # Create 3-year labels
        labels_3year = create_3year_labels(event_time, event_status)

        # Filter out unknown labels (-1)
        valid_mask = labels_3year != -1
        if valid_mask.sum() == 0:
            continue

        clinical_cat = clinical_cat[valid_mask]
        cox_omics = cox_omics[valid_mask]
        methylation = methylation[valid_mask]
        cox_mask = cox_mask[valid_mask]
        meth_mask = meth_mask[valid_mask]
        labels_3year = labels_3year[valid_mask]

        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(clinical_cat, cox_omics, methylation, cox_mask, meth_mask)
        logits = logits.squeeze()

        # Loss
        loss = criterion(logits, labels_3year)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        probs = torch.sigmoid(logits)
        all_preds.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels_3year.cpu().numpy())

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(train_loader)
    train_auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    train_acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])

    return avg_loss, train_auc, train_acc


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            clinical_cat = batch['clinical_cat'].to(device)
            cox_omics = batch['cox_omics'].to(device)
            methylation = batch['methylation'].to(device)
            cox_mask = batch['cox_mask'].squeeze(1).to(device)
            meth_mask = batch['meth_mask'].squeeze(1).to(device)
            event_time = batch['event_time'].squeeze(1).to(device)
            event_status = batch['event_status'].squeeze(1).to(device)

            # Create 3-year labels
            labels_3year = create_3year_labels(event_time, event_status)

            # Filter valid labels
            valid_mask = labels_3year != -1
            if valid_mask.sum() == 0:
                continue

            clinical_cat = clinical_cat[valid_mask]
            cox_omics = cox_omics[valid_mask]
            methylation = methylation[valid_mask]
            cox_mask = cox_mask[valid_mask]
            meth_mask = meth_mask[valid_mask]
            labels_3year = labels_3year[valid_mask]

            # Forward pass
            logits, _ = model(clinical_cat, cox_omics, methylation, cox_mask, meth_mask)
            logits = logits.squeeze()

            # Loss
            loss = criterion(logits, labels_3year)

            # Metrics
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels_3year.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    val_auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
    val_acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])

    return avg_loss, val_auc, val_acc


def train_hybrid_model(
    cox_table_path,
    meth_table_path,
    clinical_path,
    splits_path,
    output_dir,
    epochs=100,
    batch_size=32,
    lr=1e-4,
    device='cuda'
):
    """
    Train Hybrid FC-NN + TabTransformer

    Args:
        cox_table_path: Path to integrated_table_cox.parquet
        meth_table_path: Path to methylation_table.parquet
        clinical_path: Path to clinical data
        splits_path: Path to train_val_test_splits.json
        output_dir: Output directory for checkpoints
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        device: 'cuda' or 'cpu'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("HYBRID FC-NN + TABTRANSFORMER TRAINING")
    print("="*70)
    print(f"Cox table: {cox_table_path}")
    print(f"Methylation table: {meth_table_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"Device: {device}")
    print("="*70)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        cox_table_path=cox_table_path,
        meth_table_path=meth_table_path,
        clinical_path=clinical_path,
        splits_path=splits_path,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )

    # Get feature dimensions from first batch
    sample_batch = next(iter(train_loader))
    cox_input_dim = sample_batch['cox_omics'].shape[1]
    meth_input_dim = sample_batch['methylation'].shape[1]

    print(f"\nFeature dimensions:")
    print(f"  Cox omics: {cox_input_dim:,}")
    print(f"  Methylation: {meth_input_dim:,}")

    # Create model
    model = HybridMultiModalModel(
        clinical_categories=(10, 3, 8, 4, 5),
        cox_input_dim=cox_input_dim,
        cox_hidden_dims=(2048, 512, 256),
        meth_input_dim=meth_input_dim,
        meth_hidden_dims=(4096, 1024, 256),
        dim=128,
        depth=6,
        heads=8,
        attn_dropout=0.1,
        ff_dropout=0.1,
        encoder_dropout=0.3,
        dim_out=1
    ).to(device)

    # Print model info
    params = model.get_num_parameters()
    print(f"\nModel Parameters:")
    print(f"  Cox Encoder:    {params['cox_encoder']:>12,} ({params['cox_encoder_gb']:.2f} GB)")
    print(f"  Meth Encoder:   {params['meth_encoder']:>12,} ({params['meth_encoder_gb']:.2f} GB)")
    print(f"  TabTransformer: {params['tabtransformer']:>12,} ({params['tabtransformer_gb']:.2f} GB)")
    print(f"  {'─'*50}")
    print(f"  Total:          {params['total']:>12,} ({params['total_gb']:.2f} GB)")

    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7
    )

    # Training loop
    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    early_stop_patience = 15

    history = {
        'train_losses': [],
        'val_losses': [],
        'train_aucs': [],
        'val_aucs': [],
        'train_accs': [],
        'val_accs': [],
    }

    print(f"\nStarting training...")
    print("="*70)

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_auc, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Validate
        val_loss, val_auc, val_acc = validate(model, val_loader, criterion, device)

        # Scheduler
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} AUC: {train_auc:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} AUC: {val_auc:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e}")

        # Save history
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['train_aucs'].append(train_auc)
        history['val_aucs'].append(val_auc)
        history['train_accs'].append(train_acc)
        history['val_accs'].append(val_acc)

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
                'history': history
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"  ✅ New best model saved! Val AUC: {val_auc:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n⏸️ Early stopping triggered after {epoch} epochs")
            print(f"   Best Val AUC: {best_val_auc:.4f} at epoch {best_epoch}")
            break

    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Best Val AUC: {best_val_auc:.4f} at epoch {best_epoch}")

    # Save history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Test evaluation
    print(f"\nEvaluating on test set...")
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_auc, test_acc = validate(model, test_loader, criterion, device)

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  AUC:  {test_auc:.4f}")
    print(f"  Acc:  {test_acc:.4f}")
    print("="*70)

    # Save test results
    test_results = {
        'test_loss': test_loss,
        'test_auc': test_auc,
        'test_acc': test_acc,
        'best_val_auc': best_val_auc,
        'best_epoch': best_epoch
    }

    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)

    return model, history, test_results


def main():
    parser = argparse.ArgumentParser(description='Train Hybrid FC-NN + TabTransformer')
    parser.add_argument('--cox-table', type=str, required=True,
                        help='Path to integrated_table_cox.parquet')
    parser.add_argument('--meth-table', type=str, required=True,
                        help='Path to methylation_table.parquet')
    parser.add_argument('--clinical', type=str, required=True,
                        help='Path to clinical data')
    parser.add_argument('--splits', type=str, required=True,
                        help='Path to train_val_test_splits.json')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (default: cuda)')

    args = parser.parse_args()

    train_hybrid_model(
        cox_table_path=args.cox_table,
        meth_table_path=args.meth_table,
        clinical_path=args.clinical,
        splits_path=args.splits,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )


if __name__ == "__main__":
    main()
