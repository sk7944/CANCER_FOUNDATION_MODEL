"""
Sharded Methylation TabTransformer 훈련 스크립트

3단계 훈련 프로세스:
  Phase 1: 각 샤드 모델 독립 훈련 (10개)
  Phase 2: 샤드 모델 freeze, Fusion layer 훈련
  Phase 3 (optional): End-to-end fine-tuning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import json
import argparse
from pathlib import Path
import sys

# 프로젝트 루트 경로 동적 탐지
current_dir = Path(__file__).resolve().parent
project_root = current_dir
while project_root.name != 'CANCER_FOUNDATION_MODEL' and project_root.parent != project_root:
    project_root = project_root.parent
if project_root.name == 'CANCER_FOUNDATION_MODEL':
    sys.path.insert(0, str(project_root))

from src.models.sharded_methylation_tabtransformer import (
    ShardedMethylationTabTransformer,
    SingleShardModel
)


def load_shard_data(data_dir, num_shards=10):
    """
    샤드 데이터 로드

    Args:
        data_dir: 샤드 데이터가 저장된 디렉토리
        num_shards: 샤드 개수

    Returns:
        shard_dataframes: List of DataFrames
        metadata: 샤드 메타데이터
    """
    data_dir = Path(data_dir)

    # 메타데이터 로드
    metadata_file = data_dir / 'methylation_shard_metadata.json'
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    print(f"Loading {num_shards} shards from {data_dir}")

    shard_dataframes = []
    for shard_id in range(num_shards):
        shard_file = data_dir / f'methylation_shard_{shard_id}.parquet'
        shard_df = pd.read_parquet(shard_file)
        shard_dataframes.append(shard_df)

        print(f"  Shard {shard_id}: {shard_df.shape}")

    return shard_dataframes, metadata


def create_survival_labels(clinical_data, days_threshold=1095):
    """
    3년 생존 라벨 생성

    Args:
        clinical_data: 임상 데이터 DataFrame
        days_threshold: 임계값 (기본 1095일 = 3년)

    Returns:
        labels: 생존 라벨 (1=생존, 0=사망)
        valid_patient_ids: 유효한 환자 ID 리스트
    """
    required_cols = ['survival_time_clean', 'survival_event_clean']

    if not all(col in clinical_data.columns for col in required_cols):
        raise ValueError(f"Required columns missing: {required_cols}. Available: {clinical_data.columns.tolist()[:20]}")

    valid_mask = clinical_data[required_cols].notna().all(axis=1)
    clinical_filtered = clinical_data[valid_mask].copy()

    # 생존 시간 (이미 일(days) 단위)
    survival_time = clinical_filtered['survival_time_clean']

    # 생존 이벤트 (1=사망, 0=생존)
    survival_event = clinical_filtered['survival_event_clean']

    # 라벨 생성
    # 1: 3년 이상 생존 또는 3년 미만이지만 살아있음 (event=0)
    # 0: 3년 이내 사망 (event=1 and time < threshold)
    labels = np.where(
        (survival_event == 0) | (survival_time >= days_threshold), 1, 0
    )

    valid_patient_ids = clinical_filtered.index.tolist()

    print(f"\nSurvival Label Statistics:")
    print(f"  Total patients: {len(labels)}")
    print(f"  Survival (1): {(labels == 1).sum()} ({(labels == 1).sum() / len(labels) * 100:.1f}%)")
    print(f"  Death (0): {(labels == 0).sum()} ({(labels == 0).sum() / len(labels) * 100:.1f}%)")

    return labels, valid_patient_ids


def split_data_stratified(X, y, test_size=0.15, val_size=0.15, random_state=42):
    """
    Stratified split

    Args:
        X: features
        y: labels
        test_size: test set 비율
        val_size: validation set 비율
        random_state: random seed

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    from sklearn.model_selection import train_test_split

    # Train + Val vs Test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Train vs Val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
    )

    print(f"\nData Split:")
    print(f"  Train: {len(y_train)} ({len(y_train)/len(y)*100:.1f}%)")
    print(f"  Val:   {len(y_val)} ({len(y_val)/len(y)*100:.1f}%)")
    print(f"  Test:  {len(y_test)} ({len(y_test)/len(y)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_single_shard_model(model, train_loader, val_loader, epochs=30, lr=1e-4,
                              device='cuda', checkpoint_dir=None, shard_id=0):
    """
    단일 샤드 모델 훈련

    Args:
        model: SingleShardModel
        train_loader: 훈련 DataLoader
        val_loader: 검증 DataLoader
        epochs: 에폭 수
        lr: 학습률
        device: 디바이스
        checkpoint_dir: 체크포인트 저장 디렉토리
        shard_id: 샤드 ID

    Returns:
        history: 훈련 기록
    """
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-7
    )

    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc = 0.0
    history = {
        'train_losses': [],
        'val_losses': [],
        'val_aucs': [],
        'best_val_auc': 0.0
    }

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for shard_data, labels in tqdm(train_loader, desc=f"Shard {shard_id} Epoch {epoch+1}/{epochs}", leave=False):
            shard_data = shard_data.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(shard_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for shard_data, labels in val_loader:
                shard_data = shard_data.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(shard_data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs).cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds)

        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(avg_val_loss)
        history['val_aucs'].append(val_auc)

        print(f"Shard {shard_id} Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            history['best_val_auc'] = best_val_auc

            if checkpoint_dir:
                save_path = checkpoint_dir / f'best_shard_{shard_id}.pth'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_auc': val_auc,
                    'shard_id': shard_id
                }, save_path)
                print(f"  ✓ Saved best model (AUC: {val_auc:.4f})")

        scheduler.step(val_auc)

    return history


def main():
    parser = argparse.ArgumentParser(description='Sharded Methylation TabTransformer Training')

    # 데이터 경로
    parser.add_argument('--shard_dir', type=str, default='../../data/processed/methylation_shards',
                        help='샤드 데이터 디렉토리')
    parser.add_argument('--clinical_data', type=str, default='../../data/processed/processed_clinical_data_for_methylation.parquet',
                        help='임상 데이터 파일')

    # 훈련 설정
    parser.add_argument('--phase', type=str, default='all', choices=['phase1', 'phase2', 'phase3', 'all'],
                        help='훈련 단계')
    parser.add_argument('--epochs_phase1', type=int, default=30, help='Phase 1 에폭 수')
    parser.add_argument('--epochs_phase2', type=int, default=20, help='Phase 2 에폭 수')
    parser.add_argument('--epochs_phase3', type=int, default=10, help='Phase 3 에폭 수')
    parser.add_argument('--batch_size', type=int, default=16, help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-4, help='학습률')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # 출력 경로
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/methylation',
                        help='체크포인트 저장 디렉토리')
    parser.add_argument('--results_dir', type=str, default='../../results',
                        help='결과 저장 디렉토리')

    args = parser.parse_args()

    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Seed 설정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 체크포인트 디렉토리 생성
    checkpoint_dir = Path(args.checkpoint_dir) / f'seed_{args.seed}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Sharded Methylation TabTransformer Training")
    print("="*80)

    # 데이터 로드
    print("\n[1/5] Loading shard data...")
    shard_dataframes, metadata = load_shard_data(args.shard_dir, num_shards=10)

    # 임상 데이터 로드
    print("\n[2/5] Loading clinical data...")
    clinical_data = pd.read_parquet(args.clinical_data)

    # 공통 환자 찾기
    common_patients = shard_dataframes[0].index
    for shard_df in shard_dataframes[1:]:
        common_patients = common_patients.intersection(shard_df.index)

    common_patients = common_patients.intersection(clinical_data.index)
    print(f"Common patients across all shards and clinical data: {len(common_patients)}")

    # 생존 라벨 생성
    print("\n[3/5] Creating survival labels...")
    survival_labels, valid_patient_ids = create_survival_labels(clinical_data, days_threshold=1095)

    # 유효한 공통 환자
    final_patients = list(set(common_patients) & set(valid_patient_ids))
    final_patients.sort()
    print(f"Final patients with valid survival data: {len(final_patients)}")

    # 라벨 정렬
    labels_dict = dict(zip(valid_patient_ids, survival_labels))
    labels_aligned = np.array([labels_dict[pid] for pid in final_patients])

    # Phase 1: 각 샤드 모델 독립 훈련
    if args.phase in ['phase1', 'all']:
        print("\n" + "="*80)
        print("PHASE 1: Training Individual Shard Models")
        print("="*80)

        for shard_id, shard_df in enumerate(shard_dataframes):
            print(f"\n--- Training Shard {shard_id} ---")

            # 이 샤드의 데이터
            shard_data = shard_df.loc[final_patients].values
            shard_tensor = torch.tensor(shard_data, dtype=torch.float32)

            # 데이터 분할
            X_train, X_val, X_test, y_train, y_val, y_test = split_data_stratified(
                shard_tensor, labels_aligned, test_size=0.15, val_size=0.15, random_state=args.seed
            )

            # DataLoader 생성
            train_dataset = TensorDataset(X_train, torch.tensor(y_train, dtype=torch.float32))
            val_dataset = TensorDataset(X_val, torch.tensor(y_val, dtype=torch.float32))

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

            # 모델 생성
            model = SingleShardModel(
                num_probes=shard_df.shape[1],
                dim=64,
                depth=4,
                heads=8,
                attn_dropout=0.3,
                ff_dropout=0.3,
                output_dim=256
            )

            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

            # 훈련
            history = train_single_shard_model(
                model, train_loader, val_loader,
                epochs=args.epochs_phase1,
                lr=args.lr,
                device=device,
                checkpoint_dir=checkpoint_dir,
                shard_id=shard_id
            )

            print(f"Shard {shard_id} Best Val AUC: {history['best_val_auc']:.4f}")

    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
