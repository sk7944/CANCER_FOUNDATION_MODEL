import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import json
import argparse
from pathlib import Path
import sys
import warnings

# 프로젝트 루트 경로 동적 탐지
current_dir = Path(__file__).resolve().parent
project_root = current_dir
while project_root.name != 'CANCER_FOUNDATION_MODEL' and project_root.parent != project_root:
    project_root = project_root.parent
if project_root.name == 'CANCER_FOUNDATION_MODEL':
    sys.path.insert(0, str(project_root))
from src.models.cox_tabtransformer import CoxTabTransformer
from src.models.methylation_tabtransformer import MethylationTabTransformer
from src.utils.tabtransformer_utils import *

warnings.filterwarnings('ignore')

def train_cox_tabtransformer(model, train_loader, val_loader, epochs=100, lr=1e-4, device='cuda', 
                            checkpoint_dir=None, resume_from=None, target_auc=0.85):
    """
    CoxTabTransformer 훈련 함수
    
    Args:
        model: CoxTabTransformer 모델
        train_loader: 훈련 DataLoader
        val_loader: 검증 DataLoader
        epochs: 훈련 에폭 수
        lr: 학습률
        device: 디바이스 ('cuda' 또는 'cpu')
        checkpoint_dir: 체크포인트 저장 디렉토리
        resume_from: 재개할 체크포인트 파일 경로
        target_auc: 목표 AUC 점수
    
    Returns:
        history: 훈련 기록 딕셔너리
    """
    model = model.to(device)
    
    # 오버피팅 방지를 위한 보수적인 최적화 전략
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.999))  # 더 강한 weight_decay
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.2))  # 클래스 불균형 고려
    
    # 오버피팅 방지를 위한 더 빠른 학습률 감소
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.3, patience=5, min_lr=1e-8  # 더 빠른 감소, 더 짧은 patience
    )
    
    # 체크포인트 디렉토리 생성
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # 훈련 재개
    start_epoch = 0
    best_val_auc = 0.0
    if resume_from and Path(resume_from).exists():
        print(f"체크포인트에서 훈련 재개: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_auc = checkpoint.get('best_val_auc', 0.0)
        print(f"에폭 {start_epoch}부터 재개, 최고 검증 AUC: {best_val_auc:.4f}")
    
    # Early stopping 설정
    patience_counter = 0
    early_stop_patience = 10  # 10 에폭 동안 개선되지 않으면 중단
    
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_aucs': [],
        'val_aucs': [],
        'train_accs': [],
        'val_accs': [],
        'best_val_auc': best_val_auc,
        'best_epoch': 0
    }
    
    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for clinical_batch, omics_batch, targets_batch in progress_bar:
            clinical_batch = clinical_batch.to(device)
            omics_batch = omics_batch.to(device)
            targets_batch = targets_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, _ = model(clinical_batch, omics_batch)
            loss = criterion(logits.squeeze(), targets_batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            train_labels.extend(targets_batch.cpu().numpy())
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for clinical_batch, omics_batch, targets_batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False):
                clinical_batch = clinical_batch.to(device)
                omics_batch = omics_batch.to(device)
                targets_batch = targets_batch.to(device)
                
                logits, _ = model(clinical_batch, omics_batch)
                loss = criterion(logits.squeeze(), targets_batch)
                
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels.extend(targets_batch.cpu().numpy())
        
        # Metrics 계산
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_auc = roc_auc_score(train_labels, train_preds) if len(set(train_labels)) > 1 else 0
        val_auc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0
        
        # Accuracy 계산
        train_preds_binary = [1 if p > 0.5 else 0 for p in train_preds]
        val_preds_binary = [1 if p > 0.5 else 0 for p in val_preds]
        train_acc = accuracy_score(train_labels, train_preds_binary)
        val_acc = accuracy_score(val_labels, val_preds_binary)
        
        # 기록 저장
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['train_aucs'].append(train_auc)
        history['val_aucs'].append(val_auc)
        if 'train_accs' not in history:
            history['train_accs'] = []
            history['val_accs'] = []
        history['train_accs'].append(train_acc)
        history['val_accs'].append(val_acc)
        
        print(f"Epoch {epoch+1:3d}: Loss(T/V): {train_loss:.3f}/{val_loss:.3f} | "
              f"AUC(T/V): {train_auc:.3f}/{val_auc:.3f} | "
              f"Acc(T/V): {train_acc:.3f}/{val_acc:.3f}")
        
        # Best model 저장 및 체크포인트 저장
        if val_auc > history['best_val_auc']:
            history['best_val_auc'] = val_auc
            history['best_epoch'] = epoch + 1
            
            # 최고 성능 모델 저장
            best_model_path = str(checkpoint_dir / 'best_cox_tabtransformer.pth') if checkpoint_dir else 'best_cox_tabtransformer.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_auc': val_auc,
                'history': history
            }, best_model_path)
            print(f"  ✅ Best model saved (AUC: {val_auc:.4f}) at {best_model_path}")
            
            # 목표 AUC 달성 알림 (하지만 계속 훈련)
            if val_auc >= target_auc:
                print(f"  🎯 Target AUC {target_auc:.3f} achieved! Continuing training for better performance...")
            
            # Early stopping 카운터 리셋
            patience_counter = 0
        else:
            # 개선되지 않음 - patience 증가
            patience_counter += 1
        
        # Early stopping 체크
        if patience_counter >= early_stop_patience:
            print(f"\n⏹️  Early stopping triggered after {patience_counter} epochs without improvement")
            print(f"   Best Val AUC: {history['best_val_auc']:.4f} (Epoch {history['best_epoch']})")
            break
        
        # 매 에폭마다 체크포인트 저장
        if checkpoint_dir and epoch % 5 == 0:  # 5 에폭마다 저장
            checkpoint_path = str(checkpoint_dir / f'cox_checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_auc': history['best_val_auc'],
                'history': history
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")
        
        # Learning rate scheduling (val_auc 기준으로 변경)
        scheduler.step(val_auc)
        
        # Early stopping (optional)
        if epoch - history['best_epoch'] > 10 and epoch > 20:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return history

def train_methylation_tabtransformer(model, train_loader, val_loader, epochs=50, lr=5e-5, device='cuda',
                                   checkpoint_dir=None, resume_from=None, target_auc=0.85):
    """
    MethylationTabTransformer 훈련 함수
    
    Args:
        model: MethylationTabTransformer 모델
        train_loader: 훈련 DataLoader
        val_loader: 검증 DataLoader
        epochs: 훈련 에폭 수
        lr: 학습률
        device: 디바이스 ('cuda' 또는 'cpu')
    
    Returns:
        history: 훈련 기록 딕셔너리
    """
    model = model.to(device)
    
    # 오버피팅 방지를 위한 보수적인 최적화 전략
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.999))  # 더 강한 weight_decay
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.2))  # 클래스 불균형 고려
    
    # 오버피팅 방지를 위한 더 빠른 학습률 감소
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.3, patience=5, min_lr=1e-8  # 더 빠른 감소, 더 짧은 patience
    )
    
    # 체크포인트 디렉토리 생성
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # 훈련 재개
    start_epoch = 0
    best_val_auc = 0.0
    if resume_from and Path(resume_from).exists():
        print(f"체크포인트에서 훈련 재개: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_auc = checkpoint.get('best_val_auc', 0.0)
        print(f"에폭 {start_epoch}부터 재개, 최고 검증 AUC: {best_val_auc:.4f}")
    
    # Early stopping 설정
    patience_counter = 0
    early_stop_patience = 10  # 10 에폭 동안 개선되지 않으면 중단
    
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_aucs': [],
        'val_aucs': [],
        'train_accs': [],
        'val_accs': [],
        'best_val_auc': best_val_auc,
        'best_epoch': 0
    }
    
    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for meth_batch, targets_batch in progress_bar:
            meth_batch = meth_batch.to(device)
            targets_batch = targets_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, _, _ = model(meth_batch)
            loss = criterion(logits.squeeze(), targets_batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            train_labels.extend(targets_batch.cpu().numpy())
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for meth_batch, targets_batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False):
                meth_batch = meth_batch.to(device)
                targets_batch = targets_batch.to(device)
                
                logits, _, _ = model(meth_batch)
                loss = criterion(logits.squeeze(), targets_batch)
                
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels.extend(targets_batch.cpu().numpy())
        
        # Metrics 계산
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_auc = roc_auc_score(train_labels, train_preds) if len(set(train_labels)) > 1 else 0
        val_auc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0
        
        # Accuracy 계산
        train_preds_binary = [1 if p > 0.5 else 0 for p in train_preds]
        val_preds_binary = [1 if p > 0.5 else 0 for p in val_preds]
        train_acc = accuracy_score(train_labels, train_preds_binary)
        val_acc = accuracy_score(val_labels, val_preds_binary)
        
        # 기록 저장
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['train_aucs'].append(train_auc)
        history['val_aucs'].append(val_auc)
        if 'train_accs' not in history:
            history['train_accs'] = []
            history['val_accs'] = []
        history['train_accs'].append(train_acc)
        history['val_accs'].append(val_acc)
        
        print(f"Epoch {epoch+1:3d}: Loss(T/V): {train_loss:.3f}/{val_loss:.3f} | "
              f"AUC(T/V): {train_auc:.3f}/{val_auc:.3f} | "
              f"Acc(T/V): {train_acc:.3f}/{val_acc:.3f}")
        
        # Best model 저장 (체크포인트 포함)
        if val_auc > history['best_val_auc']:
            history['best_val_auc'] = val_auc
            history['best_epoch'] = epoch + 1
            
            # 최고 성능 모델 저장
            best_model_path = str(checkpoint_dir / 'best_methylation_tabtransformer.pth') if checkpoint_dir else 'best_methylation_tabtransformer.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_auc': val_auc,
                'history': history
            }, best_model_path)
            print(f"  ✅ Best model saved (AUC: {val_auc:.4f}) at {best_model_path}")
            
            # 목표 AUC 달성 알림 (하지만 계속 훈련)
            if val_auc >= target_auc:
                print(f"  🎯 Target AUC {target_auc:.3f} achieved! Continuing training for better performance...")
            
            # Early stopping 카운터 리셋
            patience_counter = 0
        else:
            # 개선되지 않음 - patience 증가
            patience_counter += 1
        
        # Early stopping 체크
        if patience_counter >= early_stop_patience:
            print(f"\n⏹️  Early stopping triggered after {patience_counter} epochs without improvement")
            print(f"   Best Val AUC: {history['best_val_auc']:.4f} (Epoch {history['best_epoch']})")
            break
        
        # Learning rate scheduling
        scheduler.step(val_auc)
    
    return history

def evaluate_model(model, test_loader, model_name, device='cuda', is_cox_model=True):
    """
    모델 평가 함수
    
    Args:
        model: 평가할 모델
        test_loader: 테스트 DataLoader
        model_name: 모델 이름
        device: 디바이스
        is_cox_model: CoxTabTransformer인지 여부
    
    Returns:
        results: 평가 결과 딕셔너리
    """
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            if is_cox_model:
                clinical_batch, omics_batch, labels_batch = batch
                clinical_batch = clinical_batch.to(device)
                omics_batch = omics_batch.to(device)
                logits, _ = model(clinical_batch, omics_batch)
            else:
                meth_batch, labels_batch = batch
                meth_batch = meth_batch.to(device)
                logits, _, _ = model(meth_batch)
            
            test_preds.extend(torch.sigmoid(logits).cpu().numpy())
            test_labels.extend(labels_batch.numpy())
    
    test_preds = np.array(test_preds).squeeze()
    test_labels = np.array(test_labels).squeeze()
    
    # 메트릭 계산
    test_auc = roc_auc_score(test_labels, test_preds)
    test_preds_binary = (test_preds > 0.5).astype(int)
    test_acc = accuracy_score(test_labels, test_preds_binary)
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds_binary)
    
    results = {
        'auc': test_auc,
        'accuracy': test_acc,
        'confusion_matrix': cm,
        'predictions': test_preds,
        'true_labels': test_labels
    }
    
    print(f"\n=== {model_name} Test Results ===")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Confusion Matrix:")
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    return results

def main():
    """
    메인 훈련 함수
    """
    args = parse_arguments()
    
    # GPU 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 결과 디렉토리 생성
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    if args.model == 'cox':
        print("=== CoxTabTransformer 훈련 ===")
        
        # 데이터 로드
        cox_data = pd.read_parquet(Path(args.data_dir) / 'integrated_table_cox.parquet')
        clinical_data = pd.read_parquet(Path(args.data_dir) / 'processed_clinical_data.parquet')
        
        # Cox 계수 로드
        _, cox_coefficients = load_cox_coefficients_by_omics(args.data_dir)
        
        # 생존 라벨 생성
        survival_labels, valid_patient_ids = create_survival_labels(clinical_data, 1095)
        
        # 데이터 전처리
        cox_data_filtered = cox_data.loc[cox_data.index.intersection(valid_patient_ids)]
        clinical_data_filtered = clinical_data.loc[clinical_data.index.intersection(valid_patient_ids)]
        
        cox_continuous, cox_feature_names = prepare_cox_data(cox_data_filtered, cox_coefficients)
        clinical_categorical, vocab_sizes, encoders, clinical_feature_names = prepare_clinical_data(clinical_data_filtered)
        
        # 라벨 정렬
        common_patients = cox_data_filtered.index.tolist()
        labels_dict = dict(zip(valid_patient_ids, survival_labels))
        labels_aligned = np.array([labels_dict[pid] for pid in common_patients])
        
        # 데이터 분할 (앙상블을 위해 seed 기반)
        combined_data = torch.cat([clinical_categorical, cox_continuous], dim=1)
        data_split_seed = getattr(args, 'seed', 42)  # 명령행에서 seed 받기
        X_train, X_val, X_test, y_train, y_val, y_test = split_data_stratified(
            combined_data, labels_aligned, test_size=0.15, val_size=0.15, random_state=data_split_seed
        )
        
        # Clinical과 Omics 부분으로 다시 분리
        clinical_dim = clinical_categorical.shape[1]
        X_train_clinical = X_train[:, :clinical_dim]
        X_train_omics = X_train[:, clinical_dim:]
        X_val_clinical = X_val[:, :clinical_dim]
        X_val_omics = X_val[:, clinical_dim:]
        X_test_clinical = X_test[:, :clinical_dim]
        X_test_omics = X_test[:, clinical_dim:]
        
        # DataLoader 생성
        train_dataset = TensorDataset(X_train_clinical.long(), X_train_omics, torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(X_val_clinical.long(), X_val_omics, torch.tensor(y_val, dtype=torch.float32))
        test_dataset = TensorDataset(X_test_clinical.long(), X_test_omics, torch.tensor(y_test, dtype=torch.float32))
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # 모델 생성
        model = CoxTabTransformer(
            clinical_categories=vocab_sizes,
            num_omics_features=len(cox_feature_names),
            dim=64,
            depth=6,
            heads=8
        )
        
        print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
        
        # 훈련
        history = train_cox_tabtransformer(model, train_loader, val_loader, args.epochs, args.lr, device)
        
        # 평가 (최고 성능 모델 로드)
        best_model_path = 'best_cox_tabtransformer.pth'
        if Path(best_model_path).exists():
            checkpoint = torch.load(best_model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Best model loaded from {best_model_path} (AUC: {checkpoint.get('best_val_auc', 'Unknown')})")
            else:
                model.load_state_dict(checkpoint)  # 구버전 호환성
        else:
            print("⚠️ No best model found, using current model for evaluation")
        results = evaluate_model(model, test_loader, 'CoxTabTransformer', device, is_cox_model=True)
        
        # 결과 저장
        final_results = {
            'model': 'CoxTabTransformer',
            'history': history,
            'test_results': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in results.items()},
            'model_config': {
                'clinical_categories': vocab_sizes,
                'num_omics_features': len(cox_feature_names),
                'dim': 64,
                'depth': 6,
                'heads': 8
            }
        }
        
        with open(results_dir / 'cox_tabtransformer_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"결과 저장 완료: {results_dir / 'cox_tabtransformer_results.json'}")
        
    elif args.model == 'methylation':
        print("=== MethylationTabTransformer 훈련 ===")
        
        # 메틸레이션 데이터 로드
        methylation_data = pd.read_parquet(Path(args.data_dir) / 'methylation_table.parquet')
        clinical_data_meth = pd.read_parquet(Path(args.data_dir) / 'processed_clinical_data_for_methylation.parquet')
        
        # 생존 라벨 생성
        survival_labels, valid_patient_ids = create_survival_labels(clinical_data_meth, 1095)
        
        # 데이터 전처리
        methylation_filtered = methylation_data.loc[methylation_data.index.intersection(valid_patient_ids)]
        methylation_tensor, selected_probe_names = prepare_methylation_data(methylation_filtered, variance_threshold=0.01)
        
        # 라벨 정렬
        common_patients = methylation_filtered.index.tolist()
        labels_dict = dict(zip(valid_patient_ids, survival_labels))
        labels_aligned = np.array([labels_dict[pid] for pid in common_patients])
        
        # 데이터 분할
        X_train, X_val, X_test, y_train, y_val, y_test = split_data_stratified(
            methylation_tensor, labels_aligned, test_size=0.15, val_size=0.15, random_state=42
        )
        
        # DataLoader 생성 (작은 배치 크기)
        train_dataset = TensorDataset(X_train, torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(X_val, torch.tensor(y_val, dtype=torch.float32))
        test_dataset = TensorDataset(X_test, torch.tensor(y_test, dtype=torch.float32))
        
        train_loader = DataLoader(train_dataset, batch_size=min(args.batch_size, 16), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=min(args.batch_size, 16), shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=min(args.batch_size, 16), shuffle=False)
        
        # 모델 생성
        model = MethylationTabTransformer(
            num_probes=methylation_tensor.shape[1],
            selected_probes=min(5000, methylation_tensor.shape[1] // 10),
            dim=64,
            depth=4,
            heads=8
        )
        
        print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
        
        # 훈련
        history = train_methylation_tabtransformer(model, train_loader, val_loader, args.epochs, args.lr, device)
        
        # 평가 (최고 성능 모델 로드)
        best_model_path = 'best_methylation_tabtransformer.pth'
        if Path(best_model_path).exists():
            checkpoint = torch.load(best_model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Best model loaded from {best_model_path} (AUC: {checkpoint.get('best_val_auc', 'Unknown')})")
            else:
                model.load_state_dict(checkpoint)  # 구버전 호환성
        else:
            print("⚠️ No best model found, using current model for evaluation")
        results = evaluate_model(model, test_loader, 'MethylationTabTransformer', device, is_cox_model=False)
        
        # 결과 저장
        final_results = {
            'model': 'MethylationTabTransformer',
            'history': history,
            'test_results': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in results.items()},
            'model_config': {
                'num_probes': methylation_tensor.shape[1],
                'selected_probes': min(5000, methylation_tensor.shape[1] // 10),
                'dim': 64,
                'depth': 4,
                'heads': 8
            }
        }
        
        with open(results_dir / 'methylation_tabtransformer_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"결과 저장 완료: {results_dir / 'methylation_tabtransformer_results.json'}")

def train_ensemble_models(args):
    """앙상블 모드로 여러 seed로 모델 훈련"""
    
    print(f"\n🎯 Starting ensemble training with {args.n_seeds} different seeds")
    print(f"Base seed: {args.seed}, Model type: {args.model}")
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = Path(args.checkpoint_dir) 
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    ensemble_results = []
    
    for i in range(args.n_seeds):
        current_seed = args.seed + i
        print(f"\n{'='*60}")
        print(f"🌱 Training model {i+1}/{args.n_seeds} with seed {current_seed}")
        print(f"{'='*60}")
        
        # seed를 현재 seed로 변경
        args.seed = current_seed
        
        # 개별 모델 훈련 실행
        try:
            if args.model == 'cox':
                history = main_cox_training(args, checkpoint_dir / f'seed_{current_seed}')
            elif args.model == 'methylation':
                history = main_methylation_training(args, checkpoint_dir / f'seed_{current_seed}')
            
            # test_results JSON 직렬화 처리
            test_results = history.get('test_results', {})
            test_results_serializable = {}
            for key, value in test_results.items():
                if hasattr(value, 'tolist'):  # numpy array인 경우
                    test_results_serializable[key] = value.tolist()
                else:
                    test_results_serializable[key] = value
            
            ensemble_results.append({
                'seed': current_seed,
                'best_val_auc': history['best_val_auc'],
                'best_epoch': history['best_epoch'],
                'final_train_auc': history['train_aucs'][-1] if history['train_aucs'] else 0,
                'final_val_auc': history['val_aucs'][-1] if history['val_aucs'] else 0,
                'test_auc': history.get('test_auc', 0.0),  # Test AUC 추가
                'test_results': test_results_serializable  # JSON 직렬화 가능한 형태
            })
            
            print(f"✅ Seed {current_seed} completed: Best Val AUC = {history['best_val_auc']:.4f}, Test AUC = {history.get('test_auc', 0.0):.4f}")
            
        except Exception as e:
            print(f"❌ Seed {current_seed} failed: {str(e)}")
            ensemble_results.append({
                'seed': current_seed,
                'best_val_auc': 0.0,
                'best_epoch': 0,
                'final_train_auc': 0.0,
                'final_val_auc': 0.0,
                'test_auc': 0.0,
                'test_results': {},
                'error': str(e)
            })
    
    # 앙상블 결과 요약
    print_ensemble_summary(ensemble_results, args.model, results_dir)
    
    return ensemble_results

def print_ensemble_summary(results, model_type, results_dir):
    """앙상블 훈련 결과 요약"""
    
    print(f"\n{'='*70}")
    print(f"🎯 ENSEMBLE TRAINING SUMMARY - {model_type.upper()} MODEL")
    print(f"{'='*70}")
    
    successful_results = [r for r in results if 'error' not in r]
    
    if successful_results:
        val_aucs = [r['best_val_auc'] for r in successful_results]
        test_aucs = [r.get('test_auc', 0.0) for r in successful_results]
        
        print(f"{'Seed':<8} {'Best Val AUC':<13} {'Test AUC':<10} {'Best Epoch':<10}")
        print(f"{'-'*60}")
        
        for r in successful_results:
            print(f"{r['seed']:<8} {r['best_val_auc']:<13.4f} {r.get('test_auc', 0.0):<10.4f} {r['best_epoch']:<10}")
        
        print(f"{'-'*60}")
        print(f"{'Mean':<8} {np.mean(val_aucs):<13.4f} {np.mean(test_aucs):<10.4f}")
        print(f"{'Std':<8} {np.std(val_aucs):<13.4f} {np.std(test_aucs):<10.4f}")
        print(f"{'Max':<8} {np.max(val_aucs):<13.4f} {np.max(test_aucs):<10.4f}")
        print(f"{'Min':<8} {np.min(val_aucs):<13.4f} {np.min(test_aucs):<10.4f}")
        
        # 목표 달성 모델 수 (Test AUC 기준)
        val_target_achieved = sum(1 for auc in val_aucs if auc >= 0.85)
        test_target_achieved = sum(1 for auc in test_aucs if auc >= 0.85)
        print(f"\n🎯 Models achieving Val AUC ≥ 0.85: {val_target_achieved}/{len(successful_results)}")
        print(f"🎯 Models achieving Test AUC ≥ 0.85: {test_target_achieved}/{len(successful_results)}")
    
    # 실패한 모델들
    failed_results = [r for r in results if 'error' in r]
    if failed_results:
        print(f"\n❌ Failed seeds: {[r['seed'] for r in failed_results]}")
    
    # 결과를 JSON으로 저장
    summary = {
        'model_type': model_type,
        'total_models': len(results),
        'successful_models': len(successful_results),
        'failed_models': len(failed_results),
        'results': results,
        'statistics': {
            'val_auc_mean': float(np.mean([r['best_val_auc'] for r in successful_results])) if successful_results else 0,
            'val_auc_std': float(np.std([r['best_val_auc'] for r in successful_results])) if successful_results else 0,
            'test_auc_mean': float(np.mean([r.get('test_auc', 0.0) for r in successful_results])) if successful_results else 0,
            'test_auc_std': float(np.std([r.get('test_auc', 0.0) for r in successful_results])) if successful_results else 0,
            'val_target_achieved_count': sum(1 for r in successful_results if r['best_val_auc'] >= 0.85),
            'test_target_achieved_count': sum(1 for r in successful_results if r.get('test_auc', 0.0) >= 0.85)
        }
    }
    
    # 파일명을 더 명확하게: cox_tabtransformer_ensemble_results.json
    summary_path = results_dir / f'{model_type}_tabtransformer_ensemble_results.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Ensemble summary saved: {summary_path}")
    print(f"{'='*70}")

def main_cox_training(args, checkpoint_dir=None):
    """Cox 모델 단일 훈련 (앙상블에서 호출)"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training with seed: {args.seed}")
    
    # 결과 디렉토리 생성
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    print("=== CoxTabTransformer 훈련 ===")
    
    # 데이터 로드
    cox_data = pd.read_parquet(Path(args.data_dir) / 'integrated_table_cox.parquet')
    clinical_data = pd.read_parquet(Path(args.data_dir) / 'processed_clinical_data.parquet')
    
    # Cox 계수 로드
    _, cox_coefficients = load_cox_coefficients_by_omics(args.data_dir)
    
    # 생존 라벨 생성
    survival_labels, valid_patient_ids = create_survival_labels(clinical_data, 1095)
    
    # 데이터 전처리
    cox_data_filtered = cox_data.loc[cox_data.index.intersection(valid_patient_ids)]
    clinical_data_filtered = clinical_data.loc[clinical_data.index.intersection(valid_patient_ids)]
    
    cox_continuous, cox_feature_names = prepare_cox_data(cox_data_filtered, cox_coefficients)
    clinical_categorical, vocab_sizes, encoders, clinical_feature_names = prepare_clinical_data(clinical_data_filtered)
    
    # 라벨 정렬
    common_patients = cox_data_filtered.index.tolist()
    labels_dict = dict(zip(valid_patient_ids, survival_labels))
    labels_aligned = np.array([labels_dict[pid] for pid in common_patients])
    
    # 데이터 분할 (앙상블을 위해 seed 기반)
    combined_data = torch.cat([clinical_categorical, cox_continuous], dim=1)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_stratified(
        combined_data, labels_aligned, test_size=0.15, val_size=0.15, random_state=args.seed
    )
    
    # Clinical과 Omics 부분으로 다시 분리
    clinical_dim = clinical_categorical.shape[1]
    X_train_clinical = X_train[:, :clinical_dim]
    X_train_omics = X_train[:, clinical_dim:]
    X_val_clinical = X_val[:, :clinical_dim]
    X_val_omics = X_val[:, clinical_dim:]
    X_test_clinical = X_test[:, :clinical_dim]
    X_test_omics = X_test[:, clinical_dim:]
    
    # DataLoader 생성
    train_dataset = TensorDataset(X_train_clinical.long(), X_train_omics, torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(X_val_clinical.long(), X_val_omics, torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(X_test_clinical.long(), X_test_omics, torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 모델 생성 (오버피팅 방지를 위해 복잡도 감소)
    model = CoxTabTransformer(
        clinical_categories=vocab_sizes,
        num_omics_features=len(cox_feature_names),
        dim=64,   # 128 → 64 (복잡도 감소)
        depth=4,  # 8 → 4 (더 얘은 모델)
        heads=8,  # 16 → 8 (어텐션 헤드 감소)
        attn_dropout=0.3,  # 더 강한 드롭아웃
        ff_dropout=0.3     # 더 강한 드롭아웃
    )
    
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 훈련
    history = train_cox_tabtransformer(
        model, train_loader, val_loader, 
        epochs=args.epochs, 
        lr=args.lr, 
        device=device,
        checkpoint_dir=checkpoint_dir,
        resume_from=args.resume_from if hasattr(args, 'resume_from') else None,
        target_auc=args.target_auc if hasattr(args, 'target_auc') else 0.85
    )
    
    # 평가 (최고 성능 모델 로드)
    best_model_name = f'best_cox_seed_{args.seed}.pth' if checkpoint_dir else 'best_cox_tabtransformer.pth'
    best_model_path = checkpoint_dir / best_model_name if checkpoint_dir else best_model_name
    
    if Path(best_model_path).exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Best model loaded from {best_model_path}")
    
    results = evaluate_model(model, test_loader, 'CoxTabTransformer', device, is_cox_model=True)
    
    # 결과에 test AUC 추가
    history['test_auc'] = results.get('auc', 0.0)
    history['test_results'] = results
    
    return history

def main_methylation_training(args, checkpoint_dir=None):
    """Methylation 모델 단일 훈련 (앙상블에서 호출)"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training with seed: {args.seed}")
    
    # 결과 디렉토리 생성
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    print("=== MethylationTabTransformer 훈련 ===")
    
    # 데이터 로드
    methylation_data = pd.read_parquet(Path(args.data_dir) / 'methylation_data_for_tabtransformer.parquet')
    clinical_data = pd.read_parquet(Path(args.data_dir) / 'processed_clinical_data.parquet')
    
    # Methylation과 clinical 데이터의 공통 환자 찾기
    common_patients_meth = methylation_data.index.intersection(clinical_data.index)
    methylation_data_filtered = methylation_data.loc[common_patients_meth]
    clinical_data_meth = clinical_data.loc[common_patients_meth]
    
    # 생존 라벨 생성
    survival_labels, valid_patient_ids = create_survival_labels(clinical_data_meth, 1095)
    
    # 유효한 환자들로 필터링
    methylation_data_filtered = methylation_data_filtered.loc[
        methylation_data_filtered.index.intersection(valid_patient_ids)
    ]
    
    # 데이터 전처리
    methylation_tensor, selected_probes = prepare_methylation_data(methylation_data_filtered, variance_threshold=0.01)
    
    # 라벨 정렬
    common_patients_final = methylation_data_filtered.index.tolist()
    labels_dict = dict(zip(valid_patient_ids, survival_labels))
    labels_aligned = np.array([labels_dict[pid] for pid in common_patients_final])
    
    # 데이터 분할 (seed 기반)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_stratified(
        methylation_tensor, labels_aligned, test_size=0.15, val_size=0.15, random_state=args.seed
    )
    
    # DataLoader 생성
    train_dataset = TensorDataset(X_train, torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(X_val, torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(X_test, torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 모델 생성 (오버피팅 방지를 위해 복잡도 감소)
    model = MethylationTabTransformer(
        num_probes=methylation_tensor.shape[1],
        selected_probes=min(5000, methylation_tensor.shape[1] // 10),  # 프로브 수 감소
        dim=64,   # 128 → 64 (복잡도 감소)
        depth=4,  # 6 → 4 (더 얕은 모델)
        heads=8,  # 16 → 8 (어텐션 헤드 감소)
        attn_dropout=0.3,  # 더 강한 드롭아웃
        ff_dropout=0.3     # 더 강한 드롭아웃
    )
    
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 훈련
    history = train_methylation_tabtransformer(
        model, train_loader, val_loader, 
        epochs=args.epochs, 
        lr=args.lr, 
        device=device,
        checkpoint_dir=checkpoint_dir,
        resume_from=args.resume_from if hasattr(args, 'resume_from') else None,
        target_auc=args.target_auc if hasattr(args, 'target_auc') else 0.85
    )
    
    # 평가 (최고 성능 모델 로드)
    best_model_name = f'best_methylation_seed_{args.seed}.pth' if checkpoint_dir else 'best_methylation_tabtransformer.pth'
    best_model_path = checkpoint_dir / best_model_name if checkpoint_dir else best_model_name
    
    if Path(best_model_path).exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Best model loaded from {best_model_path}")
    
    results = evaluate_model(model, test_loader, 'MethylationTabTransformer', device, is_cox_model=False)
    
    # 결과에 test AUC 추가
    history['test_auc'] = results.get('auc', 0.0)
    history['test_results'] = results
    
    return history

def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='TabTransformer 훈련 스크립트')
    parser.add_argument('--model', choices=['cox', 'methylation'], required=True, help='모델 타입')
    parser.add_argument('--epochs', type=int, default=50, help='훈련 에폭 수')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-4, help='학습률')
    parser.add_argument('--data_dir', type=str, default='../data/processed', help='데이터 디렉토리')
    parser.add_argument('--results_dir', type=str, default='../results', help='결과 저장 디렉토리')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드 (앙상블용)')
    parser.add_argument('--ensemble', action='store_true', help='앙상블 모드 활성화')
    parser.add_argument('--n_seeds', type=int, default=5, help='앙상블에 사용할 시드 개수')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='체크포인트 저장 디렉토리')
    parser.add_argument('--resume_from', type=str, default=None, help='재개할 체크포인트 파일 경로')
    parser.add_argument('--target_auc', type=float, default=0.85, help='목표 AUC 점수')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.ensemble:
        train_ensemble_models(args)
    else:
        main()