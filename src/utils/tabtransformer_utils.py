import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

def prepare_cox_data(omics_data, cox_coefficients_lookup):
    """
    [측정값, Cox계수] 쌍을 TabTransformer 입력 형태로 변환
    
    Args:
        omics_data: 오믹스 데이터 DataFrame (patients x features)
        cox_coefficients_lookup: Cox 계수 딕셔너리 {feature_name: cox_coeff}
    
    Returns:
        paired_data: [측정값, Cox계수] 쌍들이 flatten된 tensor
        feature_names: 특성 이름 리스트
    """
    
    # Cox 계수가 있는 특성들만 선택
    available_features = [col for col in omics_data.columns if col in cox_coefficients_lookup]
    omics_filtered = omics_data[available_features]
    
    print(f"Cox 계수가 있는 특성 수: {len(available_features)}")
    
    # Cox 계수 벡터 생성
    cox_coefficients = torch.tensor([
        cox_coefficients_lookup[feature] for feature in available_features
    ], dtype=torch.float32)
    
    # 데이터를 tensor로 변환
    omics_tensor = torch.tensor(omics_filtered.values, dtype=torch.float32)
    batch_size, num_features = omics_tensor.shape
    
    # Cox 계수를 배치 크기만큼 확장
    cox_expanded = cox_coefficients.unsqueeze(0).expand(batch_size, -1)
    
    # [측정값, Cox계수] 쌍을 interleave 방식으로 결합
    paired_data = torch.stack([omics_tensor, cox_expanded], dim=2)  # (batch, features, 2)
    flattened = paired_data.view(batch_size, -1)  # (batch, features*2)
    
    return flattened, available_features

def prepare_clinical_data(clinical_df, categorical_columns=None):
    """
    임상 데이터를 TabTransformer용 범주형 데이터로 변환
    
    Args:
        clinical_df: 임상 데이터 DataFrame
        categorical_columns: 범주형으로 처리할 컬럼 리스트 (None이면 자동 선택)
    
    Returns:
        categorical_data: 인코딩된 범주형 데이터 tensor
        vocab_sizes: 각 범주형 특성의 vocabulary size tuple
        encoders: 사용된 label encoder들의 딕셔너리
        feature_names: 범주형 특성 이름 리스트
    """
    
    if categorical_columns is None:
        # 기본 범주형 특성들 선택
        categorical_columns = []
        
        # Age를 binning하여 범주형으로 변환
        if 'age_at_initial_pathologic_diagnosis' in clinical_df.columns:
            clinical_df = clinical_df.copy()
            age_col = 'age_at_initial_pathologic_diagnosis'
            clinical_df['age_group'] = pd.cut(
                clinical_df[age_col], 
                bins=[0, 40, 50, 60, 70, 100], 
                labels=['young', 'middle_young', 'middle', 'middle_old', 'old'],
                include_lowest=True
            )
            categorical_columns.append('age_group')
        
        # Gender
        if 'gender' in clinical_df.columns:
            categorical_columns.append('gender')
        
        # Tumor stage
        if 'pathologic_stage' in clinical_df.columns:
            categorical_columns.append('pathologic_stage')
        
        # T stage
        if 'pathologic_T' in clinical_df.columns:
            categorical_columns.append('pathologic_T')
        
        # N stage
        if 'pathologic_N' in clinical_df.columns:
            categorical_columns.append('pathologic_N')
        
        # Cancer type (acronym)
        if 'acronym' in clinical_df.columns:
            categorical_columns.append('acronym')
    
    # 사용 가능한 컬럼만 선택
    available_columns = [col for col in categorical_columns if col in clinical_df.columns]
    print(f"사용할 범주형 특성: {available_columns}")
    
    if not available_columns:
        # 범주형 특성이 없으면 빈 데이터 반환
        return torch.empty(len(clinical_df), 0, dtype=torch.long), (), {}, []
    
    # 결측값 처리 및 인코딩
    categorical_data = []
    vocab_sizes = []
    encoders = {}
    
    for col in available_columns:
        # 결측값을 'Unknown'으로 채우기
        if clinical_df[col].dtype.name == 'category':
            # Categorical 컬럼인 경우 카테고리에 'Unknown' 추가
            if 'Unknown' not in clinical_df[col].cat.categories:
                clinical_df[col] = clinical_df[col].cat.add_categories(['Unknown'])
            col_data = clinical_df[col].fillna('Unknown').astype(str)
        else:
            col_data = clinical_df[col].fillna('Unknown').astype(str)
        
        # Label encoding
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(col_data)
        
        categorical_data.append(encoded)
        vocab_sizes.append(len(encoder.classes_))
        encoders[col] = encoder
        
        print(f"{col}: {len(encoder.classes_)} categories")
    
    # Tensor로 변환
    categorical_tensor = torch.tensor(
        np.column_stack(categorical_data), 
        dtype=torch.long
    )
    
    return categorical_tensor, tuple(vocab_sizes), encoders, available_columns

def create_survival_labels(clinical_data, threshold_days=1095):
    """
    3년 생존 여부 Binary Label 생성
    
    Args:
        clinical_data: 임상 데이터 DataFrame
        threshold_days: 생존 기준일 (기본 3년 = 1095일)
    
    Returns:
        survival_labels: numpy array of binary labels
        valid_indices: 유효한 환자 인덱스 리스트
    """
    survival_labels = []
    valid_indices = []
    
    for idx, (patient_id, row) in enumerate(clinical_data.iterrows()):
        if 'survival_time_clean' in row and pd.notna(row['survival_time_clean']):
            survival_days = row['survival_time_clean']
            survival_event = row.get('survival_event_clean', 0)  # 1=사망, 0=생존/censored
            
            if survival_days >= threshold_days:
                # 3년 이상 생존 → 3년 생존 성공 (라벨: 0)
                survival_labels.append(0)
                valid_indices.append(patient_id)
            else:
                # 3년 미만 기록
                if survival_event == 1:  # 사망
                    survival_labels.append(1)  # 3년 내 사망 (라벨: 1)
                    valid_indices.append(patient_id)
                else:  # 생존 또는 추적중단 (censored)
                    # Censored 데이터는 제외
                    continue
    
    return np.array(survival_labels), valid_indices

def prepare_methylation_data(methylation_df, variance_threshold=0.01):
    """
    메틸레이션 데이터 전처리
    
    Args:
        methylation_df: 메틸레이션 데이터 DataFrame
        variance_threshold: variance filtering 기준
    
    Returns:
        processed_data: 전처리된 메틸레이션 tensor
        selected_probes: 선택된 probe 이름들
    """
    print(f"원본 메틸레이션 데이터 shape: {methylation_df.shape}")
    
    # 결측값 처리 (median으로 대체)
    methylation_filled = methylation_df.fillna(methylation_df.median())
    
    # Variance filtering - 변화가 거의 없는 probe 제거
    variances = methylation_filled.var()
    high_var_probes = variances[variances > variance_threshold].index
    
    methylation_filtered = methylation_filled[high_var_probes]
    
    print(f"Variance filtering 후 shape: {methylation_filtered.shape}")
    print(f"제거된 probe 수: {len(methylation_df.columns) - len(high_var_probes)}")
    
    # 값 범위를 [0, 1]로 정규화 (β-value는 원래 0-1 범위이지만 확실히)
    methylation_normalized = methylation_filtered.clip(0, 1)
    
    # Tensor로 변환
    methylation_tensor = torch.tensor(methylation_normalized.values, dtype=torch.float32)
    
    return methylation_tensor, list(high_var_probes)

def load_cox_coefficients(cox_file_path):
    """
    Cox 계수 파일 로드
    
    Args:
        cox_file_path: Cox 계수 pickle 파일 경로
    
    Returns:
        cox_lookup: {feature_name: cox_coefficient} 딕셔너리
    """
    try:
        with open(cox_file_path, 'rb') as f:
            cox_lookup = pickle.load(f)
        print(f"Cox 계수 로드 완료: {len(cox_lookup)}개 특성")
        return cox_lookup
    except FileNotFoundError:
        print(f"Cox 계수 파일을 찾을 수 없습니다: {cox_file_path}")
        print("JSON 파일에서 Cox 계수 추출을 시도합니다...")
        
        # JSON 파일에서 Cox 계수 추출 시도
        json_path = cox_file_path.parent / 'cox_feature_info.json'
        if json_path.exists():
            import json
            with open(json_path, 'r') as f:
                cox_info = json.load(f)
            
            cox_lookup = {}
            for feature_name, info in cox_info.items():
                if 'cox_coefficient' in info:
                    cox_lookup[feature_name] = float(info['cox_coefficient'])
            
            print(f"JSON에서 Cox 계수 추출 완료: {len(cox_lookup)}개 특성")
            
            # Pickle 파일로 저장
            with open(cox_file_path, 'wb') as f:
                pickle.dump(cox_lookup, f)
            
            return cox_lookup
        else:
            raise FileNotFoundError(f"Cox 계수 파일이 없습니다: {cox_file_path}")

def split_data_stratified(data, labels, test_size=0.15, val_size=0.15, random_state=42):
    """
    층화 추출로 train/val/test 분할
    
    Args:
        data: 입력 데이터 (DataFrame 또는 tensor)
        labels: 라벨 배열
        test_size: 테스트 비율
        val_size: 검증 비율 (train+val에서의 비율)
        random_state: 랜덤 시드
    
    Returns:
        train_data, val_data, test_data, train_labels, val_labels, test_labels
    """
    
    # Train+Val vs Test 분할
    X_temp, X_test, y_temp, y_test = train_test_split(
        data, labels, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=labels
    )
    
    # Train vs Val 분할
    val_size_adjusted = val_size / (1 - test_size)  # 남은 데이터에서의 비율
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_data_loaders(train_data, val_data, test_data, 
                       train_labels, val_labels, test_labels,
                       batch_size=32, num_workers=0):
    """
    PyTorch DataLoader 생성
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    # TensorDataset 생성
    train_dataset = TensorDataset(train_data, torch.tensor(train_labels, dtype=torch.float32))
    val_dataset = TensorDataset(val_data, torch.tensor(val_labels, dtype=torch.float32))
    test_dataset = TensorDataset(test_data, torch.tensor(test_labels, dtype=torch.float32))
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader