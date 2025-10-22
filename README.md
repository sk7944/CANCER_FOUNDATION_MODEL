# 🧬 Cancer Foundation Model

**설명 가능한 AI 기반 암 예후 예측 시스템**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)

---

## 🎯 프로젝트 개요

Cancer Foundation Model은 멀티오믹스 데이터를 활용하여 암 환자의 생존 예후를 예측하는 딥러닝 시스템입니다.

### 핵심 특징

- **🧬 멀티오믹스 통합**: Expression, CNV, microRNA, RPPA, Mutation 5개 데이터 통합
- **📊 고차원 메틸레이션**: 396,065개 프로브 데이터 처리 (샤딩 예정)
- **🧠 설명 가능한 AI**: Cox 회귀계수 기반 특성 가중치 및 Attention 메커니즘
- **⚡ 높은 성능**: Test AUC **0.8495** (3년 생존 예측)
- **🔬 TCGA 데이터**: 4,504명 환자의 Pan-Cancer 데이터로 훈련

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│  입력: 환자 멀티오믹스 데이터                              │
├─────────────────────────────────────────────────────────┤
│  1. 임상 데이터 (나이, 성별, 병기 등)                      │
│     → 범주형 특성으로 인코딩                               │
│                                                         │
│  2. 멀티오믹스 데이터 (5개 오믹스)                        │
│     → [측정값, Cox계수] 쌍으로 변환                       │
│     예: [BRCA1_발현량: 5.2, BRCA1_Cox계수: 0.8]          │
│                                                         │
│  3. 메틸레이션 데이터 (396K probes)                       │
│     → 샤딩 후 독립 모델로 처리 (예정)                      │
├─────────────────────────────────────────────────────────┤
│  CoxTabTransformer (TabTransformer 기반)                │
│     • 임상 범주형 특성 임베딩                              │
│     • 멀티오믹스 [값, Cox계수] 쌍 처리                     │
│     • Self-Attention 레이어로 특성 관계 학습               │
│     • 출력: 256-dim representation                      │
├─────────────────────────────────────────────────────────┤
│  출력: 3년 생존 예측 (0-1 확률)                           │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 데이터 및 성능

### 훈련 데이터
- **데이터셋**: TCGA Pan-Cancer
- **환자 수**: 4,504명
- **암종**: 27개 타입 (BRCA, LUAD, COAD, OV, KIRC 등)
- **오믹스 특성**: 71,520개 (5개 오믹스 합산)
- **메틸레이션**: 396,065 probes (별도 처리)

### 현재 성능
| 모델 | Test AUC | 환자 수 | 특성 수 | 상태 |
|------|---------|---------|---------|------|
| **CoxTabTransformer** | **0.8495** | 4,504 | 71,520 | ✅ 완료 |
| MethylationTabTransformer | - | 8,224 | 396,065 | 🔄 샤딩 필요 |

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/your-org/CANCER_FOUNDATION_MODEL.git
cd CANCER_FOUNDATION_MODEL

# 의존성 설치
pip install -r requirements.txt
pip install tab-transformer-pytorch lifelines
```

### 2. 데이터 전처리 (한 번만 실행)

#### Step 1: Cox 회귀분석 실행
```bash
cd src/preprocessing
./run_cox_feature_engineer.sh
# 백그라운드 실행, 로그 모니터링: tail -f cox_analysis_*.log
```

**수행 작업:**
- TCGA 원본 데이터 로드 (`data/raw/`)
- **log2 변환 적용** (데이터 정규화):
  - Expression: `log2(x + 1)`
  - CNV: `log2(x - min + 1)` (음수 처리 포함)
  - microRNA: `log2(x + 1)`
  - RPPA: `log2(x - min + 1)` (음수 처리 포함)
  - ⚠️ Methylation: 변환 없음 (beta values 0-1)
  - ⚠️ Mutations: 변환 없음 (impact scores 0-2)
- 각 암종별로 5개 오믹스 타입에 대해 Cox 비례위험 회귀분석 수행
- Cox 계수 룩업 테이블 생성 (`cox_coefficients_*.parquet`)
- 전처리된 오믹스 데이터 저장 (`processed_*_data.parquet`)
- 예상 소요 시간: 1-2시간

**주요 출력물:**
```
data/processed/
├── cox_coefficients_expression.parquet    # Expression Cox 계수
├── cox_coefficients_cnv.parquet           # CNV Cox 계수
├── cox_coefficients_microrna.parquet      # microRNA Cox 계수
├── cox_coefficients_rppa.parquet          # RPPA Cox 계수
├── cox_coefficients_mutations.parquet     # Mutation Cox 계수
├── processed_expression_data.parquet      # log2 변환된 Expression
├── processed_cnv_data.parquet             # log2 변환된 CNV
├── processed_microrna_data.parquet        # log2 변환된 microRNA
├── processed_rppa_data.parquet            # log2 변환된 RPPA
├── processed_mutations_data.parquet       # Mutation impact scores
├── methylation_data_for_tabtransformer.parquet  # Methylation beta values
└── processed_clinical_data.parquet        # 임상 데이터
```

#### Step 2: 통합 데이터셋 생성
```bash
./run_integrated_dataset_builder.sh
# 백그라운드 실행, 로그 모니터링: tail -f integrated_dataset_*.log
```

**수행 작업:**
- Cox 계수와 오믹스 측정값을 **[측정값, Cox계수] 쌍으로 결합**
- **중요**: 측정값과 Cox계수를 곱하지 않고 별도 2개 값으로 유지
- 5개 오믹스 데이터를 하나의 통합 테이블로 병합
- 통합 데이터셋 저장 (`integrated_table_cox.parquet`)
- Train/Validation/Test 분할 (70%/15%/15%)
- 예상 소요 시간: 10-30분

**주요 출력물:**
```
data/processed/
├── integrated_table_cox.parquet    # 🔥 핵심 훈련 파일 (4,504 × 32,762)
│   # 각 유전자마다 _val과 _cox 2개 컬럼 포함
│   # 예: Mutations_BRCA1_val, Mutations_BRCA1_cox
├── train_val_test_splits.json      # 데이터셋 분할 정보
└── integrated_dataset_summary.json # 통계 요약
```

### 3. 모델 훈련

```bash
cd ../training

# CoxTabTransformer 훈련
python train_tabtransformer.py \
    --model cox \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --data_dir ../../data/processed \
    --results_dir ../../results
```

**훈련 옵션:**
- `--ensemble`: 여러 seed로 앙상블 모델 훈련
- `--n_seeds 5`: 앙상블 시드 개수
- `--resume_from`: 체크포인트에서 재개

---

## 💡 신규 환자 데이터 예측

### 단일 모델 예측 (Single Seed)

```python
from src.models.cox_tabtransformer import CoxTabTransformer
from src.utils.tabtransformer_utils import prepare_cox_data, prepare_clinical_data
import pandas as pd
import numpy as np
import torch

# ========================================
# Step 1: 훈련된 모델 로드 (특정 seed)
# ========================================
model = CoxTabTransformer(
    clinical_categories=(10, 3, 8, 4, 5),  # 훈련 시 사용한 vocab sizes
    num_omics_features=71520,
    dim=64, depth=4, heads=8
)
# 특정 시드 모델 로드 (예: seed_42)
checkpoint = torch.load('src/training/checkpoints/seed_42/best_cox_tabtransformer.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ========================================
# Step 2: 신규 환자 원본 데이터 로드
# ========================================
# ⚠️ 중요: 원본 데이터 (log2 변환 전)를 로드해야 합니다!
new_patient_expression = pd.read_csv('new_patient_expression.csv', index_col=0)
new_patient_cnv = pd.read_csv('new_patient_cnv.csv', index_col=0)
new_patient_mirna = pd.read_csv('new_patient_mirna.csv', index_col=0)
new_patient_rppa = pd.read_csv('new_patient_rppa.csv', index_col=0)
new_patient_mutations = pd.read_csv('new_patient_mutations.csv', index_col=0)
new_patient_clinical = pd.read_csv('new_patient_clinical.csv', index_col=0)

# ========================================
# Step 3: log2 변환 적용 (훈련 시와 동일한 방법)
# ========================================
# Expression: log2(x + 1)
expression_log2 = np.log2(new_patient_expression + 1)

# CNV: log2(x - min + 1) for negative handling
cnv_min = new_patient_cnv.min().min()
cnv_log2 = np.log2(new_patient_cnv - cnv_min + 1) if cnv_min < 0 else np.log2(new_patient_cnv + 1)

# microRNA: log2(x + 1)
mirna_log2 = np.log2(new_patient_mirna + 1)

# RPPA: log2(x - min + 1) for negative handling
rppa_min = new_patient_rppa.min().min()
rppa_log2 = np.log2(new_patient_rppa - rppa_min + 1) if rppa_min < 0 else np.log2(new_patient_rppa + 1)

# Mutations: NO transformation (already impact scores 0-2)
mutations_scores = new_patient_mutations

# ========================================
# Step 4: Cox 계수 로드 (환자의 암종에 맞춰)
# ========================================
patient_id = 'TCGA-XX-XXXX'
patient_cancer_type = new_patient_clinical.loc[patient_id, 'acronym']  # 예: 'BRCA'

# 통합 테이블의 특성 순서 로드 (훈련 시와 동일한 순서 필수!)
integrated_data = pd.read_parquet('data/processed/integrated_table_cox.parquet')
feature_columns = [col for col in integrated_data.columns if col.endswith('_val')]

# ========================================
# Step 5: [측정값, Cox계수] 쌍 생성 (특성 순서 동일하게!)
# ========================================
omics_values = []
for feat_col in feature_columns:
    # feat_col 예: 'Expression_BRCA1_val'
    cox_col = feat_col.replace('_val', '_cox')

    # 훈련된 데이터에서 이 특성의 Cox 계수 가져오기
    cox_value = integrated_data[cox_col].iloc[0]  # 모든 환자가 동일한 Cox 계수 사용

    # 환자의 측정값 가져오기 (로그 변환된 값)
    omics_type, feature_name = feat_col.split('_', 1)[0], '_'.join(feat_col.split('_')[1:-1])

    if omics_type == 'Expression':
        measured_value = expression_log2.loc[feature_name, patient_id]
    elif omics_type == 'CNV':
        measured_value = cnv_log2.loc[feature_name, patient_id]
    elif omics_type == 'microRNA':
        measured_value = mirna_log2.loc[feature_name, patient_id]
    elif omics_type == 'RPPA':
        measured_value = rppa_log2.loc[feature_name, patient_id]
    elif omics_type == 'Mutations':
        measured_value = mutations_scores.loc[feature_name, patient_id]

    omics_values.extend([measured_value, cox_value])  # [val, cox] 쌍

omics_tensor = torch.tensor(omics_values, dtype=torch.float32).unsqueeze(0)

# ========================================
# Step 6: 임상 데이터 인코딩
# ========================================
clinical_encoded, _, _, _ = prepare_clinical_data(new_patient_clinical)

# ========================================
# Step 7: 예측 수행
# ========================================
with torch.no_grad():
    survival_logit, representation = model(clinical_encoded.long(), omics_tensor)
    survival_prob = torch.sigmoid(survival_logit)

print(f"환자 ID: {patient_id}")
print(f"암종: {patient_cancer_type}")
print(f"3년 생존 확률: {survival_prob.item():.2%}")
print(f"예측 결과: {'생존 가능성 높음' if survival_prob > 0.5 else '생존 가능성 낮음'}")
```

### 앙상블 예측 (Multiple Seeds)

더 안정적인 예측을 위해 여러 시드 모델의 평균을 사용할 수 있습니다.

```python
import glob

# 모든 시드 모델 로드
seed_dirs = glob.glob('src/training/checkpoints/seed_*')
ensemble_predictions = []

for seed_dir in seed_dirs:
    model = CoxTabTransformer(
        clinical_categories=(10, 3, 8, 4, 5),
        num_omics_features=71520,
        dim=64, depth=4, heads=8
    )
    checkpoint = torch.load(f'{seed_dir}/best_cox_tabtransformer.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        survival_logit, _ = model(clinical_encoded.long(), omics_tensor)
        survival_prob = torch.sigmoid(survival_logit)
        ensemble_predictions.append(survival_prob.item())

# 앙상블 평균
mean_prob = np.mean(ensemble_predictions)
std_prob = np.std(ensemble_predictions)

print(f"앙상블 예측 (평균): {mean_prob:.2%} ± {std_prob:.2%}")
print(f"개별 모델 예측: {[f'{p:.2%}' for p in ensemble_predictions]}")
```

### 입력 데이터 형식

#### 멀티오믹스 데이터 (`new_patient_omics.csv`)
```csv
gene_id,patient_1,patient_2
BRCA1|672,5.234,4.567
TP53|7157,6.789,5.432
MYC|4609,7.123,6.890
...
```

#### 임상 데이터 (`new_patient_clinical.csv`)
```csv
patient_id,age_at_initial_pathologic_diagnosis,gender,acronym,pathologic_stage
patient_1,55,FEMALE,BRCA,Stage II
patient_2,62,MALE,LUAD,Stage III
```

### 중요 사항

1. **log2 변환 필수**: 원본 데이터에 반드시 동일한 log2 변환 적용
   - Expression, CNV, microRNA, RPPA: `log2(x + 1)` (음수는 `log2(x - min + 1)`)
   - Methylation, Mutations: 변환 없음

2. **[측정값, Cox계수] 쌍 형식**:
   - ❌ 곱셈 아님: `value * cox_coefficient`
   - ✅ 2개 값 쌍: `[value, cox_coefficient]`

3. **특성 순서 일치**: `integrated_table_cox.parquet`의 컬럼 순서와 동일하게 정렬

4. **Cox 계수 매칭**: 환자의 암종(cancer_type)에 해당하는 Cox 계수만 사용

5. **모델 체크포인트**: `src/training/checkpoints/seed_XX/best_cox_tabtransformer.pth` 형식

---

## 📁 프로젝트 구조

```
CANCER_FOUNDATION_MODEL/
├── data/
│   ├── raw/                           # TCGA 원본 데이터
│   │   ├── *_expression_whitelisted.tsv
│   │   ├── CNV.*_whitelisted.tsv
│   │   ├── *_miRNASeq_whitelisted.tsv
│   │   ├── *_RPPA_whitelisted.tsv
│   │   ├── *_whitelisted.maf.gz
│   │   ├── *_Methylation450_whitelisted.tsv
│   │   └── clinical_*_with_followup.tsv
│   └── processed/                     # 전처리된 데이터
│       ├── cox_coefficients_*.parquet           # Cox 회귀계수 룩업 테이블
│       ├── processed_*_data.parquet             # log2 변환된 오믹스 데이터
│       ├── integrated_table_cox.parquet         # 🔥 핵심 훈련 파일
│       ├── train_val_test_splits.json           # 데이터셋 분할 정보
│       └── processed_clinical_data.parquet      # 임상 데이터
│
├── src/
│   ├── preprocessing/                 # 데이터 전처리 스크립트
│   │   ├── cancer_multiomics_dataset.py      # PyTorch Dataset 클래스
│   │   ├── cox_feature_engineer.py           # Cox 회귀분석 실행
│   │   ├── integrated_dataset_builder.py     # 데이터셋 통합
│   │   ├── run_cox_feature_engineer.sh       # Cox 분석 래퍼
│   │   └── run_integrated_dataset_builder.sh # 빌더 래퍼
│   ├── models/
│   │   ├── cox_tabtransformer.py             # 멀티오믹스 모델
│   │   └── methylation_tabtransformer.py     # 메틸레이션 모델
│   ├── training/
│   │   └── train_tabtransformer.py           # 훈련 스크립트
│   └── utils/
│       ├── tabtransformer_utils.py           # 전처리 유틸리티
│       ├── feature_converter.py              # 추론용 변환
│       └── user_data_pipeline.py             # 추론용 파이프라인
│
├── notebooks/                         # 분석 노트북
├── results/                           # 훈련 결과
├── doc/
│   └── CFM.vibe_coding_guide.md       # 개발자 가이드
└── README.md                          # 사용자 가이드 (이 파일)
```

---

## 🔬 기술 세부사항

### Cox 기반 멀티오믹스 파이프라인 전체 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: 원본 TCGA 데이터 (data/raw/)                              │
├─────────────────────────────────────────────────────────────────┤
│ Expression, CNV, microRNA, RPPA, Mutations 데이터 로드            │
│ - 환자 x 유전자 형태의 매트릭스                                    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: log2 변환 (cox_feature_engineer.py)                     │
├─────────────────────────────────────────────────────────────────┤
│ Expression:  log2(x + 1)                                        │
│ CNV:         log2(x - min + 1)  [음수 처리]                      │
│ microRNA:    log2(x + 1)                                        │
│ RPPA:        log2(x - min + 1)  [음수 처리]                      │
│ Mutations:   변환 없음 (impact scores 0-2)                       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Cox 회귀분석 (cox_feature_engineer.py)                  │
├─────────────────────────────────────────────────────────────────┤
│ 암종별로 Cox 비례위험 회귀분석 수행                                │
│ - 엔드포인트: 3년 생존 여부 (OS_3yr)                              │
│ - 출력: cox_coefficients_*.parquet                              │
│         (유전자 × 암종별 Cox 계수)                                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: [측정값, Cox계수] 쌍 생성 (integrated_dataset_builder.py)│
├─────────────────────────────────────────────────────────────────┤
│ for gene in genes:                                              │
│     enhanced_features[f"{gene}_val"] = log2_transformed_value   │
│     enhanced_features[f"{gene}_cox"] = cox_coefficient          │
│                                                                 │
│ ⚠️ 중요: 곱셈 아님! 별도 2개 컬럼으로 유지                         │
│ 출력: integrated_table_cox.parquet (4,504 × 32,762)             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 5: Tensor 변환 (tabtransformer_utils.py:41-42)            │
├─────────────────────────────────────────────────────────────────┤
│ paired_data = torch.stack([values, cox], dim=2)                 │
│ # Shape: (batch, 71520, 2)                                      │
│                                                                 │
│ flattened = paired_data.view(batch, -1)                         │
│ # Shape: (batch, 143040) = 71520 * 2                            │
│                                                                 │
│ 최종 입력: [gene1_val, gene1_cox, gene2_val, gene2_cox, ...]   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 6: CoxTabTransformer 모델 (cox_tabtransformer.py)         │
├─────────────────────────────────────────────────────────────────┤
│ TabTransformer(                                                 │
│     num_continuous = 71520 * 2,  # [val, cox] 쌍이므로 *2       │
│     ...                                                         │
│ )                                                               │
│ → Transformer layers → 3년 생존 예측                             │
└─────────────────────────────────────────────────────────────────┘
```

### 입력 데이터 형식 ([측정값, Cox계수] 쌍)

**⚠️ 매우 중요**: 모델은 측정값과 Cox계수를 **곱하지 않고** 별도의 2개 값으로 입력받습니다.

```python
# 잘못된 방법 ❌
input = [gene1_value * gene1_cox, gene2_value * gene2_cox, ...]  # 곱한 값 X

# 올바른 방법 ✅
input = [
    gene1_value, gene1_cox,  # 2개 값 쌍
    gene2_value, gene2_cox,  # 2개 값 쌍
    gene3_value, gene3_cox,  # 2개 값 쌍
    ...
]
```

**구현 세부사항:**
```python
# src/utils/tabtransformer_utils.py:41-42
paired_data = torch.stack([omics_tensor, cox_expanded], dim=2)  # (batch, features, 2)
flattened = paired_data.view(batch_size, -1)  # (batch, features*2)

# src/models/cox_tabtransformer.py:31
self.base_transformer = TabTransformer(
    num_continuous=num_omics_features * 2,  # [측정값, Cox계수] 쌍이므로 *2
    ...
)
```

**데이터 흐름 예시:**
```python
# 입력 예시 (BRCA1 유전자)
raw_value = 1234.5           # 원본 Expression 값
log2_value = log2(1234.5 + 1) = 10.27  # log2 변환
cox_coef = 0.345             # BRCA 암종의 BRCA1 Cox 계수

# integrated_table_cox.parquet에 저장:
# Expression_BRCA1_val: 10.27
# Expression_BRCA1_cox: 0.345

# 모델 입력 텐서:
# [..., 10.27, 0.345, ...] ← 2개 값이 연속으로 배치
```

### 모델 아키텍처

```python
CoxTabTransformer(
    clinical_categories=(10, 3, 8, 4, 5),     # 범주형 임상 특성
    num_omics_features=71520,                 # 5개 오믹스 특성 합계
    dim=64,                                   # 임베딩 차원
    depth=4,                                  # Transformer 레이어 수
    heads=8,                                  # Attention 헤드 수
    attn_dropout=0.3,                         # Attention dropout
    ff_dropout=0.3                            # Feedforward dropout
)
```

**입력:**
- `clinical_categorical`: (batch_size, num_clinical_features)
- `omics_continuous`: (batch_size, num_omics_features * 2)

**출력:**
- `survival_logit`: (batch_size, 1) - 3년 생존 예측 로짓
- `representation`: (batch_size, 256) - 중간 임베딩 (해석용)

---

## 📈 성능 및 검증

### 훈련 환경
- **GPU**: NVIDIA RTX A6000 (48GB)
- **훈련 시간**: ~2시간 (50 epochs)
- **메모리 사용량**: ~4GB

### 훈련 설정
- **Optimizer**: AdamW (weight_decay=1e-2)
- **Learning Rate**: 1e-4 (초기값), ReduceLROnPlateau 스케줄러
- **Batch Size**: 32
- **Loss**: BCEWithLogitsLoss (pos_weight=1.2)
- **Early Stopping**: 10 epochs patience

### 검증 결과
- **Best Validation AUC**: 0.85+
- **Test AUC**: 0.8495
- **Test Accuracy**: 0.77+

---

## 🛣️ 로드맵

### ✅ Phase 1: 데이터 준비 (완료)
- [x] TCGA 데이터 다운로드 및 정제
- [x] Cox 회귀분석 (5개 오믹스)
- [x] 특성 공학 및 데이터 통합

### ✅ Phase 2: 멀티오믹스 모델 (완료)
- [x] CoxTabTransformer 구현
- [x] 훈련 파이프라인 구축
- [x] 성능 검증 (Test AUC 0.8495)

### 🔄 Phase 2: 메틸레이션 모델 (진행 중)
- [ ] 샤딩 전략 구현 (396K probes)
- [ ] 샤드별 모델 훈련
- [ ] Fusion layer 구현

### ⏳ Phase 3: 병리영상 모델 (대기)
- [ ] WSI 전처리
- [ ] Swin Transformer 구현
- [ ] MIL(Multiple Instance Learning) 적용

### ⏳ Phase 4: 멀티모달 융합 (대기)
- [ ] Cross-modal Attention
- [ ] LLM 파인튜닝
- [ ] 설명 가능성 시각화

---

## 📚 참고 문헌

### 데이터
- **TCGA Research Network** - The Cancer Genome Atlas Pan-Cancer Analysis Project
- [TCGA Data Portal](https://portal.gdc.cancer.gov/)

### 방법론
- **TabTransformer** - Huang et al., "TabTransformer: Tabular Data Modeling Using Contextual Embeddings"
- **Cox Regression** - Cox, D. R. (1972). "Regression models and life-tables"

### 구현
- **tab-transformer-pytorch** - [lucidrains/tab-transformer-pytorch](https://github.com/lucidrains/tab-transformer-pytorch)
- **lifelines** - Cox regression library for Python

---

## 🤝 기여 및 문의

### 기여 방법
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### 문의
- **이슈 등록**: GitHub Issues
- **이메일**: your-email@example.com

---

## 📄 라이선스

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 감사의 글

- **TCGA Research Network**: 포괄적인 암 유전체학 데이터 제공
- **PyTorch Team**: 딥러닝 프레임워크
- **lucidrains**: 우수한 tab-transformer-pytorch 구현
- **암 연구 커뮤니티**: 도메인 전문 지식 및 검증

---

**🔬 AI를 통한 암 연구 발전**

*Built with ❤️ for the cancer research community*
