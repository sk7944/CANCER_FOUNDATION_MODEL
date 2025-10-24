# 🧬 Cancer Foundation Model

**멀티오믹스 기반 암 예후 예측 딥러닝 시스템**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)

---

## 🎯 프로젝트 개요

Cancer Foundation Model은 멀티오믹스 데이터를 활용하여 암 환자의 3년 생존 예후를 예측하는 **Hybrid FC-NN + TabTransformer** 시스템입니다.

### 핵심 특징

- **🧬 멀티오믹스 통합**: 5개 오믹스 (Expression, CNV, microRNA, RPPA, Mutation) + Methylation
- **🔬 Missing Modality Learning**: Cox 데이터 있음/없음 모두 처리 가능
- **📊 고차원 데이터 처리**: FC-NN 기반 Dimension Reduction (143K→256, 396K→256)
- **🧠 Cox 회귀계수 활용**: 도메인 지식을 `[측정값, Cox계수]` 쌍으로 모델에 주입
- **⚡ 효율적 아키텍처**: 29.58GB 모델, 48GB GPU 메모리로 훈련 가능
- **📈 TCGA 데이터**: 8,224명 환자의 Pan-Cancer 데이터 활용

---

## 🏗️ 시스템 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│  입력: 환자 멀티오믹스 데이터 (8,224명)                         │
├──────────────────────────────────────────────────────────────┤
│  1. Clinical Categories (5개)                                 │
│     → Categorical Embedding                                   │
│                                                               │
│  2. Cox Omics (143,040 features) [value, cox] 쌍              │
│     → FC-NN (2048→512→256)                                    │
│     → Encoder Dropout (0.3)                                   │
│     → 256-dim representation                                  │
│                                                               │
│  3. Methylation (396,065 CG sites)                            │
│     → FC-NN (4096→1024→256)                                   │
│     → Encoder Dropout (0.3)                                   │
│     → 256-dim representation                                  │
├──────────────────────────────────────────────────────────────┤
│  TabTransformer (dim=128, depth=6, heads=8)                  │
│     • Clinical embedding + Cox 256-dim + Meth 256-dim         │
│     • Self-Attention layers (dropout=0.1)                     │
│     • Cross-modal feature learning                            │
├──────────────────────────────────────────────────────────────┤
│  출력: 3년 생존 예측 (0=생존, 1=사망)                          │
└──────────────────────────────────────────────────────────────┘
```

### Missing Modality Learning

```
환자 구성:
├─ Cox 데이터 있음 (4,504명): Clinical + Cox Omics + Methylation
└─ Cox 데이터 없음 (3,720명): Clinical + [ZERO] + Methylation
   → Total: 8,224명 환자 모두 활용
```

---

## 📊 데이터 및 성능

### 데이터셋

| 구분 | 환자 수 | 특성 수 | 비고 |
|------|---------|---------|------|
| **Cox Omics** | 4,504명 | 71,520 features | Expression, CNV, microRNA, RPPA, Mutation |
| **Methylation** | 8,224명 | 396,065 CG sites | 모든 환자 포함 |
| **암종** | - | 27개 타입 | BRCA, LUAD, COAD, OV, KIRC 등 |

### 모델 상세

| 항목 | 값 |
|------|-----|
| **아키텍처** | Hybrid FC-NN + TabTransformer |
| **Cox Encoder** | 2,929M params (11.18 GB) |
| **Meth Encoder** | 4,509M params (17.20 GB) |
| **TabTransformer** | 212M params (0.81 GB) |
| **Total** | 7,651M params (29.19 GB) |
| **GPU 메모리** | 48GB (RTX A6000) |
| **배치 크기** | 32 |

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
- TCGA 원본 데이터 로드 및 log2 변환
- 암종별 Cox 비례위험 회귀분석 수행
- Cox 계수 룩업 테이블 생성
- 예상 소요 시간: 1-2시간

**주요 출력물:**
```
data/processed/
├── cox_coefficients_*.parquet          # Cox 계수 (암종 × 유전자)
├── processed_*_data.parquet            # log2 변환된 오믹스 데이터
└── processed_clinical_data.parquet     # 임상 데이터
```

#### Step 2: 통합 데이터셋 생성

```bash
./run_integrated_dataset_builder.sh
# 백그라운드 실행, 로그 모니터링: tail -f integrated_dataset_*.log
```

**수행 작업:**
- Cox 계수와 측정값을 `[측정값, Cox계수]` 쌍으로 결합
- 통합 데이터셋 생성 (integrated_table_cox.parquet)
- Methylation 테이블 생성 (methylation_table.parquet)
- Train/Val/Test 분할 (70%/15%/15%)
- 예상 소요 시간: 10-30분

**주요 출력물:**
```
data/processed/
├── integrated_table_cox.parquet    # 4,504 × 143,048 (Cox omics)
├── methylation_table.parquet       # 8,224 × 396,065 (Methylation)
└── train_val_test_splits.json      # 데이터셋 분할 정보
```

### 3. 모델 훈련

```bash
cd ../training

# Hybrid 모델 훈련
bash run_hybrid_training.sh
```

**훈련 설정:**
- Epochs: 100 (early stopping patience=15)
- Batch size: 32
- Learning rate: 1e-4 (AdamW, weight_decay=1e-2)
- Optimizer scheduler: ReduceLROnPlateau (patience=5)
- Loss: BCEWithLogitsLoss

---

## 💡 모델 사용 (추론)

### 신규 환자 데이터 예측

```python
import torch
import pandas as pd
import numpy as np
from src.models.hybrid_fc_tabtransformer import HybridMultiModalModel

# 1. 훈련된 모델 로드
model = HybridMultiModalModel(
    clinical_categories=(10, 3, 8, 4, 5),
    cox_input_dim=143040,    # 71,520 * 2 ([val, cox] 쌍)
    cox_hidden_dims=(2048, 512, 256),
    meth_input_dim=396065,
    meth_hidden_dims=(4096, 1024, 256),
    dim=128, depth=6, heads=8
)

checkpoint = torch.load('results/hybrid_training_YYYYMMDD_HHMMSS/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. 신규 환자 데이터 준비
# - Clinical: [age_group, sex, race, stage, grade] (categorical)
# - Cox Omics: [val, cox] 쌍 형식 (143,040 features)
# - Methylation: beta values (396,065 CG sites)

clinical_cat = torch.tensor([[5, 1, 2, 3, 2]], dtype=torch.long)  # (1, 5)
cox_omics = torch.randn(1, 143040)  # (1, 143040)
methylation = torch.randn(1, 396065)  # (1, 396065)
cox_mask = torch.tensor([[True]], dtype=torch.bool)  # Cox 데이터 있음

# 3. 예측 수행
with torch.no_grad():
    logit, representation = model(clinical_cat, cox_omics, methylation, cox_mask)
    survival_prob = torch.sigmoid(logit)

print(f"3년 생존 확률: {survival_prob.item():.2%}")
print(f"예측 결과: {'생존 가능성 높음' if survival_prob > 0.5 else '생존 가능성 낮음'}")
```

### Cox 데이터가 없는 환자

```python
# Cox 데이터가 없는 경우
clinical_cat = torch.tensor([[5, 1, 2, 3, 2]], dtype=torch.long)
cox_omics = torch.zeros(1, 143040)  # Cox 데이터 없음 → ZERO
methylation = torch.randn(1, 396065)
cox_mask = torch.tensor([[False]], dtype=torch.bool)  # Cox 데이터 없음

with torch.no_grad():
    logit, representation = model(clinical_cat, cox_omics, methylation, cox_mask)
    survival_prob = torch.sigmoid(logit)

print(f"3년 생존 확률: {survival_prob.item():.2%}")
```

---

## 📁 프로젝트 구조

```
CANCER_FOUNDATION_MODEL/
├── data/
│   ├── raw/                           # TCGA 원본 데이터
│   └── processed/                     # 전처리된 데이터
│       ├── integrated_table_cox.parquet         # Cox 통합 테이블
│       ├── methylation_table.parquet            # Methylation 테이블
│       └── train_val_test_splits.json           # 데이터셋 분할
│
├── src/
│   ├── preprocessing/                 # 데이터 전처리
│   │   ├── cox_feature_engineer.py
│   │   ├── integrated_dataset_builder.py
│   │   ├── run_cox_feature_engineer.sh
│   │   └── run_integrated_dataset_builder.sh
│   ├── models/
│   │   ├── hybrid_fc_tabtransformer.py         # Hybrid 모델
│   │   └── [obsolete files in obsolete/]
│   ├── data/
│   │   └── hybrid_dataset.py                   # PyTorch Dataset
│   └── training/
│       ├── train_hybrid.py                     # 훈련 스크립트
│       └── run_hybrid_training.sh              # 훈련 래퍼
│
├── results/                           # 훈련 결과 (timestamped)
├── obsolete/                          # 구버전 코드/모델
└── doc/
    └── CFM.vibe_coding_guide.md       # 개발자 가이드
```

---

## 🔬 기술 세부사항

### Cox 기반 멀티오믹스 파이프라인

```
Step 1: 원본 데이터 → log2 변환
  - Expression: log2(x + 1)
  - CNV: log2(x - min + 1)  [음수 처리]
  - microRNA: log2(x + 1)
  - RPPA: log2(x - min + 1)
  - Mutations: 변환 없음 (impact scores 0-2)
  - Methylation: 변환 없음 (beta values 0-1)

Step 2: Cox 회귀분석
  - 암종별로 각 유전자에 대해 Cox 비례위험 회귀분석 수행
  - Cox 계수 룩업 테이블 생성

Step 3: [측정값, Cox계수] 쌍 생성
  - 각 유전자마다 2개 값으로 저장:
    - gene_val: log2 변환된 측정값
    - gene_cox: Cox 회귀계수
  - ⚠️ 중요: 곱셈 아님! 별도 2개 값으로 유지

Step 4: Hybrid FC-NN + TabTransformer
  - FC-NN으로 Dimension Reduction
  - TabTransformer로 Cross-modal Learning
  - 3년 생존 예측
```

### 입력 데이터 형식

**⚠️ 매우 중요**: 모델은 측정값과 Cox계수를 **곱하지 않고** 별도 2개 값으로 입력받습니다.

```python
# ❌ 잘못된 방법
input = [gene1_value * gene1_cox, gene2_value * gene2_cox, ...]

# ✅ 올바른 방법
input = [
    gene1_value, gene1_cox,  # 2개 값 쌍
    gene2_value, gene2_cox,
    gene3_value, gene3_cox,
    ...
]
```

### 모델 아키텍처 상세

```python
HybridMultiModalModel(
    clinical_categories=(10, 3, 8, 4, 5),     # Clinical categorical features
    cox_input_dim=143040,                      # 71,520 * 2 ([val, cox] 쌍)
    cox_hidden_dims=(2048, 512, 256),         # Cox FC-NN layers
    meth_input_dim=396065,                     # Methylation CG sites
    meth_hidden_dims=(4096, 1024, 256),       # Meth FC-NN layers
    dim=128,                                   # TabTransformer embedding dim
    depth=6,                                   # Transformer layers
    heads=8,                                   # Attention heads
    attn_dropout=0.1,
    ff_dropout=0.1,
    encoder_dropout=0.3,
    dim_out=1                                  # Binary classification
)
```

**입력:**
- `clinical_cat`: (batch, 5) - Categorical features
- `cox_omics`: (batch, 143040) - Cox [val, cox] 쌍
- `methylation`: (batch, 396065) - Beta values
- `cox_mask`: (batch,) - Cox 데이터 유무 (True/False)

**출력:**
- `logit`: (batch, 1) - 3년 생존 예측 로짓
- `representation`: (batch, dim) - 중간 임베딩

---

## 📈 성능 및 검증

### 훈련 환경

- **GPU**: NVIDIA RTX A6000 (48GB)
- **모델 크기**: 29.19 GB
- **훈련 시간**: ~6-8시간 (100 epochs)
- **배치 크기**: 32

### 훈련 설정

- **Optimizer**: AdamW (weight_decay=1e-2)
- **Learning Rate**: 1e-4 (ReduceLROnPlateau)
- **Loss**: BCEWithLogitsLoss
- **Early Stopping**: 15 epochs patience

---

## ⚠️ 주의사항

### 1. [측정값, Cox계수] 쌍 형식 (매우 중요!)

- ❌ **잘못된 방법**: `value * cox_coefficient` (곱셈)
- ✅ **올바른 방법**: `[value, cox_coefficient]` (2개 값 스택)

### 2. log2 변환 일관성

- Expression, CNV, microRNA, RPPA: `log2(x + 1)` 필수
- CNV, RPPA 음수 처리: `log2(x - min + 1)`
- Methylation, Mutations: 변환 없음
- **추론 시 동일한 변환 적용 필수**

### 3. Missing Modality 처리

- Cox 데이터 없는 환자: `cox_omics`를 ZERO로, `cox_mask`를 False로 설정
- 모델은 자동으로 Methylation만 사용하여 예측

### 4. GPU 메모리

- 48GB GPU 필요 (RTX A6000)
- 배치 크기 32 권장
- 메모리 부족 시 배치 크기 줄이기

### 5. 데이터 정렬

- 모든 데이터셋의 환자 ID 정렬 확인
- 특성 순서 일치 필수

---

## 🔧 Troubleshooting

### 데이터 전처리 버그 (2025-10-24 수정 완료)

#### 문제: 37,446개 중복 컬럼명

초기 구현에서 `integrated_dataset_builder.py`의 feature naming에 버그가 있었습니다:

```python
# ❌ 버그 있는 코드 (수정 전)
enhanced_features[f"{feature}_value"] = measured_values[feature]
enhanced_features[f"{feature}_cox"] = cox_coef_mean[feature]
# 문제: omics_type 접두사 없음 → 같은 유전자가 여러 omics에 있으면 중복!
```

**근본 원인**: 동일한 유전자(예: BRCA1, TP53)가 여러 omics 타입(Expression, CNV, Mutations 등)에 존재할 때, omics 타입 구분 없이 feature명을 생성하여 37,446개 중복 컬럼 발생.

#### 해결 방법

```python
# ✅ 수정된 코드 (2025-10-24)
enhanced_features[f"{omics_type}_{feature}_val"] = measured_values[feature]
enhanced_features[f"{omics_type}_{feature}_cox"] = cox_coef_mean[feature]
```

**수정 내용:**
1. `omics_type` 접두사 추가: `Expression_`, `CNV_`, `Mutations_` 등
2. `_value` → `_val`로 변경 (일관성)
3. 올바른 형식: `Expression_BRCA1|672_val`, `Expression_BRCA1|672_cox`

**영향받는 파일:**
- `src/preprocessing/integrated_dataset_builder.py:180-183`
- `data/processed/integrated_table_cox.parquet` (재생성 필요)
- `data/processed/cox_feature_info.json` (자동 수정됨)

### Feature Naming Convention

**올바른 feature 명명 규칙:**

```
{OmicsType}_{GeneSymbol}|{EntrezID}_{val|cox}

예시:
- Expression_TP53|7157_val
- Expression_TP53|7157_cox
- CNV_BRCA1|672_val
- CNV_BRCA1|672_cox
- Mutations_EGFR|1956_val
- Mutations_EGFR|1956_cox
```

**주의사항:**
- Expression omics는 `GeneSymbol|EntrezID` 형식 사용
- CNV, Mutations는 `GeneSymbol`만 사용 (Entrez ID 없음)
- microRNA는 miRNA 이름 사용 (예: `microRNA_hsa-mir-21_val`)
- RPPA는 protein 이름 사용 (예: `RPPA_p53_val`)

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

## 🙏 감사의 글

- **TCGA Research Network**: 포괄적인 암 유전체학 데이터 제공
- **PyTorch Team**: 딥러닝 프레임워크
- **lucidrains**: 우수한 tab-transformer-pytorch 구현
- **암 연구 커뮤니티**: 도메인 전문 지식 및 검증

---

**🔬 AI를 통한 암 연구 발전**

*Built with ❤️ for the cancer research community*
