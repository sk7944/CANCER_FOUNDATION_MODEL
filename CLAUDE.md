# CLAUDE.md - Cancer Foundation Model

AI 어시스턴트가 이 프로젝트를 효과적으로 이해하고 작업할 수 있도록 작성된 가이드입니다.

---

## 프로젝트 개요

**Cancer Foundation Model** - TCGA 멀티오믹스 데이터를 활용한 암 환자 3년 생존 예측 딥러닝 시스템

### 핵심 목표
- **입력**: 5개 오믹스(Expression, CNV, microRNA, RPPA, Mutation) + Methylation + 임상정보
- **출력**: 3년 생존 예측 (0=생존, 1=사망)
- **특징**: Missing modality learning (Cox 또는 Methylation 데이터 없는 환자도 예측 가능)

### 데이터 규모

| 항목 | 수치 |
|------|------|
| **총 환자 수 (Union)** | **8,577명** (27개 암종) |
| Cox + Methylation 둘 다 | 4,151명 |
| Cox만 보유 | 353명 |
| Methylation만 보유 | 4,073명 |
| Cox 특성 수 | 132,098 (66,049 genes × 2) |
| Methylation CG sites | 396,065 |

> **중요**: 총 환자 수는 8,577명 (Cox ∪ Methylation). 이전 8,224명은 Methylation만 기준이었음.

---

## 현재 진행 상황

```
Phase 1: 데이터 전처리      ✅ 100% 완료
Phase 2-A: 멀티오믹스 모델   ✅ 100% 완료 (코드 완성, 훈련 미실행)
Phase 2-B: 병리영상 (WSI)   ❌ 0% (다음 단계)
Phase 3: 멀티모달 융합       ❌ 대기 중
```

**중요**: `results/`와 `logs/` 디렉토리가 비어있음 → 실제 훈련은 아직 미실행 상태

---

## 디렉토리 구조

```
CANCER_FOUNDATION_MODEL/
├── data/
│   ├── raw/                    # TCGA 원본 (48GB)
│   └── processed/              # 전처리 완료 (31GB)
│       ├── integrated_table_cox.parquet    # 4,504 × ~132,106 (생성 필요)
│       ├── methylation_table.parquet       # 8,224 × ~396,072 (생성 필요)
│       ├── train_val_test_splits.json      # 70/15/15 분할 (생성 필요)
│       ├── processed_clinical_data.parquet                # Cox 임상 (4,504명)
│       └── processed_clinical_data_for_methylation.parquet # Meth 임상 (8,224명)
│
├── src/
│   ├── preprocessing/
│   │   ├── cox_feature_engineer.py         # Cox 회귀분석
│   │   ├── integrated_dataset_builder.py   # 통합 데이터셋 생성
│   │   └── run_integrated_dataset_builder.sh
│   ├── models/
│   │   └── hybrid_fc_tabtransformer.py     # 메인 모델
│   ├── data/
│   │   └── hybrid_dataset.py               # PyTorch Dataset
│   └── training/
│       ├── train_hybrid.py                 # 훈련 스크립트
│       └── run_hybrid_training.sh          # 실행 래퍼
│
├── results/          # 훈련 결과 저장 (현재 비어있음)
├── logs/             # 훈련 로그 (현재 비어있음)
├── obsolete/         # 구버전 코드 (731MB, 무시)
└── doc/              # 문서
```

---

## 핵심 파일 위치

### 모델 관련
- **모델 정의**: `src/models/hybrid_fc_tabtransformer.py`
- **데이터셋**: `src/data/hybrid_dataset.py`
- **훈련 스크립트**: `src/training/train_hybrid.py`

### 전처리 관련
- **Cox 분석**: `src/preprocessing/cox_feature_engineer.py`
- **데이터셋 통합**: `src/preprocessing/integrated_dataset_builder.py`

### 데이터
- **Cox omics**: `data/processed/integrated_table_cox.parquet`
- **Methylation**: `data/processed/methylation_table.parquet`
- **분할 정보**: `data/processed/train_val_test_splits.json`
- **Cox 임상**: `data/processed/processed_clinical_data.parquet` (4,504명)
- **Meth 임상**: `data/processed/processed_clinical_data_for_methylation.parquet` (8,224명)

---

## 모델 아키텍처

```
HybridMultiModalModel (총 ~29 GB, ~7.6B params)
│
├─ CoxOmicsEncoder
│  └─ FC: 132,098 → 2048 → 512 → 256
│
├─ MethylationEncoder
│  └─ FC: 396,065 → 4096 → 1024 → 256
│
└─ TabTransformer
   ├─ Clinical categorical embeddings (5개)
   ├─ Continuous inputs: cox_256 + meth_256 = 512
   └─ Output: 3-year survival (binary)
```

### 모델 파라미터
```python
HybridMultiModalModel(
    clinical_categories=(10, 3, 8, 4, 5),  # age, sex, race, stage, grade
    cox_input_dim=132098,                   # 66,049 features × 2
    cox_hidden_dims=(2048, 512, 256),
    meth_input_dim=396065,
    meth_hidden_dims=(4096, 1024, 256),
    dim=128, depth=6, heads=8,
    attn_dropout=0.1, ff_dropout=0.1, encoder_dropout=0.3
)
```

### Forward 호출
```python
# cox_mask: Cox 데이터 유무
# meth_mask: Methylation 데이터 유무
logits, features = model(clinical_cat, cox_omics, methylation, cox_mask, meth_mask)
```

---

## Missing Modality Learning

### 환자 구성 (8,577명)

```
┌─────────────────────────────────────────────────────┐
│           전체 환자: 8,577명 (Union)                  │
├─────────────────────────────────────────────────────┤
│  Cox=✅ Meth=✅ (둘 다):    4,151명                   │
│  Cox=✅ Meth=❌ (Cox만):     353명                   │
│  Cox=❌ Meth=✅ (Meth만):  4,073명                   │
└─────────────────────────────────────────────────────┘
```

### 처리 방식

```python
# Missing Cox
if cox_mask[i] == False:
    cox_encoded[i] = ZEROS(256)  # Cox 임베딩을 0으로

# Missing Methylation
if meth_mask[i] == False:
    meth_encoded[i] = ZEROS(256)  # Meth 임베딩을 0으로
```

---

## 데이터 형식 규칙

### 1. [value, cox] 쌍 형식 (가장 중요!)

```python
# ❌ 잘못된 방법 - 곱셈
input = [gene1_value * gene1_cox, gene2_value * gene2_cox, ...]

# ✅ 올바른 방법 - 2개 값을 별도로 유지
input = [gene1_val, gene1_cox, gene2_val, gene2_cox, ...]
```

### 2. Feature 명명 규칙

```
{OmicsType}_{GeneSymbol}|{EntrezID}_{val|cox}

예시:
- Expression_TP53|7157_val
- Expression_TP53|7157_cox
- CNV_BRCA1_val
- Mutations_EGFR_val
- microRNA_hsa-mir-21_val
- RPPA_p53_val
```

### 3. log2 변환 규칙

| 오믹스 타입 | 변환 |
|-------------|------|
| Expression | `log2(x + 1)` |
| CNV | `log2(x - min + 1)` (음수 처리) |
| microRNA | `log2(x + 1)` |
| RPPA | `log2(x - min + 1)` (음수 처리) |
| Mutations | 변환 없음 (0-2) |
| Methylation | 변환 없음 (0-1) |

---

## 임상 데이터 처리

### 두 파일 병합

`hybrid_dataset.py`가 자동으로 두 임상 파일을 병합하여 8,577명 전체를 커버:

```python
# 1차: methylation용 (8,224명)
processed_clinical_data_for_methylation.parquet

# 2차: cox용에서 누락된 353명 추가
processed_clinical_data.parquet (중복 제외)
```

### 컬럼 매핑 (preprocess_clinical_data 함수)

| 원본 TCGA 컬럼 | 변환 컬럼 | 인코딩 |
|----------------|----------|--------|
| `gender` | `sex` | MALE=0, FEMALE=1 |
| `age_at_initial_pathologic_diagnosis` | `age_group` | 10개 bin (0-9) |
| `pathologic_stage` | `ajcc_pathologic_stage` | Stage I-IV → 0-7 |
| `neoplasm_histologic_grade` | `grade` | G1-G4 → 0-3 |
| `survival_time_clean` | `event_time` | 그대로 |
| `survival_event_clean` | `event_status` | 그대로 |
| `race` | `race` | 6개 카테고리 (0-5) |

---

## 주요 클래스 및 함수

### Dataset
```python
from src.data.hybrid_dataset import HybridMultiOmicsDataset, create_dataloaders

# 데이터로더 생성 (임상 파일은 자동 병합됨)
train_loader, val_loader, test_loader = create_dataloaders(
    cox_table_path="data/processed/integrated_table_cox.parquet",
    meth_table_path="data/processed/methylation_table.parquet",
    clinical_path="data/processed/processed_clinical_data_for_methylation.parquet",
    splits_path="data/processed/train_val_test_splits.json",
    batch_size=32
)
```

### Dataset 반환값
```python
batch = {
    'clinical_cat': (batch, 5),      # Categorical features
    'cox_omics': (batch, 132098),    # Cox [val, cox] pairs
    'methylation': (batch, 396065),  # Beta values
    'cox_mask': (batch, 1),          # True if Cox available
    'meth_mask': (batch, 1),         # True if Meth available
    'event_time': (batch, 1),        # Survival time (days)
    'event_status': (batch, 1),      # Event indicator
    'patient_id': string
}
```

### Model
```python
from src.models.hybrid_fc_tabtransformer import HybridMultiModalModel

model = HybridMultiModalModel(...)
logit, features = model(clinical_cat, cox_omics, methylation, cox_mask, meth_mask)
```

### 3-Year Label 생성
```python
from src.training.train_hybrid import create_3year_labels

labels = create_3year_labels(event_times, event_status)
# -1: 제외 (3년 전 중도절단)
#  0: 3년 생존
#  1: 3년 내 사망
```

---

## 훈련 설정

```python
# 기본 설정
epochs = 100
batch_size = 32
learning_rate = 1e-4
weight_decay = 1e-2

# Optimizer & Scheduler
optimizer = AdamW(lr=1e-4, weight_decay=1e-2)
scheduler = ReduceLROnPlateau(patience=5)

# Loss & Metrics
loss = BCEWithLogitsLoss()
metrics = [AUC, Accuracy]

# Early Stopping
patience = 15
```

### 훈련 실행

```bash
cd src/training
bash run_hybrid_training.sh
```

---

## 버그 수정 이력

### 2025-12-31 수정 완료

#### ⚠️ [치명적] Cox 계수 암종 미매핑 버그

**이 버그는 모델의 핵심 기능을 무력화시키는 치명적 버그였음.**

- **문제**: Cox 계수를 환자의 암종에 맞게 매핑하지 않고, 전체 암종의 **평균값**을 사용
- **위치**: `integrated_dataset_builder.py` - `create_cox_enhanced_features()` 함수
- **영향**:
  - 모든 환자가 동일한 Cox 계수를 가짐 (암종 무시)
  - 암종별 Cox 회귀분석의 의미가 완전히 상실
  - 모델이 암종 특이적 생존 패턴을 학습할 수 없음

**잘못된 코드:**
```python
# ❌ 모든 암종의 평균 사용 (잘못됨!)
cox_coef_mean = cox_coefficients.loc[common_features].mean(axis=1)
enhanced_features[f"{omics_type}_{feature}_cox"] = cox_coef_mean[feature]
```

**수정된 코드:**
```python
# ✅ 환자의 암종에 맞는 계수 매핑
for patient in measured_values.index:
    cancer_type = patient_cancer_types.get(patient)  # 환자의 암종 조회
    if cancer_type is not None:
        cox_values.append(cox_coefficients.loc[feature, cancer_type])  # 해당 암종 계수
    else:
        cox_values.append(cox_coef_mean[feature])  # fallback
```

**검증 방법:**
```python
import pandas as pd
cox = pd.read_parquet('data/processed/integrated_table_cox.parquet')

# 특정 Cox 컬럼의 고유값 확인
test_col = [c for c in cox.columns if c.endswith('_cox')][0]
print(f"고유값 수: {cox[test_col].nunique()}")  # 27 이상이어야 함 (암종 수)

# 암종별 값이 다른지 확인
for cancer in ['BRCA', 'LGG', 'LUAD']:
    val = cox[cox['acronym'] == cancer][test_col].iloc[0]
    print(f"  {cancer}: {val:.6f}")  # 값이 달라야 함!
```

#### 8. Train/Val/Test Splits 누락 버그

- **문제**: Splits가 Cox 환자(4,504명)만 포함, Methylation-only 환자(4,073명) 누락
- **위치**: `integrated_dataset_builder.py` - `create_stratified_splits()` 함수
- **영향**: 8,577명 중 4,073명이 훈련에서 제외됨

**수정:**
- `create_union_stratified_splits()` 함수 추가
- 8,577명 전체를 Union 기반으로 분할
- 키 이름 변경: `train` → `train_patients` (patient ID 사용)

**새 splits 구조:**
```json
{
  "train_patients": ["TCGA-XX-XXXX", ...],  // 6,003명 (70%)
  "val_patients": [...],                     // 1,286명 (15%)
  "test_patients": [...],                    // 1,288명 (15%)
  "metadata": {
    "total_patients": 8577,
    "split_method": "random",
    "stratified_by": "cancer_type + modality"
  }
}
```

---

### 2025-12-30 수정 완료

#### 1. Missing Methylation 미지원 버그
- **문제**: 353명 Cox-only 환자(Methylation 없음) 처리 불가
- **수정**:
  - `hybrid_fc_tabtransformer.py`: `meth_mask` 파라미터 추가
  - `hybrid_dataset.py`: `meth_mask` 반환, Union 기반 환자 목록
  - `train_hybrid.py`: `meth_mask` 사용

#### 2. 임상 데이터 미병합 버그
- **문제**: 하나의 임상 파일만 로드하여 353명 누락
- **수정**: `hybrid_dataset.py`에서 두 임상 파일 자동 병합

#### 3. 임상 컬럼 매핑 미구현
- **문제**: 원본 TCGA 컬럼명과 모델 기대 컬럼명 불일치
- **수정**: `preprocess_clinical_data()` 함수 추가 (gender→sex, etc.)

#### 4. numpy.object_ 변환 에러
- **문제**: Dataset에 object dtype 컬럼 포함 시 torch 변환 실패
- **위치**: `integrated_dataset_builder.py:386-389`
- **수정**: numeric 컬럼만 필터링 후 float32 변환

#### 5. 검증 코드 접미사 불일치
- **위치**: `integrated_dataset_builder.py:581-583`
- **수정**: `_value` → `_val`

#### 6. max_features_per_omics 기본값
- **위치**: `integrated_dataset_builder.py:65, 149, 202`
- **수정**: 5000 → 0 (unlimited)

#### 7. 훈련 스크립트 clinical 경로
- **위치**: `run_hybrid_training.sh:16`
- **수정**: `tcga_pancan_clinical_processed.parquet` → `processed_clinical_data_for_methylation.parquet`

---

## 현재 이슈

### 1. 통합 데이터셋 재생성 필요 ⚠️

**Cox 계수 버그 수정 후 반드시 재실행 필요:**

```bash
cd src/preprocessing
./run_integrated_dataset_builder.sh
tail -f integrated_dataset_*.log
```

**생성 후 반드시 검증:**
```python
import pandas as pd
cox = pd.read_parquet('data/processed/integrated_table_cox.parquet')

# Cox 계수 컬럼의 고유값이 1개면 버그!
cox_col = [c for c in cox.columns if c.endswith('_cox')][0]
unique_count = cox[cox_col].nunique()
assert unique_count > 1, f"❌ Cox 계수가 모두 동일함! (고유값={unique_count})"
print(f"✅ Cox 계수 검증 통과 (고유값={unique_count})")
```

### 2. 훈련 미실행 상태
- 코드는 완성되었으나 실제 훈련이 실행되지 않음
- `results/`, `logs/` 디렉토리 비어있음

### 3. GPU 메모리 부족 시
- batch_size를 32 → 16 또는 8로 줄이기
- gradient accumulation 사용 고려

---

## 자주 사용하는 명령어

```bash
# 통합 데이터셋 생성 (가장 먼저 실행!)
cd src/preprocessing && ./run_integrated_dataset_builder.sh

# 훈련 실행
cd src/training && bash run_hybrid_training.sh

# 로그 모니터링
tail -f *.log

# GPU 사용량 확인
nvidia-smi -l 1
```

---

## 하드웨어 요구사항

- **GPU**: 48GB VRAM (RTX A6000 테스트됨)
- **RAM**: 64GB+ 권장 (대용량 parquet 로딩)
- **Storage**: 100GB+ (데이터 + 모델 체크포인트)

---

## 다음 단계 (Phase 2-B: WSI)

아직 구현되지 않은 항목:
1. WSI 데이터 다운로드 및 전처리
2. `src/preprocessing/wsi_preprocessing.py`
3. `src/models/wsi_swin_transformer.py`
4. `src/data/wsi_dataset.py`
5. `src/training/train_wsi.py`

---

## 프로젝트 난이도 및 교훈 (Lessons Learned)

### 왜 이렇게 어려웠나?

멀티오믹스 + Cox 회귀 + TabTransformer 파이프라인 구축은 예상보다 훨씬 복잡했습니다.

#### 1. 데이터 이질성 (Data Heterogeneity)

```
5개 오믹스 × 서로 다른 스케일 × 서로 다른 환자 집합 = 복잡도 폭발
```

- **Expression**: 20,000+ genes, FPKM/TPM, log2 변환 필요
- **CNV**: 음수 포함, 정규화 방법 다름
- **Methylation**: 0-1 beta values, 396,065 CG sites
- **Mutations**: 0/1/2 이산값
- **RPPA**: 단백질 발현, 음수 포함

각 오믹스마다 전처리 로직이 달라야 하고, 하나라도 잘못되면 모델이 학습하지 못함.

#### 2. 환자 집합 불일치 (Patient Set Mismatch)

```
처음 가정: Cox 환자 ⊂ Methylation 환자 (nested)
실제 현실: Cox ∩ Meth = 4,151명, Cox만 = 353명, Meth만 = 4,073명
```

- 두 데이터 소스의 환자 집합이 완전히 겹치지 않음
- "당연히 포함되어 있겠지"라는 가정이 틀림
- Union 기반으로 재설계 필요 → Missing Modality Learning 도입

#### 3. [value, cox] 쌍 형식의 함정

```python
# 직관적이지만 틀린 방법
weighted_feature = gene_value * cox_coefficient  # ❌

# 실제 필요한 방법
features = [gene_val, gene_cox, ...]  # ✅ 별도 유지
```

- Cox 계수를 곱해서 하나의 값으로 만들면 정보 손실
- 모델이 value와 cox를 각각 학습해야 최적의 가중치를 찾음
- 이 결정 하나가 feature 수를 2배로 늘림 (66,049 → 132,098)

#### 4. 임상 데이터 파편화

```
파일 1: processed_clinical_data.parquet (4,504명)
파일 2: processed_clinical_data_for_methylation.parquet (8,224명)
→ 둘 다 로드해서 병합해야 8,577명 전체 커버
```

- TCGA 원본 컬럼명 ≠ 모델 기대 컬럼명
- `gender` → `sex`, `pathologic_stage` → `ajcc_pathologic_stage`
- 매핑 테이블 직접 작성 필요

#### 5. 메모리와 차원의 저주

```
Cox encoder input:  132,098 features
Meth encoder input: 396,065 features
Total parameters:   ~7.6 billion
GPU memory:         ~29 GB (batch=32)
```

- 일반적인 GPU로는 학습 불가
- RTX A6000 (48GB) 필요
- 그래도 batch_size 제한적

#### 6. 디버깅의 어려움

```
에러: "numpy.object_ has no attribute 'astype'"
원인: parquet 파일의 한 컬럼이 object dtype
위치: 수십만 개 컬럼 중 어딘가
해결: select_dtypes(include=[np.number])로 필터링
```

- 대용량 데이터에서 하나의 잘못된 컬럼 찾기가 극도로 어려움
- 에러 메시지가 실제 원인을 숨김
- 로그 출력을 촘촘히 넣어야 디버깅 가능

#### 7. ⚠️ 설계 의도와 구현의 괴리 (가장 치명적)

```python
# 설계 의도: 환자의 암종에 맞는 Cox 계수 사용
# BRCA 환자 → BRCA의 Cox 계수
# LGG 환자 → LGG의 Cox 계수

# 실제 구현 (잘못됨): 모든 암종의 평균 사용
cox_coef_mean = cox_coefficients.mean(axis=1)  # ❌ 치명적 버그!
```

- **문제**: 코드가 "동작"하지만 "의미"가 완전히 틀림
- **증상**: 에러 없음, 모델 훈련 가능, 하지만 성능 저하
- **발견**: 출력 데이터를 직접 검사해야만 발견 가능
- **교훈**: **"동작한다 ≠ 올바르다"** - 출력 데이터 검증 필수

### 핵심 교훈

| 교훈 | 설명 |
|------|------|
| **가정하지 말 것** | "당연히 되겠지"가 가장 위험한 생각 |
| **환자 집합 먼저 확인** | 데이터 로드 후 즉시 intersection/union 확인 |
| **작은 단위로 검증** | 전체 파이프라인 전에 각 컴포넌트 단위 테스트 |
| **로그를 아끼지 말 것** | print문이 미래의 나를 구원함 |
| **문서화 즉시** | 발견한 문제와 해결책을 바로 기록 |
| **⚠️ 출력 데이터 검증** | 생성된 파일을 반드시 샘플링하여 의도대로 생성되었는지 확인 |
| **⚠️ 암종별 값 확인** | Cox 계수 등 암종별로 달라야 하는 값은 실제로 다른지 검증 |

### 필수 검증 체크리스트

```python
# 1. 환자 수 검증
assert len(cox_table) == 4504, "Cox 환자 수 불일치"
assert len(meth_table) == 8224, "Meth 환자 수 불일치"

# 2. Cox 계수 암종별 다름 검증 (가장 중요!)
cox_col = [c for c in cox_table.columns if c.endswith('_cox')][0]
assert cox_table[cox_col].nunique() > 1, "Cox 계수가 모두 동일함!"

# 3. Splits 전체 환자 포함 검증
total_in_splits = len(train) + len(val) + len(test)
assert total_in_splits == 8577, "Splits에 누락된 환자 있음"
```

### 시간 투자 분포 (체감)

```
실제 모델 코딩:     20%
데이터 전처리:      30%
버그 수정:          25%
환자/컬럼 매핑:     15%
문서화 및 검증:     10%
```

> "모델은 쉽다. 데이터가 어렵다."
> "동작한다고 올바른 게 아니다."

---

## 투명성 원칙

이 프로젝트 작업 시 다음을 준수합니다:
1. 모든 데이터 처리 과정을 로그로 기록
2. 훈련 시 실제 데이터 크기, 샘플 수 출력
3. 불확실한 부분은 미리 공유
4. 작은 단위로 작업하고 결과 확인
5. **생성된 데이터 파일을 반드시 샘플링하여 검증**

---

*Last updated: 2025-12-31*
