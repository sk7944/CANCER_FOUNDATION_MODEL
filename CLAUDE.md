# CLAUDE.md

> AI 어시스턴트와 효과적으로 협업하기 위한 프로젝트 컨텍스트 문서

---

## 한 줄 요약

**TCGA 멀티오믹스 + 병리영상(WSI)을 통합하여 설명 가능한(Explainable) 암 예후 예측 파운데이션 모델 구축**

---

## 프로젝트 목표

멀티오믹스(Tabular)와 병리 영상(Image) 데이터를 통합하여, 설명 가능한(Explainable) 암 예후 예측 파운데이션 모델을 구축한다. 모델은 최종적으로 예측에 대한 판단 근거를 **자연어 텍스트**와 **시각적 히트맵**으로 제시해야 한다.

---

## 빠른 시작

```bash
# Multi-omics 모델 훈련 (완료된 작업)
cd multiomics_model/src/training && bash run_hybrid_training.sh

# 데이터 재생성 필요 시
cd multiomics_model/src/preprocessing && ./run_integrated_dataset_builder.sh

# WSI 모델 훈련 (예정)
cd wsi_model/src/training && bash run_wsi_training.sh
```

---

## 프로젝트 로드맵

### Phase 1: 데이터 준비 및 전처리 ✅

| 작업 | 상태 | 설명 |
|------|------|------|
| 1-1. 멀티모달 데이터 다운로드 | ✅ 완료 | TCGA Pan-Cancer 데이터 |
| 1-2. Multi-omics 특성 공학 | ✅ 완료 | Cox 회귀계수 [val, cox] 쌍 |
| 1-3. 병리영상 전처리 | ⏳ 예정 | WSI 패치 분할 |

### Phase 2: 단일 모달리티 모델 개발

| 작업 | 상태 | 설명 |
|------|------|------|
| 2-1. Multi-omics 모델 | ✅ 완료 | Hybrid FC-NN + TabTransformer |
| 2-2. 병리영상 모델 | ⏳ 예정 | Swin Transformer (ROI-free) |

### Phase 3: 멀티모달 융합 및 LLM 파인튜닝 (예정)

| 작업 | 상태 | 설명 |
|------|------|------|
| 3-1. 추론 텍스트 데이터셋 구축 | ⏳ 예정 | (멀티오믹스, 병리이미지) → 전문가 추론 텍스트 |
| 3-2. 융합 아키텍처 설계 | ⏳ 예정 | 임베딩 추출 + 프로젝션 레이어 |
| 3-3. LLM 파인튜닝 | ⏳ 예정 | Llama 3 / Qwen2 등 |

### Phase 4: 모델 평가 및 시각화 (예정)

| 작업 | 상태 | 설명 |
|------|------|------|
| 4-1. 최종 LLM 모델 평가 | ⏳ 예정 | 정량/정성 평가 |
| 4-2. XAI 시각화 구현 | ⏳ 예정 | 어텐션 맵 히트맵 |

---

## 현재 상태

### Multi-omics 모델 (Phase 2-1) ✅ 완료

**훈련 결과 (2026-01-07):**
- Best Epoch: 8
- **Val AUC: 0.9234**
- **Test AUC: 0.9074**
- Test Accuracy: 82.19%
- 결과 위치: `multiomics_model/results/hybrid_training_20260107_170056/`

**생성된 데이터:**
- `integrated_table_cox.parquet`: 4,504 × 132,100 (임상 컬럼 제외, omics만)
- `methylation_table.parquet`: 8,224 × 396,065
- `train_val_test_splits.json`: 8,577명 (6,003/1,286/1,288)

### WSI 모델 (Phase 2-2) ⏳ 예정

- 아키텍처: Swin Transformer
- 목표: 3년/5년 생존 여부 분류
- 핵심: ROI 정보 없이, 이미지 전체 레이블로 훈련

---

## 프로젝트 구조

```
CANCER_FOUNDATION_MODEL/
├── CLAUDE.md                    # AI 개발자 가이드 (이 파일)
├── README.md                    # 프로젝트 소개
├── requirements.txt             # 의존성
├── doc/                         # 문서
│   └── TODO_LIST.CFM.pdf        # 프로젝트 로드맵
│
├── multiomics_model/            # 🧬 Multi-omics 모델 (Phase 2-1)
│   ├── src/
│   │   ├── models/              # HybridMultiModalModel
│   │   ├── data/                # PyTorch Dataset
│   │   ├── training/            # 훈련 스크립트
│   │   ├── preprocessing/       # 데이터 전처리
│   │   └── utils/               # 유틸리티 (추론용)
│   ├── data/
│   │   ├── raw/                 # TCGA 원본 데이터
│   │   └── processed/           # 전처리된 데이터
│   └── results/                 # 훈련 결과
│
└── wsi_model/                   # 🔬 WSI 모델 (Phase 2-2)
    ├── src/
    │   ├── models/              # Swin Transformer
    │   ├── data/                # WSI Dataset
    │   ├── training/            # 훈련 스크립트
    │   └── preprocessing/       # WSI 패치 분할
    ├── data/
    │   ├── raw/                 # WSI 원본 이미지
    │   └── processed/           # 패치 이미지
    └── results/                 # 훈련 결과
```

---

## Multi-omics 모델 상세

### 핵심 숫자

| 항목 | 값 |
|------|-----|
| 총 환자 수 | **8,577명** (27개 암종) |
| Cox + Meth 둘 다 | 4,151명 |
| Cox만 | 353명 |
| Meth만 | 4,073명 |
| Cox features | 132,100 (66,050 × 2) |
| Meth features | 396,065 |
| 모델 크기 | ~7.14GB, 1.9B params |
| GPU 요구사항 | 48GB VRAM |

### 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│ 입력: 8,577명 환자                                       │
├─────────────────────────────────────────────────────────┤
│ Clinical (5 categorical)  → Embedding                   │
│ Cox Omics (132,100)       → FC: 2048→512→256           │
│ Methylation (396,065)     → FC: 4096→1024→256          │
├─────────────────────────────────────────────────────────┤
│ TabTransformer (dim=128, depth=6, heads=8)             │
├─────────────────────────────────────────────────────────┤
│ 출력: 3년 생존 (0=생존, 1=사망)                          │
└─────────────────────────────────────────────────────────┘

Missing Modality:
- Cox 없음 → cox_encoded = ZEROS(256), cox_mask = False
- Meth 없음 → meth_encoded = ZEROS(256), meth_mask = False
```

### 핵심 파일

```
multiomics_model/src/
├── models/hybrid_fc_tabtransformer.py   # 메인 모델
├── data/hybrid_dataset.py               # PyTorch Dataset
├── training/train_hybrid.py             # 훈련 스크립트
├── preprocessing/
│   ├── cox_feature_engineer.py          # Cox 회귀
│   └── integrated_dataset_builder.py    # 데이터 통합
└── utils/                               # 추론 파이프라인
    ├── inference_pipeline.py            # 모델 추론
    ├── user_data_pipeline.py            # 사용자 데이터 전처리
    ├── feature_converter.py             # 피처 변환
    └── data_format_guide.py             # 데이터 형식 가이드
```

### Utils 사용법

```python
# 추론 파이프라인
from multiomics_model.src.utils import InferencePipeline

pipeline = InferencePipeline(
    model_checkpoint='path/to/best_model.pth',
    cox_coef_path='path/to/cox_coefficients.parquet'
)

# 단일 환자 예측
result = pipeline.predict_single(
    age=55, sex='FEMALE', race='WHITE', stage='II', grade='G2',
    cancer_type='BRCA'
)
print(f"생존 확률: {result['survival_probability'][0]:.2%}")

# 사용자 데이터 전처리
from multiomics_model.src.utils import UserDataPipeline

pipeline = UserDataPipeline(cox_coef_path='path/to/cox_coefficients.parquet')
result = pipeline.process_user_data(
    user_files={'expression': 'expr.csv', 'methylation': 'meth.csv'},
    cancer_type='BRCA'
)
```

---

## WSI 모델 상세 (예정)

### 계획

| 항목 | 값 |
|------|-----|
| 아키텍처 | Swin Transformer |
| 입력 | WSI 패치 (예: 256×256 또는 512×512) |
| 목표 | 3년/5년 생존 분류 |
| 핵심 | ROI-free (전체 레이블로 훈련) |

### 파이프라인 (예정)

```
WSI (고해상도)
    ↓
패치 분할 (Phase 1-3)
    ↓
Swin Transformer 훈련 (Phase 2-2)
    ↓
어텐션 맵 추출 (Phase 4-2)
```

---

## 절대 규칙 (MUST)

### 1. [val, cox] 쌍 형식
```python
# ❌ 절대 금지: 곱셈
input = gene_val * gene_cox

# ✅ 올바름: 별도 유지 (훈련 데이터 형식: {feature}_val, {feature}_cox)
input = [gene_val, gene_cox, gene_val, gene_cox, ...]
```

### 2. Cox 계수는 암종별로 달라야 함
```python
# 검증 필수
cox_col = [c for c in df.columns if c.endswith('_cox')][0]
assert df[cox_col].nunique() > 1  # 1이면 버그!
```

### 3. 환자 수 검증
```python
assert len(cox_table) == 4504
assert len(meth_table) == 8224
assert len(train) + len(val) + len(test) == 8577
```

---

## 데이터 변환 규칙

| 오믹스 | 변환 | 범위 |
|--------|------|------|
| Expression | `log2(x + 1)` | 연속값 |
| CNV | `log2(x - min + 1)` | 연속값 |
| microRNA | `log2(x + 1)` | 연속값 |
| RPPA | `log2(x - min + 1)` | 연속값 |
| **Mutations** | 없음 (Impact Score) | **0, 1, 2** |
| **Methylation** | 없음 (Beta value) | **0~1** |

### Mutation Impact Score

| Score | Impact | Variants |
|-------|--------|----------|
| 0 | Low | Silent, UTR, Intron |
| 1 | Moderate | Missense, In_Frame_Del/Ins |
| 2 | High (LoF) | Nonsense, Frameshift, Splice_Site |

---

## 임상 변수 (Clinical Categories)

`clinical_categories=(10, 2, 6, 5, 4)` - 각 categorical 변수의 카테고리 수

| 순서 | 변수 | 카테고리 수 | 설명 |
|------|------|-------------|------|
| 0 | age_group | 10 | 연령 구간 (0-29, 30-39, ..., 80+) |
| 1 | sex | 2 | 0=MALE, 1=FEMALE |
| 2 | race | 6 | WHITE, BLACK, ASIAN, ..., Unknown |
| 3 | ajcc_pathologic_stage | 5 | I, II, III, IV, NA |
| 4 | grade | 4 | G1, G2, G3, G4 |

### Stage 매핑 (서브스테이지 일원화)

```
Stage I, IA, IB, IC 등   → 0
Stage II, IIA, IIB 등    → 1
Stage III, IIIA, IIIB 등 → 2
Stage IV, IVA, IVB 등    → 3
[Not Available], [Unknown] 등 → 4 (NA)
```

---

## 훈련 설정

```python
epochs = 100
batch_size = 32
lr = 1e-4
optimizer = AdamW(weight_decay=1e-2)
scheduler = ReduceLROnPlateau(patience=5)
loss = BCEWithLogitsLoss()
early_stopping = 15 epochs
clinical_categories = (10, 2, 6, 5, 4)  # age, sex, race, stage, grade
```

---

## 버그 이력 (치명적)

### [2026-01-07] CUDA OOM 에러 - 체크포인트 저장/로딩

- **증상 1**: 훈련 중 best model 저장 시 CUDA OOM (`Tried to allocate 6.04 GiB`)
- **원인 1**: `model.state_dict()`가 GPU 메모리에서 직접 직렬화되어 추가 메모리 필요
- **수정 1**: state dict를 CPU로 이동 후 저장
  ```python
  # Before
  'model_state_dict': model.state_dict()
  # After
  'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()}
  ```

- **증상 2**: 테스트 평가 시 체크포인트 로딩에서 CUDA OOM
- **원인 2**: 훈련 완료 후 모델이 GPU에 남아있는 상태에서 체크포인트를 GPU로 직접 로딩
- **수정 2**: 기존 모델 삭제 후 CPU로 로드, 그 다음 GPU로 이동
  ```python
  del model
  torch.cuda.empty_cache()
  model = HybridMultiModalModel(...)  # 새로 생성
  checkpoint = torch.load(path, map_location='cpu')  # CPU로 먼저 로드
  model.load_state_dict(checkpoint['model_state_dict'])
  model = model.to(device)  # 그 다음 GPU로 이동
  ```
- **파일**: `multiomics_model/src/training/train_hybrid.py`

### [2026-01-07] NaN Loss 발생

- **증상**: 훈련 시작 직후 loss가 NaN으로 발산
- **원인**: 데이터에 NaN/Inf 값 존재 (Cox omics, Methylation 배열)
- **수정**: `_prepare_arrays()`에서 NaN/Inf를 0으로 대체
  ```python
  self.cox_omics_array = np.nan_to_num(self.cox_omics_array, nan=0.0, posinf=0.0, neginf=0.0)
  self.meth_array = np.nan_to_num(self.meth_array, nan=0.0, posinf=0.0, neginf=0.0)
  ```
- **파일**: `multiomics_model/src/data/hybrid_dataset.py`

### [2026-01-06] Cox 테이블 임상 컬럼 및 clinical_categories 수정

- **증상**: 훈련 시 `ValueError: could not convert string to float: 'FEMALE'`
- **원인 1**: `integrated_table_cox.parquet`에 문자열 임상 컬럼 포함 (레거시)
- **원인 2**: `clinical_categories=(10, 3, 8, 4, 5)` 설정이 실제 데이터 범위와 불일치
- **수정**:
  1. Cox 테이블에서 임상 컬럼 6개 제거 (132,106 → 132,100)
  2. `integrated_dataset_builder.py` 수정: 임상 데이터 제외
  3. Stage 매핑 일원화: 8개 → 5개 (I, II, III, IV, NA)
  4. `clinical_categories` 수정: `(10, 2, 6, 5, 4)`
- **파일**: `hybrid_dataset.py`, `train_hybrid.py`, `hybrid_fc_tabtransformer.py`

### [2025-12-31] Cox 계수 암종 미매핑

- **증상**: 모든 환자가 동일한 Cox 계수 (암종 무시)
- **원인**: `mean(axis=1)` 사용 → 모든 암종 평균
- **수정**: 환자별 암종(acronym) 조회 후 해당 암종 계수 매핑
- **검증**: `cox[cox_col].nunique() > 1` 확인 필수

### [2025-12-31] Splits 누락

- **증상**: Meth-only 환자 4,073명 훈련에서 제외
- **원인**: Cox 환자만 splits에 포함
- **수정**: Union 기반 splits, 키 이름 `train_patients` 사용

---

## 자주 쓰는 명령어

```bash
# Multi-omics 훈련
cd multiomics_model/src/training && bash run_hybrid_training.sh

# GPU 모니터링
nvidia-smi -l 1

# 데이터 검증
python -c "
import pandas as pd
cox = pd.read_parquet('multiomics_model/data/processed/integrated_table_cox.parquet')
print(f'Shape: {cox.shape}')
col = [c for c in cox.columns if c.endswith('_cox')][0]
print(f'Unique cox values: {cox[col].nunique()}')  # Must be > 1
"
```

---

## 협업 원칙

1. **작은 단위로 작업**: 전체 파이프라인 전에 컴포넌트 검증
2. **로그 아끼지 않기**: 디버깅을 위한 print문 충분히
3. **즉시 문서화**: 발견한 문제와 해결책 바로 기록
4. **출력 데이터 검증**: "에러 없음"이 "올바름"을 의미하지 않음

---

## 교훈

> "모델은 쉽다. 데이터가 어렵다."
> "동작한다 ≠ 올바르다."

| 교훈 | 설명 |
|------|------|
| 가정하지 말 것 | 환자 집합 겹침을 가정했다가 실패 |
| 출력 검증 필수 | 생성된 데이터를 샘플링하여 의도대로인지 확인 |
| 암종별 값 확인 | Cox 계수가 암종별로 다른지 반드시 검증 |

---

*Last updated: 2026-01-13*
