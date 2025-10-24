# Cancer Foundation Model 구축 가이드

## 📋 프로젝트 목표

멀티오믹스 + 병리영상 데이터를 통합하여 **3년 생존 예후**를 예측하는 딥러닝 시스템 구축

---

## 🎯 전체 진행 상황 (2025-10-24)

| Phase | 단계 | 상태 | 진행률 |
|-------|------|------|--------|
| **Phase 1** | **데이터 준비 및 전처리** | ✅ **완료** | **100%** |
| **Phase 2-A** | **Hybrid 멀티오믹스 모델** | ✅ **완료** | **100%** |
| **Phase 2-B** | **병리영상 모델 (WSI)** | ❌ **다음 단계** | **0%** |
| Phase 3 | 멀티모달 융합 | ⏸️ 대기 | 0% |

---

## ✅ Phase 1: 데이터 준비 (완료)

### 데이터 현황

| 구분 | 환자 수 | 특성 수 | 상태 |
|------|---------|---------|------|
| Cox Omics | 4,504명 | 71,520 features | ✅ 완료 |
| Methylation | 8,224명 | 396,065 CG sites | ✅ 완료 |
| Total | 8,224명 | - | ✅ 완료 |

### 처리 완료된 데이터

```
data/processed/
├── integrated_table_cox.parquet    # 4,504 × 143,048 (Cox [val,cox] 쌍)
├── methylation_table.parquet       # 8,224 × 396,065 (Beta values)
├── train_val_test_splits.json      # Train/Val/Test 분할
└── processed_clinical_data.parquet # 임상 데이터
```

### log2 변환 규칙

```python
# Expression, microRNA: log2(x + 1)
# CNV, RPPA: log2(x - min + 1)  [음수 처리]
# Mutations: 변환 없음 (impact scores 0-2)
# Methylation: 변환 없음 (beta values 0-1)
```

---

## ✅ Phase 2-A: Hybrid 멀티오믹스 모델 (완료)

### 모델 아키텍처

```
HybridMultiModalModel
├─ Clinical Categories (5개) → Embedding
├─ Cox Encoder: 143,040 → FC-NN(2048→512→256) → 256-dim
├─ Meth Encoder: 396,065 → FC-NN(4096→1024→256) → 256-dim
└─ TabTransformer(dim=128, depth=6, heads=8) → 3-year survival
```

### 모델 파라미터

```
Cox Encoder:    2,929M params (11.18 GB)
Meth Encoder:   4,509M params (17.20 GB)
TabTransformer:   212M params ( 0.81 GB)
────────────────────────────────────────
Total:          7,651M params (29.19 GB)
```

### Missing Modality Learning

```
환자 구성:
├─ Cox 있음 (4,504명): Clinical + Cox + Methylation
└─ Cox 없음 (3,720명): Clinical + [ZERO] + Methylation
   → Total: 8,224명 모두 활용
```

### 구현 완료

```
✅ src/models/hybrid_fc_tabtransformer.py  - Hybrid 모델
✅ src/data/hybrid_dataset.py              - Dataset (Missing modality 지원)
✅ src/training/train_hybrid.py            - 훈련 스크립트
✅ src/training/run_hybrid_training.sh     - 실행 래퍼
```

### 훈련 설정

- **Target**: 3-year survival (0=생존, 1=사망)
- **Loss**: BCEWithLogitsLoss
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-2)
- **Scheduler**: ReduceLROnPlateau (patience=5)
- **Early stopping**: patience=15
- **Batch size**: 32
- **Epochs**: 100

### 실행 방법

```bash
cd src/training
bash run_hybrid_training.sh
```

---

## ❌ Phase 2-B: 병리영상 모델 (다음 단계)

### 현재 상황

- ❌ WSI 데이터 미확보
- ❌ 패치 추출 스크립트 미작성
- ❌ Swin Transformer 모델 미구현
- ❌ MIL(Multiple Instance Learning) 미구현

### 필요 작업

**1. WSI 데이터 전처리**
```
목표:
- TCGA WSI 데이터 다운로드
- 패치 추출 (224×224 or 384×384)
- 배경 제거 및 품질 필터링
- 패치 메타데이터 생성

출력:
data/processed/wsi_patches/
├── TCGA-XX-XXXX/
│   ├── patch_0001.png
│   ├── patch_0002.png
│   └── ...
└── patch_metadata.csv
```

**2. Swin Transformer 모델 구현**
```python
# src/models/wsi_swin_transformer.py
WSISwinTransformer
├─ Swin Backbone (pretrained)
├─ MIL Aggregation (Attention pooling)
└─ Survival Head → 256-dim representation
```

**3. 훈련 파이프라인**
```bash
# src/training/train_wsi.py
# src/training/run_wsi_training.sh
```

### 권장 구현 순서

1. **WSI 전처리** (Jupyter Notebook)
   - `notebooks/02_wsi_preprocessing.ipynb`
   - openslide-python으로 패치 추출
   - 배경 제거 (HSV threshold)
   - 품질 필터링

2. **Swin Transformer 모델**
   - Pretrained weights 활용 (ImageNet or histopathology)
   - MIL 어댑터 구현
   - Attention weights 저장 (시각화용)

3. **훈련 및 검증**
   - Slide-level 3-year survival prediction
   - Attention map 시각화
   - 성능 평가

---

## 📁 프로젝트 구조

```
CANCER_FOUNDATION_MODEL/
├── data/
│   ├── raw/                           # TCGA 원본
│   └── processed/
│       ├── integrated_table_cox.parquet         ✅
│       ├── methylation_table.parquet            ✅
│       ├── train_val_test_splits.json           ✅
│       └── wsi_patches/                         ❌ 다음 단계
│
├── src/
│   ├── preprocessing/
│   │   ├── cox_feature_engineer.py              ✅
│   │   ├── integrated_dataset_builder.py        ✅
│   │   └── wsi_preprocessing.py                 ❌ 다음 단계
│   ├── models/
│   │   ├── hybrid_fc_tabtransformer.py          ✅
│   │   └── wsi_swin_transformer.py              ❌ 다음 단계
│   ├── data/
│   │   ├── hybrid_dataset.py                    ✅
│   │   └── wsi_dataset.py                       ❌ 다음 단계
│   └── training/
│       ├── train_hybrid.py                      ✅
│       ├── run_hybrid_training.sh               ✅
│       └── train_wsi.py                         ❌ 다음 단계
│
├── results/                           # 훈련 결과
├── obsolete/                          # 구버전 (731MB)
└── doc/
    └── CFM.vibe_coding_guide.md       # 이 파일
```

---

## 🔬 기술 세부사항

### Cox 기반 파이프라인

```
원본 데이터 → log2 변환 → Cox 회귀분석 → [val, cox] 쌍 생성
→ FC-NN Dimension Reduction → TabTransformer → 3-year survival
```

### 입력 데이터 형식

**⚠️ 중요**: 측정값과 Cox계수를 **곱하지 않고** 별도 2개 값으로 입력

```python
# ❌ 잘못: value * cox
# ✅ 올바: [value, cox] 쌍

input = [gene1_val, gene1_cox, gene2_val, gene2_cox, ...]
```

---

## ⚠️ 주의사항

### 1. [측정값, Cox계수] 쌍 형식
- ❌ 곱셈: `value * cox_coefficient`
- ✅ 스택: `[value, cox_coefficient]`

### 2. log2 변환 일관성
- Expression, CNV, microRNA, RPPA: `log2(x + 1)`
- CNV, RPPA 음수: `log2(x - min + 1)`
- Methylation, Mutations: 변환 없음

### 3. Missing Modality
- Cox 없는 환자: `cox_omics=ZERO`, `cox_mask=False`

### 4. GPU 메모리
- 48GB GPU 필요 (RTX A6000)
- Batch size: 32 (멀티오믹스), 1 (WSI)

---

## 📝 다음 단계

### ✅ 완료:
1. ✅ 데이터 준비 (100%)
2. ✅ Hybrid 멀티오믹스 모델 (100%)
   - 모델 구현 완료
   - Dataset 구현 완료
   - 훈련 스크립트 완료
   - Missing modality learning 완료

### 🔥 다음 단계 (병리영상):

**1. WSI 데이터 확보 및 전처리** ⭐ **최우선**
   - TCGA WSI 데이터 다운로드
   - 패치 추출 스크립트 작성
   - 배경 제거 및 품질 필터링
   - 패치 메타데이터 생성

**2. Swin Transformer 모델 구현**
   - `src/models/wsi_swin_transformer.py`
   - Pretrained backbone 활용
   - MIL aggregation 구현
   - Attention mechanism

**3. WSI 훈련 파이프라인**
   - `src/data/wsi_dataset.py`
   - `src/training/train_wsi.py`
   - Slide-level prediction
   - 성능 평가

---

## 📚 참고 문헌

### 데이터
- TCGA Research Network - The Cancer Genome Atlas

### 방법론
- TabTransformer - Huang et al.
- Cox Regression - Cox, D. R. (1972)
- Swin Transformer - Liu et al. (2021)
- Multiple Instance Learning for Histopathology

### 구현
- tab-transformer-pytorch
- lifelines
- openslide-python (WSI 처리)

---

**마지막 업데이트**: 2025년 10월 24일
**프로젝트 상태**: Phase 2-A 완료, Phase 2-B (병리영상) 다음 단계
