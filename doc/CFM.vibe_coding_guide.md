# Cancer Foundation Model 구축 가이드 (Vibe Coding 프롬프트)

## 📋 프로젝트 목표

멀티오믹스(Tabular)와 병리 영상(Image) 데이터를 통합하여, **설명 가능한(Explainable)** 암 예후 예측 파운데이션 모델을 구축한다. 모델은 최종적으로 예측에 대한 **판단 근거를 자연어 텍스트와 시각적 히트맵으로 제시**해야 한다.

---

## 🎯 전체 진행 상황 요약 (2025-10-22 기준)

| Phase | 단계 | 상태 | 진행률 | 비고 |
|-------|------|------|--------|------|
| Phase 1 | 데이터 준비 및 전처리 | ✅ **완료** | **100%** | Cox 분석 완료, 코드 리팩토링 완료 |
| Phase 2-A | Multi-omics 모델 | ✅ **완료** | **100%** | CoxTabTransformer 완료 (AUC: 0.8495) |
| Phase 2-B | Methylation 모델 | 🔄 **대기 중** | **30%** | 코드 구현 완료, 샤딩 필요 |
| Phase 2-C | 병리영상 모델 | ❌ **미착수** | **0%** | WSI 전처리 및 Swin Transformer |
| Phase 3 | 멀티모달 융합 및 LLM | ⏸️ **대기 중** | **0%** | Phase 2 완료 후 진행 |
| Phase 4 | 평가 및 시각화 | ⏸️ **대기 중** | **0%** | Phase 3 완료 후 진행 |

**📊 현재 달성 수치:**
- **Multi-omics TabTransformer**: Test AUC **0.8495** ✅ (목표: 0.85 달성)
- **데이터셋**: 4,504명 환자, 71,520개 특성 (5개 오믹스)
- **훈련 완료**: 5개 시드 앙상블 모델 (seed_42, 43, 45, 48, 52)
- **체크포인트**: `src/training/checkpoints/seed_XX/best_cox_tabtransformer.pth`
- **Methylation 데이터**: 8,224명 환자, 396,065 probes (샤딩 필요)

**📁 코드 라인 수 (핵심 모듈):**
- `cox_feature_engineer.py`: 1,279 lines
- `integrated_dataset_builder.py`: 822 lines
- `train_tabtransformer.py`: 1,001 lines
- `cancer_multiomics_dataset.py`: 274 lines
- 총 핵심 코드: 3,710 lines

---

## Phase 1: 데이터 준비 및 전처리

### ✅ 1-1. 멀티모달 데이터 다운로드 및 정제 (완료)

**상태**: ✅ **완료** (100%)

**달성 내용**:
- ✅ TCGA PANCAN 멀티오믹스 데이터 다운로드 완료
- ✅ 원발성 암(Primary Tumor) 샘플 필터링 완료
- ✅ 환자 ID 표준화 완료 (TCGA-XX-XXXX 형식)
- ✅ Whitelisted 데이터 정제 완료

**데이터 파일 위치**: `./data/raw/`

**검증 완료**:
```bash
ls data/raw/*_whitelisted.* | wc -l
# 출력: 7개 파일 (Expression, CNV, microRNA, RPPA, Mutations, Methylation, Clinical)
```

**데이터 종류**:
1. **전사체**: `unc.edu_PANCAN_IlluminaHiSeq_RNASeqV2.geneExp_whitelisted.tsv`
2. **복제수변이**: `CNV.GISTIC_call.all_data_by_genes_whitelisted.tsv`
3. **microRNA**: `bcgsc.ca_PANCAN_IlluminaHiSeq_miRNASeq.miRNAExp_whitelisted.tsv`
4. **RPPA**: `mdanderson.org_PANCAN_MDA_RPPA_Core.RPPA_whitelisted.tsv`
5. **메틸레이션**: `jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv` (396,065 probes)
6. **돌연변이**: `tcga_pancancer_082115.vep.filter_whitelisted.maf.gz`
7. **임상데이터**: `clinical_PANCAN_patient_with_followup.tsv`

---

### ✅ 1-2. Multi-omics 특성 공학 (Feature Engineering) (완료)

**상태**: ✅ **완료**

**구현 파일**: `src/preprocessing/cox_feature_engineer.py`

**실행 방법**:
```bash
cd src/preprocessing
./run_cox_feature_engineer.sh
# 로그 모니터링: tail -f cox_analysis_*.log
```

**달성 내용**:

#### 1️⃣ Cox 회귀분석 수행
- **대상**: Expression, CNV, microRNA, RPPA, Mutation (메틸레이션 제외)
- **엔드포인트**: 3년 생존 예측 (OS_3yr 이진 변수)
- **암종별 분석**: 각 암종(cancer type)별로 Cox 비례위험 회귀분석 실행
- **모든 특성 보존**: p-value와 관계없이 모든 유전자/특성의 Cox 계수 저장

**코드 구현** (`cox_feature_engineer.py`):
```python
# Line 140-180: Expression Cox Analysis
# Step 1: log2 변환
df_log = np.log2(df + 1)

# Step 2: Cox 회귀분석 (암종별)
for cancer_type in cancer_types:
    cancer_patients = clinical_cancer['acronym'] == cancer_type
    for gene in genes:
        cph = CoxPHFitter()
        cph.fit(cox_data, duration_col='OS_month', event_col='OS_3yr')
        cox_coefficients[cancer_type][gene] = cph.params_[gene]
```

#### 2️⃣ 회귀계수 룩업 테이블 생성
- **구조**: DataFrame (genes × cancer_types)
- **파일**: `./data/processed/cox_coefficients_*.parquet`
- **용도**: TabTransformer 입력 시 [측정값, Cox계수] 쌍 생성

#### 3️⃣ 데이터 변환 및 정규화 (중요!)

**⚠️ log2 변환 적용 규칙**:

```python
# src/preprocessing/cox_feature_engineer.py

# Expression (Line 160-161): log2(x + 1)
df_log = np.log2(df + 1)

# CNV (Line 207-212): 음수 처리 포함
min_val = df_values.min()
if min_val < 0:
    df_log = np.log2(df_values - min_val + 1)  # 음수를 양수로 shift
else:
    df_log = np.log2(df_values + 1)

# microRNA (Line 260): log2(x + 1)
df_log = np.log2(df + 1)

# RPPA (Line 307-312): 음수 처리 포함
min_val = df_values.min()
if min_val < 0:
    df_log = np.log2(df_values - min_val + 1)
else:
    df_log = np.log2(df_values + 1)

# Methylation (Line 358): NO 변환
# logger.info("Note: NO log2 transformation applied - beta values (0-1)")

# Mutations (Line 405): NO 변환 (impact scores 0-2)
```

#### 4️⃣ 통합 데이터셋 생성

**구현 파일**: `src/preprocessing/integrated_dataset_builder.py`

**실행 방법**:
```bash
cd src/preprocessing
./run_integrated_dataset_builder.sh
# 로그 모니터링: tail -f integrated_dataset_*.log
```

**핵심 로직** (`integrated_dataset_builder.py:186-187`):
```python
# ⚠️ 중요: [측정값, Cox계수] 쌍 생성 (곱셈 아님!)
for feature in common_features:
    enhanced_features[f"{feature}_value"] = measured_values[feature]  # log2 변환된 값
    enhanced_features[f"{feature}_cox"] = cox_coef_mean[feature]      # Cox 계수

# 예시 출력:
# Mutations_BRCA1_val: 1.234 (log2 변환된 측정값)
# Mutations_BRCA1_cox: 0.567 (Cox 회귀계수)
```

**최종 출력 파일**: `./data/processed/integrated_table_cox.parquet`
- **크기**: 4,504 환자 × 32,762 컬럼
- **형식**: 각 유전자마다 `_val`과 `_cox` 2개 컬럼
- **용도**: 🔥 **CoxTabTransformer 훈련의 핵심 입력 파일**

**주요 출력물 (검증 완료):**
```
./data/processed/
├── cox_coefficients_expression.parquet    # 5.4M - Expression Cox 계수
├── cox_coefficients_cnv.parquet           # 2.6M - CNV Cox 계수
├── cox_coefficients_microrna.parquet      # 219K - microRNA Cox 계수
├── cox_coefficients_rppa.parquet          # 75K - RPPA Cox 계수
├── cox_coefficients_mutations.parquet     # 1.3M - Mutation Cox 계수
├── processed_expression_data.parquet      # 739M - log2 변환된 Expression
├── processed_cnv_data.parquet             # 379M - log2 변환된 CNV
├── processed_microrna_data.parquet        # 21M - log2 변환된 microRNA
├── processed_rppa_data.parquet            # 8.6M - log2 변환된 RPPA
├── processed_mutations_data.parquet       # 16M - Mutation impact scores
├── methylation_data_for_tabtransformer.parquet  # 29G - Methylation beta values
├── methylation_table.parquet              # 2.2G - Methylation 통합 테이블
├── processed_clinical_data.parquet        # 1.1M - 임상 데이터 (Cox용)
├── processed_clinical_data_for_methylation.parquet  # 2.0M - 임상 (Methylation용)
└── integrated_table_cox.parquet           # 🔥 176M - Cox 적용 통합 테이블 (핵심!)
```

**추가 메타데이터 파일:**
```
./data/processed/
├── cox_feature_info.json                  # 942K - Cox 특성 정보
├── methylation_feature_info.json          # 911K - Methylation 특성 정보
└── train_val_test_splits.json             # 372K - 데이터셋 분할 정보
```

**데이터 통계 (검증 완료):**
- **Cox 분석 대상**: 4,504명 환자
- **Methylation 데이터**: 8,224명 환자 (Cox 제외, 별도 TabTransformer용)
- **특성 개수**:
  - Expression: 20,531 유전자
  - CNV: 25,128 유전자
  - microRNA: 1,071 miRNAs
  - RPPA: 387 단백질
  - Mutations: 25,423 유전자
  - **합계: 71,520개 특성** (5개 오믹스)
  - **Methylation: 396,065 probes** ⚠️ (샤딩 필요)

**디스크 사용량:**
- Cox 관련 파일: ~2.3GB
- Methylation 파일: ~31GB
- 총 데이터 크기: ~33GB

---

### ❌ 1-3. 병리영상 데이터 전처리 (미착수)

**상태**: ❌ **미착수** (0%)

**현재 상황**:
- WSI(Whole Slide Image) 데이터 다운로드 미완료
- 패치 추출 코드 미구현
- 전처리 파이프라인 미구성

**필요 작업**:
1. ❌ WSI 데이터 다운로드 (TCGA에서)
2. ❌ 패치 추출 스크립트 작성
3. ❌ 배경 제거 및 품질 필터링
4. ❌ 패치 메타데이터 생성

**다음 단계 프롬프트**:

```
Whole Slide Image(WSI) 전처리를 위한 Jupyter Notebook을 작성해주세요:

Notebook 구성:

1. 환경 설정:
   - openslide-python, PIL, opencv 설치 및 임포트
   - matplotlib, numpy, tqdm 임포트

2. WSI 데이터 탐색:
   - 샘플 WSI 파일 로드 및 메타데이터 확인
   - 슬라이드 크기, 배율 정보 출력
   - 썸네일 이미지 시각화

3. 패치 추출 함수 구현:
   - extract_patches() 함수 정의
   - 224x224 또는 384x384 패치로 분할
   - 배경 영역 제거 (HSV threshold 기반)
   - 패치 품질 필터링 (variance 기반 흐림 제거)

4. 배치 처리 및 저장:
   - 여러 WSI 파일을 순차 처리
   - 환자별 패치 폴더 생성
   - 패치 좌표 정보 CSV로 저장
   - 진행률 표시 (tqdm)

5. 패치 품질 검증:
   - 추출된 패치 샘플 시각화
   - 패치당 조직 비율 히스토그램
   - 품질 필터링 전후 비교

6. 데이터셋 통계:
   - 환자별 패치 개수 분포
   - 총 패치 수 및 저장 용량 계산
   - 메타데이터 요약 테이블 생성

파일명: notebooks/02_wsi_preprocessing.ipynb

주요 출력물:
- data/processed/wsi_patches/ (환자별 패치 이미지 폴더)
- patch_metadata.csv (패치 좌표 및 품질 정보)
- wsi_processing_summary.json (전처리 통계 요약)
```

---

## Phase 2: 단일 모달리티 모델 개발

### ✅ 2-1-A. Multi-omics 모델 (CoxTabTransformer) 훈련 (완료)

**상태**: ✅ **완료** (Test AUC: **0.8495**)

**구현 파일**:
- 모델: `src/models/cox_tabtransformer.py`
- 훈련: `src/training/train_tabtransformer.py`
- 유틸리티: `src/utils/tabtransformer_utils.py`

**모델 아키텍처**:
```python
# src/models/cox_tabtransformer.py
CoxTabTransformer(
    clinical_categories=(10, 3, 8, 4, 5),  # 임상 범주형 특성 vocab sizes
    num_omics_features=71_520,  # 5개 오믹스의 총 특성 수
    dim=64,                     # 임베딩 차원
    depth=4,                    # Transformer 레이어 수
    heads=8,                    # Attention 헤드 수
    attn_dropout=0.3,
    ff_dropout=0.3
)

# Line 31: 중요! num_continuous는 features * 2
self.base_transformer = TabTransformer(
    categories=clinical_categories,
    num_continuous=num_omics_features * 2,  # [value, cox] 쌍이므로 *2
    ...
)
```

**입력 데이터 형식 ([측정값, Cox계수] 쌍)**:

**⚠️ 매우 중요**: 모델은 측정값과 Cox계수를 **곱하지 않고** 별도 2개 값으로 처리합니다!

```python
# src/utils/tabtransformer_utils.py:41-42
# ❌ 잘못된 방법: value * cox
# ✅ 올바른 방법: [value, cox] 스택

paired_data = torch.stack([omics_tensor, cox_expanded], dim=2)  # (batch, features, 2)
flattened = paired_data.view(batch_size, -1)  # (batch, features*2)

# 최종 입력 형태:
# [gene1_val, gene1_cox, gene2_val, gene2_cox, gene3_val, gene3_cox, ...]
```

**전체 데이터 파이프라인**:

```python
# 1. 원본 데이터 → log2 변환 (cox_feature_engineer.py)
expression_log2 = np.log2(expression_raw + 1)

# 2. Cox 회귀분석 → 계수 저장 (cox_feature_engineer.py)
cox_coefficients[cancer_type][gene] = cph.params_[gene]

# 3. [value, cox] 쌍 생성 (integrated_dataset_builder.py:186-187)
enhanced_features[f"{gene}_val"] = expression_log2[gene]  # log2 변환된 값
enhanced_features[f"{gene}_cox"] = cox_coefficients[gene]  # Cox 계수

# 4. 통합 테이블 저장
integrated_table_cox.parquet  # (4,504 × 32,762)

# 5. 모델 입력용 텐서 변환 (tabtransformer_utils.py:41-42)
paired_data = torch.stack([values, cox], dim=2)  # (batch, 71520, 2)
flattened = paired_data.view(batch, -1)           # (batch, 143040)
#                                                  # = 71520 * 2

# 6. TabTransformer 처리 (cox_tabtransformer.py)
x_cont_embedded = self.continuous_embedding(omics_continuous)
# → Transformer layers → 생존 예측
```

**훈련 결과**:
- **Test AUC**: 0.8495
- **환자 수**: 4,504명
- **특성 수**: 71,520개 (5개 오믹스 합계)
- **입력 차원**: 143,040 (71,520 × 2)
- **체크포인트**: `src/training/checkpoints/seed_XX/best_cox_tabtransformer.pth`

**모델 실행 방법**:
```bash
cd src/training

# 단일 시드 훈련
python train_tabtransformer.py --model cox --epochs 50 --lr 1e-4 --batch_size 32

# 앙상블 훈련 (여러 시드)
python train_tabtransformer.py --model cox --ensemble --n_seeds 5 --epochs 50
```

**훈련 로직**:
1. `integrated_table_cox.parquet` 로드 (4,504 × 32,762)
2. `_val`과 `_cox` 컬럼을 [value, cox] 쌍으로 스택
3. TabTransformer에 입력 (batch, 143040)
4. 3년 생존 여부를 Binary Classification으로 학습
5. Best 모델을 `checkpoints/seed_XX/`에 저장

---

### 🔄 2-1-B. Methylation 모델 (MethylationTabTransformer) 훈련 (대기 중)

**상태**: 🔄 **코드 구현 완료, 샤딩 전략 필요** (30%)

**현재 상황**:
- ✅ 모델 코드 구현 완료: `src/models/methylation_tabtransformer.py` (170 lines)
- ✅ 훈련 스크립트 준비: `src/training/train_tabtransformer.py` (methylation 모드 포함)
- ✅ 데이터 준비 완료: `methylation_data_for_tabtransformer.parquet` (29GB, 8,224 환자)
- ✅ Methylation 통합 테이블: `methylation_table.parquet` (2.2GB)
- ❌ 샤딩 전략 미구현
- ❌ 실제 훈련 미수행

**이유**: 메틸레이션 데이터가 너무 큼 (396,065 probes) → **샤딩(Sharding) 필요**

**구현 파일 (검증 완료)**:
- ✅ 모델: `src/models/methylation_tabtransformer.py`
- ✅ 훈련: `src/training/train_tabtransformer.py` (methylation 모드)
- ✅ 데이터: `data/processed/methylation_data_for_tabtransformer.parquet` (29GB)
- ✅ 통합 테이블: `data/processed/methylation_table.parquet` (2.2GB)

**문제점 분석**:
- 396,065개의 probes는 메모리에 한 번에 로드 불가
- Feature selection layer만으로는 부족 (여전히 메모리 부족)
- GPU 메모리 제약: 48GB VRAM으로도 부족
- 배치 처리 시 OOM(Out of Memory) 발생

**필요 작업**:
1. ❌ 샤딩 전략 설계 및 구현
2. ❌ 샤드별 모델 훈련
3. ❌ Fusion layer 구현
4. ❌ End-to-end 파인튜닝

**다음 단계 프롬프트**:

```
메틸레이션 데이터(396,065 probes)의 샤딩(Sharding) 및 병합 전략을 구현해주세요:

요구사항:

1. 데이터 샤딩 전략:
   - 메틸레이션 데이터를 여러 샤드(shard)로 분할
   - 각 샤드 크기: 약 40,000 probes (총 10개 샤드)
   - 샤드별로 variance-based feature selection 적용 (상위 5,000개씩 선택)

2. 샤드별 모델 훈련:
   - 각 샤드마다 독립적인 MethylationTabTransformer 훈련
   - 샤드별 중간 representation (256-dim) 추출
   - 샤드별 모델 체크포인트 저장

3. 샤드 병합 전략:
   - 10개 샤드의 representation을 concat (256 x 10 = 2,560-dim)
   - 최종 fusion layer로 2,560-dim → 256-dim 압축
   - 생존 예측 헤드 추가

4. 구현 파일:
   - `src/models/sharded_methylation_tabtransformer.py`
   - `src/training/train_sharded_methylation.py`
   - `src/utils/methylation_sharding_utils.py`

5. 훈련 절차:
   - Step 1: 각 샤드별 모델 개별 훈련
   - Step 2: 모든 샤드 모델을 freeze하고 fusion layer만 훈련
   - Step 3: End-to-end fine-tuning (optional)

파일명: notebooks/04_methylation_sharding_training.ipynb

주요 출력물:
- methylation_shard_0.parquet ~ methylation_shard_9.parquet
- best_methylation_shard_0.pth ~ best_methylation_shard_9.pth
- best_methylation_fused_model.pth
```

**메틸레이션 데이터 통계**:
- **환자 수**: 8,224명 (Cox 분석보다 많음)
- **Probes 수**: 396,065개
- **데이터 크기**: 매우 큼 (샤딩 필수)
- **Cox 계수**: 없음 (별도 TabTransformer로 처리)

---

### ❌ 2-2. 병리영상 모델 (Swin Transformer) 훈련 (미착수)

**상태**: ❌ **미착수** (0%)

**현재 상황**:
- ❌ WSI 데이터 미확보
- ❌ `src/models/wsi_swin_transformer.py` 미구현
- ❌ 훈련 스크립트 미작성
- ❌ MIL(Multiple Instance Learning) 구조 미설계

**필요 작업**:
1. ❌ WSI 패치 데이터 준비 (Phase 1-3 선행 필요)
2. ❌ Swin Transformer 백본 구현
3. ❌ MIL 어댑터 구현
4. ❌ Attention pooling 구현
5. ❌ 훈련 파이프라인 구축

**의존성**: Phase 1-3 (WSI 전처리) 완료 필요

**다음 단계 프롬프트**:

```
병리영상 분석을 위한 Swin Transformer 모델을 구현해주세요:

아키텍처:

1. WSISwinTransformer 클래스 구현
   - 사전훈련된 Swin Transformer 백본 사용 (ImageNet 또는 histopathology 사전학습 모델)
   - Multiple Instance Learning (MIL) 어댑터 추가
   - Attention pooling for patch aggregation
   - ROI-free 학습 방식 (이미지 전체 레이블만 사용)

2. 핵심 기능:
   - 가변 개수의 패치 처리
   - Attention weights 저장 (시각화용)
   - 메모리 효율적 배치 처리
   - 생존 예측을 위한 출력 헤드

3. 훈련 절차:
   - Patch-level feature extraction (Swin Transformer)
   - Patch aggregation (Attention pooling 또는 MIL)
   - Slide-level survival prediction

파일명:
- src/models/wsi_swin_transformer.py
- src/training/train_wsi_swin.py

실행 예시:
python train_wsi_swin.py --epochs 50 --lr 1e-5 --batch_size 1 --num_patches 100
```

---

## Phase 3: 멀티모달 융합 및 LLM 파인튜닝 (미착수)

**상태**: ⏸️ **대기 중** (0%) - Phase 2 완료 후 진행

**의존성**:
- Phase 2-A ✅ 완료 (CoxTabTransformer)
- Phase 2-B 🔄 대기 중 (Methylation - 샤딩 필요)
- Phase 2-C ❌ 미착수 (WSI Swin Transformer)

**블로커**: Phase 2-B와 2-C 완료 필요

---

### ⏸️ 3-1. 추론 텍스트 데이터셋 구축 (미착수)

**상태**: ⏸️ **미착수** (0%)

**필요 작업**:
1. ❌ 템플릿 기반 추론 텍스트 생성 스크립트
2. ❌ LLM을 활용한 초기 추론 생성
3. ❌ 전문가 검토 및 수정 프로세스 구축
4. ❌ (멀티모달 데이터) → (추론 텍스트) 쌍 데이터셋 생성

**목표**: LLM 파인튜닝을 위한 (멀티모달 데이터) → (전문가 추론 텍스트) 쌍 생성

**방법**:
1. **템플릿 기반 생성**:
   - "유전자 [GENE] 의 Cox 계수가 [COEF]로 높아 위험 요인으로 작용하며..."
   - "이미지상 [PATTERN] 패턴이 관찰됩니다..."
   - 전문가 감수

2. **LLM 활용**:
   - GPT-4로 초기 추론 생성
   - 전문가 검토 및 수정

---

### ⏸️ 3-2. 융합 아키텍처 설계 및 구현 (미착수)

**상태**: ⏸️ **미착수** (0%)

**필요 작업**:
1. ❌ Cross-modal Attention 모듈 구현
2. ❌ Projection Layer 설계
3. ❌ 멀티모달 융합 모델 통합
4. ❌ 훈련 파이프라인 구축

**구조**:
```
[CoxTabTransformer] → 256-dim representation ✅ (완료)
[MethylationTabTransformer (Fused)] → 256-dim representation 🔄 (대기)
[SwinTransformer] → 256-dim representation ❌ (미착수)
           ↓
[Cross-modal Attention] → 768-dim fused representation ❌ (미구현)
           ↓
[Projection Layer] → LLM token embedding space ❌ (미구현)
           ↓
[LLM Input] ❌ (미구현)
```

---

### ⏸️ 3-3. 공개 LLM 선정 및 파인튜닝 (미착수)

**상태**: ⏸️ **미착수** (0%)

**필요 작업**:
1. ❌ LLM 모델 선정 (Llama 3 / Qwen2 / Mistral-7B)
2. ❌ LoRA 설정 및 구현
3. ❌ 파인튜닝 스크립트 작성
4. ❌ 평가 메트릭 설정

**추천 모델**: Llama 3, Qwen2, Mistral-7B

**파인튜닝 방법**:
- LoRA (Low-Rank Adaptation)
- [융합 임베딩] + 텍스트 프롬프트 → 추론 텍스트 생성

---

## Phase 4: 모델 평가 및 시각화 (미착수)

**상태**: ⏸️ **대기 중** (0%) - Phase 3 완료 후 진행

**의존성**: Phase 3 (멀티모달 융합 및 LLM) 완료 필요

**블로커**: Phase 2-B, 2-C, Phase 3 전체 완료 필요

---

### ⏸️ 4-1. 최종 LLM 모델 평가 (미착수)

**상태**: ⏸️ **미착수** (0%)

**필요 작업**:
1. ❌ 정량 평가 메트릭 설정 (AUC, Accuracy, F1-score)
2. ❌ 정성 평가 프로토콜 설계
3. ❌ 의료 전문가 리뷰 프로세스 구축
4. ❌ 평가 결과 분석 및 보고서 작성

**평가 항목**:
- 정량 평가: AUC, Accuracy, Precision, Recall, F1-score
- 정성 평가: 의료 전문가 리뷰, 추론 텍스트 품질 평가

---

### ⏸️ 4-2. 설명 가능성(XAI) 시각화 구현 (미착수)

**상태**: ⏸️ **미착수** (0%)

**필요 작업**:
1. ❌ Attention 맵 시각화 모듈 구현
2. ❌ 병리 이미지 히트맵 오버레이 구현
3. ❌ 통합 대시보드 개발
4. ❌ 사용자 인터페이스 구축

**구현 목표**:
- Attention 맵 시각화 (Swin Transformer)
- 병리 이미지 위 히트맵 오버레이
- 생성된 추론 텍스트와 어텐션 히트맵 함께 제공
- 유전자/단백질 중요도 시각화

---

## 📝 다음 단계 우선순위 (2025-10-22 기준)

### ✅ 완료된 작업:

1. ✅ **Phase 1: 데이터 준비 및 전처리** (100%)
   - ✅ TCGA 데이터 다운로드 및 정제
   - ✅ Cox 회귀분석 (1,279 lines 코드)
   - ✅ 통합 데이터셋 생성 (822 lines 코드)
   - ✅ 코드 리팩토링 (`notebooks/` → `src/preprocessing/`)

2. ✅ **Phase 2-A: CoxTabTransformer** (100%)
   - ✅ 모델 구현 (128 lines)
   - ✅ 훈련 완료 (1,001 lines 훈련 스크립트)
   - ✅ 5개 시드 앙상블 훈련 완료
   - ✅ Test AUC 0.8495 달성 (목표: 0.85)
   - ✅ 체크포인트 저장 (`checkpoints/seed_XX/`)

---

### 🔥 즉시 진행해야 할 작업 (우선순위 1):

**1. 메틸레이션 모델 샤딩 구현 및 훈련** ⭐ **최우선**
   - ❌ 샤딩 전략 설계 및 구현
   - ❌ `src/models/sharded_methylation_tabtransformer.py` 작성
   - ❌ 샤드별 모델 개별 훈련 (10개 샤드, 각 40K probes)
   - ❌ Fusion layer 구현 및 훈련
   - ❌ End-to-end 파인튜닝
   - **현재 블로커**: 396,065 probes OOM 문제
   - **목표**: Methylation 모델 훈련 완료 및 성능 검증

---

### 📅 중기 목표 (우선순위 2):

**2. 병리영상 전처리 및 모델 구현**
   - ❌ WSI 데이터 다운로드
   - ❌ 패치 추출 스크립트 작성 (`src/preprocessing/wsi_preprocessing.py`)
   - ❌ 배경 제거 및 품질 필터링
   - ❌ Swin Transformer 모델 구현 (`src/models/wsi_swin_transformer.py`)
   - ❌ MIL(Multiple Instance Learning) 구조 구현
   - ❌ Attention pooling 구현
   - ❌ 훈련 파이프라인 구축
   - **의존성**: WSI 데이터 확보 필요
   - **목표**: WSI 모델 훈련 완료 및 성능 검증

---

### 🎯 장기 목표 (우선순위 3):

**3. 멀티모달 융합 및 LLM 통합**
   - ❌ 추론 텍스트 데이터셋 구축
   - ❌ Cross-modal Attention 구현
   - ❌ Projection Layer 설계
   - ❌ LLM 모델 선정 (Llama 3 / Qwen2 / Mistral-7B)
   - ❌ LoRA 파인튜닝
   - **의존성**: Phase 2-B, 2-C 완료 필요
   - **목표**: 설명 가능한 멀티모달 예측 시스템 완성

**4. 평가 및 시각화**
   - ❌ 정량/정성 평가 메트릭 설정
   - ❌ Attention 맵 시각화
   - ❌ 병리 이미지 히트맵 오버레이
   - ❌ 통합 대시보드 구현
   - **의존성**: Phase 3 완료 필요
   - **목표**: 의료 전문가 리뷰 및 논문 출판

---

## 🔬 모델 구조 요약

### 최종 아키텍처:

```
┌─────────────────────────────────────────────────────────┐
│  Input Layer: Patient Data                              │
├─────────────────────────────────────────────────────────┤
│  1. Clinical + 5 Omics (Cox-based)                      │
│     → CoxTabTransformer → 256-dim                       │
│  2. Methylation (396K probes, sharded)                  │
│     → 10 Shard Models → Fusion → 256-dim               │
│  3. WSI Patches (H&E images)                            │
│     → Swin Transformer → MIL → 256-dim                  │
├─────────────────────────────────────────────────────────┤
│  Cross-Modal Attention Fusion                           │
│     (256 + 256 + 256) → 768-dim fused representation   │
├─────────────────────────────────────────────────────────┤
│  Projection to LLM Token Space                          │
│     768-dim → LLM embedding dimension                   │
├─────────────────────────────────────────────────────────┤
│  LLM Fine-tuning (Llama 3 / Qwen2 / Mistral)           │
│     Input: [Fused Embedding] + [Text Prompt]           │
│     Output: Reasoning Text + Survival Prediction       │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 현재 성능 지표 (2025-10-22 기준)

| 모델 | 환자 수 | 특성 수 | Test AUC | 체크포인트 | 상태 |
|------|---------|---------|----------|-----------|------|
| **CoxTabTransformer** | **4,504** | **71,520** | **0.8495** | ✅ 5개 시드 | ✅ **완료** |
| MethylationTabTransformer | 8,224 | 396,065 | - | ❌ 없음 | 🔄 샤딩 필요 |
| SwinTransformer | - | - | - | ❌ 없음 | ❌ 미착수 |
| Multimodal LLM | - | - | - | ❌ 없음 | ⏸️ 미착수 |

**CoxTabTransformer 상세 결과**:
- **Test AUC**: 0.8495 (목표 0.85 달성)
- **훈련 환자**: 4,504명 (train 70%, val 15%, test 15%)
- **입력 차원**: 143,040 (71,520 특성 × 2 [value, cox])
- **앙상블 모델**: 5개 시드 (42, 43, 45, 48, 52)
- **체크포인트 경로**: `src/training/checkpoints/seed_XX/best_cox_tabtransformer.pth`
- **결과 파일**: `results/cox_tabtransformer_ensemble_results.json` (87KB)

**데이터 통계**:
- **총 환자 수**: 4,504명 (Cox 분석), 8,224명 (Methylation)
- **총 특성 수**: 71,520개 (5개 오믹스), 396,065개 (Methylation probes)
- **데이터 크기**: ~33GB (processed)
- **체크포인트 크기**: ~100MB per seed

---

## 💡 중요 설계 원칙

1. **독립적 모델 훈련**:
   - CoxTabTransformer, MethylationTabTransformer, SwinTransformer는 각각 독립적으로 훈련
   - 각 모델이 단독으로도 예측 가능하도록 설계
   - 최종 융합 단계에서 3개 모델 결합

2. **Cox 계수 활용**:
   - Cox 회귀계수를 "생존 예후에 대한 사전 지식"으로 활용
   - `[측정값, Cox계수]` 쌍으로 모델에 도메인 지식 주입
   - 메틸레이션은 Cox 제외 (데이터 크기 문제)

3. **Explainability 우선**:
   - Attention weights 저장 및 시각화
   - LLM을 통한 자연어 추론 생성
   - 의료 전문가가 이해 가능한 설명 제공

4. **확장 가능한 구조**:
   - 각 모달리티는 256-dim representation으로 통일
   - 새로운 모달리티 추가 용이
   - Modular design

---

## 📁 주요 파일 위치

### 데이터:
```
./data/
├── raw/                                      # 원본 TCGA 데이터
│   ├── *_expression_whitelisted.tsv
│   ├── CNV.*_whitelisted.tsv
│   ├── *_miRNASeq_whitelisted.tsv
│   ├── *_RPPA_whitelisted.tsv
│   ├── *_whitelisted.maf.gz
│   ├── *_Methylation450_whitelisted.tsv
│   └── clinical_*_with_followup.tsv
├── processed/
│   ├── cox_coefficients_*.parquet            # Cox 계수 룩업 테이블
│   ├── processed_*_data.parquet              # log2 변환된 오믹스 데이터
│   ├── integrated_table_cox.parquet          # 🔥 핵심 훈련 파일 (4,504 × 32,762)
│   ├── train_val_test_splits.json            # 데이터셋 분할 정보
│   ├── methylation_data_for_tabtransformer.parquet  # Methylation 데이터
│   └── processed_clinical_data.parquet       # 임상 데이터
```

### 코드:
```
./src/
├── preprocessing/                            # ✅ 데이터 전처리 (리팩토링 완료)
│   ├── cox_feature_engineer.py               # Cox 회귀분석 실행
│   ├── integrated_dataset_builder.py         # 통합 데이터셋 생성
│   ├── cancer_multiomics_dataset.py          # PyTorch Dataset 클래스
│   ├── run_cox_feature_engineer.sh           # Cox 분석 래퍼
│   └── run_integrated_dataset_builder.sh     # 빌더 래퍼
├── models/
│   ├── cox_tabtransformer.py                 # ✅ CoxTabTransformer 모델
│   ├── methylation_tabtransformer.py         # 🔄 MethylationTabTransformer 모델
│   └── wsi_swin_transformer.py               # ⏸️ (미구현)
├── training/
│   ├── train_tabtransformer.py               # ✅ 훈련 스크립트
│   ├── run_tabtransformer_training.sh        # 훈련 래퍼
│   └── checkpoints/                          # 모델 체크포인트
│       ├── seed_42/
│       │   └── best_cox_tabtransformer.pth
│       ├── seed_45/
│       │   └── best_cox_tabtransformer.pth
│       └── seed_XX/...
└── utils/
    ├── tabtransformer_utils.py               # ✅ [value, cox] 쌍 변환 유틸리티
    ├── feature_converter.py                  # ✅ 추론용 데이터 변환
    └── user_data_pipeline.py                 # ✅ 추론용 파이프라인
```

### Notebooks:
```
./notebooks/
├── 02_wsi_preprocessing.ipynb                # ⏸️ (미구현)
└── 04_methylation_sharding_training.ipynb    # 🔄 (다음 단계)

⚠️ Note: Cox 분석 및 통합 데이터셋 코드는 src/preprocessing/로 이동 완료
```

---

## 🚀 실행 가이드

### 데이터 전처리 (한 번만 실행):

```bash
# Step 1: Cox 회귀분석 실행
cd src/preprocessing
./run_cox_feature_engineer.sh
# 로그 모니터링: tail -f cox_analysis_*.log
# 예상 소요 시간: 1-2시간

# Step 2: 통합 데이터셋 생성
./run_integrated_dataset_builder.sh
# 로그 모니터링: tail -f integrated_dataset_*.log
# 예상 소요 시간: 10-30분

# 출력 확인
ls -lh ../../data/processed/integrated_table_cox.parquet
# 예상 크기: ~176MB (4,504 × 32,762)
```

### CoxTabTransformer 훈련:

```bash
cd src/training

# 단일 시드 훈련
python train_tabtransformer.py --model cox --epochs 50 --lr 1e-4 --batch_size 32

# 앙상블 훈련 (여러 시드)
python train_tabtransformer.py --model cox --ensemble --n_seeds 5 --epochs 50

# 훈련 결과 확인
ls -lh checkpoints/seed_*/best_cox_tabtransformer.pth
```

### MethylationTabTransformer 훈련 (샤딩 후 실행):
```bash
cd src/training
python train_sharded_methylation.py --epochs 50 --lr 5e-5 --batch_size 16
```

---

## ⚠️ 주의사항

1. **[측정값, Cox계수] 쌍 형식 (매우 중요!)**:
   - ❌ **잘못된 방법**: `value * cox_coefficient` (곱셈)
   - ✅ **올바른 방법**: `[value, cox_coefficient]` (2개 값 스택)
   - 코드: `torch.stack([values, cox], dim=2).view(batch, -1)`
   - 모델 입력 차원: `num_continuous = num_features * 2`

2. **log2 변환 일관성**:
   - Expression, CNV, microRNA, RPPA: `log2(x + 1)` 필수
   - CNV, RPPA 음수 처리: `log2(x - min + 1)`
   - Methylation, Mutations: 변환 없음
   - **추론 시 동일한 변환 적용 필수**

3. **특성 순서 일치**:
   - `integrated_table_cox.parquet`의 컬럼 순서와 동일하게 유지
   - 훈련 시와 추론 시 특성 순서가 다르면 예측 실패

4. **Cox 계수 매칭**:
   - 환자의 암종(cancer_type)에 해당하는 Cox 계수만 사용
   - 암종별로 Cox 계수가 다르므로 주의

5. **메틸레이션 데이터 크기**:
   - 396,065 probes는 메모리에 한 번에 로드 불가
   - 반드시 샤딩 전략 구현 후 훈련

6. **데이터 정렬**:
   - 모든 데이터셋의 환자 ID 정렬 확인
   - Common patients만 사용

7. **GPU 메모리**:
   - Methylation 모델은 배치 크기를 작게 설정 (16 이하)
   - WSI 모델은 배치 크기 1로 시작

8. **모델 체크포인트 경로**:
   - 체크포인트는 `src/training/checkpoints/seed_XX/` 형식
   - 앙상블 시 여러 시드 모델 사용 가능

---

## 📚 참고 문헌 및 라이브러리

### 주요 라이브러리:
- **tab-transformer-pytorch**: TabTransformer 구현체
- **lifelines**: Cox 회귀분석
- **torch**: PyTorch
- **pandas, numpy**: 데이터 처리
- **openslide-python**: WSI 처리 (예정)

### 참고 논문:
- TabTransformer: Tabular Data Modeling Using Contextual Embeddings
- Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
- Multiple Instance Learning for Histopathology Images

---

**마지막 업데이트**: 2025년 10월 22일
**작성자**: Cancer Foundation Model 개발팀
**프로젝트 상태**: Phase 2-A 완료 (CoxTabTransformer), Phase 2-B 대기 중 (Methylation 샤딩 필요)

---

## 📝 주요 업데이트 내역

### 2025-10-22: 중간 정리 - Cox 파이프라인 검증 및 문서화 완료

**1. 코드 검증 및 구조 확인**
- ✅ 전체 코드베이스 검증 완료 (3,710 lines 핵심 코드)
- ✅ 파일 구조 검증: preprocessing, models, training, utils
- ✅ 데이터 파일 검증: 33GB processed data, 5개 시드 체크포인트
- ✅ 실행 가능한 모든 스크립트 확인 (shell wrappers)

**2. Cox 기반 파이프라인 문서화**
- ✅ Cox 기반 멀티오믹스 파이프라인 전체 검증 완료
- ✅ [측정값, Cox계수] 쌍 형식 명확화 (곱셈 아님!)
- ✅ log2 변환 규칙 상세 문서화 (코드 라인 번호 포함)
- ✅ 데이터 흐름 6단계 시각화 (원본 → 모델 입력)
- ✅ `integrated_table_cox.parquet` 핵심 파일 확인 (176MB)

**3. 코드 리팩토링**
- ✅ `notebooks/` → `src/preprocessing/` 이동 완료
- ✅ `01_cox_feature_engineering.py` → `cox_feature_engineer.py`
- ✅ `03_integrated_dataset.py` → `integrated_dataset_builder.py`
- ✅ `integrated_dataset.py` → `cancer_multiomics_dataset.py`
- ✅ Shell wrapper 스크립트 생성 및 경로 수정

**4. 문서 업데이트**
- ✅ README.md 업데이트: 사용자 가이드, 예측 워크플로우
- ✅ CFM.vibe_coding_guide.md 전면 개편
  - ✅ 진행 상황 요약 테이블 (Phase별 진행률)
  - ✅ 완료/미완료 작업 명확히 표기 (✅/❌/🔄/⏸️)
  - ✅ 필요 작업 체크리스트 추가
  - ✅ 의존성 및 블로커 명시
  - ✅ 다음 단계 우선순위 정리
  - ✅ 파일 크기 및 통계 검증
  - ✅ 체크포인트 경로 수정 (`checkpoints/seed_XX/`)

**5. 현재 상태 명확화**
- ✅ Phase 1: 100% 완료
- ✅ Phase 2-A: 100% 완료 (CoxTabTransformer, AUC 0.8495)
- 🔄 Phase 2-B: 30% (Methylation - 코드 완료, 샤딩 필요)
- ❌ Phase 2-C: 0% (WSI 미착수)
- ⏸️ Phase 3: 0% (블로커: Phase 2-B, 2-C)
- ⏸️ Phase 4: 0% (블로커: Phase 3)

**6. 다음 단계 로드맵**
- ⭐ **최우선**: Methylation 샤딩 전략 구현 및 훈련
- 📅 **중기**: WSI 전처리 및 Swin Transformer 구현
- 🎯 **장기**: 멀티모달 융합 및 LLM 파인튜닝

---

### 이전 주요 업데이트

**2025-08-20**: CoxTabTransformer 앙상블 훈련 완료
- ✅ 5개 시드 앙상블 훈련 완료
- ✅ Test AUC 0.8495 달성

**2025-08-18**: 통합 데이터셋 생성 완료
- ✅ `integrated_table_cox.parquet` 생성 (176MB)
- ✅ Train/Val/Test 분할 완료

**2025-08-14**: Cox 회귀분석 완료
- ✅ 5개 오믹스에 대한 Cox 분석 완료
- ✅ Cox 계수 룩업 테이블 생성
