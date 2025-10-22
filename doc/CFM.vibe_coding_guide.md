# Cancer Foundation Model 구축 가이드 (Vibe Coding 프롬프트)

## 📋 프로젝트 목표

멀티오믹스(Tabular)와 병리 영상(Image) 데이터를 통합하여, **설명 가능한(Explainable)** 암 예후 예측 파운데이션 모델을 구축한다. 모델은 최종적으로 예측에 대한 **판단 근거를 자연어 텍스트와 시각적 히트맵으로 제시**해야 한다.

---

## 🎯 전체 진행 상황 요약

| Phase | 단계 | 상태 | 비고 |
|-------|------|------|------|
| Phase 1 | 데이터 준비 및 전처리 | ✅ 완료 (일부 진행 중) | Cox 분석 완료, WSI 전처리 미착수 |
| Phase 2 | 단일 모달리티 모델 개발 | 🔄 진행 중 | CoxTabTransformer 완료 (AUC: 0.8495) |
| Phase 3 | 멀티모달 융합 및 LLM | ⏸️ 대기 중 | Phase 2 완료 후 진행 |
| Phase 4 | 평가 및 시각화 | ⏸️ 대기 중 | Phase 3 완료 후 진행 |

**📊 현재 달성 수치:**
- **Multi-omics TabTransformer**: Test AUC **0.8495** ✅ (목표: 0.85)
- **데이터셋**: 4,504명 환자, 5개 오믹스 데이터 (Expression, CNV, microRNA, RPPA, Mutation)
- **Methylation 데이터**: 8,224명 환자, 396,065 probes (샤딩 필요)

---

## Phase 1: 데이터 준비 및 전처리

### ✅ 1-1. 멀티모달 데이터 다운로드 및 정제 (완료)

**상태**: ✅ **완료**

**달성 내용**:
- TCGA PANCAN 멀티오믹스 데이터 다운로드 완료
- 원발성 암(Primary Tumor) 샘플 필터링 완료
- 환자 ID 표준화 완료 (TCGA-XX-XXXX 형식)

**데이터 파일 위치**: `./data/raw/`

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

**구현 파일**: `notebooks/01_cox_feature_engineering.ipynb`

**달성 내용**:

#### 1️⃣ Cox 회귀분석 수행
- **대상**: Expression, CNV, microRNA, RPPA, Mutation (메틸레이션 제외)
- **엔드포인트**: 3년 생존 예측
- **암종별 분석**: 각 암종(cancer type)별로 Cox 비례위험 회귀분석 실행
- **모든 특성 보존**: p-value와 관계없이 모든 유전자/특성의 Cox 계수 저장

#### 2️⃣ 회귀계수 룩업 테이블 생성
- **구조**: `{cancer_type: {omics_type: {feature_name: cox_coefficient}}}`
- **파일**: `./data/processed/cox_coefficients_*.parquet`
- **용도**: TabTransformer 입력 시 [측정값, Cox계수] 쌍 생성

#### 3️⃣ 데이터 변환 및 정규화
- **log2 변환 적용**: Expression, CNV, microRNA, RPPA
  - Expression: `log2(x + 1)`
  - CNV: `log2(|x| + 1) × sign(x)` (음수 고려)
  - microRNA: `log2(x + 1)`
  - RPPA: `log2(|x| + 1) × sign(x)` (음수 고려)
- **NO 변환**: Methylation (beta values 0-1), Mutation (impact scores 0-2)

#### 4️⃣ 최종 입력 테이블 생성
- **형식**: 각 환자에 대해 `[유전자1_발현량, 유전자1_Cox계수], [유전자2_발현량, 유전자2_Cox계수], ...`
- **저장 위치**: `./data/processed/integrated_table_cox.parquet`

**주요 출력물**:
```
./data/processed/
├── cox_coefficients_expression.parquet    # Expression Cox 계수
├── cox_coefficients_cnv.parquet           # CNV Cox 계수
├── cox_coefficients_microrna.parquet      # microRNA Cox 계수
├── cox_coefficients_rppa.parquet          # RPPA Cox 계수
├── cox_coefficients_mutations.parquet     # Mutation Cox 계수
├── processed_expression_data.parquet      # log2 변환된 Expression 데이터
├── processed_cnv_data.parquet             # log2 변환된 CNV 데이터
├── processed_microrna_data.parquet        # log2 변환된 microRNA 데이터
├── processed_rppa_data.parquet            # log2 변환된 RPPA 데이터
├── processed_mutations_data.parquet       # Mutation 데이터 (impact scores)
├── methylation_data_for_tabtransformer.parquet  # Methylation 데이터 (beta values)
├── processed_clinical_data.parquet        # 임상 데이터
└── integrated_table_cox.parquet           # Cox 적용 오믹스 통합 테이블
```

**데이터 통계**:
- **Cox 분석 대상**: 4,504명 환자
- **Methylation 데이터**: 8,224명 환자 (Cox 제외, 별도 TabTransformer용)
- **특성 개수**:
  - Expression: 20,531 유전자
  - CNV: 25,128 유전자
  - microRNA: 1,071 miRNAs
  - RPPA: 387 단백질
  - Mutations: 25,423 유전자
  - **Methylation: 396,065 probes** ⚠️ (샤딩 필요)

---

### ⏸️ 1-3. 병리영상 데이터 전처리 (미착수)

**상태**: ⏸️ **미착수**

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
CoxTabTransformer(
    clinical_categories=(age_groups, gender, stage, ...),  # 임상 범주형 특성
    num_omics_features=71,520,  # 5개 오믹스의 모든 특성
    dim=64,
    depth=4,
    heads=8,
    attn_dropout=0.3,
    ff_dropout=0.3
)
```

**입력 데이터 형식**:
- **Clinical categorical**: 범주형 임상 변수 (나이, 성별, 병기 등)
- **Omics continuous**: `[측정값, Cox계수]` 쌍으로 flatten된 형태
  - 예: `[BRCA1_expression, BRCA1_cox_coef, TP53_expression, TP53_cox_coef, ...]`

**훈련 결과**:
- **Test AUC**: 0.8495
- **환자 수**: 4,504명
- **특성 수**: 71,520개 (5개 오믹스 합계)

**모델 실행 방법**:
```bash
cd src/training
python train_tabtransformer.py --model cox --epochs 50 --lr 1e-4 --batch_size 32
```

**훈련 로직**:
1. Cox 계수 룩업 테이블 로드
2. 각 환자의 오믹스 측정값과 해당 암종의 Cox 계수를 매칭
3. `[측정값, Cox계수]` 쌍을 TabTransformer에 입력
4. 3년 생존 여부를 Binary Classification으로 학습

---

### 🔄 2-1-B. Methylation 모델 (MethylationTabTransformer) 훈련 (미완성)

**상태**: 🔄 **코드 구현 완료, 훈련 미완성**

**이유**: 메틸레이션 데이터가 너무 큼 (396,065 probes) → **샤딩(Sharding) 필요**

**구현 파일**:
- 모델: `src/models/methylation_tabtransformer.py`
- 훈련: `src/training/train_tabtransformer.py` (methylation 모드)

**문제점**:
- 396,065개의 probes는 메모리에 한 번에 로드 불가
- Feature selection layer만으로는 부족 (여전히 메모리 부족)

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

### ⏸️ 2-2. 병리영상 모델 (Swin Transformer) 훈련 (미착수)

**상태**: ⏸️ **미착수**

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

**상태**: ⏸️ **대기 중** (Phase 2 완료 후 진행)

### 3-1. 추론 텍스트 데이터셋 구축

**목표**: LLM 파인튜닝을 위한 (멀티모달 데이터) → (전문가 추론 텍스트) 쌍 생성

**방법**:
1. **템플릿 기반 생성**:
   - "유전자 [GENE] 의 Cox 계수가 [COEF]로 높아 위험 요인으로 작용하며..."
   - "이미지상 [PATTERN] 패턴이 관찰됩니다..."
   - 전문가 감수

2. **LLM 활용**:
   - GPT-4로 초기 추론 생성
   - 전문가 검토 및 수정

### 3-2. 융합 아키텍처 설계 및 구현

**구조**:
```
[CoxTabTransformer] → 256-dim representation
[MethylationTabTransformer (Fused)] → 256-dim representation
[SwinTransformer] → 256-dim representation
           ↓
[Cross-modal Attention] → 768-dim fused representation
           ↓
[Projection Layer] → LLM token embedding space
           ↓
[LLM Input]
```

### 3-3. 공개 LLM 선정 및 파인튜닝

**추천 모델**: Llama 3, Qwen2, Mistral-7B

**파인튜닝 방법**:
- LoRA (Low-Rank Adaptation)
- [융합 임베딩] + 텍스트 프롬프트 → 추론 텍스트 생성

---

## Phase 4: 모델 평가 및 시각화 (미착수)

**상태**: ⏸️ **대기 중** (Phase 3 완료 후 진행)

### 4-1. 최종 LLM 모델 평가

- 정량 평가: AUC, Accuracy
- 정성 평가: 의료 전문가 리뷰

### 4-2. 설명 가능성(XAI) 시각화 구현

- Attention 맵 시각화 (Swin Transformer)
- 병리 이미지 위 히트맵 오버레이
- 생성된 추론 텍스트와 어텐션 히트맵 함께 제공

---

## 📝 다음 단계 우선순위

### 🔥 즉시 진행해야 할 작업:

1. **메틸레이션 모델 샤딩 구현 및 훈련**
   - 샤딩 전략 구현
   - 샤드별 모델 훈련
   - Fusion layer 훈련

2. **병리영상 전처리**
   - WSI 패치 추출
   - 패치 품질 검증

3. **Swin Transformer 모델 구현 및 훈련**
   - ROI-free MIL 방식
   - Attention pooling

### 📅 중기 목표:

4. **추론 텍스트 데이터셋 구축**
5. **멀티모달 융합 아키텍처 구현**
6. **LLM 파인튜닝**

### 🎯 최종 목표:

7. **설명 가능한 예측 시스템 완성**
   - 자연어 추론 텍스트 생성
   - 병리 이미지 어텐션 히트맵
   - 통합 대시보드 구현

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

## 📊 현재 성능 지표

| 모델 | 환자 수 | 특성 수 | Test AUC | 상태 |
|------|---------|---------|----------|------|
| CoxTabTransformer | 4,504 | 71,520 | **0.8495** | ✅ 완료 |
| MethylationTabTransformer | 8,224 | 396,065 | - | 🔄 샤딩 필요 |
| SwinTransformer | - | - | - | ⏸️ 미착수 |
| Multimodal LLM | - | - | - | ⏸️ 미착수 |

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
├── raw/                           # 원본 TCGA 데이터
├── processed/
│   ├── cox_coefficients_*.parquet      # Cox 계수 룩업 테이블
│   ├── processed_*_data.parquet        # 전처리된 오믹스 데이터
│   ├── integrated_table_cox.parquet    # Cox 적용 통합 테이블
│   └── methylation_data_for_tabtransformer.parquet  # Methylation 데이터
```

### 코드:
```
./src/
├── models/
│   ├── cox_tabtransformer.py           # ✅ CoxTabTransformer 모델
│   ├── methylation_tabtransformer.py   # 🔄 MethylationTabTransformer 모델
│   └── wsi_swin_transformer.py         # ⏸️ (미구현)
├── training/
│   └── train_tabtransformer.py         # ✅ 훈련 스크립트
└── utils/
    ├── tabtransformer_utils.py         # ✅ 유틸리티 함수
    └── feature_converter.py            # ✅ 데이터 변환 함수
```

### Notebooks:
```
./notebooks/
├── 01_cox_feature_engineering.ipynb    # ✅ Cox 분석 및 전처리
├── 02_wsi_preprocessing.ipynb          # ⏸️ (미구현)
├── 03_integrated_dataset.ipynb         # ✅ 통합 데이터셋 생성
└── 04_methylation_sharding_training.ipynb  # 🔄 (다음 단계)
```

---

## 🚀 실행 가이드

### CoxTabTransformer 훈련 (완료):
```bash
cd src/training
python train_tabtransformer.py --model cox --epochs 50 --lr 1e-4 --batch_size 32
```

### MethylationTabTransformer 훈련 (샤딩 후 실행):
```bash
cd src/training
python train_sharded_methylation.py --epochs 50 --lr 5e-5 --batch_size 16
```

### 앙상블 모델 훈련 (여러 seed):
```bash
python train_tabtransformer.py --model cox --ensemble --n_seeds 5 --epochs 50
```

---

## ⚠️ 주의사항

1. **메틸레이션 데이터 크기**:
   - 396,065 probes는 메모리에 한 번에 로드 불가
   - 반드시 샤딩 전략 구현 후 훈련

2. **Cox 계수 매칭**:
   - 환자의 암종(cancer_type)에 해당하는 Cox 계수만 사용
   - 암종별로 Cox 계수가 다르므로 주의

3. **데이터 정렬**:
   - 모든 데이터셋의 환자 ID 정렬 확인
   - Common patients만 사용

4. **GPU 메모리**:
   - Methylation 모델은 배치 크기를 작게 설정 (16 이하)
   - WSI 모델은 배치 크기 1로 시작

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
**프로젝트 상태**: Phase 2 진행 중 (CoxTabTransformer 완료, Methylation 샤딩 필요)
