# Cancer Foundation Model 구축을 위한 Claude Code 프롬프트 (수정판)

## Phase 1: 데이터 준비 및 전처리

### 1-1. 프로젝트 초기 설정
```
Cancer Foundation Model 프로젝트를 시작합니다. 다음 구조로 프로젝트 디렉토리를 생성하고 필요한 라이브러리들을 설치해주세요:

프로젝트 구조:
- cfm_project/
  - data/
    - raw/
    - processed/
  - src/
    - preprocessing/
    - models/
    - utils/
  - notebooks/
  - results/
  - requirements.txt

필요한 주요 라이브러리:
- torch, transformers
- pandas, numpy, scikit-learn
- lifelines (Cox regression)
- openslide-python (WSI 처리)
- matplotlib, seaborn
- tqdm, wandb

requirements.txt 파일을 생성하고 기본 설정 파일들을 만들어주세요.
```

### 1-2. 멀티오믹스 데이터 로더 및 Cox 회귀분석 통합 구현 (Jupyter Notebook)
```
TCGA PANCAN 멀티오믹스 데이터를 로드하고 Cox 회귀분석을 통한 특성 공학을 수행하는 Jupyter Notebook을 작성해주세요:

데이터 파일 구조 (./data/raw/ 폴더):
1. 전사체: unc.edu_PANCAN_IlluminaHiSeq_RNASeqV2.geneExp_whitelisted.tsv
   - 첫 번째 컬럼: 유전자심볼|Entrez_ID 형태
   - 나머지 컬럼: 환자 ID별 발현값
   
2. 복제수변이: CNV.GISTIC_call.all_data_by_genes_whitelisted.tsv
   - 처음 3개 컬럼: 주석 정보 (Gene Symbol, Locus ID, Cytoband)
   - 나머지 컬럼: 환자 ID별 CNV 값
   
3. microRNA: bcgsc.ca_PANCAN_IlluminaHiSeq_miRNASeq.miRNAExp_whitelisted.tsv
   - 첫 번째 컬럼: miRNA ID
   - 나머지 컬럼: 환자 ID별 발현값
   
4. RPPA: mdanderson.org_PANCAN_MDA_RPPA_Core.RPPA_whitelisted.tsv
   - 첫 번째 컬럼: 단백질 ID
   - 나머지 컬럼: 환자 ID별 단백질 발현값

5. 메틸레이션: jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv (45만 probes)
   - 첫 번째 컬럼: Probe ID 
   - 나머지 컬럼: 환자 ID별 메틸레이션 값
   - **별도 TabTransformer로 처리 예정 (Cox 분석 제외)**
   
6. 돌연변이: tcga_pancancer_082115.vep.filter_whitelisted.maf.gz
   - MAF 형식 (gzip 압축)
   - 주요 컬럼: Hugo_Symbol, Tumor_Sample_Barcode, Variant_Classification 등
   
7. 임상데이터: clinical_PANCAN_patient_with_followup.tsv
   - 환자별 임상정보 및 생존 데이터

Notebook 구성:

1. 환경 설정 및 라이브러리 임포트:
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from lifelines import CoxPHFitter
   from lifelines.statistics import logrank_test
   from lifelines import KaplanMeierFitter
   import gzip
   from tqdm import tqdm
   import pickle
   import warnings
   warnings.filterwarnings('ignore')
   ```

2. 데이터 로딩 및 구조 파악:
   a) 전사체 데이터:
      - 첫 번째 컬럼을 파싱하여 Gene_Symbol과 Entrez_ID 분리
      - 환자 ID 컬럼들을 행으로 transpose
      - **중요**: Raw count 데이터이므로 log2 변환 수행: log2(x + 1)
      - 결측치 분포 확인 및 변환 전후 분포 비교 시각화
   
   b) CNV 데이터:
      - 처음 3개 주석 컬럼 제외하고 로드
      - Gene Symbol을 인덱스로 설정
      - transpose하여 환자 × 유전자 형태로 변환
      - **log2 변환 적용**: log2(|x| + 1) × sign(x) (음수 고려)
   
   c) microRNA 데이터:
      - miRNA ID를 인덱스로 설정
      - transpose하여 환자 × miRNA 형태로 변환
      - **log2 변환 적용**: log2(x + 1)
   
   d) RPPA 데이터:
      - 단백질 ID를 인덱스로 설정  
      - transpose하여 환자 × 단백질 형태로 변환
      - **log2 변환 적용**: log2(|x| + 1) × sign(x) (음수 가능성 고려)
   
   e) 메틸레이션 데이터:
      - Probe ID를 인덱스로 설정
      - transpose하여 환자 × Probe 형태로 변환
      - **Cox 분석 제외**: 별도 TabTransformer에서 처리 예정
      - 결측치 처리 및 기본 통계 확인
      
   f) 돌연변이 데이터 (MAF):
      - gzip으로 압축된 MAF 파일 읽기
      - 환자별, 유전자별로 변이 타입을 정수 인코딩
      - Variant_Classification: Missense=1, Nonsense=2, Silent=0 등
      - 환자 × 유전자 행렬로 변환 (pivot table)
   
   g) 임상 데이터:
      - 생존 시간(days), 생존 상태(vital_status) 확인
      - 암 타입(cancer_type) 분포 확인
      - 환자 ID 표준화 (TCGA barcode 처리)

3. 환자 ID 표준화 및 매칭:
   - TCGA barcode를 12자리로 표준화 (TCGA-XX-XXXX)
   - 모든 데이터셋의 공통 환자 ID 추출
   - 교집합 환자 목록 생성 및 통계 출력

4. 데이터 품질 체크 및 시각화:
   - 각 오믹스별 특성 개수 및 환자 수
   - **모든 오믹스 데이터 변환 검증**: 
     - Raw data 히스토그램 vs log2 변환 후 히스토그램 비교
     - 변환 전후 분포의 정규성 확인 (Q-Q plot)
     - CNV, RPPA의 음수값 처리 결과 확인
   - 결측치 히트맵 생성
   - 암종별 생존 데이터 분포
   - Kaplan-Meier 생존 곡선 (전체 및 암종별)

5. Cox 회귀분석 수행 (메틸레이션 제외):
   - **적용 대상**: Expression, CNV, miRNA, RPPA, Mutation만
   - 각 암종별로 분리하여 Cox 회귀 수행
   - 오믹스별 유니바리어트 Cox regression:
     ```python
     def perform_cox_analysis(omics_data, clinical_data, omics_type, cancer_type):
         cox_results = []
         for feature in tqdm(omics_data.columns):
             # Cox regression for each feature
             # 모든 특성에 대해 계수 계산 (p-value 필터링 없음)
             # Return: feature_name, hazard_ratio, p_value, confidence_interval
         return cox_results
     ```
   - **중요**: p-value와 관계없이 모든 특성의 Cox 계수 보존
   - 결과를 DataFrame으로 정리

6. Cox 계수 룩업 테이블 생성:
   - **메틸레이션 제외**: Expression, CNV, miRNA, RPPA, Mutation만 포함
   - 암종별, 오믹스별 Cox 계수 딕셔너리 생성
   - 다중 인덱스 구조: {cancer_type: {omics_type: {feature: cox_coef}}}
   - **모든 특성 포함**: 유의성과 관계없이 전체 Cox 계수 저장
   - pickle 파일로 저장

7. [측정값, Cox계수] 쌍 생성 및 검증:
   - **Cox 적용 오믹스**: 원본 측정값과 Cox 계수를 매칭
   - **메틸레이션**: 측정값만 저장 (별도 TabTransformer용)
   - 샘플 환자 데이터로 변환 결과 확인
   - 암종별 특성 중요도 top 20 시각화

8. 전처리된 데이터 저장:
   - 각 오믹스를 환자 × 특성 형태로 저장
   - **모든 수치형 오믹스**: log2 변환된 값으로 저장
   - **메틸레이션**: Cox 계수 없이 측정값만 저장 (별도 TabTransformer용)
   - 표준화된 환자 ID로 통일
   - parquet 형식으로 저장 (압축률 및 속도 고려)

파일명: notebooks/01_cox_feature_engineering.ipynb

주요 출력물:
- cox_coefficients_lookup.pkl (암종별 Cox 계수 룩업 테이블, 메틸레이션 제외)
- processed_expression.parquet (log2 변환된 전사체 데이터, 환자 × 유전자)
- processed_cnv.parquet (log2 변환된 CNV 데이터, 환자 × 유전자)  
- processed_mirna.parquet (log2 변환된 miRNA 데이터, 환자 × miRNA)
- processed_rppa.parquet (log2 변환된 RPPA 데이터, 환자 × 단백질)
- processed_methylation.parquet (메틸레이션 데이터, 환자 × Probe, Cox 제외)
- processed_mutation.parquet (변이 데이터, 환자 × 유전자)
- processed_clinical.parquet (표준화된 임상 데이터)
- feature_importance_summary.csv (암종별 특성 중요도 요약, 메틸레이션 제외)
- omics_transformation_stats.json (모든 오믹스 log2 변환 전후 통계)
- data_processing_log.txt (처리 과정 로그 및 통계)
```

### 1-3. WSI 병리영상 전처리 (Jupyter Notebook)
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

### 1-4. 통합 데이터셋 생성 (Jupyter Notebook) - **수정됨**
```
통합 데이터셋을 생성하고 PyTorch Dataset 클래스를 구현하는 Jupyter Notebook을 작성해주세요:

Notebook 구성:
1. 이전 단계 결과물 로드:
   - cox_coefficients_lookup.pkl 로드
   - processed_omics_data.parquet 로드  
   - patch_metadata.csv 로드
   - 생존 데이터 로드

2. Wide Format 통합 테이블 생성:
   - 환자 ID를 행, 모든 특성을 열로 하는 테이블 구축
   - [임상 블록] + [Expression 블록] + [CNV 블록] + [miRNA 블록] + [RPPA 블록] + [Mutation 블록]
   - **Methylation은 별도 처리**: WSI와 마찬가지로 독립적인 TabTransformer 적용 예정
   - Cox 적용 오믹스 특성을 [측정값, Cox계수] 쌍으로 구성
   - 결측치 처리 및 정규화

3. [측정값, Cox계수] 쌍 구성 방식:
   ```python
   # TabTransformer 입력 형태
   # 각 특성마다 2차원 벡터: [measured_value, cox_coefficient]
   
   # 훈련 시
   feature_input = [log2_expression_value, cox_coef_for_this_gene_in_cancer_type]
   
   # 테스트 시 (새로운 환자)
   test_input = [new_patient_measured_value, 
                 cox_lookup_table[cancer_type][omics_type][feature_name]]
   ```

4. 테이블 구조 예시 시각화:
   ```
   | patient_id | age | gender | stage | BRCA1_val | BRCA1_cox | TP53_val | TP53_cox | ... |
   |------------|-----|--------|-------|-----------|-----------|----------|----------|-----|
   ```

5. IntegratedDataset 클래스 구현:
   - PyTorch Dataset 상속
   - __getitem__에서 통합 테이블 + 생존 라벨 반환
   - **3개 모달리티 분리 처리**:
     a) TabTransformer용 (Clinical + Expression + CNV + miRNA + RPPA + Mutation)
     b) TabTransformer용 (Methylation only)  
     c) SwinTransformer용 (WSI patches)
   - 환자별 가변 패치 개수 처리
   - 데이터 분할 (train/val/test) 기능

6. 데이터 검증 및 통계:
   - 최종 테이블 shape 및 특성 개수 확인
   - 생존 라벨 분포 (3년, 5년) 시각화
   - 암종별 데이터 분포 확인
   - 샘플 배치 출력으로 데이터 로더 테스트

7. 데이터 저장:
   - 최종 통합 테이블을 parquet으로 저장
   - Dataset 클래스를 별도 .py 파일로 export
   - 데이터 분할 인덱스 저장

파일명: notebooks/03_integrated_dataset.ipynb

주요 출력물:
- integrated_table_cox.parquet (Cox 적용 오믹스 통합 테이블)
- methylation_table.parquet (메틸레이션 전용 테이블)
- src/preprocessing/integrated_dataset.py (Dataset 클래스)
- train_val_test_splits.json (데이터 분할 인덱스)
```

## Phase 2: 단일 모달리티 모델 개발 - **수정됨**

### 2-1. TabTransformer 구현 (기존 구현체 활용) - 기반: lucidrains/tab-transformer-pytorch
```
기존 검증된 TabTransformer 구현체를 활용하여 우리 도메인에 맞게 수정해주세요:

기존 구현체 설치:
pip install tab-transformer-pytorch

구현 계획:

1. 환경 설정 및 기존 구현체 탐색:
   ```python
   from tab_transformer_pytorch import TabTransformer
   import torch
   import torch.nn as nn
   
   # 기본 사용법 확인
   model = TabTransformer(
       categories = (10, 5, 6, 5, 8),     # 범주형 특성별 unique 값 개수
       num_continuous = 10,               # 수치형 특성 개수
       dim = 32,                          # 임베딩 차원
       dim_out = 1,                       # 출력 차원
       depth = 6,                         # Transformer layers
       heads = 8,                         # Attention heads
       attn_dropout = 0.1,
       ff_dropout = 0.1
   )
   ```

2. CoxTabTransformer 클래스 구현 (기존 구조 확장):
   ```python
   class CoxTabTransformer(nn.Module):
       def __init__(self, 
                    clinical_categories,        # 임상 범주형 특성 vocab sizes
                    num_omics_features,         # 오믹스 특성 개수
                    dim=32,
                    depth=6, 
                    heads=8,
                    attn_dropout=0.1,
                    ff_dropout=0.1):
           super().__init__()
           
           # 기존 TabTransformer를 베이스로 사용
           self.base_transformer = TabTransformer(
               categories=clinical_categories,
               num_continuous=num_omics_features * 2,  # [측정값, Cox계수] 쌍
               dim=dim,
               depth=depth,
               heads=heads,
               attn_dropout=attn_dropout,
               ff_dropout=ff_dropout,
               dim_out=256  # 중간 representation
           )
           
           # 생존 예측용 최종 헤드
           self.survival_head = nn.Sequential(
               nn.Linear(256, 128),
               nn.ReLU(),
               nn.Dropout(0.1),
               nn.Linear(128, 1)
           )
   
       def forward(self, clinical_categorical, omics_continuous):
           # clinical_categorical: 범주형 임상 변수
           # omics_continuous: [측정값, Cox계수] 쌍들 flatten된 형태
           
           representation = self.base_transformer(clinical_categorical, omics_continuous)
           survival_logit = self.survival_head(representation)
           
           return survival_logit, representation
   ```

3. MethylationTabTransformer 클래스 구현:
   ```python
   class MethylationTabTransformer(nn.Module):
       def __init__(self, 
                    num_probes=450000,
                    selected_probes=45000,  # 10% 선택
                    dim=32,
                    depth=4,
                    heads=8):
           super().__init__()
           
           # Feature selection layer
           self.feature_selector = nn.Linear(num_probes, selected_probes)
           
           # 선택된 probe들을 "연속형" 변수로 취급
           self.methylation_transformer = TabTransformer(
               categories=(),  # 범주형 없음
               num_continuous=selected_probes,
               dim=dim,
               depth=depth, 
               heads=heads,
               dim_out=256  # 다른 모달리티와 동일한 출력 차원
           )
   
       def forward(self, methylation_data):
           # Feature selection
           selected_features = torch.relu(self.feature_selector(methylation_data))
           
           # TabTransformer 처리 (범주형은 빈 텐서)
           empty_categorical = torch.empty(methylation_data.size(0), 0, dtype=torch.long)
           representation = self.methylation_transformer(empty_categorical, selected_features)
           
           return representation
   ```

4. 데이터 전처리 함수 구현:
   ```python
   def prepare_cox_data(omics_data, cox_coefficients):
       """
       [측정값, Cox계수] 쌍을 TabTransformer 입력 형태로 변환
       """
       # omics_data: (batch, num_features)
       # cox_coefficients: (num_features,) - 각 특성별 Cox 계수
       
       batch_size, num_features = omics_data.shape
       
       # Cox 계수를 배치 크기만큼 확장
       cox_expanded = cox_coefficients.unsqueeze(0).expand(batch_size, -1)
       
       # [측정값, Cox계수] 쌍을 interleave 방식으로 결합
       paired_data = torch.stack([omics_data, cox_expanded], dim=2)  # (batch, features, 2)
       flattened = paired_data.view(batch_size, -1)  # (batch, features*2)
       
       return flattened
   
   def prepare_clinical_data(clinical_df):
       """
       임상 데이터를 범주형으로 변환
       """
       # Age → age_group, Stage → stage_numeric 등
       categorical_data = []
       vocab_sizes = []
       
       # 예시: age binning
       age_groups = pd.cut(clinical_df['age'], bins=[0, 40, 60, 80, 100], labels=[0,1,2,3])
       categorical_data.append(age_groups)
       vocab_sizes.append(4)
       
       # Gender encoding
       gender_encoded = clinical_df['gender'].map({'M': 0, 'F': 1})
       categorical_data.append(gender_encoded)
       vocab_sizes.append(2)
       
       return torch.stack(categorical_data, dim=1), tuple(vocab_sizes)
   ```

5. 통합 훈련 함수:
   ```python
   def train_cox_tabtransformer(model, train_loader, val_loader, epochs=100):
       optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
       criterion = nn.BCEWithLogitsLoss()  # 또는 Cox loss
       
       for epoch in range(epochs):
           model.train()
           for batch in train_loader:
               clinical_cat, omics_cont, targets = batch
               
               # Forward pass
               logits, representations = model(clinical_cat, omics_cont)
               loss = criterion(logits.squeeze(), targets.float())
               
               # Backward pass
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
               
           # Validation
           if epoch % 10 == 0:
               val_loss = validate_model(model, val_loader, criterion)
               print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}")
   ```

6. Attention 시각화 함수:
   ```python
   def extract_attention_weights(model, input_data):
       """
       기존 TabTransformer에서 attention weights 추출
       """
       model.eval()
       with torch.no_grad():
           # TabTransformer 내부 transformer 접근
           transformer = model.base_transformer.transformer
           
           # Hook 등록하여 attention weights 추출
           attention_weights = []
           def hook(module, input, output):
               if hasattr(output, 'attention_weights'):
                   attention_weights.append(output.attention_weights)
           
           # Feature importance 계산
           # ... attention analysis ...
   ```

주요 장점:
- **검증된 구현체**: lucidrains의 안정적이고 최적화된 코드 활용
- **최소한의 수정**: 기존 구조를 유지하면서 도메인 특화 기능만 추가
- **유지보수성**: 원본 라이브러리 업데이트 혜택 지속 활용
- **성능 최적화**: 이미 최적화된 attention 메커니즘 사용

구현 우선순위:
1. 기존 TabTransformer 동작 확인 및 이해
2. CoxTabTransformer 구현 (Clinical + 5개 오믹스)
3. MethylationTabTransformer 구현
4. 데이터 전처리 파이프라인 구축
5. 훈련 및 평가 함수 구현

7. Jupyter Notebook 구성 - 통합 구현 및 테스트:
   ```python
   # Notebook: notebooks/04_tabtransformer_implementation.ipynb
   
   # 1. 환경 설정 및 라이브러리 설치
   !pip install tab-transformer-pytorch
   
   import torch
   import torch.nn as nn
   import pandas as pd
   import numpy as np
   from tab_transformer_pytorch import TabTransformer
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import LabelEncoder
   import matplotlib.pyplot as plt
   import seaborn as sns
   
   # 2. 기존 TabTransformer 동작 확인
   # 간단한 예제로 라이브러리 테스트
   
   # 3. 전처리된 데이터 로드
   expression_data = pd.read_parquet('data/processed/processed_expression.parquet')
   clinical_data = pd.read_parquet('data/processed/processed_clinical.parquet')
   cox_lookup = pickle.load(open('data/processed/cox_coefficients_lookup.pkl', 'rb'))
   
   # 4. CoxTabTransformer 클래스 구현 및 테스트
   class CoxTabTransformer(nn.Module):
       # ... 위에서 정의한 구현체 ...
   
   # 5. MethylationTabTransformer 클래스 구현 및 테스트
   class MethylationTabTransformer(nn.Module):
       # ... 위에서 정의한 구현체 ...
   
   # 6. 데이터 전처리 파이프라인 테스트
   def prepare_cox_data_test():
       # 샘플 데이터로 [측정값, Cox계수] 쌍 생성 테스트
       # 변환 전후 데이터 shape 확인
       # 시각화로 변환 과정 검증
   
   # 7. 모델 인스턴스 생성 및 forward pass 테스트
   # 배치 크기, 특성 개수 등 차원 확인
   
   # 8. 작은 데이터셋으로 프로토타입 훈련 (빠른 검증용)
   # overfitting test로 모델 동작 확인
   # 5-10 에폭 정도의 간단한 훈련
   
   # 9. Attention weights 추출 및 시각화
   def visualize_attention():
       # 특성별 attention score 히트맵
       # 임상변수 vs 오믹스 특성 attention 패턴 분석
   
   # 10. 성능 메트릭 계산 및 비교
   # C-index, AUC 계산
   # 기존 방법론과 비교
   ```

8. train_tabtransformer.py (본격적인 훈련용 스크립트):
   ```python
   # 주피터 노트북에서 검증된 코드를 기반으로 한 프로덕션 훈련 스크립트
   # - 긴 시간의 full training (100+ epochs)
   # - K-fold cross validation
   # - WandB 로깅
   # - 체크포인트 저장/로드
   # - 하이퍼파라미터 튜닝
   # - 배치 처리 최적화
   # - 멀티GPU 지원
   
   def main():
       parser = argparse.ArgumentParser()
       parser.add_argument('--model', choices=['cox', 'methylation'], required=True)
       parser.add_argument('--epochs', type=int, default=100)
       parser.add_argument('--batch_size', type=int, default=32)
       parser.add_argument('--lr', type=float, default=1e-4)
       parser.add_argument('--k_folds', type=int, default=5)
       
   # 사용법:
   # python src/training/train_tabtransformer.py --model cox --epochs 100 --lr 1e-4
   # python src/training/train_tabtransformer.py --model methylation --epochs 50 --k_folds 3
   ```

파일명:
- notebooks/04_tabtransformer_implementation.ipynb (프로토타입 구현 및 빠른 테스트)
- src/models/cox_tabtransformer.py
- src/models/methylation_tabtransformer.py  
- src/utils/tabtransformer_utils.py (전처리 함수들)
- src/training/train_tabtransformer.py (본격적인 프로덕션 훈련)
```

### 2-2. Swin Transformer for WSI 구현
```
병리영상 분석을 위한 Swin Transformer 모델을 구현해주세요:

아키텍처:
1. WSISwinTransformer 클래스 구현
2. 사전훈련된 Swin Transformer 백본 사용
3. Multiple Instance Learning (MIL) 어댑터 추가
4. Attention pooling for patch aggregation
5. ROI-free 학습 방식

핵심 기능:
- 가변 개수의 패치 처리
- Attention weights 저장 (시각화용)
- 메모리 효율적 배치 처리
- 생존 예측을 위한 출력 헤드

파일명: src/models/wsi_swin_transformer.py
```

### 2-3. 모델 훈련 스크립트
```
단일 모달리티 모델들을 훈련하는 스크립트를 작성해주세요:

기능:
1. 각 모델별 훈련 함수 구현
2. 하이퍼파라미터 설정 (learning rate, batch size 등)
3. 손실 함수 및 평가 메트릭 (C-index, AUC) 계산
4. 모델 체크포인트 저장
5. WandB 로깅 연동
6. K-fold cross validation 옵션

실행 예시:
python train_unimodal.py --model tabular_cox --epochs 100 --lr 1e-4
python train_unimodal.py --model tabular_methylation --epochs 100 --lr 1e-4
python train_unimodal.py --model wsi --epochs 50 --lr 1e-5

파일명: src/train_unimodal.py
```

## Phase 3: 멀티모달 융합 및 LLM 파인튜닝

### 3-1. 추론 텍스트 데이터셋 생성
```
LLM 훈련을 위한 추론 텍스트 데이터셋을 생성해주세요:

기능:
1. ReasoningDatasetGenerator 클래스 구현
2. 환자 데이터를 의료 전문가 스타일의 추론 텍스트로 변환
3. 템플릿 기반 텍스트 생성:
   - "이 환자는 [나이]세 [성별]로, [암 타입] 진단을 받았습니다..."
   - "유전자 발현 분석 결과, [주요 유전자들]의 발현이 특이적으로..."
   - "병리 소견상 [조직학적 특징]이 관찰되어..."
   - "종합적으로 고려할 때, 3년 생존 가능성은 [HIGH/LOW]로 예상됩니다."

4. 다양한 추론 패턴 및 의학 용어 사용
5. 전문가 검토를 위한 샘플 출력 기능

파일명: src/preprocessing/reasoning_dataset.py
```

### 3-2. 멀티모달 융합 아키텍처 - **수정됨**
```
세 개 모달리티를 융합하는 아키텍처를 구현해주세요:

구조:
1. MultimodalFusion 클래스 구현
2. 사전 훈련된 3개 모델에서 임베딩 추출:
   - TabTransformer_Cox (Clinical + 5개 오믹스)
   - TabTransformer_Methylation (메틸레이션)
   - SwinTransformer (WSI patches)
3. Cross-modal attention mechanism
4. 융합된 임베딩을 텍스트 토큰 공간으로 프로젝션
5. LLM 입력 형식으로 변환

핵심 기능:
- Frozen backbone (사전훈련 모델 고정)
- Learnable projection layers
- 3-way 모달리티 간 상호작용 모델링
- 그래디언트 체크포인팅으로 메모리 최적화

파일명: src/models/multimodal_fusion.py
```

### 3-3. LLM 파인튜닝 구현
```
공개 LLM을 파인튜닝하는 코드를 구현해주세요:

기능:
1. CancerFoundationLLM 클래스 구현
2. HuggingFace transformers 기반
3. 추천 모델: LLaMA-2-7B 또는 Mistral-7B
4. LoRA (Low-Rank Adaptation) 적용으로 효율적 파인튜닝
5. 생존 예측과 추론 텍스트 생성을 동시에 학습

훈련 설정:
- Mixed precision training (fp16)
- Gradient accumulation
- Custom loss: prediction loss + text generation loss
- 정규 표현식으로 예측 결과 추출

파일명: src/models/cancer_foundation_llm.py
```

## Phase 4: 모델 평가 및 시각화

### 4-1. 모델 평가 스크립트
```
최종 모델의 성능을 평가하는 종합 평가 스크립트를 작성해주세요:

평가 항목:
1. 정량 평가:
   - C-index, AUC, Accuracy
   - 암종별 성능 분석
   - 생존 곡선 (Kaplan-Meier) 생성
   
2. 정성 평가:
   - 생성된 추론 텍스트의 의학적 타당성
   - 키워드 분석 및 용어 사용 빈도
   - 예측 일관성 체크

3. 비교 분석:
   - 단일 모달리티 vs 멀티모달 성능 비교
   - 기존 방법론과의 벤치마크
   - 통계적 유의성 검정

파일명: src/evaluation/model_evaluation.py
```

### 4-2. 시각화 및 해석 도구
```
모델 해석과 시각화 도구를 구현해주세요:

기능:
1. AttentionVisualizer 클래스 구현
2. WSI 어텐션 맵 히트맵 생성
3. 특성 중요도 시각화 (tabular data)
4. 생성된 추론 텍스트와 예측 근거 매칭
5. 대화형 대시보드 (Streamlit 또는 Gradio)

시각화 요소:
- 병리 이미지 위 어텐션 히트맵 오버레이
- 유전자/단백질 발현 레벨 막대 그래프
- 생존 확률 시각화
- 추론 과정 단계별 표시

파일명: src/visualization/attention_visualizer.py
```

### 4-3. 데모 애플리케이션
```
최종 Cancer Foundation Model의 데모 웹 애플리케이션을 구현해주세요:

기능:
1. Streamlit 기반 웹 인터페이스
2. 환자 데이터 입력 폼 (임상정보, 오믹스 데이터)
3. WSI 이미지 업로드 기능
4. 실시간 예측 및 추론 텍스트 생성
5. 시각화 결과 표시
6. 결과 리포트 PDF 출력

UI 구성:
- 사이드바: 입력 데이터 업로드
- 메인 패널: 예측 결과 및 추론
- 시각화 탭: 어텐션 맵, 특성 중요도 등
- 설정 탭: 모델 파라미터 조정

파일명: src/app/demo_app.py
```

## 실행 순서 가이드

각 단계별로 위 프롬프트를 Claude Code에 순차적으로 입력하여 구현을 진행하세요:

1. **Phase 1**: 1-1 → 1-2 → 1-3 → 1-4 순으로 데이터 파이프라인 구축
2. **Phase 2**: 2-1 → 2-2 → 2-3 순으로 단일 모델 훈련 (3개 모델)
3. **Phase 3**: 3-1 → 3-2 → 3-3 순으로 멀티모달 LLM 구축  
4. **Phase 4**: 4-1 → 4-2 → 4-3 순으로 평가 및 시각화

각 단계 완료 후 다음 단계로 진행하며, 필요시 이전 단계 코드를 수정/보완하세요.

## 주요 수정사항 요약

1. **Methylation 처리 명시**: 별도 TabTransformer로 처리, Cox 분석 제외
2. **[측정값, Cox계수] 쌍 구성**: TabTransformer가 2D 입력으로 처리
3. **3개 모달리티 아키텍처**: Cox적용 오믹스 + Methylation + WSI
4. **테스트 시 Cox 계수**: 미리 계산된 룩업 테이블에서 가져옴
5. **모델 훈련**: 3개의 단일 모달리티 모델 별도 훈련