# CLAUDE.md

> AI 어시스턴트와 효과적으로 협업하기 위한 프로젝트 컨텍스트 문서

---

## 한 줄 요약

**TCGA 멀티오믹스 + Cox 회귀계수를 [value, cox] 쌍으로 TabTransformer에 주입하여 암 환자 3년 생존 예측**

---

## 빠른 시작

```bash
# 훈련 실행 (데이터 전처리 완료 상태)
cd src/training && bash run_hybrid_training.sh

# 데이터 재생성 필요 시
cd src/preprocessing && ./run_integrated_dataset_builder.sh
```

---

## 현재 상태

| Phase | 상태 | 설명 |
|-------|------|------|
| 데이터 전처리 | ✅ 완료 | Cox 계수 암종별 매핑 완료 |
| 모델 코드 | ✅ 완료 | Hybrid FC-NN + TabTransformer |
| 훈련 | ⏳ 대기 | 첫 훈련 실행 전 |
| WSI (병리영상) | ❌ 미착수 | Phase 2-B |

**생성된 데이터:**
- `integrated_table_cox.parquet`: 4,504 × 132,106 (검증 완료)
- `methylation_table.parquet`: 8,224 × 396,072
- `train_val_test_splits.json`: 8,577명 (6,003/1,286/1,288)

---

## 핵심 숫자

| 항목 | 값 |
|------|-----|
| 총 환자 수 | **8,577명** (27개 암종) |
| Cox + Meth 둘 다 | 4,151명 |
| Cox만 | 353명 |
| Meth만 | 4,073명 |
| Cox features | 132,098 (66,049 × 2) |
| Meth features | 396,065 |
| 모델 크기 | ~29GB, 7.6B params |
| GPU 요구사항 | 48GB VRAM |

---

## 핵심 파일

```
src/
├── models/hybrid_fc_tabtransformer.py   # 메인 모델
├── data/hybrid_dataset.py               # PyTorch Dataset
├── training/train_hybrid.py             # 훈련 스크립트
└── preprocessing/
    ├── cox_feature_engineer.py          # Cox 회귀 (이미 실행됨)
    └── integrated_dataset_builder.py    # 데이터 통합

data/processed/
├── integrated_table_cox.parquet         # Cox omics [val, cox] 쌍
├── methylation_table.parquet            # Beta values
└── train_val_test_splits.json           # 환자 ID 기반 분할
```

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│ 입력: 8,577명 환자                                       │
├─────────────────────────────────────────────────────────┤
│ Clinical (5 categorical)  → Embedding                   │
│ Cox Omics (132,098)       → FC: 2048→512→256           │
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

---

## 절대 규칙 (MUST)

### 1. [value, cox] 쌍 형식
```python
# ❌ 절대 금지: 곱셈
input = gene_value * gene_cox

# ✅ 올바름: 별도 유지
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

집계: 환자-유전자 쌍별 `max()`

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
```

---

## 자주 쓰는 명령어

```bash
# 훈련
cd src/training && bash run_hybrid_training.sh

# GPU 모니터링
nvidia-smi -l 1

# 로그 확인
tail -f *.log

# 데이터 검증
python -c "
import pandas as pd
cox = pd.read_parquet('data/processed/integrated_table_cox.parquet')
print(f'Shape: {cox.shape}')
col = [c for c in cox.columns if c.endswith('_cox')][0]
print(f'Unique cox values: {cox[col].nunique()}')  # Must be > 1
"
```

---

## 버그 이력 (치명적)

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

## 교훈

> "모델은 쉽다. 데이터가 어렵다."
> "동작한다 ≠ 올바르다."

| 교훈 | 설명 |
|------|------|
| 가정하지 말 것 | 환자 집합 겹침을 가정했다가 실패 |
| 출력 검증 필수 | 생성된 데이터를 샘플링하여 의도대로인지 확인 |
| 암종별 값 확인 | Cox 계수가 암종별로 다른지 반드시 검증 |

---

## 다음 단계

1. **현재**: Hybrid 모델 훈련 실행
2. **다음**: WSI (병리영상) 파이프라인 구축
3. **최종**: 멀티모달 융합 (Omics + WSI)

---

## 협업 원칙

1. **작은 단위로 작업**: 전체 파이프라인 전에 컴포넌트 검증
2. **로그 아끼지 않기**: 디버깅을 위한 print문 충분히
3. **즉시 문서화**: 발견한 문제와 해결책 바로 기록
4. **출력 데이터 검증**: "에러 없음"이 "올바름"을 의미하지 않음

---

*Last updated: 2025-12-31*
