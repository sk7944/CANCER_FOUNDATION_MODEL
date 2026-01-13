# Cancer Foundation Model

**Explainable AI for Cancer Prognosis Prediction**

멀티오믹스(Multi-omics)와 병리영상(WSI)을 통합하여 암 환자의 예후를 예측하고, 판단 근거를 자연어와 시각적 히트맵으로 제시하는 파운데이션 모델

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

Cancer Foundation Model은 TCGA Pan-Cancer 데이터를 활용하여 암 환자의 3년 생존 예후를 예측합니다.

### Key Features

- **Multi-omics Integration**: 5개 오믹스 데이터 통합 (Expression, CNV, microRNA, RPPA, Mutation) + Methylation
- **Cox Coefficient Injection**: 도메인 지식을 `[value, cox]` 쌍으로 모델에 주입
- **Missing Modality Learning**: Cox 또는 Methylation 데이터 누락 환자도 학습 가능
- **Explainable AI**: LLM 기반 자연어 추론 + Attention Heatmap (예정)

---

## Project Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Data Preparation | ✅ Complete |
| **Phase 2-1** | Multi-omics Model (TabTransformer) | ✅ Complete |
| **Phase 2-2** | WSI Model (Swin Transformer) | ⏳ In Progress |
| **Phase 3** | Multimodal Fusion + LLM Fine-tuning | Planned |
| **Phase 4** | Evaluation + XAI Visualization | Planned |

---

## Performance

### Multi-omics Model (Phase 2-1)

| Metric | Value |
|--------|-------|
| **Test AUC** | 0.9074 |
| **Test Accuracy** | 82.19% |
| **Best Val AUC** | 0.9234 |
| Patients | 8,577 (27 cancer types) |
| Parameters | 1.9B (~7.14 GB) |

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/your-org/CANCER_FOUNDATION_MODEL.git
cd CANCER_FOUNDATION_MODEL

# Create conda environment
conda create -n cfm python=3.10
conda activate cfm

# Install dependencies
pip install -r requirements.txt
pip install tab-transformer-pytorch lifelines
```

### 2. Training Multi-omics Model

```bash
cd multiomics_model/src/training
bash run_hybrid_training.sh
```

### 3. Training WSI Model (Coming Soon)

```bash
cd wsi_model/src/training
bash run_wsi_training.sh
```

---

## Project Structure

```
CANCER_FOUNDATION_MODEL/
├── multiomics_model/           # Multi-omics Model (Phase 2-1)
│   ├── src/
│   │   ├── models/             # HybridMultiModalModel
│   │   ├── data/               # PyTorch Dataset
│   │   ├── training/           # Training scripts
│   │   └── preprocessing/      # Data preprocessing
│   ├── data/                   # TCGA data
│   └── results/                # Training results
│
├── wsi_model/                  # WSI Model (Phase 2-2)
│   ├── src/
│   │   ├── models/             # Swin Transformer
│   │   ├── data/               # WSI Dataset
│   │   └── training/           # Training scripts
│   └── data/                   # WSI patches
│
├── doc/                        # Documentation
├── CLAUDE.md                   # AI Developer Guide
└── README.md                   # This file
```

---

## Architecture

### Multi-omics Model

```
Input: 8,577 patients (27 cancer types)
├── Clinical (5 categorical) → Embedding
├── Cox Omics (132,100) → FC: 2048→512→256
└── Methylation (396,065) → FC: 4096→1024→256
         ↓
    TabTransformer (dim=128, depth=6, heads=8)
         ↓
    Output: 3-year survival prediction
```

### WSI Model (Planned)

```
Input: Whole Slide Images
         ↓
    Patch Extraction (256×256 or 512×512)
         ↓
    Swin Transformer (ROI-free training)
         ↓
    Output: 3-year survival prediction + Attention Map
```

---

## Data

### Multi-omics Data

| Data Type | Features | Transformation |
|-----------|----------|----------------|
| Expression | ~20,000 genes | log2(x + 1) |
| CNV | ~20,000 genes | log2(x - min + 1) |
| microRNA | ~1,800 miRNAs | log2(x + 1) |
| RPPA | ~200 proteins | log2(x - min + 1) |
| Mutations | ~20,000 genes | Impact score (0-2) |
| Methylation | ~400,000 CpG sites | Beta values (0-1) |

### Patient Distribution

| Category | Count |
|----------|-------|
| Cox + Methylation | 4,151 |
| Cox only | 353 |
| Methylation only | 4,073 |
| **Total** | **8,577** |

---

## Citation

If you use this work, please cite:

```bibtex
@article{cancer_foundation_model,
  title={Cancer Foundation Model: Explainable AI for Cancer Prognosis},
  author={Your Team},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **TCGA Research Network** - Pan-Cancer genomics data
- **tab-transformer-pytorch** - TabTransformer implementation
- **lifelines** - Cox regression analysis

---

*Built for the cancer research community*
