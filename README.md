# 🧬 Cancer Foundation Model

**A Comprehensive Multi-Modal AI System for Cancer Prognosis and Analysis**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)

## 🎯 Overview

The Cancer Foundation Model is a state-of-the-art multimodal AI system designed to revolutionize cancer prognosis prediction through the integration of:

- **🧬 Multi-Omics Data**: Expression, CNV, miRNA, RPPA, Mutation profiles
- **🔬 Methylation Profiles**: High-dimensional β-value analysis  
- **📊 Clinical Data**: Patient demographics and treatment history
- **🖼️ Histopathology Images**: Whole Slide Image (WSI) analysis

## 🏗️ Architecture

### Phase 1: Data Engineering ✅
- **Raw Data Processing**: TCGA data download and preprocessing
- **Feature Engineering**: Cox regression coefficient integration
- **Data Integration**: Unified patient-centric data tables
- **Quality Control**: Missing value handling and normalization

### Phase 2: Core Models 🚧
- **2-1: TabTransformer** ✅ - Multi-omics and methylation analysis (lucidrains/tab-transformer-pytorch)
- **2-2: Swin Transformer** ⏳ - Histopathology image processing
- **2-3: Training Pipeline** ⏳ - End-to-end model training

### Phase 3: Multimodal Fusion ⏳
- **Modality Integration**: Cross-attention mechanisms
- **LLM Fine-tuning**: Foundation model enhancement
- **Knowledge Distillation**: Model compression

### Phase 4: Deployment ⏳
- **Model Evaluation**: Comprehensive benchmarking
- **Visualization Tools**: Interactive analysis dashboard
- **Clinical Integration**: Real-world deployment pipeline

## 🔬 Technical Innovation

### TabularTransformer Architecture
```python
# Cox-Enhanced Multi-Omics Processing (lucidrains/tab-transformer-pytorch based)
CoxTabTransformer(
    clinical_categories=(5, 3, 4),           # Categorical clinical features
    num_omics_features=1000,                 # [measured_value, cox_coefficient] pairs
    architecture="Real TabTransformer + Cox integration",
    output="3-year survival prediction"
)

# High-Dimensional Methylation Analysis with Feature Selection
MethylationTabTransformer(
    num_probes=450000,                       # Full methylation array
    selected_probes=5000,                    # Learnable feature selection
    feature_selection="Neural attention-based top-k selection",
    output="Binary survival classification"
)
```

### Key Features
- **🎯 Survival-Focused**: 3-year survival binary classification endpoint
- **⚡ Efficient Processing**: Dynamic feature selection for high-dimensional data
- **🧠 Interpretable**: Attention-based feature importance extraction
- **🔄 Modular Design**: Easy integration of additional data modalities

## 📊 Dataset

### TCGA Integration
- **Patients**: 4,504 total → 2,444 valid (after censoring)
- **Survival Distribution**: 62.8% survived ≥3 years, 37.2% died within 3 years
- **Data Types**:
  - Multi-omics: 175.1 MB (Cox-enhanced features)
  - Methylation: 2.2 GB (450K probes)
  - Clinical: 1.1 MB (749 clinical variables)

### Data Processing Pipeline
```
Raw TCGA Data → Quality Control → Feature Engineering → Cox Integration → Model Input
```

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/your-org/cancer-foundation-model.git
cd cancer-foundation-model
pip install -r requirements.txt
```

### Training TabularTransformers
```python
from src.models.cox_tabtransformer import CoxTabTransformer
from src.models.methylation_tabtransformer import MethylationTabTransformer

# Cox-enhanced multi-omics model (real TabTransformer)
cox_model = CoxTabTransformer(
    clinical_categories=(10, 5, 8, 4),      # Clinical categorical vocab sizes
    num_omics_features=1000,                # Multi-omics features count
    dim=64, depth=4, heads=8
)

# High-dimensional methylation model with feature selection
meth_model = MethylationTabTransformer(
    num_probes=450000,                      # Full methylation array
    selected_probes=5000,                   # Learnable selection count
    dim=64, depth=3, heads=8
)
```

### Testing
```bash
python test/test_tabtransformers.py
```

## 📁 Project Structure

```
cancer-foundation-model/
├── 📊 data/
│   ├── raw/                    # Raw TCGA downloads
│   └── processed/              # Processed and integrated data
├── 📓 notebooks/               # Jupyter analysis notebooks  
├── 🧬 src/
│   ├── models/                 # Model architectures
│   │   ├── cox_tabtransformer.py
│   │   └── methylation_tabtransformer.py
│   └── utils/
│       └── tabtransformer_utils.py    # TabTransformer data preprocessing
├── 🧪 test/                    # Model testing scripts
├── 📈 results/                 # Training outputs and metrics
└── 📋 README.md               # This file
```

## 🎯 Current Status

### ✅ Completed
- [x] **Data Pipeline**: TCGA data processing and integration
- [x] **Cox Integration**: Survival coefficient incorporation  
- [x] **CoxTabTransformer**: Multi-omics survival prediction (lucidrains/tab-transformer-pytorch)
- [x] **MethylationTabTransformer**: High-dimensional methylation analysis with feature selection
- [x] **Training Pipeline**: End-to-end model training framework with real TabTransformer
- [x] **Model Testing**: Comprehensive validation with real data (4/4 tests passed)

### 🚧 In Progress  
- [ ] **Swin Transformer**: Histopathology image analysis
- [ ] **Multimodal Fusion**: Cross-modality integration
- [ ] **Model Optimization**: Performance and memory improvements

### ⏳ Planned
- [ ] **LLM Integration**: Foundation model enhancement
- [ ] **Clinical Deployment**: Real-world integration tools
- [ ] **Interactive Dashboard**: Visualization and analysis interface

## 📈 Performance

### Model Specifications
| Model | Parameters | Input Dim | Memory | GPU Support |
|-------|------------|-----------|---------|-------------|
| CoxTabTransformer | ~1.3M | Clinical Cat + Omics×2 | ~2GB | ✅ RTX A6000 |
| MethylationTabTransformer | ~25M | 450K→5K probes | ~4GB | ✅ CUDA Enabled |

### Training Configuration
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 1e-4 (Cox), 5e-5 (Methylation)
- **Batch Size**: 32 (Cox), 16 (Methylation)  
- **Regularization**: Dropout 0.1, Gradient clipping

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style standards
- Testing requirements
- Documentation expectations
- Pull request process

## 📜 Citation

```bibtex
@misc{cancer-foundation-model,
  title={Cancer Foundation Model: Multimodal AI for Cancer Prognosis},
  author={Your Team},
  year={2025},
  url={https://github.com/your-org/cancer-foundation-model}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TCGA Research Network** for comprehensive cancer genomics data
- **PyTorch Team** for the deep learning framework
- **lucidrains** for the excellent tab-transformer-pytorch implementation
- **Cancer Research Community** for domain expertise and validation

---

**🔬 Advancing Cancer Research Through AI Innovation**

*Built with ❤️ for the cancer research community*