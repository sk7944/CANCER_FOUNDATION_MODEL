"""
Preprocessing Module for Cancer Foundation Model
================================================

This module contains data preprocessing scripts and utilities for TCGA PANCAN data.

Main Components:
- CancerMultiomicsDataset: PyTorch Dataset class for integrated multi-omics data
- cox_feature_engineer.py: Cox regression analysis script (run once)
- integrated_dataset_builder.py: Integrated dataset creation script (run once)

Execution Scripts:
- run_cox_feature_engineer.sh: Shell wrapper for Cox analysis
- run_integrated_dataset_builder.sh: Shell wrapper for dataset building

Quick Usage:
    # Step 1: Run Cox regression analysis
    cd src/preprocessing
    ./run_cox_feature_engineer.sh

    # Step 2: Build integrated datasets
    ./run_integrated_dataset_builder.sh

    # Step 3: Use Dataset in training
    from src.preprocessing import IntegratedCancerDataset
    dataset = IntegratedCancerDataset('path/to/data.parquet')
"""

from .cancer_multiomics_dataset import IntegratedCancerDataset

__version__ = "1.0.0"
__author__ = "Cancer Foundation Model Team"

__all__ = [
    "IntegratedCancerDataset",
]
