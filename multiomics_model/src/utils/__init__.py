"""
User Data Processing Utilities for HybridMultiModalModel
=========================================================

This package provides comprehensive utilities for preprocessing user data
to match the trained HybridMultiModalModel format.

Main Components:
- InferencePipeline: Complete inference pipeline for HybridMultiModalModel
- UserDataPipeline: Complete data preprocessing pipeline
- FeatureConverter: Convert user features to [value, cox] pair format
- ProteinMapper: Map protein names for RPPA data
- DataFormatGuide: Validate and guide data formatting

Quick Usage:
    # For inference with trained model
    from multiomics_model.src.utils import InferencePipeline

    pipeline = InferencePipeline(
        model_checkpoint='path/to/best_model.pth',
        cox_coef_path='path/to/cox_coefficients.parquet'
    )
    result = pipeline.predict_single(
        age=55, sex='FEMALE', race='WHITE', stage='II', grade='G2',
        cancer_type='BRCA'
    )

    # For data preprocessing
    from multiomics_model.src.utils import UserDataPipeline

    pipeline = UserDataPipeline(cox_coef_path='path/to/cox_coefficients.parquet')
    result = pipeline.process_user_data(
        user_files={'expression': 'expr.csv', 'methylation': 'meth.csv'},
        cancer_type='BRCA'
    )
"""

from .inference_pipeline import InferencePipeline
from .feature_converter import FeatureConverter
from .protein_mapper import ProteinMapper
from .data_format_guide import DataFormatGuide
from .user_data_pipeline import UserDataPipeline

__version__ = "2.0.0"  # Updated for HybridMultiModalModel
__author__ = "Cancer Foundation Model Team"

__all__ = [
    "InferencePipeline",
    "UserDataPipeline",
    "FeatureConverter",
    "ProteinMapper",
    "DataFormatGuide",
]
