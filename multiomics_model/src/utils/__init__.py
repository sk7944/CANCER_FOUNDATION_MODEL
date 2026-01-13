"""
User Data Processing Utilities for TabTransformer Cancer Foundation Model
========================================================================

This package provides comprehensive utilities for preprocessing user data
to match the trained TabTransformer model format.

Main Components:
- FeatureConverter: Convert user features to model format
- ProteinMapper: Map protein names for RPPA data  
- DataFormatGuide: Validate and guide data formatting
- UserDataPipeline: Complete preprocessing pipeline

Quick Usage:
    from src.utils import UserDataPipeline
    
    pipeline = UserDataPipeline('model_metadata_path')
    result = pipeline.process_user_data(user_files)
"""

from .feature_converter import FeatureConverter
from .protein_mapper import ProteinMapper  
from .data_format_guide import DataFormatGuide
from .user_data_pipeline import UserDataPipeline

__version__ = "1.0.0"
__author__ = "Cancer Foundation Model Team"

__all__ = [
    "FeatureConverter",
    "ProteinMapper", 
    "DataFormatGuide",
    "UserDataPipeline"
]