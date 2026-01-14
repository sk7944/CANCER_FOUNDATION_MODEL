"""
WSI Preprocessing Pipeline
===========================

This package provides comprehensive preprocessing utilities for Whole Slide Images (WSI).

Main Components:
- TissueDetector: Detect tissue regions and filter background
- PatchExtractor: Extract patches from WSI at specified magnification
- StainNormalizer: Stain normalization and augmentation (Macenko method)
- FeatureExtractor: Extract features using pretrained models
- WSIPreprocessor: Main pipeline orchestrating all components

Usage:
    from wsi_model.src.preprocessing import WSIPreprocessor

    preprocessor = WSIPreprocessor(
        patch_size=256,
        feature_extractor='resnet50',
        output_dir='path/to/output'
    )
    preprocessor.process_wsi('path/to/slide.svs')
"""

from .tissue_detector import TissueDetector
from .patch_extractor import PatchExtractor
from .stain_normalizer import StainNormalizer
from .feature_extractor import FeatureExtractor
from .wsi_preprocessor import WSIPreprocessor

__version__ = "1.0.0"
__all__ = [
    "TissueDetector",
    "PatchExtractor",
    "StainNormalizer",
    "FeatureExtractor",
    "WSIPreprocessor",
]
