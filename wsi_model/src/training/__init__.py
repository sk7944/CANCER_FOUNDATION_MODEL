"""
WSI Model Training Utilities
============================

Training scripts and utilities for MIL models.
"""

from .train_mil import train_mil_model, MILTrainer, TrainingConfig

__all__ = [
    "train_mil_model",
    "MILTrainer",
    "TrainingConfig",
]
