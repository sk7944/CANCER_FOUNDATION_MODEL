#!/usr/bin/env python3
"""
TCGA PANCAN Integrated Dataset Creation for TabTransformer
==========================================================

Background execution script for creating integrated multi-omics datasets
with Cox regression coefficients for TabTransformer deep learning models.

Usage:
    python 03_integrated_dataset.py [options]
    nohup python 03_integrated_dataset.py > integrated_dataset.log 2>&1 &

"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for background execution
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union
import gc

# Disable warnings for cleaner output
warnings.filterwarnings('ignore')

def setup_logging(log_file=None):
    """Setup logging configuration"""
    if log_file is None:
        log_file = f"integrated_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='TCGA PANCAN Integrated Dataset Creation')
    parser.add_argument('--data-dir', default='../data/processed', help='Directory containing processed data files')
    parser.add_argument('--output-dir', default='../data/processed', help='Output directory for integrated datasets')
    parser.add_argument('--results-dir', default='../results', help='Results directory')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Test set ratio')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--max-features-per-omics', type=int, default=5000, help='Maximum features per omics type')
    return parser.parse_args()

def load_cox_results_and_coefficients(data_dir: Path, logger) -> Dict:
    """Load Cox regression results and create coefficient lookup tables"""
    logger.info("=" * 60)
    logger.info("LOADING COX REGRESSION RESULTS")
    logger.info("=" * 60)
    
    omics_types = ['expression', 'cnv', 'microrna', 'rppa', 'mutations']
    cox_coefficients = {}
    
    for omics_type in omics_types:
        try:
            # Load coefficient matrix
            coef_file = data_dir / f'cox_coefficients_{omics_type}.parquet'
            if coef_file.exists():
                coef_df = pd.read_parquet(coef_file)
                cox_coefficients[omics_type] = coef_df
                logger.info(f"✅ {omics_type.capitalize()}: {coef_df.shape[0]:,} features × {coef_df.shape[1]} cancer types")
            else:
                logger.warning(f"⚠️  Cox coefficients not found for {omics_type}: {coef_file}")
                cox_coefficients[omics_type] = pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Failed to load {omics_type} coefficients: {e}")
            cox_coefficients[omics_type] = pd.DataFrame()
    
    # Display summary
    total_features = sum(len(df) for df in cox_coefficients.values() if not df.empty)
    logger.info(f"📊 Total Cox features loaded: {total_features:,}")
    
    return cox_coefficients

def load_processed_omics_data(data_dir: Path, logger) -> Dict[str, pd.DataFrame]:
    """Load all processed omics data"""
    logger.info("=" * 60)
    logger.info("LOADING PROCESSED OMICS DATA")
    logger.info("=" * 60)
    
    omics_data = {}
    omics_files = {
        'expression': 'processed_expression_data.parquet',
        'cnv': 'processed_cnv_data.parquet', 
        'microrna': 'processed_microrna_data.parquet',
        'rppa': 'processed_rppa_data.parquet',
        'mutations': 'processed_mutations_data.parquet',
        'methylation': 'methylation_data_for_tabtransformer.parquet'
    }
    
    for omics_type, filename in omics_files.items():
        file_path = data_dir / filename
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                omics_data[omics_type] = df
                logger.info(f"✅ {omics_type.capitalize()}: {df.shape[0]} patients × {df.shape[1]:,} features")
            except Exception as e:
                logger.error(f"❌ Failed to load {omics_type}: {e}")
                omics_data[omics_type] = pd.DataFrame()
        else:
            logger.warning(f"⚠️  File not found: {file_path}")
            omics_data[omics_type] = pd.DataFrame()
    
    return omics_data

def load_clinical_data(data_dir: Path, logger) -> pd.DataFrame:
    """Load processed clinical data"""
    logger.info("Loading clinical data...")
    
    clinical_file = data_dir / 'processed_clinical_data.parquet'
    if clinical_file.exists():
        clinical_df = pd.read_parquet(clinical_file)
        logger.info(f"✅ Clinical data: {clinical_df.shape[0]} patients × {clinical_df.shape[1]} features")
        return clinical_df
    else:
        logger.error(f"❌ Clinical data not found: {clinical_file}")
        return pd.DataFrame()

def create_cox_enhanced_features(
    omics_data: pd.DataFrame, 
    cox_coefficients: pd.DataFrame,
    omics_type: str,
    logger,
    max_features: int = 5000
) -> pd.DataFrame:
    """
    Create Cox-enhanced features by combining measured values with coefficients
    
    Returns DataFrame with columns: [feature_name]_value, [feature_name]_cox
    """
    if omics_data.empty or cox_coefficients.empty:
        logger.warning(f"⚠️  Empty data for {omics_type}")
        return pd.DataFrame()
    
    # Find common features between omics data and Cox coefficients
    common_features = list(set(omics_data.columns).intersection(set(cox_coefficients.index)))
    
    if len(common_features) == 0:
        logger.warning(f"⚠️  No common features between {omics_type} data and Cox coefficients")
        return pd.DataFrame()
    
    # FC-NN 사용: 모든 features 사용 (제한 없음)
    logger.info(f"🔬 Creating Cox-enhanced features for {omics_type}: {len(common_features):,} features (ALL features)")
    
    # Get data for common features
    measured_values = omics_data[common_features].copy()
    
    # Calculate mean coefficient across cancer types for each feature
    cox_coef_mean = cox_coefficients.loc[common_features].mean(axis=1)
    
    # Create enhanced feature matrix
    enhanced_features = pd.DataFrame(index=measured_values.index)
    
    # Add measured values with omics_type prefix to prevent duplicates
    # Format: {omics_type}_{gene_symbol|entrez_id}_val and _cox
    for feature in common_features:
        enhanced_features[f"{omics_type}_{feature}_val"] = measured_values[feature]
        enhanced_features[f"{omics_type}_{feature}_cox"] = cox_coef_mean[feature]
    
    logger.info(f"✅ Enhanced features created: {enhanced_features.shape[0]} patients × {enhanced_features.shape[1]} features")
    
    return enhanced_features

def create_integrated_cox_table(
    omics_data_dict: Dict[str, pd.DataFrame],
    cox_coefficients_dict: Dict[str, pd.DataFrame], 
    clinical_data: pd.DataFrame,
    logger,
    max_features_per_omics: int = 5000
) -> pd.DataFrame:
    """Create integrated table with Cox-enhanced features (excluding methylation)"""
    logger.info("=" * 60)
    logger.info("CREATING INTEGRATED COX TABLE")
    logger.info("=" * 60)
    
    # Define omics types for Cox analysis (exclude methylation)
    cox_omics_types = ['expression', 'cnv', 'microrna', 'rppa', 'mutations']
    
    # Find common patients across all Cox omics types and clinical data
    patient_sets = []
    for omics_type in cox_omics_types:
        if omics_type in omics_data_dict and not omics_data_dict[omics_type].empty:
            patient_sets.append(set(omics_data_dict[omics_type].index))
    
    if not clinical_data.empty:
        patient_sets.append(set(clinical_data.index))
    
    if not patient_sets:
        logger.error("❌ No valid patient data found")
        return pd.DataFrame()
    
    # Get intersection of all patient sets
    common_patients = set.intersection(*patient_sets)
    common_patients = sorted(list(common_patients))
    
    logger.info(f"📊 Common patients across all Cox omics types: {len(common_patients)}")
    
    # Start with clinical data
    integrated_table = clinical_data.loc[common_patients].copy()
    
    # Add essential clinical features
    essential_clinical_features = [
        'age_at_initial_pathologic_diagnosis', 'gender', 'race', 'ethnicity',
        'survival_time_clean', 'survival_event_clean', 'acronym', 'vital_status'
    ]
    
    available_clinical_features = [f for f in essential_clinical_features if f in integrated_table.columns]
    integrated_table = integrated_table[available_clinical_features]
    
    logger.info(f"🏥 Starting with clinical features: {len(available_clinical_features)}")
    
    # Process each omics type
    for omics_type in cox_omics_types:
        if omics_type not in omics_data_dict or omics_data_dict[omics_type].empty:
            logger.warning(f"⚠️  Skipping {omics_type}: no data available")
            continue
            
        if omics_type not in cox_coefficients_dict or cox_coefficients_dict[omics_type].empty:
            logger.warning(f"⚠️  Skipping {omics_type}: no Cox coefficients available")
            continue
        
        # Create Cox-enhanced features
        enhanced_features = create_cox_enhanced_features(
            omics_data_dict[omics_type].loc[common_patients],
            cox_coefficients_dict[omics_type],
            omics_type,
            logger,
            max_features_per_omics
        )
        
        if not enhanced_features.empty:
            # Add to integrated table
            integrated_table = pd.concat([integrated_table, enhanced_features], axis=1)
            logger.info(f"✅ Added {omics_type}: {enhanced_features.shape[1]} enhanced features")
        else:
            logger.warning(f"⚠️  No enhanced features created for {omics_type}")
    
    logger.info(f"🎯 Final integrated Cox table: {integrated_table.shape[0]} patients × {integrated_table.shape[1]} features")

    # Check for duplicate columns
    if integrated_table.columns.duplicated().any():
        logger.warning(f"⚠️  Found {integrated_table.columns.duplicated().sum()} duplicate column names")
        logger.warning(f"   Removing duplicates...")
        integrated_table = integrated_table.loc[:, ~integrated_table.columns.duplicated()]
        logger.info(f"✅ After removing duplicates: {integrated_table.shape[0]} patients × {integrated_table.shape[1]} features")

    # Convert data types for memory efficiency
    for col in integrated_table.columns:
        col_data = integrated_table[col]
        if isinstance(col_data, pd.Series):  # Ensure it's a Series
            if col_data.dtype == 'float64':
                integrated_table[col] = col_data.astype('float32')
            elif col_data.dtype == 'int64':
                integrated_table[col] = col_data.astype('int32')

    logger.info("✅ Data types optimized for memory efficiency")
    
    return integrated_table

def create_methylation_table(
    methylation_data: pd.DataFrame,
    clinical_data: pd.DataFrame,
    logger
) -> pd.DataFrame:
    """Create methylation table with clinical data (separate from Cox analysis)"""
    logger.info("=" * 60)
    logger.info("CREATING METHYLATION TABLE")
    logger.info("=" * 60)
    
    if methylation_data.empty:
        logger.error("❌ No methylation data available")
        return pd.DataFrame()
    
    # For methylation, we need separate clinical data that includes all patients
    # Load methylation-specific clinical data
    methylation_patients = set(methylation_data.index)
    clinical_patients = set(clinical_data.index)
    common_patients = methylation_patients.intersection(clinical_patients)
    
    logger.info(f"📊 Methylation patients: {len(methylation_patients)}")
    logger.info(f"📊 Clinical patients: {len(clinical_patients)}")
    logger.info(f"📊 Common patients: {len(common_patients)}")
    
    if len(common_patients) == 0:
        logger.error("❌ No common patients between methylation and clinical data")
        return pd.DataFrame()
    
    common_patients = sorted(list(common_patients))
    
    # Create methylation-specific clinical data
    methylation_clinical_features = [
        'age_at_initial_pathologic_diagnosis', 'gender', 'race', 
        'acronym', 'vital_status', 'days_to_death', 'days_to_last_followup'
    ]
    
    available_features = [f for f in methylation_clinical_features if f in clinical_data.columns]
    methylation_clinical = clinical_data.loc[common_patients, available_features].copy()
    
    # FC-NN 사용: 모든 CG 사이트 사용 (제한 없음)
    methylation_subset = methylation_data.loc[common_patients]

    logger.info(f"🧬 Using ALL {methylation_subset.shape[1]:,} methylation CG sites (no sampling)")
    
    # Combine clinical and methylation data
    methylation_table = pd.concat([methylation_clinical, methylation_subset], axis=1)
    
    logger.info(f"🎯 Final methylation table: {methylation_table.shape[0]} patients × {methylation_table.shape[1]} features")

    # Check for duplicate columns
    if methylation_table.columns.duplicated().any():
        logger.warning(f"⚠️  Found {methylation_table.columns.duplicated().sum()} duplicate column names")
        logger.warning(f"   Removing duplicates...")
        methylation_table = methylation_table.loc[:, ~methylation_table.columns.duplicated()]
        logger.info(f"✅ After removing duplicates: {methylation_table.shape[0]} patients × {methylation_table.shape[1]} features")

    # Convert data types for memory efficiency
    for col in methylation_table.columns:
        col_data = methylation_table[col]
        if isinstance(col_data, pd.Series):  # Ensure it's a Series
            if col_data.dtype == 'float64':
                methylation_table[col] = col_data.astype('float32')
            elif col_data.dtype == 'int64':
                methylation_table[col] = col_data.astype('int32')

    logger.info("✅ Data types optimized for memory efficiency")
    
    return methylation_table

class IntegratedCancerDataset(Dataset):
    """PyTorch Dataset for integrated multi-omics cancer data with Cox features"""
    
    def __init__(self, data_df: pd.DataFrame, target_column: str = 'acronym'):
        """
        Args:
            data_df: Integrated DataFrame with features and target
            target_column: Column name for classification target
        """
        self.data_df = data_df.copy()
        
        # Separate features and target
        if target_column in self.data_df.columns:
            self.target_column = target_column
            
            # Encode cancer types
            self.label_encoder = LabelEncoder()
            self.targets = self.label_encoder.fit_transform(self.data_df[target_column])
            self.cancer_types = self.label_encoder.classes_
            
            # Remove target and non-feature columns from features
            non_feature_cols = [target_column, 'vital_status', 'survival_time_clean', 'survival_event_clean']
            feature_cols = [col for col in self.data_df.columns if col not in non_feature_cols]
            self.features = self.data_df[feature_cols].values
            
        else:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Store metadata
        self.n_samples = len(self.data_df)
        self.n_features = self.features.shape[1]
        self.n_classes = len(self.cancer_types)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        target = torch.LongTensor([self.targets[idx]])[0]
        return features, target
    
    def get_info(self):
        """Get dataset information"""
        return {
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'cancer_types': self.cancer_types.tolist(),
            'feature_columns': [col for col in self.data_df.columns 
                              if col not in [self.target_column, 'vital_status', 'survival_time_clean', 'survival_event_clean']]
        }

class MultiModalCancerDataset(Dataset):
    """PyTorch Dataset for methylation data (separate modal)"""
    
    def __init__(self, data_df: pd.DataFrame, target_column: str = 'acronym'):
        """
        Args:
            data_df: Methylation DataFrame with features and target
            target_column: Column name for classification target
        """
        self.data_df = data_df.copy()
        
        # Separate features and target
        if target_column in self.data_df.columns:
            self.target_column = target_column
            
            # Encode cancer types
            self.label_encoder = LabelEncoder()
            self.targets = self.label_encoder.fit_transform(self.data_df[target_column])
            self.cancer_types = self.label_encoder.classes_
            
            # Remove target and non-feature columns from features
            non_feature_cols = [target_column, 'vital_status', 'days_to_death', 'days_to_last_followup']
            feature_cols = [col for col in self.data_df.columns if col not in non_feature_cols]
            self.features = self.data_df[feature_cols].values
            
        else:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Store metadata
        self.n_samples = len(self.data_df)
        self.n_features = self.features.shape[1]
        self.n_classes = len(self.cancer_types)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        target = torch.LongTensor([self.targets[idx]])[0]
        return features, target
    
    def get_info(self):
        """Get dataset information"""
        return {
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'cancer_types': self.cancer_types.tolist(),
            'feature_columns': [col for col in self.data_df.columns 
                              if col not in [self.target_column, 'vital_status', 'days_to_death', 'days_to_last_followup']]
        }

def create_stratified_splits(
    dataset: Dataset, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15,
    random_seed: int = 42,
    logger=None
) -> Dict[str, List[int]]:
    """Create stratified train/validation/test splits"""
    
    if logger:
        logger.info("=" * 60)
        logger.info("CREATING STRATIFIED DATA SPLITS")
        logger.info("=" * 60)
    
    # Validate ratios
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    # Get targets for stratification
    targets = [dataset[i][1].item() for i in range(len(dataset))]
    targets = np.array(targets)
    
    if logger:
        logger.info(f"📊 Total samples: {len(targets)}")
        logger.info(f"📊 Number of classes: {len(np.unique(targets))}")
        
        # Show class distribution
        unique_targets, counts = np.unique(targets, return_counts=True)
        for target, count in zip(unique_targets, counts):
            cancer_type = dataset.cancer_types[target] if hasattr(dataset, 'cancer_types') else f"Class_{target}"
            logger.info(f"  {cancer_type}: {count} samples")
    
    # Create indices array
    indices = np.arange(len(targets))
    
    try:
        # First split: train vs (val + test)
        train_test_ratio = val_ratio + test_ratio
        
        sss1 = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=train_test_ratio, 
            random_state=random_seed
        )
        train_indices, temp_indices = next(sss1.split(indices, targets))
        
        # Second split: val vs test from temp
        val_test_ratio = test_ratio / train_test_ratio
        
        sss2 = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=val_test_ratio, 
            random_state=random_seed
        )
        val_indices, test_indices = next(sss2.split(temp_indices, targets[temp_indices]))
        
        # Convert to actual indices
        val_indices = temp_indices[val_indices]
        test_indices = temp_indices[test_indices]
        
        splits = {
            'train': train_indices.tolist(),
            'val': val_indices.tolist(), 
            'test': test_indices.tolist()
        }
        
        if logger:
            logger.info(f"✅ Train: {len(train_indices)} samples ({len(train_indices)/len(targets)*100:.1f}%)")
            logger.info(f"✅ Val: {len(val_indices)} samples ({len(val_indices)/len(targets)*100:.1f}%)")
            logger.info(f"✅ Test: {len(test_indices)} samples ({len(test_indices)/len(targets)*100:.1f}%)")
        
        return splits
        
    except Exception as e:
        if logger:
            logger.warning(f"⚠️  Stratified split failed: {e}")
            logger.info("🔄 Falling back to random split")
        
        # Fallback to random split
        np.random.seed(random_seed)
        shuffled_indices = np.random.permutation(len(targets))
        
        n_train = int(len(targets) * train_ratio)
        n_val = int(len(targets) * val_ratio)
        
        train_indices = shuffled_indices[:n_train]
        val_indices = shuffled_indices[n_train:n_train + n_val]
        test_indices = shuffled_indices[n_train + n_val:]
        
        splits = {
            'train': train_indices.tolist(),
            'val': val_indices.tolist(),
            'test': test_indices.tolist()
        }
        
        if logger:
            logger.info(f"🔄 Random Train: {len(train_indices)} samples")
            logger.info(f"🔄 Random Val: {len(val_indices)} samples") 
            logger.info(f"🔄 Random Test: {len(test_indices)} samples")
        
        return splits

def validate_and_summarize_results(
    integrated_cox_table: pd.DataFrame,
    methylation_table: pd.DataFrame,
    cox_dataset: IntegratedCancerDataset,
    methylation_dataset: MultiModalCancerDataset,
    splits: Dict[str, List[int]],
    logger
):
    """Validate and summarize all results"""
    logger.info("=" * 60)
    logger.info("VALIDATION AND SUMMARY")
    logger.info("=" * 60)
    
    # Validate integrated Cox table
    logger.info("🔍 Cox integrated table validation:")
    logger.info(f"  Shape: {integrated_cox_table.shape}")
    logger.info(f"  Missing values: {integrated_cox_table.isnull().sum().sum()}")
    logger.info(f"  Memory usage: {integrated_cox_table.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Check feature composition
    value_cols = [col for col in integrated_cox_table.columns if col.endswith('_value')]
    cox_cols = [col for col in integrated_cox_table.columns if col.endswith('_cox')]
    clinical_cols = [col for col in integrated_cox_table.columns if not (col.endswith('_value') or col.endswith('_cox'))]
    
    logger.info(f"  Clinical features: {len(clinical_cols)}")
    logger.info(f"  Measured value features: {len(value_cols)}")
    logger.info(f"  Cox coefficient features: {len(cox_cols)}")
    
    # Validate methylation table
    logger.info("🔍 Methylation table validation:")
    logger.info(f"  Shape: {methylation_table.shape}")
    logger.info(f"  Missing values: {methylation_table.isnull().sum().sum()}")
    logger.info(f"  Memory usage: {methylation_table.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Validate datasets
    cox_info = cox_dataset.get_info()
    methylation_info = methylation_dataset.get_info()
    
    logger.info("🔍 PyTorch dataset validation:")
    logger.info(f"  Cox dataset: {cox_info['n_samples']} samples × {cox_info['n_features']} features → {cox_info['n_classes']} classes")
    logger.info(f"  Methylation dataset: {methylation_info['n_samples']} samples × {methylation_info['n_features']} features → {methylation_info['n_classes']} classes")
    
    # Validate splits
    logger.info("🔍 Data splits validation:")
    total_samples = len(splits['train']) + len(splits['val']) + len(splits['test'])
    logger.info(f"  Total samples in splits: {total_samples}")
    logger.info(f"  Expected samples: {cox_info['n_samples']}")
    logger.info(f"  Splits match: {'✅' if total_samples == cox_info['n_samples'] else '❌'}")
    
    # Cancer type distribution
    logger.info("🔍 Cancer type distribution:")
    for i, cancer_type in enumerate(cox_info['cancer_types']):
        logger.info(f"  {i}: {cancer_type}")
    
    return {
        'integrated_cox_validation': {
            'shape': integrated_cox_table.shape,
            'missing_values': int(integrated_cox_table.isnull().sum().sum()),
            'memory_mb': integrated_cox_table.memory_usage(deep=True).sum() / 1024**2,
            'clinical_features': len(clinical_cols),
            'value_features': len(value_cols),
            'cox_features': len(cox_cols)
        },
        'methylation_validation': {
            'shape': methylation_table.shape,
            'missing_values': int(methylation_table.isnull().sum().sum()),
            'memory_mb': methylation_table.memory_usage(deep=True).sum() / 1024**2
        },
        'dataset_info': {
            'cox_dataset': cox_info,
            'methylation_dataset': methylation_info
        },
        'splits_validation': {
            'total_samples_in_splits': total_samples,
            'expected_samples': cox_info['n_samples'],
            'splits_valid': total_samples == cox_info['n_samples']
        }
    }

def main():
    """Main execution function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    # Set pandas display options
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 100)
    
    # Set random seeds for reproducibility
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # Define paths
    DATA_DIR = Path(args.data_dir)
    OUTPUT_DIR = Path(args.output_dir)
    RESULTS_DIR = Path(args.results_dir)
    
    # Create directories if they don't exist
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    
    logger.info("=" * 80)
    logger.info("🚀 TCGA PANCAN INTEGRATED DATASET CREATION STARTED")
    logger.info("=" * 80)
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Results directory: {RESULTS_DIR}")
    logger.info(f"Max features per omics: {args.max_features_per_omics:,}")
    logger.info(f"Split ratios - Train: {args.train_ratio}, Val: {args.val_ratio}, Test: {args.test_ratio}")
    logger.info(f"Random seed: {args.random_seed}")
    
    try:
        # 1. Load Cox regression results and coefficients
        cox_coefficients = load_cox_results_and_coefficients(DATA_DIR, logger)
        
        # 2. Load processed omics data
        omics_data = load_processed_omics_data(DATA_DIR, logger)
        
        # 3. Load clinical data
        clinical_data = load_clinical_data(DATA_DIR, logger)
        
        if clinical_data.empty:
            logger.error("❌ No clinical data available - cannot proceed")
            sys.exit(1)
        
        # 4. Create integrated Cox table (multi-omics with Cox coefficients)
        integrated_cox_table = create_integrated_cox_table(
            omics_data, cox_coefficients, clinical_data, logger, 
            args.max_features_per_omics
        )
        
        if integrated_cox_table.empty:
            logger.error("❌ Failed to create integrated Cox table")
            sys.exit(1)
        
        # 5. Create methylation table (separate processing)
        methylation_table = create_methylation_table(
            omics_data.get('methylation', pd.DataFrame()), 
            clinical_data, logger
        )
        
        if methylation_table.empty:
            logger.warning("⚠️  No methylation table created")
        
        # 6. Create PyTorch datasets
        logger.info("=" * 60)
        logger.info("CREATING PYTORCH DATASETS")
        logger.info("=" * 60)
        
        # Cox integrated dataset
        cox_dataset = IntegratedCancerDataset(integrated_cox_table, target_column='acronym')
        logger.info(f"✅ Cox dataset created: {len(cox_dataset)} samples")
        
        # Methylation dataset (if available)
        methylation_dataset = None
        if not methylation_table.empty:
            methylation_dataset = MultiModalCancerDataset(methylation_table, target_column='acronym')
            logger.info(f"✅ Methylation dataset created: {len(methylation_dataset)} samples")
        
        # 7. Create stratified splits
        splits = create_stratified_splits(
            cox_dataset, 
            args.train_ratio, args.val_ratio, args.test_ratio,
            args.random_seed, logger
        )
        
        # 8. Validate and summarize results
        validation_results = validate_and_summarize_results(
            integrated_cox_table, methylation_table,
            cox_dataset, methylation_dataset, splits, logger
        )
        
        # 9. Save all results
        logger.info("=" * 60)
        logger.info("💾 SAVING RESULTS")
        logger.info("=" * 60)
        
        # Save integrated Cox table
        cox_table_file = OUTPUT_DIR / 'integrated_table_cox.parquet'
        integrated_cox_table.to_parquet(cox_table_file)
        logger.info(f"✅ Saved Cox table: {cox_table_file}")
        logger.info(f"    Size: {cox_table_file.stat().st_size / 1024**2:.1f} MB")
        
        # Save methylation table
        if not methylation_table.empty:
            methylation_table_file = OUTPUT_DIR / 'methylation_table.parquet'
            methylation_table.to_parquet(methylation_table_file)
            logger.info(f"✅ Saved methylation table: {methylation_table_file}")
            logger.info(f"    Size: {methylation_table_file.stat().st_size / 1024**2:.1f} MB")
        
        # Save train/val/test splits
        splits_file = OUTPUT_DIR / 'train_val_test_splits.json'
        with open(splits_file, 'w') as f:
            json.dump(splits, f, indent=2)
        logger.info(f"✅ Saved data splits: {splits_file}")
        
        # Save dataset info
        dataset_info = {
            'cox_dataset': cox_dataset.get_info() if cox_dataset else {},
            'methylation_dataset': methylation_dataset.get_info() if methylation_dataset else {},
            'splits': {
                'train_size': len(splits['train']),
                'val_size': len(splits['val']),
                'test_size': len(splits['test']),
                'ratios': {
                    'train': args.train_ratio,
                    'val': args.val_ratio,
                    'test': args.test_ratio
                }
            },
            'processing_info': {
                'max_features_per_omics': args.max_features_per_omics,
                'random_seed': args.random_seed,
                'creation_time': datetime.now().isoformat()
            }
        }
        
        dataset_info_file = RESULTS_DIR / 'integrated_dataset_summary.json'
        with open(dataset_info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        logger.info(f"✅ Saved dataset info: {dataset_info_file}")
        
        # Save validation results
        validation_file = RESULTS_DIR / 'integrated_dataset_validation.json'
        with open(validation_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
                
            validation_json = convert_numpy_types(validation_results)
            json.dump(validation_json, f, indent=2)
        logger.info(f"✅ Saved validation results: {validation_file}")
        
        # Final summary
        logger.info("=" * 80)
        logger.info("🎉 INTEGRATED DATASET CREATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("📊 Final Results Summary:")
        logger.info(f"  • Cox integrated table: {integrated_cox_table.shape[0]} patients × {integrated_cox_table.shape[1]:,} features")
        if not methylation_table.empty:
            logger.info(f"  • Methylation table: {methylation_table.shape[0]} patients × {methylation_table.shape[1]:,} features")
        logger.info(f"  • Train/Val/Test splits: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
        logger.info(f"  • Number of cancer types: {cox_dataset.n_classes}")
        logger.info(f"📁 Output files saved to: {OUTPUT_DIR}")
        logger.info(f"📈 Summary files saved to: {RESULTS_DIR}")
        logger.info("")
        logger.info("🔮 Ready for TabTransformer training!")
        logger.info("    - Use integrated_table_cox.parquet for multi-omics model")
        if not methylation_table.empty:
            logger.info("    - Use methylation_table.parquet for methylation model")
        logger.info("    - Use train_val_test_splits.json for consistent data splits")
        
    except Exception as e:
        logger.error(f"❌ Dataset creation failed with error: {e}")
        logger.error("Stack trace:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()