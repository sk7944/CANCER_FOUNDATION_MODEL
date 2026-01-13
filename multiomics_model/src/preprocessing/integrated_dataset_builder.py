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
    parser.add_argument('--max-features-per-omics', type=int, default=0, help='Maximum features per omics type (0=unlimited, uses all features)')
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
    clinical_data: pd.DataFrame,
    omics_type: str,
    logger,
    max_features: int = 0
) -> pd.DataFrame:
    """
    Create Cox-enhanced features by combining measured values with cancer-type-specific coefficients

    Args:
        omics_data: DataFrame with omics measurements (patients × features)
        cox_coefficients: DataFrame with Cox regression coefficients (features × cancer_types)
        clinical_data: DataFrame with patient clinical data (must have 'acronym' column)
        omics_type: Type of omics (Expression, CNV, etc.)
        logger: Logger instance
        max_features: Maximum features to use (0=unlimited, uses all features)

    Returns:
        DataFrame with columns: {omics_type}_{feature}_val, {omics_type}_{feature}_cox
        Cox coefficients are cancer-type-specific for each patient.
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

    # Get available cancer types in Cox coefficients
    available_cancer_types = set(cox_coefficients.columns)
    logger.info(f"   Available cancer types in Cox coefficients: {len(available_cancer_types)}")

    # Create enhanced feature matrix
    enhanced_features = pd.DataFrame(index=measured_values.index)

    # Get patient cancer types
    patient_cancer_types = {}
    for patient in measured_values.index:
        if patient in clinical_data.index and 'acronym' in clinical_data.columns:
            cancer = clinical_data.loc[patient, 'acronym']
            if pd.notna(cancer) and cancer in available_cancer_types:
                patient_cancer_types[patient] = cancer
            else:
                patient_cancer_types[patient] = None  # Will use mean
        else:
            patient_cancer_types[patient] = None

    # Count patients with valid cancer type mapping
    valid_mapping = sum(1 for v in patient_cancer_types.values() if v is not None)
    logger.info(f"   Patients with valid cancer type mapping: {valid_mapping}/{len(patient_cancer_types)}")

    # Calculate mean coefficient as fallback for unknown cancer types
    cox_coef_mean = cox_coefficients.loc[common_features].mean(axis=1)

    # Add measured values and cancer-type-specific Cox coefficients
    # Format: {omics_type}_{gene_symbol|entrez_id}_val and _cox
    for feature in common_features:
        # Add measured value column
        enhanced_features[f"{omics_type}_{feature}_val"] = measured_values[feature]

        # Add Cox coefficient column (cancer-type-specific)
        cox_values = []
        for patient in measured_values.index:
            cancer_type = patient_cancer_types.get(patient)
            if cancer_type is not None:
                # Use cancer-type-specific coefficient
                cox_values.append(cox_coefficients.loc[feature, cancer_type])
            else:
                # Fallback to mean coefficient
                cox_values.append(cox_coef_mean[feature])

        enhanced_features[f"{omics_type}_{feature}_cox"] = cox_values

    logger.info(f"✅ Enhanced features created: {enhanced_features.shape[0]} patients × {enhanced_features.shape[1]} features")
    logger.info(f"   Cox coefficients are cancer-type-specific for {valid_mapping} patients")

    return enhanced_features

def create_integrated_cox_table(
    omics_data_dict: Dict[str, pd.DataFrame],
    cox_coefficients_dict: Dict[str, pd.DataFrame], 
    clinical_data: pd.DataFrame,
    logger,
    max_features_per_omics: int = 0  # 0=unlimited, uses all features
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

    # Start with empty DataFrame (omics only, no clinical data)
    # Clinical data is loaded separately in hybrid_dataset.py during training
    integrated_table = pd.DataFrame(index=common_patients)

    logger.info(f"🏥 Starting with empty DataFrame (clinical data excluded from Cox table)")
    
    # Process each omics type
    for omics_type in cox_omics_types:
        if omics_type not in omics_data_dict or omics_data_dict[omics_type].empty:
            logger.warning(f"⚠️  Skipping {omics_type}: no data available")
            continue
            
        if omics_type not in cox_coefficients_dict or cox_coefficients_dict[omics_type].empty:
            logger.warning(f"⚠️  Skipping {omics_type}: no Cox coefficients available")
            continue
        
        # Create Cox-enhanced features (with cancer-type-specific coefficients)
        enhanced_features = create_cox_enhanced_features(
            omics_data_dict[omics_type].loc[common_patients],
            cox_coefficients_dict[omics_type],
            clinical_data,  # Pass clinical data for cancer-type-specific Cox mapping
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
    
    # FC-NN 사용: 모든 CG 사이트 사용 (제한 없음)
    # 임상 데이터는 별도 파일에서 로드하므로 methylation 테이블에는 포함하지 않음
    methylation_subset = methylation_data.loc[common_patients]

    logger.info(f"🧬 Using ALL {methylation_subset.shape[1]:,} methylation CG sites (no sampling)")
    logger.info(f"🏥 Clinical data excluded from methylation table (loaded separately during training)")

    # Methylation data only (no clinical columns)
    methylation_table = methylation_subset.copy()
    
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

            # Ensure all features are numeric (filter out object dtype columns)
            feature_df = self.data_df[feature_cols]
            numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
            self.features = feature_df[numeric_cols].values.astype(np.float32)
            
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

def create_union_stratified_splits(
    cox_table: pd.DataFrame,
    methylation_table: pd.DataFrame,
    clinical_df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    logger=None
) -> Dict[str, List[str]]:
    """
    Create stratified train/validation/test splits for Union of Cox and Methylation patients.

    Returns patient IDs (not indices) with keys: train_patients, val_patients, test_patients
    Stratifies by cancer type + modality availability to ensure balanced distribution.
    """

    if logger:
        logger.info("=" * 60)
        logger.info("CREATING UNION-BASED STRATIFIED DATA SPLITS")
        logger.info("=" * 60)

    # Validate ratios
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    # Get patient sets
    cox_patients = set(cox_table.index)
    meth_patients = set(methylation_table.index)
    all_patients = sorted(cox_patients | meth_patients)

    if logger:
        logger.info(f"📊 Cox patients: {len(cox_patients)}")
        logger.info(f"📊 Methylation patients: {len(meth_patients)}")
        logger.info(f"📊 Union (total): {len(all_patients)}")
        logger.info(f"📊 Intersection (both): {len(cox_patients & meth_patients)}")
        logger.info(f"📊 Cox only: {len(cox_patients - meth_patients)}")
        logger.info(f"📊 Meth only: {len(meth_patients - cox_patients)}")

    # Create stratification key (cancer_type + modality)
    strat_info = []
    for patient in all_patients:
        # Cancer type from clinical data
        if patient in clinical_df.index and 'acronym' in clinical_df.columns:
            cancer = clinical_df.loc[patient, 'acronym']
            if pd.isna(cancer):
                cancer = 'UNKNOWN'
        else:
            cancer = 'UNKNOWN'

        # Modality availability
        has_cox = patient in cox_patients
        has_meth = patient in meth_patients
        if has_cox and has_meth:
            modality = 'BOTH'
        elif has_cox:
            modality = 'COX_ONLY'
        else:
            modality = 'METH_ONLY'

        strat_key = f"{cancer}_{modality}"
        strat_info.append({
            'patient_id': patient,
            'cancer': cancer,
            'modality': modality,
            'strat_key': strat_key
        })

    df = pd.DataFrame(strat_info)

    if logger:
        logger.info(f"📊 Number of cancer types: {df['cancer'].nunique()}")
        logger.info(f"📊 Modality distribution:")
        logger.info(f"    BOTH: {(df['modality'] == 'BOTH').sum()}")
        logger.info(f"    COX_ONLY: {(df['modality'] == 'COX_ONLY').sum()}")
        logger.info(f"    METH_ONLY: {(df['modality'] == 'METH_ONLY').sum()}")

    # Handle rare classes (merge classes with < 5 samples)
    strat_counts = df['strat_key'].value_counts()
    rare_classes = strat_counts[strat_counts < 5].index.tolist()

    if logger and len(rare_classes) > 0:
        logger.info(f"📊 Merging {len(rare_classes)} rare classes (< 5 samples)")

    df['strat_key_merged'] = df.apply(
        lambda x: f"RARE_{x['modality']}" if x['strat_key'] in rare_classes else x['strat_key'],
        axis=1
    )

    # Check for remaining rare classes
    merged_counts = df['strat_key_merged'].value_counts()
    still_rare = merged_counts[merged_counts < 2].index.tolist()
    if len(still_rare) > 0:
        df['strat_key_merged'] = df['strat_key_merged'].apply(
            lambda x: 'SUPER_RARE' if x in still_rare else x
        )

    # Perform split
    patients = df['patient_id'].values
    strat_keys = df['strat_key_merged'].values

    try:
        # First split: train vs temp
        sss1 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=(val_ratio + test_ratio),
            random_state=random_seed
        )
        train_idx, temp_idx = next(sss1.split(patients, strat_keys))

        # Second split: val vs test from temp
        temp_patients = patients[temp_idx]
        temp_strat = strat_keys[temp_idx]
        sss2 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.5,
            random_state=random_seed
        )
        val_idx, test_idx = next(sss2.split(temp_patients, temp_strat))

        train_patients = patients[train_idx].tolist()
        val_patients = temp_patients[val_idx].tolist()
        test_patients = temp_patients[test_idx].tolist()

        split_method = "stratified"

        if logger:
            logger.info(f"✅ Stratified split successful")

    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Stratified split failed: {e}")
            logger.info("🔄 Falling back to random split")

        # Fallback to random split
        np.random.seed(random_seed)
        indices = np.random.permutation(len(patients))

        n_train = int(train_ratio * len(patients))
        n_val = int(val_ratio * len(patients))

        train_patients = patients[indices[:n_train]].tolist()
        val_patients = patients[indices[n_train:n_train + n_val]].tolist()
        test_patients = patients[indices[n_train + n_val:]].tolist()

        split_method = "random"

    if logger:
        logger.info(f"✅ Train: {len(train_patients)} patients ({len(train_patients)/len(all_patients)*100:.1f}%)")
        logger.info(f"✅ Val: {len(val_patients)} patients ({len(val_patients)/len(all_patients)*100:.1f}%)")
        logger.info(f"✅ Test: {len(test_patients)} patients ({len(test_patients)/len(all_patients)*100:.1f}%)")

        # Verify modality distribution in each split
        for split_name, split_pats in [('Train', train_patients), ('Val', val_patients), ('Test', test_patients)]:
            split_df = df[df['patient_id'].isin(split_pats)]
            both = (split_df['modality'] == 'BOTH').sum()
            cox_only = (split_df['modality'] == 'COX_ONLY').sum()
            meth_only = (split_df['modality'] == 'METH_ONLY').sum()
            logger.info(f"    {split_name}: BOTH={both}, COX_ONLY={cox_only}, METH_ONLY={meth_only}")

    splits = {
        'train_patients': train_patients,
        'val_patients': val_patients,
        'test_patients': test_patients,
        'metadata': {
            'total_patients': len(all_patients),
            'train_size': len(train_patients),
            'val_size': len(val_patients),
            'test_size': len(test_patients),
            'split_method': split_method,
            'stratified_by': 'cancer_type + modality',
            'random_seed': random_seed
        }
    }

    return splits


def create_stratified_splits(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    logger=None
) -> Dict[str, List[int]]:
    """
    [DEPRECATED] Create stratified train/validation/test splits based on Cox dataset only.
    Use create_union_stratified_splits() instead for Union-based splits.
    """

    if logger:
        logger.info("=" * 60)
        logger.info("CREATING STRATIFIED DATA SPLITS (Cox only - deprecated)")
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
    value_cols = [col for col in integrated_cox_table.columns if col.endswith('_val')]
    cox_cols = [col for col in integrated_cox_table.columns if col.endswith('_cox')]
    clinical_cols = [col for col in integrated_cox_table.columns if not (col.endswith('_val') or col.endswith('_cox'))]
    
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
    
    # Validate splits (Union-based: patient IDs)
    logger.info("🔍 Data splits validation (Union of Cox + Methylation):")
    train_patients = splits.get('train_patients', splits.get('train', []))
    val_patients = splits.get('val_patients', splits.get('val', []))
    test_patients = splits.get('test_patients', splits.get('test', []))
    total_samples = len(train_patients) + len(val_patients) + len(test_patients)

    # Union size = Cox + Meth - Intersection
    cox_patients = set(integrated_cox_table.index)
    meth_patients = set(methylation_table.index)
    union_size = len(cox_patients | meth_patients)

    logger.info(f"  Total samples in splits: {total_samples}")
    logger.info(f"  Expected (Union): {union_size}")
    logger.info(f"  Splits match: {'✅' if total_samples == union_size else '❌'}")
    
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
            'expected_samples_union': union_size,
            'splits_valid': total_samples == union_size
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
        # Load methylation-specific clinical data (includes all 8,224 patients)
        methylation_clinical_file = DATA_DIR / 'processed_clinical_data_for_methylation.parquet'
        if methylation_clinical_file.exists():
            methylation_clinical_data = pd.read_parquet(methylation_clinical_file)
            logger.info(f"✅ Loaded methylation clinical data: {methylation_clinical_data.shape[0]} patients")
        else:
            logger.warning(f"⚠️  Methylation clinical file not found: {methylation_clinical_file}")
            logger.warning("   Falling back to standard clinical data (fewer patients)")
            methylation_clinical_data = clinical_data

        methylation_table = create_methylation_table(
            omics_data.get('methylation', pd.DataFrame()),
            methylation_clinical_data, logger
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
        
        # 7. Create Union-based stratified splits (8,577 patients)
        # Merge clinical data for all patients
        merged_clinical = pd.concat([methylation_clinical_data, clinical_data]).drop_duplicates()
        merged_clinical = merged_clinical[~merged_clinical.index.duplicated(keep='first')]

        splits = create_union_stratified_splits(
            integrated_cox_table, methylation_table, merged_clinical,
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
            'splits': splits.get('metadata', {
                'train_size': len(splits.get('train_patients', [])),
                'val_size': len(splits.get('val_patients', [])),
                'test_size': len(splits.get('test_patients', [])),
                'ratios': {
                    'train': args.train_ratio,
                    'val': args.val_ratio,
                    'test': args.test_ratio
                }
            }),
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
        logger.info(f"  • Train/Val/Test splits: {len(splits['train_patients'])}/{len(splits['val_patients'])}/{len(splits['test_patients'])}")
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