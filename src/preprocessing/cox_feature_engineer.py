#!/usr/bin/env python3
"""
TCGA PANCAN Multi-Omics Cox Regression Feature Engineering
==========================================================

Background execution script for comprehensive Cox regression analysis
on TCGA PANCAN multi-omics data with parallel processing.

Usage:
    python 01_cox_feature_engineering.py [options]
    nohup python 01_cox_feature_engineering.py > cox_analysis.log 2>&1 &

"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for background execution
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
import gzip
from tqdm import tqdm
import pickle
import warnings
import json
import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from collections import defaultdict

# 병렬처리를 위한 라이브러리 추가
from multiprocessing import Pool, cpu_count
from functools import partial
import concurrent.futures

# Set up logging for background execution
def setup_logging(log_file=None):
    """Setup logging configuration"""
    if log_file is None:
        log_file = f"cox_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
    parser = argparse.ArgumentParser(description='TCGA PANCAN Cox Regression Analysis')
    parser.add_argument('--data-dir', default='../data/raw', help='Directory containing raw data files')
    parser.add_argument('--output-dir', default='../data/processed', help='Output directory for processed data')
    parser.add_argument('--results-dir', default='../results', help='Results directory')
    parser.add_argument('--max-workers', type=int, default=3, help='Maximum parallel workers for cancer processing')
    parser.add_argument('--use-gpu', action='store_true', help='Enable GPU acceleration (if available)')
    parser.add_argument('--min-patients', type=int, default=20, help='Minimum patients per cancer type')
    parser.add_argument('--p-threshold', type=float, default=0.05, help='P-value significance threshold')
    parser.add_argument('--log-file', help='Log file path')
    parser.add_argument('--skip-visualization', action='store_true', help='Skip visualization (faster for background)')
    return parser.parse_args()

# Disable warnings for cleaner output
warnings.filterwarnings('ignore')

def standardize_patient_id(patient_id):
    """Standardize TCGA patient IDs to 12-character format (TCGA-XX-XXXX)"""
    if isinstance(patient_id, str):
        # Remove any trailing parts after the sample type (e.g., -01A, -11A)
        parts = patient_id.split('-')
        if len(parts) >= 3:
            return f"{parts[0]}-{parts[1]}-{parts[2]}"
    return patient_id

def filter_primary_samples(data, logger):
    """
    Filter TCGA data to keep only primary tumor samples
    
    TCGA Sample Types (4th part of barcode):
    - 01: Primary Solid Tumor (원발성 고형암)
    - 03: Primary Blood Derived Cancer - Peripheral Blood (원발성 혈액암 - 말초혈액)
    - 09: Primary Blood Derived Cancer - Bone Marrow (원발성 혈액암 - 골수)
    """
    primary_sample_types = ['01', '03', '09']
    
    # Extract sample type from full barcode
    def get_sample_type(barcode):
        parts = str(barcode).split('-')
        if len(parts) >= 4:
            return parts[3][:2]  # First 2 digits of 4th part
        return None
    
    # Filter for primary samples only
    sample_types = data.index.map(get_sample_type)
    primary_mask = sample_types.isin(primary_sample_types)
    
    before_count = len(data)
    filtered_data = data[primary_mask]
    after_count = len(filtered_data)
    
    logger.info(f"  원발성 암 필터링: {before_count} → {after_count} 샘플 ({before_count - after_count}개 제거)")
    
    return filtered_data

def load_transcriptome_data(file_path, logger):
    """Load and preprocess transcriptome data with log2 transformation"""
    logger.info("Loading transcriptome data...")
    
    # Load data
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    
    # Parse gene symbols from first column (Gene_Symbol|Entrez_ID)
    gene_info = df.index.str.split('|', expand=True)
    
    # Handle potential IndexError
    if hasattr(gene_info, 'shape') and len(gene_info.shape) > 1 and gene_info.shape[1] >= 2:
        gene_symbols = gene_info.iloc[:, 0]
        entrez_ids = gene_info.iloc[:, 1]
        gene_symbols = gene_symbols.where(gene_symbols != '?', 'Gene_' + entrez_ids.astype(str))
    else:
        # No "|" separator found, use original index
        gene_symbols = df.index
        logger.info("Gene symbols not split - using original index")
    
    # Set gene symbols as index
    df.index = gene_symbols
    
    # Transpose to get patients as rows
    df = df.T
    
    # Filter for primary tumor samples only
    df = filter_primary_samples(df, logger)
    
    # Standardize patient IDs
    df.index = [standardize_patient_id(pid) for pid in df.index]
    
    # Remove duplicated patients (keep first occurrence)
    df = df[~df.index.duplicated(keep='first')]
    
    # Store original values for comparison
    original_stats = {
        'mean': df.values.mean(),
        'std': df.values.std(),
        'min': df.values.min(),
        'max': df.values.max(),
        'zeros': (df.values == 0).sum()
    }
    
    # Apply log2 transformation: log2(x + 1)
    df_log = np.log2(df + 1)
    
    # Store transformed stats
    transformed_stats = {
        'mean': df_log.values.mean(),
        'std': df_log.values.std(),
        'min': df_log.values.min(),
        'max': df_log.values.max(),
        'zeros': (df_log.values == 0).sum()
    }
    
    transformation_stats = {
        'original': original_stats,
        'transformed': transformed_stats,
        'n_patients': df_log.shape[0],
        'n_genes': df_log.shape[1]
    }
    
    logger.info(f"Transcriptome data loaded: {df_log.shape[0]} patients × {df_log.shape[1]} genes (log2 transformed)")
    
    return df_log, transformation_stats

def load_cnv_data(file_path, logger):
    """Load and preprocess CNV data with log2 transformation"""
    logger.info("Loading CNV data...")
    
    # Load data
    df = pd.read_csv(file_path, sep='\t')
    
    # Skip first 3 annotation columns and set gene symbol as index
    gene_symbols = df.iloc[:, 0]  # First column is Gene Symbol
    df_values = df.iloc[:, 3:]  # Skip first 3 columns (Gene Symbol, Locus ID, Cytoband)
    df_values.index = gene_symbols
    
    # Transpose to get patients as rows
    df_values = df_values.T
    
    # Filter for primary tumor samples only
    df_values = filter_primary_samples(df_values, logger)
    
    # Standardize patient IDs
    df_values.index = [standardize_patient_id(pid) for pid in df_values.index]
    
    # Remove duplicated patients (keep first occurrence)
    df_values = df_values[~df_values.index.duplicated(keep='first')]
    
    # Apply log2 transformation: log2(x + 1) for positive values, handle negatives
    min_val = df_values.values.min()
    if min_val < 0:
        # Shift negative values to make all positive before log transformation
        df_log = np.log2(df_values - min_val + 1)
        logger.info(f"Applied log2(x - {min_val:.3f} + 1) transformation for negative CNV values")
    else:
        df_log = np.log2(df_values + 1)
        logger.info("Applied log2(x + 1) transformation")
    
    logger.info(f"CNV data loaded: {df_log.shape[0]} patients × {df_log.shape[1]} genes (log2 transformed)")
    
    return df_log

def load_mirna_data(file_path, logger):
    """Load and preprocess microRNA data with log2 transformation"""
    logger.info("Loading microRNA data...")
    
    # Load data
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    
    # Transpose to get patients as rows
    df = df.T
    
    # Filter for primary tumor samples only
    df = filter_primary_samples(df, logger)
    
    # Standardize patient IDs
    df.index = [standardize_patient_id(pid) for pid in df.index]
    
    # Remove duplicated patients (keep first occurrence)
    df = df[~df.index.duplicated(keep='first')]
    
    # Apply log2 transformation: log2(x + 1)
    df_log = np.log2(df + 1)
    
    logger.info(f"microRNA data loaded: {df_log.shape[0]} patients × {df_log.shape[1]} miRNAs (log2 transformed)")
    
    return df_log

def load_rppa_data(file_path, logger):
    """Load and preprocess RPPA protein data with log2 transformation"""
    logger.info("Loading RPPA data...")
    
    # Load data
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    
    # Transpose to get patients as rows
    df = df.T
    
    # Filter for primary tumor samples only
    df = filter_primary_samples(df, logger)
    
    # Standardize patient IDs
    df.index = [standardize_patient_id(pid) for pid in df.index]
    
    # Remove duplicated patients (keep first occurrence)
    df = df[~df.index.duplicated(keep='first')]
    
    # Apply log2 transformation: log2(x + 1) for positive values, handle negatives
    min_val = df.values.min()
    if min_val < 0:
        # Shift negative values to make all positive before log transformation
        df_log = np.log2(df - min_val + 1)
        logger.info(f"Applied log2(x - {min_val:.3f} + 1) transformation for negative RPPA values")
    else:
        df_log = np.log2(df + 1)
        logger.info("Applied log2(x + 1) transformation")
    
    logger.info(f"RPPA data loaded: {df_log.shape[0]} patients × {df_log.shape[1]} proteins (log2 transformed)")
    
    return df_log

def load_methylation_data(file_path, logger):
    """Load methylation data for tab-transformer (NO log2 transformation)"""
    logger.info("Loading methylation data...")
    logger.info("Note: NO log2 transformation applied - beta values (0-1) for tab-transformer")
    
    # Load data - try both .csv and .tsv extensions
    try:
        df = pd.read_csv(file_path, sep='\t', index_col=0)
    except FileNotFoundError:
        # Try with .csv extension
        csv_path = str(file_path).replace('.tsv', '.csv')
        df = pd.read_csv(csv_path, sep='\t', index_col=0)
    
    # Transpose to get patients as rows
    df = df.T
    
    # Filter for primary tumor samples only
    df = filter_primary_samples(df, logger)
    
    # Standardize patient IDs
    df.index = [standardize_patient_id(pid) for pid in df.index]
    
    # Remove duplicated patients (keep first occurrence)
    df = df[~df.index.duplicated(keep='first')]
    
    # Check data quality
    missing_values = df.isna().sum().sum()
    total_values = df.shape[0] * df.shape[1]
    missing_percentage = (missing_values / total_values) * 100
    
    logger.info(f"Methylation data loaded: {df.shape[0]} patients × {df.shape[1]} probes")
    logger.info(f"Missing values: {missing_values:,} ({missing_percentage:.2f}% of total)")
    logger.info("This data is prepared for tab-transformer network (beta values preserved)")
    
    return df

def load_mutation_data(file_path, logger):
    """Load and preprocess mutation data from MAF format (NO log2 transformation)"""
    logger.info("Loading mutation data...")
    logger.info("Note: Impact scores (0-2), NO log2 transformation")
    
    # Load MAF file with encoding handling, skipping the version line
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            # Skip the first line (#version 2.4)
            first_line = f.readline()
            if first_line.startswith('#version'):
                # Read the rest of the file
                df = pd.read_csv(f, sep='\t', low_memory=False)
            else:
                # Reset file pointer and read normally
                f.seek(0)
                df = pd.read_csv(f, sep='\t', low_memory=False)
    except UnicodeDecodeError:
        with gzip.open(file_path, 'rt', encoding='latin-1') as f:
            first_line = f.readline()
            if first_line.startswith('#version'):
                df = pd.read_csv(f, sep='\t', low_memory=False)
            else:
                f.seek(0)
                df = pd.read_csv(f, sep='\t', low_memory=False)
    
    logger.info(f"Raw MAF data: {df.shape[0]} mutations")
    
    # Filter for primary tumor samples only
    def get_sample_type_from_barcode(barcode):
        parts = str(barcode).split('-')
        if len(parts) >= 4:
            return parts[3][:2]
        return None
    
    primary_sample_types = ['01', '03', '09']
    sample_types = df['Tumor_Sample_Barcode'].map(get_sample_type_from_barcode)
    primary_mask = sample_types.isin(primary_sample_types)
    
    before_count = len(df)
    df = df[primary_mask]
    after_count = len(df)
    logger.info(f"  원발성 암 필터링: {before_count} → {after_count} 돌연변이 ({before_count - after_count}개 제거)")
    
    # Define mutation impact scoring
    variant_impact = {
        'Silent': 0,
        'Missense_Mutation': 1,
        'Nonsense_Mutation': 2,
        'Frame_Shift_Del': 2,
        'Frame_Shift_Ins': 2,
        'Splice_Site': 2,
        'Translation_Start_Site': 1,
        'Nonstop_Mutation': 1,
        'In_Frame_Del': 1,
        'In_Frame_Ins': 1,
        "3'UTR": 0,
        "5'UTR": 0,
        'Intron': 0,
        'RNA': 0
    }
    
    # Filter for relevant columns
    required_cols = ['Hugo_Symbol', 'Tumor_Sample_Barcode', 'Variant_Classification']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Missing required columns. Available columns: {list(df.columns[:10])}...")
        return pd.DataFrame()
    
    # Standardize patient IDs
    df['Patient_ID'] = df['Tumor_Sample_Barcode'].apply(standardize_patient_id)
    
    # Map variant classifications to impact scores
    df['Impact_Score'] = df['Variant_Classification'].map(variant_impact).fillna(0)
    
    # Aggregate mutations per patient-gene pair (take maximum impact)
    mutation_matrix = df.groupby(['Patient_ID', 'Hugo_Symbol'])['Impact_Score'].max().unstack(fill_value=0)
    
    logger.info(f"Mutation matrix: {mutation_matrix.shape[0]} patients × {mutation_matrix.shape[1]} genes (impact scores)")
    
    return mutation_matrix

def load_clinical_data(file_path, logger):
    """Load and preprocess clinical data"""
    logger.info("Loading clinical data...")
    
    # Try different encodings to handle problematic characters
    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    df = None
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, sep='\t', encoding=encoding, low_memory=False)
            logger.info(f"Successfully loaded clinical data with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        # Last resort: ignore problematic characters
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8', errors='ignore', low_memory=False)
        logger.info("Loaded clinical data with UTF-8 encoding, ignoring problematic characters")
    
    # Standardize patient IDs
    df['bcr_patient_barcode'] = df['bcr_patient_barcode'].apply(standardize_patient_id)
    
    # Set patient ID as index
    df = df.set_index('bcr_patient_barcode')
    
    logger.info(f"Clinical data loaded: {df.shape[0]} patients × {df.shape[1]} features")
    
    return df

def clean_survival_data(clinical_df, logger):
    """생존 데이터를 정리하여 올바른 숫자형으로 변환"""
    
    logger.info("=== 생존 데이터 정리 ===")
    
    clinical_clean = clinical_df.copy()
    
    # 1. days_to_death 정리
    logger.info("1. days_to_death 정리:")
    death_col = clinical_clean['days_to_death'].copy()
    
    # 비수치 값들을 NaN으로 변환
    invalid_death = death_col.isin(['[Not Applicable]', '[Not Available]', '[Discrepancy]', '[Unknown]'])
    logger.info(f"   • 비수치 값: {invalid_death.sum()}개 → NaN으로 변환")
    
    death_col[invalid_death] = np.nan
    death_col = pd.to_numeric(death_col, errors='coerce')
    clinical_clean['days_to_death_clean'] = death_col
    
    # 2. days_to_last_followup 정리
    logger.info("2. days_to_last_followup 정리:")
    followup_col = clinical_clean['days_to_last_followup'].copy()
    
    invalid_followup = followup_col.isin(['[Not Applicable]', '[Not Available]', '[Discrepancy]', '[Unknown]'])
    logger.info(f"   • 비수치 값: {invalid_followup.sum()}개 → NaN으로 변환")
    
    followup_col[invalid_followup] = np.nan
    followup_col = pd.to_numeric(followup_col, errors='coerce')
    
    # 음수값 제거 (잘못된 데이터)
    negative_followup = followup_col < 0
    logger.info(f"   • 음수 값: {negative_followup.sum()}개 → NaN으로 변환")
    followup_col[negative_followup] = np.nan
    
    clinical_clean['days_to_last_followup_clean'] = followup_col
    
    # 3. vital_status 정리
    logger.info("3. vital_status 정리:")
    vital_status_counts = clinical_clean['vital_status'].value_counts()
    logger.info(f"   • {vital_status_counts.to_dict()}")
    
    # 올바른 vital_status만 유지
    valid_vital_status = clinical_clean['vital_status'].isin(['Alive', 'Dead'])
    logger.info(f"   • 유효한 vital_status: {valid_vital_status.sum()}개")
    
    # 4. 새로운 생존 시간과 이벤트 생성
    logger.info("4. 새로운 survival_time과 survival_event 생성:")
    
    # survival_time 재계산
    survival_time_new = np.where(
        (clinical_clean['vital_status'] == 'Dead') & clinical_clean['days_to_death_clean'].notna(),
        clinical_clean['days_to_death_clean'],
        clinical_clean['days_to_last_followup_clean']
    )
    
    # survival_event 재계산
    survival_event_new = (clinical_clean['vital_status'] == 'Dead').astype(int)
    
    # 유효하지 않은 vital_status는 제외
    survival_event_new[~valid_vital_status] = np.nan
    survival_time_new[~valid_vital_status] = np.nan
    
    clinical_clean['survival_time_clean'] = survival_time_new
    clinical_clean['survival_event_clean'] = survival_event_new
    
    # 5. 유효한 생존 데이터만 남기기
    valid_survival = (
        pd.notna(clinical_clean['survival_time_clean']) & 
        pd.notna(clinical_clean['survival_event_clean']) &
        (clinical_clean['survival_time_clean'] >= 0)
    )
    
    logger.info(f"   • 유효한 생존 데이터: {valid_survival.sum()}개")
    logger.info(f"   • 사망 이벤트: {clinical_clean.loc[valid_survival, 'survival_event_clean'].sum()}개")
    logger.info(f"   • 평균 생존 시간: {clinical_clean.loc[valid_survival, 'survival_time_clean'].mean():.1f}일")
    
    return clinical_clean, valid_survival

def process_single_feature_cox_safe(args):
    """
    Process a single feature for Cox regression (safer version for multiprocessing)
    
    Parameters:
    - args: tuple containing (feature_name, feature_data, survival_times, survival_events)
    
    Returns:
    - dict with Cox regression results or None if failed
    """
    try:
        feature_name, feature_data, survival_times, survival_events = args
        
        # Create dataframe for Cox regression
        cox_data = pd.DataFrame({
            'T': survival_times,
            'E': survival_events,
            'feature': feature_data
        })
        
        # Remove rows with missing data
        cox_data = cox_data.dropna()
        
        if len(cox_data) < 10:  # Need at least 10 observations for stable results
            return None
        
        # Check for variance in feature
        if cox_data['feature'].var() == 0:
            return None
            
        # Fit Cox model
        cph = CoxPHFitter()
        cph.fit(cox_data, duration_col='T', event_col='E')
        
        # Extract results
        coef = cph.summary.loc['feature', 'coef']
        p_value = cph.summary.loc['feature', 'p']
        hr = np.exp(coef)
        ci_lower = np.exp(cph.summary.loc['feature', 'coef lower 95%'])
        ci_upper = np.exp(cph.summary.loc['feature', 'coef upper 95%'])
        
        return {
            'feature': feature_name,
            'coef': float(coef),
            'hr': float(hr),
            'p_value': float(p_value),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_patients': int(len(cox_data))
        }
        
    except Exception as e:
        # Skip problematic features silently
        return None

def process_cancer_cox_optimized(cancer_data_tuple, use_gpu=True, logger=None):
    """
    Process Cox regression for a single cancer type using optimized approach
    
    Parameters:
    - cancer_data_tuple: tuple containing (cancer, cancer_omics, cancer_survival, valid_features)
    - use_gpu: whether to attempt GPU acceleration
    - logger: logger instance
    
    Returns:
    - DataFrame with Cox regression results
    """
    cancer, cancer_omics, cancer_survival, valid_features = cancer_data_tuple
    
    if logger:
        logger.info(f"🔬 Processing {cancer}: {len(cancer_omics)} patients, {len(valid_features):,} features")
    
    # Check GPU availability
    gpu_available = False
    device = 'cpu'
    
    if use_gpu:
        try:
            import torch
            import pycox
            from pycox.models import CoxPH
            import torchtuples as tt
            
            gpu_available = torch.cuda.is_available()
            device = 'cuda' if gpu_available else 'cpu'
            if logger:
                logger.info(f"  🖥️  GPU available: {gpu_available}")
                logger.info(f"  🎯 Device: {device}")
            
            if gpu_available and logger:
                logger.info(f"  🚀 GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"  💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")
                
        except Exception as e:
            if logger:
                logger.info(f"  ⚠️  GPU libraries error: {str(e)[:50]}...")
            gpu_available = False
            device = 'cpu'
    
    # Try GPU acceleration for large feature sets
    if gpu_available and len(valid_features) > 1000:
        if logger:
            logger.info(f"  🚀 Attempting GPU acceleration for {len(valid_features):,} features")
        
        try:
            # GPU processing code (simplified for background execution)
            # For now, fall back to CPU as GPU implementation needs more testing
            gpu_available = False
            if logger:
                logger.info("  🔄 GPU processing disabled for stability, using CPU")
                
        except Exception as e:
            if logger:
                logger.info(f"  ❌ GPU processing error: {str(e)[:100]}...")
                logger.info(f"  🔄 Falling back to CPU multiprocessing")
            gpu_available = False
            
            # Clear GPU cache on error
            if 'torch' in locals():
                torch.cuda.empty_cache()
    
    # CPU multiprocessing (optimized)
    if logger:
        logger.info(f"  🔧 Using optimized CPU multiprocessing")
    
    # Prepare data for safer multiprocessing
    survival_times = cancer_survival['survival_time_clean'].values
    survival_events = cancer_survival['survival_event_clean'].values
    
    # Prepare arguments for multiprocessing
    feature_results = []
    num_processes = min(6, cpu_count())
    
    if logger:
        logger.info(f"  ⚡ Using {num_processes} CPU processes")
    
    # Process features in optimized batches
    batch_size_cpu = 800
    total_batches = (len(valid_features) - 1) // batch_size_cpu + 1
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size_cpu
        end_idx = min(start_idx + batch_size_cpu, len(valid_features))
        batch_features = valid_features[start_idx:end_idx]
        
        if logger:
            logger.info(f"    📦 CPU Batch {batch_idx + 1}/{total_batches}: {len(batch_features)} features")
        
        # Prepare batch arguments
        batch_args = []
        for feature in batch_features:
            feature_data = cancer_omics[feature].values
            batch_args.append((feature, feature_data, survival_times, survival_events))
        
        # Process batch with multiprocessing
        try:
            with Pool(processes=num_processes) as pool:
                batch_results = pool.map(process_single_feature_cox_safe, batch_args)
            
            # Filter out None results and add to main results
            valid_results = [r for r in batch_results if r is not None]
            feature_results.extend(valid_results)
            
            if logger:
                logger.info(f"      ✅ {len(valid_results)} valid results from batch")
            
        except Exception as e:
            if logger:
                logger.info(f"      ❌ Batch failed: {str(e)[:50]}..., trying sequential processing")
            
            # Fallback to sequential processing for this batch
            for args in batch_args:
                result = process_single_feature_cox_safe(args)
                if result is not None:
                    feature_results.append(result)
    
    # Convert to DataFrame and sort by p-value
    if feature_results:
        results_df = pd.DataFrame(feature_results)
        results_df = results_df.sort_values('p_value')
        
        if logger:
            logger.info(f"  🎯 CPU processing completed: {len(results_df):,} features processed")
        return results_df
    else:
        if logger:
            logger.info(f"  ❌ No valid results for {cancer}")
        return pd.DataFrame()

def perform_cox_regression_by_cancer_parallel(omics_data, clinical_data, omics_type, 
                                            min_patients=20, p_threshold=0.05, 
                                            use_gpu=True, max_workers=2, logger=None):
    """
    Perform Cox regression analysis by cancer type with ACTUAL parallel processing
    
    Parameters:
    - omics_data: DataFrame with patients as rows, features as columns
    - clinical_data: DataFrame with survival information
    - omics_type: String identifier for the omics type
    - min_patients: Minimum number of patients required per cancer type
    - p_threshold: P-value threshold for significance
    - use_gpu: Whether to use GPU acceleration
    - max_workers: Maximum number of parallel workers for cancer processing
    - logger: Logger instance
    
    Returns:
    - cox_results: Dictionary with results by cancer type
    - summary_stats: Overall summary statistics
    """
    
    if logger:
        logger.info(f"🚀 PARALLEL Cox regression analysis for {omics_type}")
        logger.info(f"Cancer-level workers: {max_workers} 🔥")
    
    # Filter for common patients
    common_patients = list(set(omics_data.index).intersection(set(clinical_data.index)))
    
    # Get survival data for common patients
    survival_data = clinical_data.loc[common_patients, ['survival_time_clean', 'survival_event_clean', 'acronym']].copy()
    survival_data = survival_data.dropna()
    
    # Filter omics data for patients with survival data
    omics_filtered = omics_data.loc[survival_data.index].copy()
    
    if logger:
        logger.info(f"📊 Analysis dataset: {len(survival_data)} patients with {omics_filtered.shape[1]:,} features")
    
    # Group by cancer type
    cancer_types = survival_data['acronym'].value_counts()
    valid_cancers = cancer_types[cancer_types >= min_patients].index
    
    if logger:
        logger.info(f"🎯 Cancer types with >= {min_patients} patients: {len(valid_cancers)}")
    
    # Prepare cancer data for processing
    cancer_data_list = []
    
    for cancer in valid_cancers:
        # Get patients for this cancer type
        cancer_patients = survival_data[survival_data['acronym'] == cancer].index
        
        # Get omics and survival data for this cancer
        cancer_omics = omics_filtered.loc[cancer_patients]
        cancer_survival = survival_data.loc[cancer_patients, ['survival_time_clean', 'survival_event_clean']]
        
        # Remove features with low variance
        feature_vars = cancer_omics.var()
        valid_features = feature_vars[feature_vars > 1e-6].index.tolist()
        
        if len(valid_features) > 0:
            cancer_data_list.append((cancer, cancer_omics, cancer_survival, valid_features))
    
    if logger:
        logger.info(f"📋 Prepared {len(cancer_data_list)} cancer types for processing")
    
    # Process cancers with ACTUAL parallel processing
    cox_results = {}
    summary_stats = {
        'total_features': omics_filtered.shape[1],
        'total_patients': len(survival_data),
        'cancer_types': len(valid_cancers),
        'significant_features': {},
        'top_features': {}
    }
    
    # USE THE max_workers PARAMETER!
    if max_workers > 1 and len(cancer_data_list) > 1:
        if logger:
            logger.info(f"🔥 Using {max_workers} workers for PARALLEL cancer processing!")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all cancer processing tasks
            future_to_cancer = {
                executor.submit(process_cancer_cox_optimized, cancer_data, use_gpu, logger): cancer_data[0] 
                for cancer_data in cancer_data_list
            }
            
            # Collect results as they complete
            completed = 0
            for future in tqdm(concurrent.futures.as_completed(future_to_cancer), 
                              total=len(future_to_cancer), 
                              desc=f"Processing {omics_type} cancers in parallel",
                              disable=(logger is None)):
                
                cancer = future_to_cancer[future]
                completed += 1
                
                try:
                    results_df = future.result()
                    
                    if not results_df.empty:
                        # Count significant features
                        significant_count = sum(results_df['p_value'] < p_threshold)
                        
                        cox_results[cancer] = results_df
                        summary_stats['significant_features'][cancer] = significant_count
                        summary_stats['top_features'][cancer] = results_df.head(10)
                        
                        if logger:
                            logger.info(f"✅ {cancer} ({completed}/{len(cancer_data_list)}): {significant_count:,} significant features")
                    else:
                        if logger:
                            logger.info(f"❌ {cancer} ({completed}/{len(cancer_data_list)}): No valid results")
                        
                except Exception as e:
                    if logger:
                        logger.info(f"❌ {cancer} failed: {str(e)[:50]}...")
    else:
        # Sequential processing fallback
        if logger:
            logger.info(f"🔄 Using sequential processing (max_workers={max_workers} or single cancer)")
        
        for i, cancer_data in enumerate(cancer_data_list):
            cancer = cancer_data[0]
            if logger:
                logger.info(f"🔬 Processing {cancer} ({i+1}/{len(cancer_data_list)})")
            
            try:
                results_df = process_cancer_cox_optimized(cancer_data, use_gpu, logger)
                
                if not results_df.empty:
                    # Count significant features
                    significant_count = sum(results_df['p_value'] < p_threshold)
                    
                    cox_results[cancer] = results_df
                    summary_stats['significant_features'][cancer] = significant_count
                    summary_stats['top_features'][cancer] = results_df.head(10)
                    
                    if logger:
                        logger.info(f"✅ {cancer}: {significant_count:,} significant features")
                else:
                    if logger:
                        logger.info(f"❌ {cancer}: No valid results")
                    
            except Exception as e:
                if logger:
                    logger.info(f"❌ {cancer} failed: {str(e)[:50]}...")
    
    return cox_results, summary_stats

def create_cox_coefficient_lookup(cox_results_dict, omics_types, logger=None):
    """Create a comprehensive Cox coefficient lookup table"""
    
    if logger:
        logger.info("🔧 Creating Cox coefficient lookup tables...")
    
    # Initialize lookup dictionary
    lookup_tables = {}
    
    for omics_type in omics_types:
        if omics_type in cox_results_dict:
            if logger:
                logger.info(f"📋 Processing {omics_type} results...")
            
            # Combine all cancer type results
            all_results = []
            
            for cancer, results_df in cox_results_dict[omics_type].items():
                if not results_df.empty:
                    results_copy = results_df.copy()
                    results_copy['cancer_type'] = cancer
                    results_copy['omics_type'] = omics_type
                    all_results.append(results_copy)
            
            if all_results:
                combined_df = pd.concat(all_results, ignore_index=True)
                
                # Create pivot table: features × cancer_types with coefficients
                pivot_coef = combined_df.pivot_table(
                    index='feature', 
                    columns='cancer_type', 
                    values='coef', 
                    fill_value=0
                )
                
                # Create pivot table for p-values
                pivot_pval = combined_df.pivot_table(
                    index='feature', 
                    columns='cancer_type', 
                    values='p_value', 
                    fill_value=1
                )
                
                # Create summary statistics per feature
                feature_stats = combined_df.groupby('feature').agg({
                    'coef': ['mean', 'std', 'count'],
                    'p_value': ['min', 'mean'],
                    'hr': ['mean']
                }).round(4)
                
                # Flatten column names
                feature_stats.columns = ['_'.join(col).strip() for col in feature_stats.columns]
                
                lookup_tables[omics_type] = {
                    'coefficients': pivot_coef,
                    'p_values': pivot_pval,
                    'feature_stats': feature_stats,
                    'raw_results': combined_df
                }
                
                if logger:
                    logger.info(f"  ✅ {omics_type}: {len(pivot_coef):,} features across {len(pivot_coef.columns)} cancer types")
    
    return lookup_tables

def save_progress(all_cox_results, all_summary_stats, output_dir, logger=None):
    """Save intermediate progress"""
    progress_file = Path(output_dir) / f'cox_progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
    
    try:
        with open(progress_file, 'wb') as f:
            pickle.dump({
                'cox_results': all_cox_results,
                'summary_stats': all_summary_stats,
                'timestamp': datetime.now()
            }, f)
        
        if logger:
            logger.info(f"💾 Progress saved: {progress_file}")
    except Exception as e:
        if logger:
            logger.error(f"❌ Failed to save progress: {e}")

def main():
    """Main execution function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    # Set pandas display options
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 100)
    
    # Define paths
    DATA_RAW_PATH = Path(args.data_dir)
    DATA_PROCESSED_PATH = Path(args.output_dir)
    RESULTS_PATH = Path(args.results_dir)
    
    # Create directories if they don't exist
    DATA_PROCESSED_PATH.mkdir(exist_ok=True, parents=True)
    RESULTS_PATH.mkdir(exist_ok=True, parents=True)
    
    logger.info("=" * 60)
    logger.info("🚀 TCGA PANCAN Cox Regression Analysis STARTED")
    logger.info("=" * 60)
    logger.info(f"Raw data path: {DATA_RAW_PATH}")
    logger.info(f"Processed data path: {DATA_PROCESSED_PATH}")
    logger.info(f"Results path: {RESULTS_PATH}")
    logger.info(f"Available CPU cores: {cpu_count()}")
    logger.info(f"Max workers: {args.max_workers}")
    logger.info(f"GPU enabled: {args.use_gpu}")
    
    try:
        # Load all datasets
        logger.info("=" * 60)
        logger.info("LOADING TCGA PANCAN MULTI-OMICS DATA")
        logger.info("=" * 60)
        
        # Load transcriptome data (log2 transformed)
        expression_data, transformation_stats = load_transcriptome_data(
            DATA_RAW_PATH / 'unc.edu_PANCAN_IlluminaHiSeq_RNASeqV2.geneExp_whitelisted.tsv',
            logger
        )
        
        # Load CNV data (log2 transformed)
        cnv_data = load_cnv_data(
            DATA_RAW_PATH / 'CNV.GISTIC_call.all_data_by_genes_whitelisted.tsv',
            logger
        )
        
        # Load microRNA data (log2 transformed)
        mirna_data = load_mirna_data(
            DATA_RAW_PATH / 'bcgsc.ca_PANCAN_IlluminaHiSeq_miRNASeq.miRNAExp_whitelisted.tsv',
            logger
        )
        
        # Load RPPA data (log2 transformed)
        rppa_data = load_rppa_data(
            DATA_RAW_PATH / 'mdanderson.org_PANCAN_MDA_RPPA_Core.RPPA_whitelisted.tsv',
            logger
        )
        
        # Load methylation data (NO transformation - for tab-transformer)
        methylation_data = load_methylation_data(
            DATA_RAW_PATH / 'jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv',
            logger
        )
        
        # Load mutation data (impact scores - NO transformation)
        mutation_data = load_mutation_data(
            DATA_RAW_PATH / 'tcga_pancancer_082115.vep.filter_whitelisted.maf.gz',
            logger
        )
        
        # Load clinical data
        clinical_data = load_clinical_data(
            DATA_RAW_PATH / 'clinical_PANCAN_patient_with_followup.tsv',
            logger
        )
        
        logger.info("=" * 60)
        logger.info("DATA LOADING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Expression: {expression_data.shape[0]} patients × {expression_data.shape[1]} genes (log2 transformed)")
        logger.info(f"CNV: {cnv_data.shape[0]} patients × {cnv_data.shape[1]} genes (log2 transformed)")
        logger.info(f"microRNA: {mirna_data.shape[0]} patients × {mirna_data.shape[1]} miRNAs (log2 transformed)")
        logger.info(f"RPPA: {rppa_data.shape[0]} patients × {rppa_data.shape[1]} proteins (log2 transformed)")
        logger.info(f"Methylation: {methylation_data.shape[0]} patients × {methylation_data.shape[1]} probes (NO transformation)")
        logger.info(f"Mutations: {mutation_data.shape[0]} patients × {mutation_data.shape[1]} genes (impact scores)")
        logger.info(f"Clinical: {clinical_data.shape[0]} patients × {clinical_data.shape[1]} features")
        
        # Clean survival data
        clinical_data_clean, valid_survival_mask = clean_survival_data(clinical_data, logger)
        valid_survival_patients = clinical_data_clean.index[valid_survival_mask]
        
        logger.info(f"유효한 생존 데이터 환자: {len(valid_survival_patients)}명")
        
        # Patient ID matching and data integration
        datasets = {
            'Expression': set(expression_data.index).intersection(set(valid_survival_patients)),
            'CNV': set(cnv_data.index).intersection(set(valid_survival_patients)),
            'microRNA': set(mirna_data.index).intersection(set(valid_survival_patients)),
            'RPPA': set(rppa_data.index).intersection(set(valid_survival_patients)),
            'Methylation': set(methylation_data.index).intersection(set(valid_survival_patients)),
            'Mutations': set(mutation_data.index).intersection(set(valid_survival_patients)),
            'Clinical': set(valid_survival_patients)
        }
        
        logger.info("Patient counts per dataset:")
        for name, patients in datasets.items():
            logger.info(f"{name}: {len(patients)} patients")
        
        # Cox regression patients (excluding methylation)
        cox_datasets = {k: v for k, v in datasets.items() if k != 'Methylation'}
        cox_patients = set.intersection(*cox_datasets.values())
        final_cox_patients = sorted(list(cox_patients))
        
        # Methylation patients
        methylation_patients = datasets['Methylation'].intersection(datasets['Clinical'])
        final_methylation_patients = sorted(list(methylation_patients))
        
        logger.info(f"Cox 분석 대상: {len(final_cox_patients)} 환자")
        logger.info(f"메틸레이션 대상: {len(final_methylation_patients)} 환자")
        
        # Prepare filtered datasets for analysis
        filtered_data = {}
        
        logger.info("=== 최종 데이터 준비 ===")
        
        for name, data in [
            ('Expression', expression_data),
            ('CNV', cnv_data),
            ('microRNA', mirna_data),
            ('RPPA', rppa_data),
            ('Mutations', mutation_data)
        ]:
            actual_common = data.index.intersection(final_cox_patients)
            common_patients_data = data.loc[actual_common]
            filtered_data[name] = common_patients_data
            logger.info(f"{name}: {common_patients_data.shape[0]} patients × {common_patients_data.shape[1]} features")
        
        filtered_clinical = clinical_data_clean.loc[final_cox_patients]
        logger.info(f"Clinical: {filtered_clinical.shape[0]} patients × {filtered_clinical.shape[1]} features")
        
        # Perform Cox regression analysis
        logger.info("=" * 60)
        logger.info("🚀 PERFORMING PARALLEL COX REGRESSION ANALYSIS")
        logger.info("=" * 60)
        
        all_cox_results = {}
        all_summary_stats = {}
        
        omics_data_map = {
            'Expression': filtered_data['Expression'],
            'CNV': filtered_data['CNV'], 
            'microRNA': filtered_data['microRNA'],
            'RPPA': filtered_data['RPPA'],
            'Mutations': filtered_data['Mutations']
        }
        
        logger.info(f"🔥 PARALLEL Configuration:")
        logger.info(f"• GPU acceleration: {'✅ ENABLED' if args.use_gpu else '❌ DISABLED'}")
        logger.info(f"• CPU cores available: {cpu_count()}")
        logger.info(f"• Cancer-level workers: {args.max_workers}")
        logger.info(f"• Feature-level workers: 6 processes per cancer")
        logger.info(f"• Total CPU processes: {args.max_workers} × 6 = {args.max_workers * 6} processes")
        
        # Process each omics type
        for omics_type, omics_data in omics_data_map.items():
            logger.info(f"🎯 Starting PARALLEL {omics_type} analysis...")
            logger.info(f"📊 Data shape: {omics_data.shape[0]} patients × {omics_data.shape[1]:,} features")
            
            try:
                cox_results, summary_stats = perform_cox_regression_by_cancer_parallel(
                    omics_data=omics_data,
                    clinical_data=filtered_clinical,
                    omics_type=omics_type,
                    min_patients=args.min_patients,
                    p_threshold=args.p_threshold,
                    use_gpu=args.use_gpu,
                    max_workers=args.max_workers,
                    logger=logger
                )
                
                all_cox_results[omics_type] = cox_results
                all_summary_stats[omics_type] = summary_stats
                
                # Display summary
                logger.info(f"📈 {omics_type} Analysis Summary:")
                logger.info(f"  Total features analyzed: {summary_stats['total_features']:,}")
                logger.info(f"  Total patients: {summary_stats['total_patients']:,}")
                logger.info(f"  Cancer types analyzed: {summary_stats['cancer_types']}")
                
                if summary_stats['significant_features']:
                    total_significant = sum(summary_stats['significant_features'].values())
                    logger.info(f"  🎯 Total significant features: {total_significant:,}")
                    
                    # Show top cancer types
                    sorted_cancers = sorted(summary_stats['significant_features'].items(), 
                                          key=lambda x: x[1], reverse=True)
                    logger.info(f"  🏆 Top cancer types with significant features:")
                    for i, (cancer, count) in enumerate(sorted_cancers[:5]):
                        percentage = (count / summary_stats['total_features']) * 100
                        logger.info(f"    {i+1}. {cancer}: {count:,} features ({percentage:.1f}%)")
                
                logger.info(f"  ✅ {omics_type} processing completed!")
                
                # Save intermediate progress
                save_progress(all_cox_results, all_summary_stats, DATA_PROCESSED_PATH, logger)
                
            except Exception as e:
                logger.error(f"  ❌ {omics_type} processing failed: {str(e)}")
                all_cox_results[omics_type] = {}
                all_summary_stats[omics_type] = {
                    'total_features': omics_data.shape[1],
                    'total_patients': 0,
                    'cancer_types': 0,
                    'significant_features': {},
                    'top_features': {}
                }
        
        logger.info("=" * 60)
        logger.info("🎉 PARALLEL COX REGRESSION ANALYSIS COMPLETED!")
        logger.info("=" * 60)
        
        # Create lookup tables
        logger.info("🔧 Creating Cox coefficient lookup tables...")
        omics_types = list(all_cox_results.keys())
        lookup_tables = create_cox_coefficient_lookup(all_cox_results, omics_types, logger)
        
        # Save all results
        logger.info("=" * 60)
        logger.info("💾 SAVING PROCESSED DATA AND RESULTS")
        logger.info("=" * 60)
        
        # Save lookup tables
        for omics_type, tables in lookup_tables.items():
            # Save coefficient matrix
            coef_file = DATA_PROCESSED_PATH / f'cox_coefficients_{omics_type.lower()}.parquet'
            tables['coefficients'].to_parquet(coef_file)
            logger.info(f"Saved {omics_type} coefficients: {coef_file}")
            
            # Save p-values matrix
            pval_file = DATA_PROCESSED_PATH / f'cox_pvalues_{omics_type.lower()}.parquet'
            tables['p_values'].to_parquet(pval_file)
            logger.info(f"Saved {omics_type} p-values: {pval_file}")
            
            # Save feature statistics
            stats_file = DATA_PROCESSED_PATH / f'cox_feature_stats_{omics_type.lower()}.parquet'
            tables['feature_stats'].to_parquet(stats_file)
            logger.info(f"Saved {omics_type} feature stats: {stats_file}")
            
            # Save raw results
            raw_file = DATA_PROCESSED_PATH / f'cox_raw_results_{omics_type.lower()}.parquet'
            tables['raw_results'].to_parquet(raw_file)
            logger.info(f"Saved {omics_type} raw results: {raw_file}")
        
        # Save processed omics data
        for omics_type, data in filtered_data.items():
            processed_file = DATA_PROCESSED_PATH / f'processed_{omics_type.lower()}_data.parquet'
            data.to_parquet(processed_file)
            logger.info(f"Saved processed {omics_type} data: {processed_file}")
        
        # Save methylation data for tab-transformer
        if len(final_methylation_patients) > 0:
            methylation_file = DATA_PROCESSED_PATH / 'methylation_data_for_tabtransformer.parquet'
            methylation_filtered = methylation_data.loc[final_methylation_patients]
            methylation_filtered.to_parquet(methylation_file)
            logger.info(f"Saved methylation data: {methylation_file}")
            logger.info(f"  Shape: {methylation_filtered.shape[0]} patients × {methylation_filtered.shape[1]} probes")
        
        # Save processed clinical data
        clinical_file = DATA_PROCESSED_PATH / 'processed_clinical_data.parquet'
        filtered_clinical.to_parquet(clinical_file)
        logger.info(f"Saved processed clinical data: {clinical_file}")
        
        # Save analysis summary
        summary_file = RESULTS_PATH / 'cox_analysis_summary.json'
        with open(summary_file, 'w') as f:
            summary_for_json = {}
            for omics_type, stats in all_summary_stats.items():
                summary_for_json[omics_type] = {
                    'total_features': int(stats['total_features']),
                    'total_patients': int(stats['total_patients']),
                    'cancer_types': int(stats['cancer_types']),
                    'significant_features': {k: int(v) for k, v in stats['significant_features'].items()}
                }
            json.dump(summary_for_json, f, indent=2)
        logger.info(f"Saved analysis summary: {summary_file}")
        
        # Save transformation statistics
        transform_file = RESULTS_PATH / 'transformation_stats.json'
        with open(transform_file, 'w') as f:
            transform_for_json = {}
            for key, value in transformation_stats.items():
                if isinstance(value, dict):
                    transform_for_json[key] = {k: float(v) for k, v in value.items()}
                else:
                    transform_for_json[key] = int(value) if isinstance(value, (int, np.integer)) else float(value)
            json.dump(transform_for_json, f, indent=2)
        logger.info(f"Saved transformation stats: {transform_file}")
        
        # Save metadata
        metadata = {
            'data_processing_info': {
                'total_datasets': len(datasets),
                'log2_transformed': ['Expression', 'CNV', 'microRNA', 'RPPA'],
                'no_transformation': ['Methylation', 'Mutations'],
                'cox_analysis_patients': len(final_cox_patients),
                'methylation_patients': len(final_methylation_patients),
                'processing_time': datetime.now().isoformat(),
                'max_workers': args.max_workers,
                'gpu_enabled': args.use_gpu
            }
        }
        
        metadata_file = RESULTS_PATH / 'data_processing_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata: {metadata_file}")
        
        # Final summary
        successful_analyses = [k for k, v in all_cox_results.items() if v]
        total_features_processed = sum(stats['total_features'] for omics_type, stats in all_summary_stats.items() if omics_type in successful_analyses)
        total_significant_features = sum(
            sum(stats['significant_features'].values()) 
            for omics_type, stats in all_summary_stats.items() if omics_type in successful_analyses
        )
        
        logger.info("=" * 60)
        logger.info("🎊 ALL DATA PROCESSING AND ANALYSIS COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"✅ Successfully processed omics types: {len(successful_analyses)}/{len(omics_data_map)}")
        logger.info(f"📊 Total features processed: {total_features_processed:,}")
        logger.info(f"🎯 Total significant features found: {total_significant_features:,}")
        logger.info(f"📁 Processed data saved to: {DATA_PROCESSED_PATH}")
        logger.info(f"📈 Analysis results saved to: {RESULTS_PATH}")
        
        logger.info(f"\n🏁 Final Analysis Summary:")
        for omics_type, stats in all_summary_stats.items():
            if omics_type in successful_analyses:
                total_significant = sum(stats['significant_features'].values())
                logger.info(f"  {omics_type}: {stats['total_features']:,} features → {total_significant:,} significant")
        
        logger.info("🎉 Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed with error: {e}")
        logger.error("Stack trace:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()