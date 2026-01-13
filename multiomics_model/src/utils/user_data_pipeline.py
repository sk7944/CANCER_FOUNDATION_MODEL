#!/usr/bin/env python3
"""
User Data Preprocessing Pipeline for HybridMultiModalModel
==========================================================

This module provides a complete pipeline for preprocessing user data to match
the trained HybridMultiModalModel format.

Key Features:
- Multi-omics data integration (Expression, CNV, microRNA, RPPA, Mutations)
- [value, cox] pair creation with cancer-type-specific Cox coefficients
- Methylation data processing
- Clinical data encoding
- Missing modality handling

Usage:
    from multiomics_model.src.utils.user_data_pipeline import UserDataPipeline

    pipeline = UserDataPipeline(
        cox_coef_path='path/to/cox_coefficients.parquet',
        feature_metadata_path='path/to/feature_metadata.json'
    )

    result = pipeline.process_user_data(
        user_files={'expression': 'expr.csv', 'methylation': 'meth.csv'},
        cancer_type='BRCA'
    )
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from datetime import datetime

# Import our custom utilities
from .feature_converter import FeatureConverter
from .protein_mapper import ProteinMapper
from .data_format_guide import DataFormatGuide

warnings.filterwarnings('ignore')


class UserDataPipeline:
    """
    Complete pipeline for user data preprocessing for HybridMultiModalModel

    Supports:
    - Multi-omics: Expression, CNV, microRNA, RPPA, Mutations
    - Methylation: 450K/EPIC array beta values
    - Clinical: age, sex, race, stage, grade
    - [value, cox] pair format for omics data
    - Missing modality handling
    """

    # Clinical categories (must match model training)
    CLINICAL_CATEGORIES = (10, 2, 6, 5, 4)  # age_group, sex, race, stage, grade
    CLINICAL_COLS = ['age_group', 'sex', 'race', 'ajcc_pathologic_stage', 'grade']

    # Data transformations
    LOG2_TRANSFORM_TYPES = ['expression', 'microrna']
    LOG2_SHIFT_TRANSFORM_TYPES = ['cnv', 'rppa']  # log2(x - min + 1)
    NO_TRANSFORM_TYPES = ['mutations', 'methylation']  # Already in correct format

    def __init__(self,
                 cox_coef_path: Optional[str] = None,
                 feature_metadata_path: Optional[str] = None,
                 custom_protein_mappings: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize preprocessing pipeline for HybridMultiModalModel

        Args:
            cox_coef_path: Path to Cox coefficients parquet file (cancer_type columns)
            feature_metadata_path: Path to feature metadata JSON from training
            custom_protein_mappings: Optional custom protein mappings file
            logger: Optional logger instance
        """
        self.logger = logger or self._setup_logger()

        # Load Cox coefficients if provided
        self.cox_coef_path = Path(cox_coef_path) if cox_coef_path else None
        self.cox_coefficients = self._load_cox_coefficients() if cox_coef_path else None

        # Initialize component utilities
        self._initialize_components(feature_metadata_path, custom_protein_mappings)

        # Expected user data format
        self.expected_formats = self._define_expected_user_formats()

        self.logger.info("UserDataPipeline initialized for HybridMultiModalModel")
        if self.cox_coefficients is not None:
            self.logger.info(f"Cox coefficients loaded: {self.cox_coefficients.shape}")

    def _setup_logger(self) -> logging.Logger:
        """Setup default logger"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _load_cox_coefficients(self) -> Optional[pd.DataFrame]:
        """Load Cox coefficients lookup table"""
        if self.cox_coef_path is None or not self.cox_coef_path.exists():
            self.logger.warning(f"Cox coefficients not found: {self.cox_coef_path}")
            return None

        try:
            cox_coef = pd.read_parquet(self.cox_coef_path)
            self.logger.info(f"Loaded Cox coefficients: {cox_coef.shape}")
            return cox_coef
        except Exception as e:
            self.logger.error(f"Failed to load Cox coefficients: {e}")
            return None

    def _initialize_components(self, feature_metadata_path, custom_protein_mappings):
        """Initialize component utilities"""
        try:
            # Initialize format guide
            self.format_guide = DataFormatGuide(self.logger)

            # Initialize protein mapper
            self.protein_mapper = ProteinMapper(custom_protein_mappings, self.logger)

            # Feature converter (if metadata available)
            self.feature_converter = None
            if feature_metadata_path and Path(feature_metadata_path).exists():
                try:
                    self.feature_converter = FeatureConverter(feature_metadata_path, self.logger)
                    self.logger.info("FeatureConverter initialized with metadata")
                except Exception as e:
                    self.logger.warning(f"Could not initialize FeatureConverter: {e}")

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise

    def _define_expected_user_formats(self) -> Dict:
        """Define expected user input formats for HybridMultiModalModel"""
        return {
            'model_type': 'HybridMultiModalModel',
            'input_format': {
                'orientation': 'samples_as_rows',  # Rows = patients, Columns = features
                'file_types': ['.csv', '.tsv', '.txt', '.xlsx', '.parquet'],
                'index_column': 'patient_id or sample_id'
            },
            'data_types': {
                'expression': {
                    'features': 'Gene symbols (TP53, BRCA1, etc.) or SYMBOL|ENTREZ_ID',
                    'values': 'log2(FPKM/TPM/counts + 1) or raw values',
                    'transformation': 'log2(x + 1) applied automatically',
                    'output_format': '[value, cox] pairs'
                },
                'cnv': {
                    'features': 'Gene symbols',
                    'values': 'log2 copy number ratios or raw values',
                    'transformation': 'log2(x - min + 1) applied automatically',
                    'output_format': '[value, cox] pairs'
                },
                'microrna': {
                    'features': 'miRNA names (hsa-mir-21, etc.)',
                    'values': 'log2(normalized_reads + 1) or raw values',
                    'transformation': 'log2(x + 1) applied automatically',
                    'output_format': '[value, cox] pairs'
                },
                'rppa': {
                    'features': 'Protein names (p53, EGFR, p-AKT-S473, etc.)',
                    'values': 'log2 normalized protein abundance or raw values',
                    'transformation': 'log2(x - min + 1) applied automatically',
                    'output_format': '[value, cox] pairs'
                },
                'mutations': {
                    'features': 'Gene symbols',
                    'values': 'Impact scores: 0=none/silent, 1=missense, 2=LoF',
                    'transformation': 'None (categorical)',
                    'output_format': '[value, cox] pairs'
                },
                'methylation': {
                    'features': 'CpG probe IDs (cg00000292, etc.)',
                    'values': 'Beta values (0-1)',
                    'transformation': 'None (keep beta values)',
                    'output_format': 'Direct values (no Cox pairing)'
                },
                'clinical': {
                    'features': 'age, sex/gender, race, stage, grade',
                    'values': 'Categorical or numerical',
                    'transformation': 'Automatic encoding to categorical indices',
                    'output_format': '5 categorical variables'
                }
            },
            'clinical_encoding': {
                'age_group': '0-9 (10 bins: 0-29, 30-39, ..., 80+)',
                'sex': '0=MALE, 1=FEMALE',
                'race': '0-5 (WHITE, BLACK, ASIAN, NATIVE, PACIFIC, UNKNOWN)',
                'stage': '0-4 (I, II, III, IV, NA)',
                'grade': '0-3 (G1, G2, G3, G4)'
            }
        }

    def show_user_format_guide(self):
        """Display comprehensive format guide for HybridMultiModalModel users"""
        print("=" * 80)
        print("USER DATA FORMAT GUIDE FOR HybridMultiModalModel")
        print("=" * 80)
        print()
        print("GENERAL REQUIREMENTS:")
        print("   - Data format: CSV/TSV/Parquet with samples as rows, features as columns")
        print("   - Index: Patient/Sample IDs")
        print("   - Missing values: Allowed (handled via missing modality masks)")
        print()

        print("MULTI-OMICS DATA (will be converted to [value, cox] pairs):")
        print("-" * 60)

        for data_type in ['expression', 'cnv', 'microrna', 'rppa', 'mutations']:
            spec = self.expected_formats['data_types'][data_type]
            print(f"\n  {data_type.upper()}:")
            print(f"    Features: {spec['features']}")
            print(f"    Values: {spec['values']}")
            print(f"    Transform: {spec['transformation']}")
            print(f"    Output: {spec['output_format']}")

        print("\n" + "-" * 60)
        print("\nMETHYLATION DATA:")
        meth_spec = self.expected_formats['data_types']['methylation']
        print(f"    Features: {meth_spec['features']}")
        print(f"    Values: {meth_spec['values']}")
        print(f"    Transform: {meth_spec['transformation']}")

        print("\n" + "-" * 60)
        print("\nCLINICAL DATA ENCODING:")
        for col, encoding in self.expected_formats['clinical_encoding'].items():
            print(f"    {col}: {encoding}")

        print("\n" + "=" * 80)
        print("IMPORTANT: Cox coefficients are cancer-type specific!")
        print("Always provide the correct cancer_type when processing data.")
        print("=" * 80)

    def process_user_data(self,
                         user_files: Dict[str, str],
                         cancer_type: str,
                         validate_first: bool = True,
                         save_intermediate: bool = False,
                         output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete user data processing pipeline for HybridMultiModalModel

        Args:
            user_files: Dict mapping data_type -> file_path
                      e.g., {'expression': 'expr.csv', 'clinical': 'clin.csv'}
            cancer_type: Cancer type acronym (e.g., 'BRCA', 'LUAD')
            validate_first: Whether to validate data before processing
            save_intermediate: Whether to save intermediate processing results
            output_dir: Directory to save intermediate results

        Returns:
            processing_result: Complete processing results with model-ready tensors
        """

        self.logger.info("Starting user data processing for HybridMultiModalModel...")
        self.logger.info(f"Cancer type: {cancer_type}")
        self.logger.info(f"Input files: {list(user_files.keys())}")

        start_time = datetime.now()

        # Initialize result structure
        result = {
            'success': False,
            'processing_time': None,
            'cancer_type': cancer_type,
            'input_files': user_files.copy(),
            'data_validation': {},
            'loaded_data': {},
            'cox_omics_data': None,  # [value, cox] pairs
            'methylation_data': None,
            'clinical_data': None,
            'masks': {
                'cox_mask': None,
                'meth_mask': None
            },
            'statistics': {},
            'warnings': [],
            'errors': []
        }

        try:
            # Step 1: Load and validate user data
            self.logger.info("Step 1: Loading and validating user data...")
            loaded_data = {}

            for data_type, file_path in user_files.items():
                try:
                    data = self._load_user_file(file_path)
                    loaded_data[data_type] = data
                    self.logger.info(f"  Loaded {data_type}: {data.shape}")

                    if validate_first:
                        validation = self.format_guide.validate_user_data(data_type, data)
                        result['data_validation'][data_type] = validation

                        if not validation['is_valid']:
                            result['warnings'].extend([f"{data_type}: {err}" for err in validation['errors']])
                        if validation.get('warnings'):
                            result['warnings'].extend([f"{data_type}: {warn}" for warn in validation['warnings']])

                except Exception as e:
                    error_msg = f"Failed to load {data_type} from {file_path}: {e}"
                    result['errors'].append(error_msg)
                    self.logger.error(f"  {error_msg}")

            if not loaded_data:
                raise ValueError("No data files were successfully loaded")

            result['loaded_data'] = {k: v.shape for k, v in loaded_data.items()}

            # Step 2: Process Cox omics data (create [value, cox] pairs)
            self.logger.info("Step 2: Processing Cox omics data...")
            cox_omics_types = ['expression', 'cnv', 'microrna', 'rppa', 'mutations']
            cox_omics_dict = {k: loaded_data.get(k) for k in cox_omics_types}

            # Filter out None values
            cox_omics_dict = {k: v for k, v in cox_omics_dict.items() if v is not None}

            if cox_omics_dict:
                cox_omics_array, cox_features = self._create_cox_omics_array(
                    cox_omics_dict, cancer_type
                )
                result['cox_omics_data'] = {
                    'array': cox_omics_array,
                    'features': cox_features,
                    'shape': cox_omics_array.shape
                }
                result['masks']['cox_mask'] = True
                self.logger.info(f"  Cox omics array: {cox_omics_array.shape}")
            else:
                result['masks']['cox_mask'] = False
                result['warnings'].append("No Cox omics data provided")

            # Step 3: Process methylation data
            self.logger.info("Step 3: Processing methylation data...")
            if 'methylation' in loaded_data:
                meth_array = self._process_methylation_data(loaded_data['methylation'])
                result['methylation_data'] = {
                    'array': meth_array,
                    'shape': meth_array.shape
                }
                result['masks']['meth_mask'] = True
                self.logger.info(f"  Methylation array: {meth_array.shape}")
            else:
                result['masks']['meth_mask'] = False
                result['warnings'].append("No methylation data provided")

            # Step 4: Process clinical data
            self.logger.info("Step 4: Processing clinical data...")
            if 'clinical' in loaded_data:
                clinical_array = self._process_clinical_data(loaded_data['clinical'])
                result['clinical_data'] = {
                    'array': clinical_array,
                    'columns': self.CLINICAL_COLS,
                    'shape': clinical_array.shape
                }
                self.logger.info(f"  Clinical array: {clinical_array.shape}")
            else:
                result['warnings'].append("No clinical data provided - using defaults")
                # Create default clinical array
                n_samples = self._get_sample_count(loaded_data)
                result['clinical_data'] = {
                    'array': self._create_default_clinical_array(n_samples),
                    'columns': self.CLINICAL_COLS,
                    'shape': (n_samples, 5)
                }

            # Step 5: Generate statistics
            self.logger.info("Step 5: Generating statistics...")
            result['statistics'] = self._generate_processing_statistics(result)

            # Step 6: Save intermediate results if requested
            if save_intermediate and output_dir:
                self._save_intermediate_results(result, output_dir)

            # Mark as successful
            result['success'] = True
            result['processing_time'] = (datetime.now() - start_time).total_seconds()

            self.logger.info("User data processing completed successfully!")
            self.logger.info(f"  Processing time: {result['processing_time']:.2f} seconds")

        except Exception as e:
            result['errors'].append(f"Pipeline error: {e}")
            result['success'] = False
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Pipeline failed: {e}")

        return result

    def _load_user_file(self, file_path: str) -> pd.DataFrame:
        """Load user data file with automatic format detection"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            if file_path.suffix.lower() == '.csv':
                return pd.read_csv(file_path, index_col=0)
            elif file_path.suffix.lower() in ['.tsv', '.txt']:
                return pd.read_csv(file_path, sep='\t', index_col=0)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                return pd.read_excel(file_path, index_col=0)
            elif file_path.suffix.lower() == '.parquet':
                return pd.read_parquet(file_path)
            else:
                # Try CSV first, then TSV
                try:
                    return pd.read_csv(file_path, index_col=0)
                except:
                    return pd.read_csv(file_path, sep='\t', index_col=0)

        except Exception as e:
            raise ValueError(f"Could not read file {file_path}: {e}")

    def _create_cox_omics_array(
        self,
        omics_dict: Dict[str, pd.DataFrame],
        cancer_type: str
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create Cox omics array with [value, cox] pairs

        Args:
            omics_dict: Dict of data_type -> DataFrame
            cancer_type: Cancer type acronym for Cox coefficient lookup

        Returns:
            cox_omics_array: (n_samples, n_features * 2) array with [value, cox] pairs
            feature_names: List of feature names
        """
        # Determine sample count and alignment
        patient_ids = None
        for data_type, df in omics_dict.items():
            if df is not None and len(df) > 0:
                if patient_ids is None:
                    patient_ids = df.index.tolist()
                break

        if patient_ids is None:
            raise ValueError("No valid omics data to process")

        n_samples = len(patient_ids)

        # Get Cox coefficients for this cancer type
        cox_coef_series = None
        if self.cox_coefficients is not None:
            if cancer_type in self.cox_coefficients.columns:
                cox_coef_series = self.cox_coefficients[cancer_type]
            else:
                self.logger.warning(f"Cancer type {cancer_type} not found in Cox coefficients")
                cox_coef_series = self.cox_coefficients.mean(axis=1)

        # Build [value, cox] pairs
        all_values = []
        feature_names = []

        for data_type in ['expression', 'cnv', 'microrna', 'rppa', 'mutations']:
            if data_type not in omics_dict or omics_dict[data_type] is None:
                continue

            df = omics_dict[data_type].copy()

            # Align samples
            df = df.reindex(patient_ids)

            # Apply transformation
            if data_type in self.LOG2_TRANSFORM_TYPES:
                # log2(x + 1)
                df = np.log2(df.clip(lower=0) + 1)
            elif data_type in self.LOG2_SHIFT_TRANSFORM_TYPES:
                # log2(x - min + 1)
                min_val = df.min().min()
                df = np.log2(df - min_val + 1)
            # NO_TRANSFORM_TYPES: keep as is

            # Create [value, cox] pairs for each feature
            for col in df.columns:
                # Feature name (for Cox lookup)
                prefix_map = {
                    'expression': 'Expression',
                    'cnv': 'CNV',
                    'microrna': 'microRNA',
                    'rppa': 'RPPA',
                    'mutations': 'Mutations'
                }
                feature_key = f"{prefix_map.get(data_type, data_type)}_{col}"

                # Get value
                values = df[col].values.astype(np.float32)

                # Get Cox coefficient
                cox_coef = 0.0
                if cox_coef_series is not None and feature_key in cox_coef_series.index:
                    cox_coef = float(cox_coef_series[feature_key])

                cox_values = np.full(n_samples, cox_coef, dtype=np.float32)

                # Add [val, cox] pair
                all_values.append(values)
                all_values.append(cox_values)
                feature_names.append(f"{col}_val")
                feature_names.append(f"{col}_cox")

        if not all_values:
            raise ValueError("No features extracted from omics data")

        # Stack into array
        cox_omics_array = np.column_stack(all_values)

        # Handle NaN/Inf
        cox_omics_array = np.nan_to_num(cox_omics_array, nan=0.0, posinf=0.0, neginf=0.0)

        return cox_omics_array, feature_names

    def _process_methylation_data(self, meth_df: pd.DataFrame) -> np.ndarray:
        """
        Process methylation data (beta values)

        Args:
            meth_df: DataFrame with CpG sites as columns

        Returns:
            meth_array: (n_samples, n_cpg_sites) array
        """
        meth_array = meth_df.values.astype(np.float32)

        # Ensure beta values are in [0, 1]
        meth_array = np.nan_to_num(meth_array, nan=0.5)
        meth_array = np.clip(meth_array, 0, 1)

        return meth_array

    def _process_clinical_data(self, clinical_df: pd.DataFrame) -> np.ndarray:
        """
        Process clinical data to categorical indices

        Args:
            clinical_df: DataFrame with clinical columns

        Returns:
            clinical_array: (n_samples, 5) array with categorical indices
        """
        df = clinical_df.copy()
        n_samples = len(df)
        clinical_array = np.zeros((n_samples, 5), dtype=np.int64)

        # Age group encoding (column 0)
        if 'age_group' in df.columns:
            clinical_array[:, 0] = df['age_group'].values
        elif 'age' in df.columns or 'age_at_initial_pathologic_diagnosis' in df.columns:
            age_col = 'age' if 'age' in df.columns else 'age_at_initial_pathologic_diagnosis'
            age = pd.to_numeric(df[age_col], errors='coerce').fillna(60)
            bins = [0, 30, 40, 50, 55, 60, 65, 70, 75, 80, 120]
            clinical_array[:, 0] = pd.cut(age, bins=bins, labels=range(10), right=False).astype(int).values
        else:
            clinical_array[:, 0] = 5  # Default: 60-65 age group

        # Sex encoding (column 1)
        if 'sex' in df.columns:
            if df['sex'].dtype == object:
                sex_map = {'MALE': 0, 'FEMALE': 1, 'M': 0, 'F': 1, 'male': 0, 'female': 1}
                clinical_array[:, 1] = df['sex'].map(sex_map).fillna(0).astype(int).values
            else:
                clinical_array[:, 1] = df['sex'].values
        elif 'gender' in df.columns:
            sex_map = {'MALE': 0, 'FEMALE': 1, 'M': 0, 'F': 1, 'male': 0, 'female': 1}
            clinical_array[:, 1] = df['gender'].map(sex_map).fillna(0).astype(int).values
        else:
            clinical_array[:, 1] = 0  # Default: MALE

        # Race encoding (column 2)
        if 'race' in df.columns:
            if df['race'].dtype == object:
                race_map = {
                    'WHITE': 0, 'BLACK OR AFRICAN AMERICAN': 1, 'ASIAN': 2,
                    'AMERICAN INDIAN OR ALASKA NATIVE': 3,
                    'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 4,
                    '[Not Available]': 5, '[Not Evaluated]': 5, '[Unknown]': 5,
                    'white': 0, 'black': 1, 'asian': 2
                }
                clinical_array[:, 2] = df['race'].map(race_map).fillna(5).astype(int).values
            else:
                clinical_array[:, 2] = df['race'].values
        else:
            clinical_array[:, 2] = 5  # Default: Unknown

        # Stage encoding (column 3)
        stage_col = None
        for col in ['ajcc_pathologic_stage', 'pathologic_stage', 'stage']:
            if col in df.columns:
                stage_col = col
                break

        if stage_col:
            if df[stage_col].dtype == object:
                stage_map = {
                    'Stage I': 0, 'Stage IA': 0, 'Stage IB': 0, 'Stage IC': 0, 'I': 0,
                    'Stage II': 1, 'Stage IIA': 1, 'Stage IIB': 1, 'Stage IIC': 1, 'II': 1,
                    'Stage III': 2, 'Stage IIIA': 2, 'Stage IIIB': 2, 'Stage IIIC': 2, 'III': 2,
                    'Stage IV': 3, 'Stage IVA': 3, 'Stage IVB': 3, 'Stage IVC': 3, 'IV': 3,
                    '[Not Available]': 4, '[Unknown]': 4, 'NA': 4
                }
                clinical_array[:, 3] = df[stage_col].map(stage_map).fillna(4).astype(int).values
            else:
                clinical_array[:, 3] = df[stage_col].values
        else:
            clinical_array[:, 3] = 4  # Default: NA

        # Grade encoding (column 4)
        grade_col = None
        for col in ['grade', 'neoplasm_histologic_grade', 'histologic_grade']:
            if col in df.columns:
                grade_col = col
                break

        if grade_col:
            if df[grade_col].dtype == object:
                grade_map = {'G1': 0, 'G2': 1, 'G3': 2, 'G4': 3, 'GX': 2,
                            '1': 0, '2': 1, '3': 2, '4': 3}
                clinical_array[:, 4] = df[grade_col].map(grade_map).fillna(2).astype(int).values
            else:
                clinical_array[:, 4] = df[grade_col].values
        else:
            clinical_array[:, 4] = 2  # Default: G3

        # Clip to valid ranges
        for i, max_val in enumerate(self.CLINICAL_CATEGORIES):
            clinical_array[:, i] = np.clip(clinical_array[:, i], 0, max_val - 1)

        return clinical_array

    def _create_default_clinical_array(self, n_samples: int) -> np.ndarray:
        """Create default clinical array when clinical data is not provided"""
        clinical_array = np.zeros((n_samples, 5), dtype=np.int64)
        clinical_array[:, 0] = 5  # age_group: 60-65
        clinical_array[:, 1] = 0  # sex: MALE
        clinical_array[:, 2] = 5  # race: Unknown
        clinical_array[:, 3] = 4  # stage: NA
        clinical_array[:, 4] = 2  # grade: G3
        return clinical_array

    def _get_sample_count(self, loaded_data: Dict[str, pd.DataFrame]) -> int:
        """Get sample count from loaded data"""
        for df in loaded_data.values():
            if df is not None and len(df) > 0:
                return len(df)
        return 1

    def _generate_processing_statistics(self, result: Dict) -> Dict:
        """Generate comprehensive processing statistics"""
        stats = {
            'cancer_type': result['cancer_type'],
            'n_samples': None,
            'cox_omics': {},
            'methylation': {},
            'clinical': {},
            'modality_availability': {
                'cox_omics': result['masks']['cox_mask'],
                'methylation': result['masks']['meth_mask']
            }
        }

        # Cox omics stats
        if result['cox_omics_data']:
            cox_data = result['cox_omics_data']
            stats['n_samples'] = cox_data['shape'][0]
            stats['cox_omics'] = {
                'n_features': cox_data['shape'][1] // 2,  # [value, cox] pairs
                'total_dimensions': cox_data['shape'][1]
            }

        # Methylation stats
        if result['methylation_data']:
            meth_data = result['methylation_data']
            if stats['n_samples'] is None:
                stats['n_samples'] = meth_data['shape'][0]
            stats['methylation'] = {
                'n_cpg_sites': meth_data['shape'][1]
            }

        # Clinical stats
        if result['clinical_data']:
            clinical_data = result['clinical_data']
            if stats['n_samples'] is None:
                stats['n_samples'] = clinical_data['shape'][0]
            stats['clinical'] = {
                'n_features': clinical_data['shape'][1],
                'categories': self.CLINICAL_CATEGORIES
            }

        return stats

    def _save_intermediate_results(self, result: Dict, output_dir: str):
        """Save intermediate processing results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Save Cox omics data
            if result['cox_omics_data']:
                np.save(
                    output_path / f"cox_omics_{timestamp}.npy",
                    result['cox_omics_data']['array']
                )

            # Save methylation data
            if result['methylation_data']:
                np.save(
                    output_path / f"methylation_{timestamp}.npy",
                    result['methylation_data']['array']
                )

            # Save clinical data
            if result['clinical_data']:
                np.save(
                    output_path / f"clinical_{timestamp}.npy",
                    result['clinical_data']['array']
                )

            # Save processing report
            report = {
                'success': result['success'],
                'processing_time': result['processing_time'],
                'cancer_type': result['cancer_type'],
                'statistics': result['statistics'],
                'warnings': result['warnings'],
                'errors': result['errors'],
                'masks': result['masks']
            }

            with open(output_path / f"processing_report_{timestamp}.json", 'w') as f:
                json.dump(report, f, indent=2)

            self.logger.info(f"Saved intermediate results to {output_path}")

        except Exception as e:
            self.logger.warning(f"Could not save intermediate results: {e}")

    def validate_user_files(self, user_files: Dict[str, str]) -> Dict:
        """Quick validation of user files before processing"""
        validation_results = {}

        for data_type, file_path in user_files.items():
            try:
                data = self._load_user_file(file_path)
                validation = self.format_guide.validate_user_data(data_type, data)
                validation_results[data_type] = validation

                status = "Valid" if validation['is_valid'] else "Invalid"
                self.logger.info(f"  {data_type}: {status}")

            except Exception as e:
                validation_results[data_type] = {
                    'is_valid': False,
                    'errors': [str(e)],
                    'warnings': []
                }
                self.logger.error(f"  {data_type}: {e}")

        return validation_results


def main():
    """CLI interface for user data pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description='User Data Processing Pipeline for HybridMultiModalModel')
    parser.add_argument('--cox-coef', help='Cox coefficients parquet file')
    parser.add_argument('--cancer-type', default='BRCA', help='Cancer type acronym')
    parser.add_argument('--expression', help='Expression data file')
    parser.add_argument('--methylation', help='Methylation data file')
    parser.add_argument('--clinical', help='Clinical data file')
    parser.add_argument('--output-dir', help='Output directory for results')
    parser.add_argument('--show-guide', action='store_true', help='Show format guide')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = UserDataPipeline(cox_coef_path=args.cox_coef)

    if args.show_guide:
        pipeline.show_user_format_guide()
        return

    # Build user files dict
    user_files = {}
    if args.expression:
        user_files['expression'] = args.expression
    if args.methylation:
        user_files['methylation'] = args.methylation
    if args.clinical:
        user_files['clinical'] = args.clinical

    if user_files:
        result = pipeline.process_user_data(
            user_files,
            cancer_type=args.cancer_type,
            save_intermediate=bool(args.output_dir),
            output_dir=args.output_dir
        )

        print(f"\nProcessing Result: {'Success' if result['success'] else 'Failed'}")
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        print(f"Statistics: {result['statistics']}")

        if result['warnings']:
            print(f"\nWarnings ({len(result['warnings'])}):")
            for warn in result['warnings'][:5]:
                print(f"  - {warn}")

        if result['errors']:
            print(f"\nErrors ({len(result['errors'])}):")
            for err in result['errors']:
                print(f"  - {err}")


if __name__ == "__main__":
    main()
