#!/usr/bin/env python3
"""
Feature Format Converter for HybridMultiModalModel
===================================================

This module converts user input features to match the trained HybridMultiModalModel format.
Handles gene symbols, Entrez IDs, protein names, and [value, cox] pair generation.

Key Features:
- Convert user gene symbols to model format (SYMBOL|ENTREZ)
- Create [value, cox] pairs with cancer-type-specific Cox coefficients
- Support for all omics types: Expression, CNV, microRNA, RPPA, Mutations

Usage:
    from multiomics_model.src.utils.feature_converter import FeatureConverter

    converter = FeatureConverter('path/to/feature_metadata.json')
    converted_data, missing_mask, report = converter.convert_user_data(user_df, 'expression')
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import re
from collections import defaultdict


class FeatureConverter:
    """
    Convert user features to HybridMultiModalModel format

    Supports:
    - Gene symbol/Entrez ID mapping for Expression
    - Gene symbol mapping for CNV, Mutations
    - microRNA name mapping
    - Protein name mapping for RPPA
    - [value, cox] pair generation
    """

    # Omics type prefixes used in training data
    OMICS_PREFIXES = {
        'expression': 'Expression',
        'cnv': 'CNV',
        'microrna': 'microRNA',
        'rppa': 'RPPA',
        'mutations': 'Mutations'
    }

    def __init__(self,
                 feature_metadata_path: str,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize converter with trained model feature metadata

        Args:
            feature_metadata_path: Path to feature_metadata.json from training
            logger: Optional logger instance
        """
        self.logger = logger or self._setup_logger()
        self.feature_metadata_path = Path(feature_metadata_path)

        # Load feature metadata
        self.metadata = self._load_metadata()

        # Create mapping dictionaries for each omics type
        self.gene_mappings = self._create_gene_mappings()
        self.protein_mappings = self._create_protein_mappings()
        self.clinical_mappings = self._create_clinical_mappings()

        # Extract trained feature lists by omics type
        self.trained_features_by_type = self._categorize_trained_features()

        self.logger.info("FeatureConverter initialized for HybridMultiModalModel")
        self.logger.info(f"  Total trained features: {len(self.metadata.get('feature_columns', []))}")

    def _setup_logger(self) -> logging.Logger:
        """Setup default logger"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def _load_metadata(self) -> Dict:
        """Load feature metadata from training"""
        try:
            with open(self.feature_metadata_path, 'r') as f:
                metadata = json.load(f)
            self.logger.info(f"Loaded feature metadata: {self.feature_metadata_path}")
            return metadata
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            return {}

    def _categorize_trained_features(self) -> Dict[str, List[str]]:
        """Categorize trained features by omics type"""
        features_by_type = {
            'expression': [],
            'cnv': [],
            'microrna': [],
            'rppa': [],
            'mutations': [],
            'methylation': [],
            'clinical': []
        }

        for feature in self.metadata.get('feature_columns', []):
            # Identify omics type by prefix
            for omics_type, prefix in self.OMICS_PREFIXES.items():
                if feature.startswith(f"{prefix}_"):
                    features_by_type[omics_type].append(feature)
                    break
            else:
                # Check for methylation (cg prefix) or clinical
                if feature.startswith('cg'):
                    features_by_type['methylation'].append(feature)
                else:
                    features_by_type['clinical'].append(feature)

        return features_by_type

    def _create_gene_mappings(self) -> Dict[str, Dict[str, str]]:
        """
        Create comprehensive gene symbol/ID mappings for each omics type

        Returns:
            Dict mapping omics_type -> {user_symbol -> trained_feature_base}
        """
        gene_mappings = {
            'expression': {},
            'cnv': {},
            'mutations': {},
            'microrna': {}
        }

        for feature in self.metadata.get('feature_columns', []):
            # Expression features: Expression_SYMBOL|ENTREZ_val/cox
            if feature.startswith('Expression_') and (feature.endswith('_val') or feature.endswith('_cox')):
                gene_part = feature.replace('Expression_', '').replace('_val', '').replace('_cox', '')

                if '|' in gene_part:
                    symbol, entrez = gene_part.split('|', 1)
                    # Map multiple variations
                    for key in [symbol, symbol.upper(), symbol.lower(), entrez, gene_part]:
                        if key:
                            gene_mappings['expression'][key] = gene_part

            # CNV features: CNV_SYMBOL_val/cox
            elif feature.startswith('CNV_') and (feature.endswith('_val') or feature.endswith('_cox')):
                gene_symbol = feature.replace('CNV_', '').replace('_val', '').replace('_cox', '')
                for key in [gene_symbol, gene_symbol.upper(), gene_symbol.lower()]:
                    gene_mappings['cnv'][key] = gene_symbol

            # Mutations features: Mutations_SYMBOL_val/cox
            elif feature.startswith('Mutations_') and (feature.endswith('_val') or feature.endswith('_cox')):
                gene_symbol = feature.replace('Mutations_', '').replace('_val', '').replace('_cox', '')
                for key in [gene_symbol, gene_symbol.upper(), gene_symbol.lower()]:
                    gene_mappings['mutations'][key] = gene_symbol

            # microRNA features: microRNA_hsa-mir-XX_val/cox
            elif feature.startswith('microRNA_') and (feature.endswith('_val') or feature.endswith('_cox')):
                mirna_name = feature.replace('microRNA_', '').replace('_val', '').replace('_cox', '')
                for key in [mirna_name, mirna_name.upper(), mirna_name.lower()]:
                    gene_mappings['microrna'][key] = mirna_name

        for omics_type, mappings in gene_mappings.items():
            if mappings:
                self.logger.info(f"  {omics_type} mappings: {len(mappings)}")

        return gene_mappings

    def _create_protein_mappings(self) -> Dict[str, str]:
        """Create protein name mappings for RPPA data"""
        protein_mappings = {}

        # Common RPPA protein name variations
        protein_aliases = {
            'p53': ['TP53', 'P53', 'p53', 'Tp53'],
            'EGFR': ['EGFR', 'EGF-R', 'ERBB1'],
            'HER2': ['ERBB2', 'HER2', 'Her2', 'NEU'],
            'AKT': ['AKT1', 'PKB', 'AKT', 'Akt'],
            'mTOR': ['MTOR', 'mTOR', 'FRAP1'],
            'PTEN': ['PTEN', 'Pten'],
            'ERK': ['MAPK1', 'ERK2', 'ERK'],
        }

        # Add canonical mappings
        for canonical, aliases in protein_aliases.items():
            for alias in aliases:
                protein_mappings[alias.upper()] = canonical
                protein_mappings[alias.lower()] = canonical
                protein_mappings[alias] = canonical

        # Extract from trained RPPA features
        for feature in self.metadata.get('feature_columns', []):
            if feature.startswith('RPPA_') and (feature.endswith('_val') or feature.endswith('_cox')):
                protein_part = feature.replace('RPPA_', '').replace('_val', '').replace('_cox', '')
                # Extract base protein name (before any vendor codes)
                base_protein = protein_part.split('-')[0] if '-' in protein_part else protein_part
                protein_mappings[base_protein.upper()] = protein_part
                protein_mappings[base_protein.lower()] = protein_part
                protein_mappings[protein_part] = protein_part

        self.logger.info(f"  Protein mappings: {len(protein_mappings)}")
        return protein_mappings

    def _create_clinical_mappings(self) -> Dict[str, str]:
        """Create clinical feature mappings"""
        clinical_mappings = {
            # Age mappings
            'age': 'age_group',
            'age_at_diagnosis': 'age_group',
            'age_at_initial_pathologic_diagnosis': 'age_group',
            'patient_age': 'age_group',

            # Sex mappings
            'sex': 'sex',
            'gender': 'sex',
            'patient_gender': 'sex',
            'patient_sex': 'sex',

            # Race mappings
            'race': 'race',
            'ethnicity': 'race',
            'patient_race': 'race',

            # Stage mappings
            'stage': 'ajcc_pathologic_stage',
            'pathologic_stage': 'ajcc_pathologic_stage',
            'tumor_stage': 'ajcc_pathologic_stage',
            'ajcc_pathologic_stage': 'ajcc_pathologic_stage',

            # Grade mappings
            'grade': 'grade',
            'tumor_grade': 'grade',
            'histologic_grade': 'grade',
            'neoplasm_histologic_grade': 'grade',
        }

        return clinical_mappings

    def convert_user_data(self,
                         user_data: pd.DataFrame,
                         data_type: str,
                         cancer_type: Optional[str] = None,
                         cox_coefficients: Optional[pd.DataFrame] = None,
                         missing_strategy: str = 'zero'
                         ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Convert user data to HybridMultiModalModel format with [value, cox] pairs

        Args:
            user_data: User's input data (samples x features)
            data_type: 'expression', 'cnv', 'microrna', 'rppa', 'mutations', 'methylation', 'clinical'
            cancer_type: Cancer type acronym for Cox coefficient lookup (optional)
            cox_coefficients: DataFrame with Cox coefficients (optional)
            missing_strategy: 'zero', 'mean', or 'special_token'

        Returns:
            converted_data: Data in [value, cox] pair format (for omics) or processed format
            missing_mask: Boolean mask of missing features
            conversion_report: Report of conversion process
        """
        self.logger.info(f"Converting {data_type} data to HybridMultiModalModel format...")
        self.logger.info(f"  Input shape: {user_data.shape}")

        if data_type == 'auto':
            data_type = self._detect_data_type(user_data)
            self.logger.info(f"  Auto-detected type: {data_type}")

        # Get appropriate conversion method
        if data_type == 'clinical':
            return self._convert_clinical_data(user_data, missing_strategy)
        elif data_type == 'methylation':
            return self._convert_methylation_data(user_data, missing_strategy)
        else:
            # Omics data with [value, cox] pairs
            return self._convert_omics_data(
                user_data, data_type, cancer_type, cox_coefficients, missing_strategy
            )

    def _convert_omics_data(self,
                           user_data: pd.DataFrame,
                           data_type: str,
                           cancer_type: Optional[str],
                           cox_coefficients: Optional[pd.DataFrame],
                           missing_strategy: str
                           ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Convert omics data to [val, cox] pair format

        For each gene/protein/miRNA, creates two columns:
        - {feature}_val: The transformed measurement value
        - {feature}_cox: The Cox regression coefficient for this cancer type
        """
        n_samples = len(user_data)
        prefix = self.OMICS_PREFIXES.get(data_type, data_type)

        # Get gene mappings for this type
        if data_type in ['expression', 'cnv', 'mutations', 'microrna']:
            mappings = self.gene_mappings.get(data_type, {})
        elif data_type == 'rppa':
            mappings = self.protein_mappings
        else:
            mappings = {}

        # Get Cox coefficients for cancer type
        cox_coef_series = None
        if cox_coefficients is not None and cancer_type is not None:
            if cancer_type in cox_coefficients.columns:
                cox_coef_series = cox_coefficients[cancer_type]
            else:
                self.logger.warning(f"Cancer type {cancer_type} not found in Cox coefficients")

        # Build [value, cox] pairs
        converted_columns = {}
        feature_mapping = {}
        unmapped_features = []

        for user_col in user_data.columns:
            # Find matching trained feature
            mapped_feature = None

            # Try direct mapping
            if user_col in mappings:
                mapped_feature = mappings[user_col]
            elif user_col.upper() in mappings:
                mapped_feature = mappings[user_col.upper()]
            elif user_col.lower() in mappings:
                mapped_feature = mappings[user_col.lower()]

            if mapped_feature:
                # Create [val, cox] pair
                val_col_name = f"{prefix}_{mapped_feature}_val"
                cox_col_name = f"{prefix}_{mapped_feature}_cox"

                # Get values
                values = user_data[user_col].values.astype(np.float32)

                # Get Cox coefficient
                cox_coef = 0.0
                if cox_coef_series is not None:
                    cox_key = f"{prefix}_{mapped_feature}"
                    if cox_key in cox_coef_series.index:
                        cox_coef = float(cox_coef_series[cox_key])

                cox_values = np.full(n_samples, cox_coef, dtype=np.float32)

                converted_columns[val_col_name] = values
                converted_columns[cox_col_name] = cox_values
                feature_mapping[user_col] = {
                    'val_col': val_col_name,
                    'cox_col': cox_col_name,
                    'cox_coef': cox_coef
                }
            else:
                unmapped_features.append(user_col)

        # Create DataFrame
        if converted_columns:
            converted_data = pd.DataFrame(converted_columns, index=user_data.index)
        else:
            converted_data = pd.DataFrame(index=user_data.index)

        # Create missing mask
        trained_features = self.trained_features_by_type.get(data_type, [])
        missing_mask = self._create_missing_mask(converted_data, trained_features)

        # Handle missing values
        converted_data = self._handle_missing_values(converted_data, missing_strategy)

        # Create report
        report = {
            'data_type': data_type,
            'input_features': len(user_data.columns),
            'mapped_features': len(feature_mapping),
            'unmapped_features': len(unmapped_features),
            'output_columns': len(converted_data.columns),
            'value_cox_pairs': len(feature_mapping),
            'missing_strategy': missing_strategy,
            'cancer_type': cancer_type,
            'feature_mapping': feature_mapping,
            'unmapped': unmapped_features[:10]  # First 10 unmapped
        }

        self.logger.info(f"  Mapped: {report['mapped_features']}/{report['input_features']}")
        self.logger.info(f"  Output: {report['output_columns']} columns ({report['value_cox_pairs']} [value, cox] pairs)")

        return converted_data, missing_mask, report

    def _convert_methylation_data(self,
                                  user_data: pd.DataFrame,
                                  missing_strategy: str
                                  ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Convert methylation data (no Cox pairing needed)

        Methylation data is processed separately without [value, cox] pairs
        """
        # Get trained methylation features
        trained_meth_features = set(self.trained_features_by_type.get('methylation', []))

        # Map user CpG sites to trained features
        converted_columns = {}
        feature_mapping = {}

        for user_col in user_data.columns:
            if user_col in trained_meth_features:
                converted_columns[user_col] = user_data[user_col].values.astype(np.float32)
                feature_mapping[user_col] = user_col

        # Create DataFrame
        if converted_columns:
            converted_data = pd.DataFrame(converted_columns, index=user_data.index)
        else:
            converted_data = pd.DataFrame(index=user_data.index)

        # Create missing mask
        missing_mask = self._create_missing_mask(converted_data, list(trained_meth_features))

        # Handle missing values
        converted_data = self._handle_missing_values(converted_data, missing_strategy, default_fill=0.5)

        # Ensure beta values are in [0, 1]
        converted_data = converted_data.clip(0, 1)

        report = {
            'data_type': 'methylation',
            'input_features': len(user_data.columns),
            'mapped_features': len(feature_mapping),
            'output_columns': len(converted_data.columns),
            'missing_strategy': missing_strategy,
            'feature_mapping_count': len(feature_mapping)
        }

        self.logger.info(f"  Mapped: {report['mapped_features']}/{report['input_features']}")

        return converted_data, missing_mask, report

    def _convert_clinical_data(self,
                               user_data: pd.DataFrame,
                               missing_strategy: str
                               ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Convert clinical data to categorical indices

        Returns categorical indices for: age_group, sex, race, stage, grade
        """
        converted_columns = {}
        feature_mapping = {}

        for user_col in user_data.columns:
            mapped_col = self.clinical_mappings.get(user_col.lower())
            if mapped_col:
                converted_columns[mapped_col] = user_data[user_col]
                feature_mapping[user_col] = mapped_col

        if converted_columns:
            converted_data = pd.DataFrame(converted_columns, index=user_data.index)
        else:
            converted_data = pd.DataFrame(index=user_data.index)

        # Create missing mask for expected clinical columns
        expected_clinical = ['age_group', 'sex', 'race', 'ajcc_pathologic_stage', 'grade']
        missing_mask = pd.DataFrame(
            {col: col not in converted_data.columns for col in expected_clinical},
            index=user_data.index
        )

        report = {
            'data_type': 'clinical',
            'input_features': len(user_data.columns),
            'mapped_features': len(feature_mapping),
            'missing_strategy': missing_strategy,
            'feature_mapping': feature_mapping
        }

        return converted_data, missing_mask, report

    def _detect_data_type(self, user_data: pd.DataFrame) -> str:
        """Auto-detect the type of user data"""
        columns = [str(col).upper() for col in user_data.columns]

        # Check for clinical features
        clinical_indicators = ['AGE', 'GENDER', 'SEX', 'RACE', 'CANCER', 'STAGE', 'GRADE']
        if any(indicator in ' '.join(columns) for indicator in clinical_indicators):
            return 'clinical'

        # Check for methylation (CpG sites)
        if any(col.startswith('CG') for col in columns):
            return 'methylation'

        # Check for microRNA
        if any('MIR' in col or 'MIRNA' in col or col.startswith('HSA') for col in columns):
            return 'microrna'

        # Check for mutations (integer values 0, 1, 2)
        numeric_data = user_data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            unique_vals = set(numeric_data.values.flatten())
            unique_vals.discard(np.nan)
            if unique_vals.issubset({0, 1, 2, 0.0, 1.0, 2.0}):
                return 'mutations'

        # Check for RPPA (protein names, phosphorylation)
        if any('-' in col or 'P-' in col.upper() or '_P' in col.upper() for col in columns):
            return 'rppa'

        # Default to expression for gene symbols
        return 'expression'

    def _create_missing_mask(self, converted_data: pd.DataFrame, trained_features: List[str]) -> pd.DataFrame:
        """Create missing feature mask"""
        if not trained_features:
            return pd.DataFrame(index=converted_data.index, dtype=bool)

        missing_mask = pd.DataFrame(
            index=converted_data.index,
            columns=trained_features,
            dtype=bool
        )

        # Mark existing features as not missing (unless they have NaN values)
        for col in trained_features:
            if col in converted_data.columns:
                missing_mask[col] = converted_data[col].isna()
            else:
                missing_mask[col] = True

        return missing_mask

    def _handle_missing_values(self,
                               data: pd.DataFrame,
                               strategy: str,
                               default_fill: float = 0.0) -> pd.DataFrame:
        """Handle missing values according to strategy"""
        if data.empty:
            return data

        if strategy == 'special_token':
            return data.fillna(-999)
        elif strategy == 'zero':
            return data.fillna(0)
        elif strategy == 'mean':
            # Use training means if available
            means = self.metadata.get('feature_stats', {}).get('means', {})
            for col in data.columns:
                if col in means:
                    data[col] = data[col].fillna(means[col])
                else:
                    data[col] = data[col].fillna(default_fill)
            return data
        else:
            return data.fillna(default_fill)

    def get_conversion_summary(self, conversion_report: Dict) -> str:
        """Generate human-readable conversion summary"""
        summary = f"""
Feature Conversion Summary for HybridMultiModalModel
=====================================================
Data Type: {conversion_report.get('data_type', 'Unknown')}
Input Features: {conversion_report.get('input_features', 0)}
Mapped Features: {conversion_report.get('mapped_features', 0)}
Output Columns: {conversion_report.get('output_columns', 0)}
[value, cox] Pairs: {conversion_report.get('value_cox_pairs', 0)}
Cancer Type: {conversion_report.get('cancer_type', 'Not specified')}
Missing Strategy: {conversion_report.get('missing_strategy', 'Unknown')}

Mapping Ratio: {conversion_report.get('mapped_features', 0)}/{conversion_report.get('input_features', 0)} ({100*conversion_report.get('mapped_features', 0)/max(1, conversion_report.get('input_features', 1)):.1f}%)
"""

        if conversion_report.get('unmapped'):
            summary += f"\nUnmapped Features (first 10):\n"
            for feat in conversion_report['unmapped'][:10]:
                summary += f"  - {feat}\n"

        return summary


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Convert user data features for HybridMultiModalModel')
    parser.add_argument('--metadata', required=True, help='Feature metadata JSON file')
    parser.add_argument('--input', required=True, help='User data file')
    parser.add_argument('--output', required=True, help='Output converted data file')
    parser.add_argument('--data-type', default='auto', help='Data type (auto, expression, rppa, etc.)')
    parser.add_argument('--cancer-type', help='Cancer type for Cox coefficient lookup')

    args = parser.parse_args()

    # Load user data
    user_data = pd.read_csv(args.input, index_col=0)

    # Convert
    converter = FeatureConverter(args.metadata)
    converted_data, missing_mask, report = converter.convert_user_data(
        user_data, args.data_type, args.cancer_type
    )

    # Save results
    converted_data.to_csv(args.output)
    missing_mask.to_csv(args.output.replace('.csv', '_missing_mask.csv'))

    # Print summary
    print(converter.get_conversion_summary(report))


if __name__ == "__main__":
    main()
