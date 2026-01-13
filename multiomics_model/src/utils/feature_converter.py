#!/usr/bin/env python3
"""
Feature Format Converter for User Data
=====================================

This module converts user input features to match the trained model format.
Handles gene symbols, Entrez IDs, protein names, and other feature mappings.

Usage:
    from src.utils.feature_converter import FeatureConverter
    
    converter = FeatureConverter('path/to/feature_metadata.json')
    converted_data = converter.convert_user_data(user_df)
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
    """Convert user features to trained model format"""
    
    def __init__(self, feature_metadata_path: str, logger=None):
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
        
        # Create mapping dictionaries
        self.gene_mappings = self._create_gene_mappings()
        self.protein_mappings = self._create_protein_mappings()
        self.clinical_mappings = self._create_clinical_mappings()
        
        self.logger.info("✅ FeatureConverter initialized successfully")
        self.logger.info(f"📊 Loaded {len(self.metadata.get('feature_columns', []))} trained features")
    
    def _setup_logger(self):
        """Setup default logger"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def _load_metadata(self) -> Dict:
        """Load feature metadata from training"""
        try:
            with open(self.feature_metadata_path, 'r') as f:
                metadata = json.load(f)
            self.logger.info(f"✅ Loaded feature metadata: {self.feature_metadata_path}")
            return metadata
        except Exception as e:
            self.logger.error(f"❌ Failed to load metadata: {e}")
            return {}
    
    def _create_gene_mappings(self) -> Dict[str, str]:
        """Create comprehensive gene symbol/ID mappings"""
        gene_mappings = {}
        
        # Extract from feature columns (SYMBOL|ENTREZ format)
        for feature in self.metadata.get('feature_columns', []):
            # Process different omics types
            if feature.startswith('Expression_') and (feature.endswith('_value') or feature.endswith('_cox')):
                # Remove Expression_ prefix and _value/_cox suffix
                gene_part = feature.replace('Expression_', '').replace('_value', '').replace('_cox', '')
                
                if '|' in gene_part:
                    symbol, entrez = gene_part.split('|', 1)
                    # Map both symbol and entrez to the original gene_part
                    gene_mappings[symbol.upper()] = gene_part
                    gene_mappings[entrez] = gene_part
                    # Also try without case sensitivity
                    gene_mappings[symbol.lower()] = gene_part
                    # Also map symbol without suffix (for user convenience)
                    gene_mappings[symbol] = gene_part
                else:
                    # Single identifier
                    gene_mappings[gene_part.upper()] = gene_part
                    gene_mappings[gene_part.lower()] = gene_part
                    gene_mappings[gene_part] = gene_part
            
            # Process CNV and Mutations (Symbol only, no Entrez)
            elif (feature.startswith('CNV_') or feature.startswith('Mutations_')) and (feature.endswith('_value') or feature.endswith('_cox')):
                # Extract the omics prefix
                prefix = feature.split('_')[0] + '_'
                gene_symbol = feature.replace(prefix, '').replace('_value', '').replace('_cox', '')
                
                # CNV and Mutations use Symbol only (no Entrez ID)
                gene_mappings[gene_symbol.upper()] = gene_symbol
                gene_mappings[gene_symbol.lower()] = gene_symbol
                gene_mappings[gene_symbol] = gene_symbol
            
            # Process microRNA (specific miRNA names)
            elif feature.startswith('microRNA_') and (feature.endswith('_value') or feature.endswith('_cox')):
                # Extract miRNA name
                mirna_name = feature.replace('microRNA_', '').replace('_value', '').replace('_cox', '')
                
                # microRNA names are typically like hsa-mir-21
                gene_mappings[mirna_name.upper()] = mirna_name
                gene_mappings[mirna_name.lower()] = mirna_name
                gene_mappings[mirna_name] = mirna_name
        
        self.logger.info(f"📊 Created {len(gene_mappings)} gene mappings")
        return gene_mappings
    
    def _create_protein_mappings(self) -> Dict[str, str]:
        """Create protein name mappings for RPPA data"""
        protein_mappings = {}
        
        # Common RPPA protein name variations
        protein_aliases = {
            # Antibody name variations commonly seen in RPPA data
            'p53': ['TP53', 'P53', 'p53', 'Tp53'],
            'EGFR': ['EGFR', 'EGF-R', 'ERBB1'],
            'HER2': ['ERBB2', 'HER2', 'Her2', 'NEU'],
            'AKT': ['AKT1', 'PKB', 'AKT', 'Akt'],
            'mTOR': ['MTOR', 'mTOR', 'FRAP1'],
            'p70S6K1': ['RPS6KB1', 'S6K1', 'p70S6K1'],
            'PTEN': ['PTEN', 'Pten'],
            'GSK3': ['GSK3B', 'GSK3', 'GSK3-beta'],
            'beta-Catenin': ['CTNNB1', 'CTNBB1', 'beta-Catenin', 'β-Catenin'],
            'E-Cadherin': ['CDH1', 'E-Cadherin', 'E_Cadherin'],
            'Vimentin': ['VIM', 'Vimentin'],
            'Snail': ['SNAI1', 'Snail'],
            'Slug': ['SNAI2', 'Slug'],
            'ZEB1': ['ZEB1', 'Zeb1'],
            'Twist': ['TWIST1', 'Twist'],
            # Phosphorylation states
            'p-AKT': ['AKT_pS473', 'AKT_pT308', 'pAKT', 'p-Akt'],
            'p-mTOR': ['MTOR_pS2448', 'p-mTOR', 'pmTOR'],
            'p-p70S6K1': ['RPS6KB1_pT389', 'p-p70S6K1', 'pp70S6K1'],
            'p-S6': ['RPS6_pS235_S236', 'p-S6', 'pS6'],
        }
        
        # Create bidirectional mappings
        for canonical, aliases in protein_aliases.items():
            for alias in aliases:
                protein_mappings[alias.upper()] = canonical
                protein_mappings[alias.lower()] = canonical
                protein_mappings[alias] = canonical
        
        # Extract from trained features
        for feature in self.metadata.get('feature_columns', []):
            if feature.endswith('_value') or feature.endswith('_cox'):
                protein_part = feature.replace('_value', '').replace('_cox', '')
                protein_mappings[protein_part.upper()] = protein_part
                protein_mappings[protein_part.lower()] = protein_part
        
        self.logger.info(f"📊 Created {len(protein_mappings)} protein mappings")
        return protein_mappings
    
    def _create_clinical_mappings(self) -> Dict[str, str]:
        """Create clinical feature mappings"""
        clinical_mappings = {}
        
        # Common clinical feature name variations
        clinical_aliases = {
            'age_at_initial_pathologic_diagnosis': [
                'age', 'age_at_diagnosis', 'diagnosis_age', 'patient_age'
            ],
            'gender': [
                'sex', 'gender', 'patient_gender', 'patient_sex'
            ],
            'race': [
                'race', 'ethnicity', 'patient_race'
            ],
            'acronym': [
                'cancer_type', 'tumor_type', 'cancer', 'type', 'diagnosis'
            ],
            'vital_status': [
                'vital_status', 'status', 'survival_status'
            ]
        }
        
        # Create mappings
        for canonical, aliases in clinical_aliases.items():
            for alias in aliases:
                clinical_mappings[alias.upper()] = canonical
                clinical_mappings[alias.lower()] = canonical
                clinical_mappings[alias] = canonical
        
        self.logger.info(f"📊 Created {len(clinical_mappings)} clinical mappings")
        return clinical_mappings
    
    def convert_user_data(self, 
                         user_data: pd.DataFrame,
                         data_type: str = 'auto',
                         missing_strategy: str = 'special_token') -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Convert user data to trained model format
        
        Args:
            user_data: User's input data (samples × features)
            data_type: 'expression', 'cnv', 'microrna', 'rppa', 'mutations', 'methylation', 'clinical', 'auto'
            missing_strategy: 'special_token', 'zero', 'mean'
            
        Returns:
            converted_data: Data in trained model format
            missing_mask: Boolean mask of missing features
            conversion_report: Report of conversion process
        """
        self.logger.info("🔄 Converting user data to trained model format...")
        self.logger.info(f"📊 Input data shape: {user_data.shape}")
        self.logger.info(f"📊 Data type: {data_type}")
        
        # Auto-detect data type if needed
        if data_type == 'auto':
            data_type = self._detect_data_type(user_data)
            self.logger.info(f"🎯 Auto-detected data type: {data_type}")
        
        # Get trained features for this data type
        trained_features = self._get_trained_features_for_type(data_type)
        
        # Convert feature names
        converted_data, feature_mapping = self._convert_features(user_data, data_type, trained_features)
        
        # Create missing mask
        missing_mask = self._create_missing_mask(converted_data, trained_features)
        
        # Handle missing values
        converted_data = self._handle_missing_values(converted_data, missing_strategy)
        
        # Create conversion report
        conversion_report = {
            'input_shape': user_data.shape,
            'output_shape': converted_data.shape,
            'data_type': data_type,
            'mapped_features': len(feature_mapping),
            'missing_features': missing_mask.sum().sum(),
            'missing_ratio': missing_mask.sum().sum() / (missing_mask.shape[0] * missing_mask.shape[1]),
            'feature_mapping': feature_mapping,
            'missing_strategy': missing_strategy
        }
        
        self.logger.info("✅ User data conversion completed")
        self.logger.info(f"📊 Output shape: {converted_data.shape}")
        self.logger.info(f"📊 Missing ratio: {conversion_report['missing_ratio']:.1%}")
        
        return converted_data, missing_mask, conversion_report
    
    def _detect_data_type(self, user_data: pd.DataFrame) -> str:
        """Auto-detect the type of user data"""
        columns = [col.upper() for col in user_data.columns]
        
        # Check for clinical features
        clinical_indicators = ['AGE', 'GENDER', 'SEX', 'RACE', 'CANCER', 'TYPE', 'VITAL']
        if any(indicator in ' '.join(columns) for indicator in clinical_indicators):
            return 'clinical'
        
        # Check for methylation (CpG sites)
        if any(col.startswith('CG') for col in columns):
            return 'methylation'
        
        # Check for microRNA
        if any('MIR' in col or 'MIRNA' in col or col.startswith('HSA') for col in columns):
            return 'microrna'
        
        # Check for mutations (gene symbols with mutation info)
        if user_data.dtypes.apply(lambda x: x in ['int64', 'int32', 'bool']).any():
            return 'mutations'
        
        # Check for RPPA (protein names, phosphorylation)
        if any('-' in col or 'P-' in col or '_P' in col for col in columns):
            return 'rppa'
        
        # Default to expression for gene symbols
        return 'expression'
    
    def _get_trained_features_for_type(self, data_type: str) -> List[str]:
        """Get trained features for specific data type"""
        all_features = self.metadata.get('feature_columns', [])
        
        if data_type == 'clinical':
            clinical_features = self.metadata.get('feature_types', {}).get('clinical', [])
            return clinical_features
        else:
            # Get value and cox features for omics types
            pattern_map = {
                'expression': 'expression',
                'cnv': 'cnv', 
                'microrna': 'microrna',
                'rppa': 'rppa',
                'mutations': 'mutations',
                'methylation': 'methylation'
            }
            
            # This is simplified - in reality, you'd need to identify which features
            # belong to which omics type from the feature names
            return [f for f in all_features if f.endswith('_value') or f.endswith('_cox')]
    
    def _convert_features(self, user_data: pd.DataFrame, data_type: str, trained_features: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """Convert user feature names to trained format"""
        converted_data = pd.DataFrame(index=user_data.index)
        feature_mapping = {}
        
        if data_type == 'clinical':
            # Handle clinical features
            for user_col in user_data.columns:
                mapped_col = self.clinical_mappings.get(user_col.lower())
                if mapped_col and mapped_col in trained_features:
                    converted_data[mapped_col] = user_data[user_col]
                    feature_mapping[user_col] = mapped_col
        
        elif data_type == 'expression':
            # Expression uses SYMBOL|ENTREZ format
            for user_col in user_data.columns:
                mapped = False
                
                # Try to find matching gene
                # User might provide: TP53, 7157, or TP53|7157
                gene_part = None
                
                # Check if user provided symbol
                gene_part = self.gene_mappings.get(user_col.upper())
                if not gene_part:
                    gene_part = self.gene_mappings.get(user_col)
                if not gene_part and user_col.isdigit():
                    # User provided Entrez ID
                    gene_part = self.gene_mappings.get(user_col)
                
                if gene_part:
                    # For Expression, gene_part should be SYMBOL|ENTREZ
                    value_col = f"Expression_{gene_part}_value"
                    cox_col = f"Expression_{gene_part}_cox"
                    
                    if value_col in trained_features:
                        converted_data[value_col] = user_data[user_col]
                        feature_mapping[user_col] = value_col
                        mapped = True
                    
                    if cox_col in trained_features:
                        converted_data[cox_col] = 0  # Placeholder
        
        elif data_type in ['cnv', 'mutations']:
            # CNV and Mutations use Symbol only
            prefix = 'CNV_' if data_type == 'cnv' else 'Mutations_'
            
            for user_col in user_data.columns:
                # User provides gene symbol
                gene_symbol = None
                
                # Try different cases
                gene_symbol = self.gene_mappings.get(user_col.upper())
                if not gene_symbol:
                    gene_symbol = self.gene_mappings.get(user_col)
                if not gene_symbol:
                    gene_symbol = self.gene_mappings.get(user_col.lower())
                
                # For CNV/Mutations, also try direct match
                if not gene_symbol:
                    # Maybe user already provided exact format
                    gene_symbol = user_col
                
                if gene_symbol:
                    value_col = f"{prefix}{gene_symbol}_value"
                    cox_col = f"{prefix}{gene_symbol}_cox"
                    
                    if value_col in trained_features:
                        converted_data[value_col] = user_data[user_col]
                        feature_mapping[user_col] = value_col
                    
                    if cox_col in trained_features:
                        converted_data[cox_col] = 0  # Placeholder
        
        elif data_type == 'microrna':
            # microRNA uses specific miRNA names like hsa-mir-21
            for user_col in user_data.columns:
                mirna_name = None
                
                # Try to find matching miRNA
                mirna_name = self.gene_mappings.get(user_col.lower())  # miRNAs are usually lowercase
                if not mirna_name:
                    mirna_name = self.gene_mappings.get(user_col)
                if not mirna_name:
                    mirna_name = self.gene_mappings.get(user_col.upper())
                
                # Also try direct match
                if not mirna_name:
                    mirna_name = user_col
                
                if mirna_name:
                    value_col = f"microRNA_{mirna_name}_value"
                    cox_col = f"microRNA_{mirna_name}_cox"
                    
                    if value_col in trained_features:
                        converted_data[value_col] = user_data[user_col]
                        feature_mapping[user_col] = value_col
                    
                    if cox_col in trained_features:
                        converted_data[cox_col] = 0  # Placeholder
        
        elif data_type == 'rppa':
            # Handle protein features with TCGA RPPA format
            # TCGA format: RPPA_ProteinName[-phosphoSite]-Vendor-Validation_value/cox
            
            for user_col in user_data.columns:
                mapped = False
                
                # First try direct TCGA format mapping
                # Check if user already provided TCGA format
                for trained_feature in trained_features:
                    if trained_feature.startswith('RPPA_') and trained_feature.endswith('_value'):
                        # Extract protein part from RPPA_XXX_value format
                        protein_part = trained_feature[5:-6]  # Remove RPPA_ prefix and _value suffix
                        
                        # Check if user column matches (ignoring vendor codes)
                        user_clean = self._clean_protein_name(user_col)
                        tcga_base = protein_part.split('-')[0] if '-' in protein_part else protein_part
                        
                        if user_clean.upper() == tcga_base.upper() or user_col.upper() == tcga_base.upper():
                            # Found a match - add the value
                            value_col = trained_feature
                            cox_col = trained_feature.replace('_value', '_cox')
                            
                            if value_col in trained_features:
                                # If multiple matches exist (different vendor codes), use the first or aggregate
                                if value_col not in converted_data:
                                    converted_data[value_col] = user_data[user_col]
                                    feature_mapping[user_col] = value_col
                                    
                                    if cox_col in trained_features:
                                        converted_data[cox_col] = 0  # Placeholder
                                    
                                    mapped = True
                                    break
                
                # If not mapped, try protein mapper
                if not mapped:
                    protein_part = self.protein_mappings.get(user_col.upper())
                    if protein_part:
                        # Look for any RPPA feature with this protein base
                        for trained_feature in trained_features:
                            if trained_feature.startswith(f'RPPA_{protein_part}') and trained_feature.endswith('_value'):
                                value_col = trained_feature
                                cox_col = trained_feature.replace('_value', '_cox')
                                
                                converted_data[value_col] = user_data[user_col]
                                feature_mapping[user_col] = value_col
                                
                                if cox_col in trained_features:
                                    converted_data[cox_col] = 0  # Placeholder
                                
                                mapped = True
                                break
        
        elif data_type == 'methylation':
            # Handle methylation probes directly
            for user_col in user_data.columns:
                if user_col in trained_features:
                    converted_data[user_col] = user_data[user_col]
                    feature_mapping[user_col] = user_col
        
        return converted_data, feature_mapping
    
    def _create_missing_mask(self, converted_data: pd.DataFrame, trained_features: List[str]) -> pd.DataFrame:
        """Create missing feature mask"""
        missing_mask = pd.DataFrame(
            index=converted_data.index,
            columns=trained_features,
            dtype=bool
        )
        
        # Mark existing features as not missing
        for col in converted_data.columns:
            if col in missing_mask.columns:
                missing_mask[col] = converted_data[col].isna()
        
        # Mark completely missing features as missing
        for col in trained_features:
            if col not in converted_data.columns:
                missing_mask[col] = True
        
        return missing_mask
    
    def _handle_missing_values(self, data: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Handle missing values according to strategy"""
        if strategy == 'special_token':
            return data.fillna(-999)  # Special missing token
        elif strategy == 'zero':
            return data.fillna(0)
        elif strategy == 'mean':
            # Use training means if available
            means = self.metadata.get('feature_stats', {}).get('means', {})
            for col in data.columns:
                if col in means:
                    data[col] = data[col].fillna(means[col])
                else:
                    data[col] = data[col].fillna(0)
            return data
        else:
            return data
    
    def get_conversion_summary(self, conversion_report: Dict) -> str:
        """Generate human-readable conversion summary"""
        summary = f"""
🔄 Feature Conversion Summary
============================
📊 Input Data: {conversion_report['input_shape'][0]} samples × {conversion_report['input_shape'][1]} features
📊 Output Data: {conversion_report['output_shape'][0]} samples × {conversion_report['output_shape'][1]} features
📊 Data Type: {conversion_report['data_type']}
📊 Mapped Features: {conversion_report['mapped_features']}
📊 Missing Features: {conversion_report['missing_features']} ({conversion_report['missing_ratio']:.1%})
📊 Missing Strategy: {conversion_report['missing_strategy']}

🎯 Feature Mapping Examples:
"""
        
        # Show first few mappings
        mappings = conversion_report['feature_mapping']
        for i, (user_feat, trained_feat) in enumerate(list(mappings.items())[:5]):
            summary += f"   {user_feat} → {trained_feat}\n"
        
        if len(mappings) > 5:
            summary += f"   ... and {len(mappings) - 5} more\n"
        
        return summary

def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert user data features')
    parser.add_argument('--metadata', required=True, help='Feature metadata JSON file')
    parser.add_argument('--input', required=True, help='User data file')
    parser.add_argument('--output', required=True, help='Output converted data file')
    parser.add_argument('--data-type', default='auto', help='Data type (auto, expression, rppa, etc.)')
    
    args = parser.parse_args()
    
    # Load user data
    user_data = pd.read_csv(args.input, index_col=0)
    
    # Convert
    converter = FeatureConverter(args.metadata)
    converted_data, missing_mask, report = converter.convert_user_data(user_data, args.data_type)
    
    # Save results
    converted_data.to_csv(args.output)
    missing_mask.to_csv(args.output.replace('.csv', '_missing_mask.csv'))
    
    # Print summary
    print(converter.get_conversion_summary(report))

if __name__ == "__main__":
    main()