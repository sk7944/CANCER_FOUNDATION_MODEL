#!/usr/bin/env python3
"""
Data Format Guide and Validation for User Input
===============================================

This module provides comprehensive guidance on data formats expected by the model
and validation utilities to ensure user data is properly formatted.

Key Requirements:
- Expression, CNV, RPPA: log2(value + 1) transformation
- Mutations: Binary/categorical values (0-2 impact scores)  
- Methylation: Beta values (0-1 range)
- Clinical: Categorical/numerical as appropriate

Usage:
    from src.utils.data_format_guide import DataFormatGuide
    
    guide = DataFormatGuide()
    guide.validate_user_data('expression', user_df)
    guide.show_format_requirements('expression')
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

class DataFormatGuide:
    """Comprehensive data format guide and validation"""
    
    def __init__(self, logger=None):
        """Initialize data format guide"""
        self.logger = logger or self._setup_logger()
        
        # Define expected data formats
        self.format_specs = self._define_format_specifications()
        
        self.logger.info("✅ DataFormatGuide initialized")
    
    def _setup_logger(self):
        """Setup default logger"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def _define_format_specifications(self) -> Dict:
        """Define comprehensive format specifications for each data type"""
        
        specs = {
            'expression': {
                'name': 'Gene Expression Data',
                'transformation': 'log2(raw_value + 1)',
                'expected_range': (0, 25),
                'typical_range': (0, 20),
                'expected_format': 'Samples × Genes',
                'feature_names': 'Gene symbols (e.g., TP53, BRCA1) or SYMBOL|ENTREZ_ID',
                'data_type': 'Continuous (float)',
                'missing_values': 'Allowed, will be handled as special tokens',
                'example_values': [0.0, 2.5, 8.3, 15.2, 0.1],
                'description': """
Gene expression data should be log2 transformed RNA-seq or microarray data.
Raw counts or FPKM values should be transformed as log2(value + 1).
                """,
                'validation_rules': {
                    'min_value': -1,  # Allow some negative values due to normalization
                    'max_value': 30,  # Very high expression should be rare
                    'typical_median': (2, 10),
                    'zero_ratio_max': 0.3  # Max 30% zeros acceptable
                }
            },
            
            'cnv': {
                'name': 'Copy Number Variation Data',
                'transformation': 'log2(raw_value + 1) or log2(raw_value - min_value + 1)',
                'expected_range': (-3, 3),
                'typical_range': (-2, 2),
                'expected_format': 'Samples × Genes',
                'feature_names': 'Gene symbols (e.g., TP53, BRCA1)',
                'data_type': 'Continuous (float)',
                'missing_values': 'Allowed',
                'example_values': [-1.2, -0.5, 0.0, 0.8, 1.5],
                'description': """
Copy number variation data representing gene-level copy number alterations.
Values typically range from -2 (deletion) to +2 (amplification).
Should be log2 transformed relative to diploid state.
                """,
                'validation_rules': {
                    'min_value': -5,
                    'max_value': 5,
                    'typical_median': (-0.5, 0.5),
                    'zero_ratio_max': 0.4
                }
            },
            
            'microrna': {
                'name': 'MicroRNA Expression Data',
                'transformation': 'log2(raw_value + 1)',
                'expected_range': (0, 20),
                'typical_range': (0, 15),
                'expected_format': 'Samples × microRNAs',
                'feature_names': 'miRNA names (e.g., hsa-mir-21, hsa-let-7a)',
                'data_type': 'Continuous (float)',
                'missing_values': 'Allowed',
                'example_values': [0.0, 1.2, 4.5, 9.8, 0.3],
                'description': """
MicroRNA expression data from RNA-seq or microarray platforms.
Should be log2 transformed: log2(normalized_reads + 1).
                """,
                'validation_rules': {
                    'min_value': 0,
                    'max_value': 25,
                    'typical_median': (1, 8),
                    'zero_ratio_max': 0.5  # miRNAs can have many zeros
                }
            },
            
            'rppa': {
                'name': 'Reverse Phase Protein Array Data',
                'transformation': 'log2(raw_value + 1) or log2(raw_value - min_value + 1)',
                'expected_range': (-2, 3),
                'typical_range': (-1, 2),
                'expected_format': 'Samples × Proteins',
                'feature_names': 'Protein names (e.g., p53, EGFR, p-AKT-S473)',
                'data_type': 'Continuous (float)',
                'missing_values': 'Allowed',
                'example_values': [-0.8, -0.2, 0.0, 0.5, 1.2],
                'description': """
RPPA protein abundance data, typically normalized and centered.
May include phosphorylated proteins (e.g., p-AKT-S473).
Should be log2 transformed for model compatibility.
                """,
                'validation_rules': {
                    'min_value': -5,
                    'max_value': 5,
                    'typical_median': (-1, 1),
                    'zero_ratio_max': 0.2
                }
            },
            
            'mutations': {
                'name': 'Mutation Data',
                'transformation': 'None (categorical impact scores)',
                'expected_range': (0, 2),
                'typical_range': (0, 2),
                'expected_format': 'Samples × Genes',
                'feature_names': 'Gene symbols (e.g., TP53, BRCA1)',
                'data_type': 'Integer (0, 1, 2)',
                'missing_values': 'Use 0 for no mutation',
                'example_values': [0, 0, 1, 0, 2],
                'description': """
Mutation impact scores per gene:
- 0: No mutation or silent mutation
- 1: Missense mutation or low impact
- 2: Nonsense, frameshift, or high impact mutation
                """,
                'validation_rules': {
                    'min_value': 0,
                    'max_value': 2,
                    'allowed_values': [0, 1, 2],
                    'zero_ratio_min': 0.7  # Most genes should have no mutations
                }
            },
            
            'methylation': {
                'name': 'DNA Methylation Data',
                'transformation': 'None (beta values)',
                'expected_range': (0, 1),
                'typical_range': (0, 1),
                'expected_format': 'Samples × CpG Sites',
                'feature_names': 'CpG probe IDs (e.g., cg00000292, cg00002426)',
                'data_type': 'Continuous (float, 0-1)',
                'missing_values': 'Allowed but minimize',
                'example_values': [0.05, 0.23, 0.67, 0.89, 0.42],
                'description': """
DNA methylation beta values from Illumina 450K or EPIC arrays.
Beta values represent methylation percentage (0 = unmethylated, 1 = fully methylated).
No transformation needed - use raw beta values.
                """,
                'validation_rules': {
                    'min_value': 0,
                    'max_value': 1,
                    'typical_median': (0.2, 0.8),
                    'zero_ratio_max': 0.1
                }
            },
            
            'clinical': {
                'name': 'Clinical Data',
                'transformation': 'Appropriate encoding for each feature',
                'expected_format': 'Samples × Clinical Features',
                'feature_names': 'age, gender, race, cancer_type, etc.',
                'data_type': 'Mixed (numerical and categorical)',
                'missing_values': 'Allowed, handle appropriately',
                'description': """
Clinical and demographic data including:
- Age: Numerical (years)
- Gender/Sex: Categorical (Male/Female or M/F)
- Race/Ethnicity: Categorical
- Cancer Type: Categorical (if known)
- Stage, Grade: Categorical or ordinal
                """,
                'validation_rules': {
                    'age_range': (0, 120),
                    'categorical_encoding': 'String labels preferred'
                }
            }
        }
        
        return specs
    
    def show_format_requirements(self, data_type: str = 'all') -> None:
        """Display comprehensive format requirements"""
        
        if data_type == 'all':
            print("=" * 80)
            print("📋 COMPREHENSIVE DATA FORMAT REQUIREMENTS")
            print("=" * 80)
            print()
            
            for dtype in self.format_specs.keys():
                self._print_single_format(dtype)
                print("\n" + "-" * 60 + "\n")
        else:
            if data_type in self.format_specs:
                print("=" * 80)
                print(f"📋 {self.format_specs[data_type]['name'].upper()} FORMAT REQUIREMENTS")
                print("=" * 80)
                self._print_single_format(data_type)
            else:
                print(f"❌ Unknown data type: {data_type}")
                print(f"Available types: {list(self.format_specs.keys())}")
    
    def _print_single_format(self, data_type: str) -> None:
        """Print format requirements for single data type"""
        spec = self.format_specs[data_type]
        
        print(f"🔬 {spec['name']}")
        print(f"📊 Expected Format: {spec['expected_format']}")
        print(f"🏷️  Feature Names: {spec['feature_names']}")
        print(f"📈 Data Type: {spec['data_type']}")
        
        if 'transformation' in spec:
            print(f"🔄 Transformation: {spec['transformation']}")
        
        if 'expected_range' in spec:
            print(f"📏 Expected Range: {spec['expected_range']}")
        
        print(f"❓ Missing Values: {spec['missing_values']}")
        
        if 'example_values' in spec:
            print(f"💡 Example Values: {spec['example_values']}")
        
        print(f"📝 Description: {spec['description'].strip()}")
    
    def validate_user_data(self, data_type: str, data: pd.DataFrame, 
                          show_plots: bool = False) -> Dict[str, any]:
        """
        Comprehensive validation of user data
        
        Args:
            data_type: Type of data ('expression', 'cnv', etc.)
            data: User data DataFrame
            show_plots: Whether to display validation plots
            
        Returns:
            validation_result: Comprehensive validation report
        """
        
        if data_type not in self.format_specs:
            return {'error': f"Unknown data type: {data_type}"}
        
        spec = self.format_specs[data_type]
        result = {
            'data_type': data_type,
            'data_shape': data.shape,
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': [],
            'statistics': {}
        }
        
        self.logger.info(f"🔍 Validating {spec['name']} data...")
        self.logger.info(f"📊 Data shape: {data.shape}")
        
        # Basic shape validation
        if data.empty:
            result['errors'].append("Data is empty")
            result['is_valid'] = False
            return result
        
        # Data type validation
        numeric_data = data.select_dtypes(include=[np.number])
        if data_type != 'clinical' and len(numeric_data.columns) == 0:
            result['errors'].append(f"No numeric columns found for {data_type} data")
            result['is_valid'] = False
        
        # Specific validations by data type
        if data_type == 'expression':
            result.update(self._validate_expression_data(data, spec))
        elif data_type == 'cnv':
            result.update(self._validate_cnv_data(data, spec))
        elif data_type == 'microrna':
            result.update(self._validate_microrna_data(data, spec))
        elif data_type == 'rppa':
            result.update(self._validate_rppa_data(data, spec))
        elif data_type == 'mutations':
            result.update(self._validate_mutations_data(data, spec))
        elif data_type == 'methylation':
            result.update(self._validate_methylation_data(data, spec))
        elif data_type == 'clinical':
            result.update(self._validate_clinical_data(data, spec))
        
        # Generate plots if requested
        if show_plots and data_type != 'clinical':
            self._create_validation_plots(data, data_type, spec)
        
        # Summary
        if result['is_valid']:
            self.logger.info("✅ Data validation passed!")
        else:
            self.logger.warning("⚠️  Data validation failed!")
            
        if result['warnings']:
            self.logger.warning(f"⚠️  {len(result['warnings'])} warnings found")
            
        return result
    
    def _validate_expression_data(self, data: pd.DataFrame, spec: Dict) -> Dict:
        """Validate gene expression data"""
        result = {'warnings': [], 'errors': [], 'recommendations': [], 'statistics': {}}
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Value range check
        min_val = numeric_data.min().min()
        max_val = numeric_data.max().max()
        
        result['statistics']['value_range'] = (float(min_val), float(max_val))
        result['statistics']['median_value'] = float(numeric_data.median().median())
        
        if min_val < spec['validation_rules']['min_value']:
            result['warnings'].append(f"Minimum value ({min_val:.2f}) is unusually low for log2-transformed expression data")
        
        if max_val > spec['validation_rules']['max_value']:
            result['warnings'].append(f"Maximum value ({max_val:.2f}) is unusually high for expression data")
        
        # Check if data appears to be log-transformed
        if min_val >= 0 and max_val > 50:
            result['recommendations'].append("Data appears to be raw counts/FPKM. Consider log2(x + 1) transformation")
        
        # Zero ratio check
        zero_ratio = (numeric_data == 0).sum().sum() / (numeric_data.shape[0] * numeric_data.shape[1])
        result['statistics']['zero_ratio'] = float(zero_ratio)
        
        if zero_ratio > spec['validation_rules']['zero_ratio_max']:
            result['warnings'].append(f"High proportion of zeros ({zero_ratio:.1%}). Check if transformation is correct")
        
        return result
    
    def _validate_cnv_data(self, data: pd.DataFrame, spec: Dict) -> Dict:
        """Validate copy number variation data"""
        result = {'warnings': [], 'errors': [], 'recommendations': [], 'statistics': {}}
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        min_val = numeric_data.min().min()
        max_val = numeric_data.max().max()
        median_val = numeric_data.median().median()
        
        result['statistics']['value_range'] = (float(min_val), float(max_val))
        result['statistics']['median_value'] = float(median_val)
        
        # CNV should be centered around 0 (diploid state)
        if abs(median_val) > 0.5:
            result['recommendations'].append("CNV data should be centered around 0 (diploid state)")
        
        # Extreme values check
        if min_val < -5 or max_val > 5:
            result['warnings'].append("Extreme CNV values detected. Check for outliers")
        
        return result
    
    def _validate_microrna_data(self, data: pd.DataFrame, spec: Dict) -> Dict:
        """Validate microRNA expression data"""
        result = {'warnings': [], 'errors': [], 'recommendations': [], 'statistics': {}}
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        min_val = numeric_data.min().min()
        max_val = numeric_data.max().max()
        
        result['statistics']['value_range'] = (float(min_val), float(max_val))
        
        # Check for negative values (shouldn't exist in log2+1 transformed data)
        if min_val < 0:
            result['warnings'].append("Negative values found in microRNA data. Check transformation")
        
        # High zero ratio is normal for miRNA
        zero_ratio = (numeric_data == 0).sum().sum() / (numeric_data.shape[0] * numeric_data.shape[1])
        result['statistics']['zero_ratio'] = float(zero_ratio)
        
        if zero_ratio > 0.8:
            result['warnings'].append(f"Very high zero ratio ({zero_ratio:.1%}) - this is high even for miRNA data")
        
        return result
    
    def _validate_rppa_data(self, data: pd.DataFrame, spec: Dict) -> Dict:
        """Validate RPPA protein data"""
        result = {'warnings': [], 'errors': [], 'recommendations': [], 'statistics': {}}
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        min_val = numeric_data.min().min()
        max_val = numeric_data.max().max()
        median_val = numeric_data.median().median()
        
        result['statistics']['value_range'] = (float(min_val), float(max_val))
        result['statistics']['median_value'] = float(median_val)
        
        # RPPA data is typically normalized and centered
        if abs(median_val) > 1:
            result['recommendations'].append("RPPA data should typically be centered around 0")
        
        # Check for reasonable range
        if max_val - min_val > 10:
            result['warnings'].append("Very large value range for RPPA data. Check normalization")
        
        return result
    
    def _validate_mutations_data(self, data: pd.DataFrame, spec: Dict) -> Dict:
        """Validate mutation data"""
        result = {'warnings': [], 'errors': [], 'recommendations': [], 'statistics': {}}
        
        # Check if values are in expected range (0, 1, 2)
        allowed_values = set(spec['validation_rules']['allowed_values'])
        actual_values = set()
        
        for col in data.columns:
            unique_vals = set(data[col].dropna().unique())
            actual_values.update(unique_vals)
        
        result['statistics']['unique_values'] = sorted(list(actual_values))
        
        invalid_values = actual_values - allowed_values
        if invalid_values:
            result['errors'].append(f"Invalid mutation values found: {invalid_values}. Expected: {allowed_values}")
        
        # Check zero ratio (most genes should have no mutations)
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            zero_ratio = (numeric_data == 0).sum().sum() / (numeric_data.shape[0] * numeric_data.shape[1])
            result['statistics']['zero_ratio'] = float(zero_ratio)
            
            if zero_ratio < spec['validation_rules']['zero_ratio_min']:
                result['warnings'].append(f"Low zero ratio ({zero_ratio:.1%}). Most genes should have no mutations")
        
        return result
    
    def _validate_methylation_data(self, data: pd.DataFrame, spec: Dict) -> Dict:
        """Validate methylation data"""
        result = {'warnings': [], 'errors': [], 'recommendations': [], 'statistics': {}}
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        min_val = numeric_data.min().min()
        max_val = numeric_data.max().max()
        
        result['statistics']['value_range'] = (float(min_val), float(max_val))
        
        # Beta values should be between 0 and 1
        if min_val < 0 or max_val > 1:
            result['errors'].append(f"Methylation values outside [0,1] range: [{min_val:.3f}, {max_val:.3f}]")
        
        # Check for reasonable distribution
        median_val = numeric_data.median().median()
        result['statistics']['median_value'] = float(median_val)
        
        if median_val < 0.1 or median_val > 0.9:
            result['warnings'].append("Unusual methylation distribution. Check data quality")
        
        return result
    
    def _validate_clinical_data(self, data: pd.DataFrame, spec: Dict) -> Dict:
        """Validate clinical data"""
        result = {'warnings': [], 'errors': [], 'recommendations': [], 'statistics': {}}
        
        # Check for common clinical features
        common_features = ['age', 'gender', 'sex', 'race', 'ethnicity', 'cancer_type', 'type']
        found_features = [col for col in data.columns if any(feat in col.lower() for feat in common_features)]
        
        result['statistics']['clinical_features_found'] = found_features
        
        if not found_features:
            result['warnings'].append("No common clinical features detected")
        
        # Age validation if present
        age_cols = [col for col in data.columns if 'age' in col.lower()]
        for age_col in age_cols:
            if age_col in data.columns:
                age_data = pd.to_numeric(data[age_col], errors='coerce')
                if age_data.min() < 0 or age_data.max() > 120:
                    result['warnings'].append(f"Unusual age values in {age_col}")
        
        return result
    
    def _create_validation_plots(self, data: pd.DataFrame, data_type: str, spec: Dict) -> None:
        """Create validation plots"""
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'{spec["name"]} Data Validation', fontsize=14, fontweight='bold')
            
            # Distribution plot
            sample_data = numeric_data.iloc[:, 0] if not numeric_data.empty else []
            if len(sample_data) > 0:
                axes[0, 0].hist(sample_data.dropna(), bins=50, alpha=0.7)
                axes[0, 0].set_title('Value Distribution (First Feature)')
                axes[0, 0].set_xlabel('Value')
                axes[0, 0].set_ylabel('Frequency')
            
            # Box plot
            if numeric_data.shape[1] > 1:
                sample_features = numeric_data.iloc[:, :min(10, numeric_data.shape[1])]
                axes[0, 1].boxplot([sample_features[col].dropna() for col in sample_features.columns])
                axes[0, 1].set_title('Value Range (First 10 Features)')
                axes[0, 1].set_xlabel('Features')
                axes[0, 1].set_ylabel('Value')
            
            # Missing values heatmap
            missing_data = numeric_data.isnull()
            if missing_data.any().any():
                sns.heatmap(missing_data.iloc[:50, :20], ax=axes[1, 0], cbar=True)
                axes[1, 0].set_title('Missing Values Pattern')
            else:
                axes[1, 0].text(0.5, 0.5, 'No Missing Values', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Missing Values Pattern')
            
            # Summary statistics
            stats_text = f"""
Data Shape: {data.shape}
Numeric Features: {numeric_data.shape[1]}
Missing Values: {numeric_data.isnull().sum().sum()}
Min Value: {numeric_data.min().min():.3f}
Max Value: {numeric_data.max().max():.3f}
Median: {numeric_data.median().median():.3f}
            """
            axes[1, 1].text(0.1, 0.5, stats_text.strip(), 
                           transform=axes[1, 1].transAxes, fontfamily='monospace')
            axes[1, 1].set_title('Summary Statistics')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.warning(f"Could not create validation plots: {e}")
    
    def create_format_examples(self, data_type: str, output_dir: str = '.') -> None:
        """Create example files showing correct format"""
        
        if data_type not in self.format_specs:
            self.logger.error(f"Unknown data type: {data_type}")
            return
        
        spec = self.format_specs[data_type]
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create example data
        if data_type == 'expression':
            example_data = pd.DataFrame({
                'TP53': [8.2, 6.5, 9.1, 7.8, 5.2],
                'BRCA1': [5.1, 4.8, 6.2, 5.9, 4.1],
                'EGFR': [12.5, 10.2, 15.8, 11.9, 9.8],
                'MYC': [7.8, 8.9, 6.1, 9.2, 8.5]
            }, index=['Patient_1', 'Patient_2', 'Patient_3', 'Patient_4', 'Patient_5'])
            
        elif data_type == 'mutations':
            example_data = pd.DataFrame({
                'TP53': [2, 0, 1, 0, 2],
                'BRCA1': [0, 1, 0, 0, 0],
                'EGFR': [0, 0, 0, 1, 0],
                'KRAS': [1, 0, 0, 0, 1]
            }, index=['Patient_1', 'Patient_2', 'Patient_3', 'Patient_4', 'Patient_5'])
            
        elif data_type == 'methylation':
            example_data = pd.DataFrame({
                'cg00000292': [0.234, 0.567, 0.123, 0.789, 0.456],
                'cg00002426': [0.678, 0.234, 0.890, 0.345, 0.567],
                'cg00003994': [0.123, 0.789, 0.456, 0.234, 0.678]
            }, index=['Patient_1', 'Patient_2', 'Patient_3', 'Patient_4', 'Patient_5'])
        
        else:  # Default numeric example
            example_data = pd.DataFrame(
                np.random.normal(0, 1, (5, 4)),
                columns=[f'Feature_{i+1}' for i in range(4)],
                index=[f'Patient_{i+1}' for i in range(5)]
            )
        
        # Save example
        example_file = output_path / f'example_{data_type}_data.csv'
        example_data.to_csv(example_file)
        
        # Create README
        readme_content = f"""
# Example {spec['name']} Data Format

## File: {example_file.name}

{spec['description']}

## Format Requirements:
- **Transformation**: {spec.get('transformation', 'See specification')}
- **Expected Range**: {spec.get('expected_range', 'See specification')}
- **Data Type**: {spec['data_type']}
- **Feature Names**: {spec['feature_names']}
- **Missing Values**: {spec['missing_values']}

## Example Values:
{spec.get('example_values', 'See example file')}

## Usage:
```python
import pandas as pd
data = pd.read_csv('{example_file.name}', index_col=0)
```
        """
        
        readme_file = output_path / f'{data_type}_format_README.md'
        with open(readme_file, 'w') as f:
            f.write(readme_content.strip())
        
        self.logger.info(f"✅ Created example files:")
        self.logger.info(f"   📄 {example_file}")
        self.logger.info(f"   📋 {readme_file}")

def main():
    """Example usage and CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Format Guide and Validation')
    parser.add_argument('--show-requirements', choices=['all'] + list(DataFormatGuide()._define_format_specifications().keys()),
                       help='Show format requirements')
    parser.add_argument('--validate', help='Validate user data file')
    parser.add_argument('--data-type', help='Data type for validation')
    parser.add_argument('--create-examples', help='Create example files in specified directory')
    parser.add_argument('--example-type', choices=list(DataFormatGuide()._define_format_specifications().keys()),
                       help='Type of example to create')
    
    args = parser.parse_args()
    
    guide = DataFormatGuide()
    
    if args.show_requirements:
        guide.show_format_requirements(args.show_requirements)
    
    if args.validate and args.data_type:
        data = pd.read_csv(args.validate, index_col=0)
        result = guide.validate_user_data(args.data_type, data, show_plots=True)
        
        print("\n" + "="*60)
        print("🔍 VALIDATION RESULT")
        print("="*60)
        print(f"✅ Valid: {result['is_valid']}")
        if result['warnings']:
            print(f"⚠️  Warnings: {len(result['warnings'])}")
            for warning in result['warnings']:
                print(f"   • {warning}")
        if result['errors']:
            print(f"❌ Errors: {len(result['errors'])}")
            for error in result['errors']:
                print(f"   • {error}")
    
    if args.create_examples and args.example_type:
        guide.create_format_examples(args.example_type, args.create_examples)

if __name__ == "__main__":
    main()