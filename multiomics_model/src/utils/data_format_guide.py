#!/usr/bin/env python3
"""
Data Format Guide and Validation for HybridMultiModalModel
==========================================================

This module provides comprehensive guidance on data formats expected by the
HybridMultiModalModel and validation utilities to ensure user data is properly formatted.

Key Requirements:
- Multi-omics data: Converted to [value, cox] pairs
- Expression, CNV, microRNA, RPPA: log2 transformation
- Mutations: Impact scores (0-2)
- Methylation: Beta values (0-1)
- Clinical: Categorical encoding (age_group, sex, race, stage, grade)

Usage:
    from multiomics_model.src.utils.data_format_guide import DataFormatGuide

    guide = DataFormatGuide()
    guide.validate_user_data('expression', user_df)
    guide.show_format_requirements('expression')
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings


class DataFormatGuide:
    """
    Comprehensive data format guide and validation for HybridMultiModalModel

    Supports:
    - Format specifications for all data types
    - Data validation with detailed error/warning reporting
    - Example data generation
    - Clinical encoding guidelines
    """

    # Clinical categories for HybridMultiModalModel
    CLINICAL_CATEGORIES = {
        'age_group': {
            'n_categories': 10,
            'encoding': '0=<30, 1=30-39, 2=40-49, 3=50-54, 4=55-59, 5=60-64, 6=65-69, 7=70-74, 8=75-79, 9=80+'
        },
        'sex': {
            'n_categories': 2,
            'encoding': '0=MALE, 1=FEMALE'
        },
        'race': {
            'n_categories': 6,
            'encoding': '0=WHITE, 1=BLACK, 2=ASIAN, 3=NATIVE_AMERICAN, 4=PACIFIC_ISLANDER, 5=UNKNOWN'
        },
        'ajcc_pathologic_stage': {
            'n_categories': 5,
            'encoding': '0=Stage I, 1=Stage II, 2=Stage III, 3=Stage IV, 4=NA/Unknown'
        },
        'grade': {
            'n_categories': 4,
            'encoding': '0=G1, 1=G2, 2=G3, 3=G4'
        }
    }

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize data format guide for HybridMultiModalModel"""
        self.logger = logger or self._setup_logger()

        # Define format specifications
        self.format_specs = self._define_format_specifications()

        self.logger.info("DataFormatGuide initialized for HybridMultiModalModel")

    def _setup_logger(self) -> logging.Logger:
        """Setup default logger"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def _define_format_specifications(self) -> Dict:
        """Define comprehensive format specifications for HybridMultiModalModel"""

        specs = {
            'expression': {
                'name': 'Gene Expression Data',
                'model_format': '[val, cox] pairs',
                'transformation': 'log2(raw_value + 1)',
                'expected_range': (0, 25),
                'typical_range': (0, 20),
                'expected_format': 'Samples x Genes',
                'feature_names': 'Gene symbols (TP53, BRCA1) or SYMBOL|ENTREZ_ID',
                'data_type': 'Continuous (float)',
                'missing_values': 'Allowed - handled via missing mask',
                'example_values': [0.0, 2.5, 8.3, 15.2, 0.1],
                'description': """
Gene expression data should be RNA-seq (FPKM/TPM/counts).
Raw values will be log2-transformed: log2(x + 1).
Already log2-transformed data is also accepted.
Each gene will be paired with its cancer-type-specific Cox coefficient.
                """,
                'validation_rules': {
                    'min_value': -1,
                    'max_value': 30,
                    'typical_median': (2, 10),
                    'zero_ratio_max': 0.3
                }
            },

            'cnv': {
                'name': 'Copy Number Variation Data',
                'model_format': '[val, cox] pairs',
                'transformation': 'log2(raw_value - min + 1)',
                'expected_range': (-3, 3),
                'typical_range': (-2, 2),
                'expected_format': 'Samples x Genes',
                'feature_names': 'Gene symbols (TP53, BRCA1)',
                'data_type': 'Continuous (float)',
                'missing_values': 'Allowed',
                'example_values': [-1.2, -0.5, 0.0, 0.8, 1.5],
                'description': """
Copy number variation data representing gene-level CNV.
Values centered around 0 (diploid state).
Negative = deletion, Positive = amplification.
Each gene will be paired with its cancer-type-specific Cox coefficient.
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
                'model_format': '[val, cox] pairs',
                'transformation': 'log2(raw_value + 1)',
                'expected_range': (0, 20),
                'typical_range': (0, 15),
                'expected_format': 'Samples x microRNAs',
                'feature_names': 'miRNA names (hsa-mir-21, hsa-let-7a)',
                'data_type': 'Continuous (float)',
                'missing_values': 'Allowed',
                'example_values': [0.0, 1.2, 4.5, 9.8, 0.3],
                'description': """
MicroRNA expression from RNA-seq or microarray.
Raw values will be log2-transformed: log2(x + 1).
Each miRNA will be paired with its cancer-type-specific Cox coefficient.
                """,
                'validation_rules': {
                    'min_value': 0,
                    'max_value': 25,
                    'typical_median': (1, 8),
                    'zero_ratio_max': 0.5
                }
            },

            'rppa': {
                'name': 'Reverse Phase Protein Array Data',
                'model_format': '[val, cox] pairs',
                'transformation': 'log2(raw_value - min + 1)',
                'expected_range': (-2, 3),
                'typical_range': (-1, 2),
                'expected_format': 'Samples x Proteins',
                'feature_names': 'Protein names (p53, EGFR, p-AKT-S473)',
                'data_type': 'Continuous (float)',
                'missing_values': 'Allowed',
                'example_values': [-0.8, -0.2, 0.0, 0.5, 1.2],
                'description': """
RPPA protein abundance data, typically normalized.
May include phosphorylated proteins (e.g., p-AKT-S473).
Each protein will be paired with its cancer-type-specific Cox coefficient.
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
                'model_format': '[val, cox] pairs',
                'transformation': 'None (categorical impact scores)',
                'expected_range': (0, 2),
                'typical_range': (0, 2),
                'expected_format': 'Samples x Genes',
                'feature_names': 'Gene symbols (TP53, BRCA1, KRAS)',
                'data_type': 'Integer (0, 1, 2)',
                'missing_values': 'Use 0 for no mutation',
                'example_values': [0, 0, 1, 0, 2],
                'description': """
Mutation impact scores per gene:
- 0: No mutation or silent/low impact
- 1: Missense or moderate impact
- 2: Nonsense, frameshift, or high impact (LoF)

For multiple mutations per gene, use max(impact_score).
Each gene will be paired with its cancer-type-specific Cox coefficient.
                """,
                'validation_rules': {
                    'min_value': 0,
                    'max_value': 2,
                    'allowed_values': [0, 1, 2],
                    'zero_ratio_min': 0.7
                }
            },

            'methylation': {
                'name': 'DNA Methylation Data',
                'model_format': 'Direct values (no Cox pairing)',
                'transformation': 'None (beta values)',
                'expected_range': (0, 1),
                'typical_range': (0, 1),
                'expected_format': 'Samples x CpG Sites',
                'feature_names': 'CpG probe IDs (cg00000292, cg00002426)',
                'data_type': 'Continuous (float, 0-1)',
                'missing_values': 'Allowed - filled with 0.5 (neutral)',
                'example_values': [0.05, 0.23, 0.67, 0.89, 0.42],
                'description': """
DNA methylation beta values from Illumina 450K or EPIC arrays.
Beta = methylated / (methylated + unmethylated + 100)
Range: 0 (unmethylated) to 1 (fully methylated)
No transformation applied - use raw beta values.
Note: Methylation is processed separately, not paired with Cox coefficients.
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
                'model_format': '5 categorical variables',
                'transformation': 'Automatic encoding to indices',
                'expected_format': 'Samples x Clinical Features',
                'feature_names': 'age, sex/gender, race, stage, grade',
                'data_type': 'Mixed (numerical and categorical)',
                'missing_values': 'Allowed - filled with defaults',
                'description': """
Clinical and demographic data including:
- Age: Numerical (years) -> encoded to 10 age groups
- Sex/Gender: Categorical (MALE/FEMALE) -> 0/1
- Race: Categorical -> 0-5
- Stage: Categorical (I/II/III/IV/NA) -> 0-4
- Grade: Categorical (G1/G2/G3/G4) -> 0-3

See CLINICAL_CATEGORIES for detailed encoding.
                """,
                'validation_rules': {
                    'age_range': (0, 120),
                    'categorical_encoding': 'Automatic'
                }
            }
        }

        return specs

    def show_format_requirements(self, data_type: str = 'all') -> None:
        """Display comprehensive format requirements for HybridMultiModalModel"""

        if data_type == 'all':
            print("=" * 80)
            print("DATA FORMAT REQUIREMENTS FOR HybridMultiModalModel")
            print("=" * 80)
            print()
            print("KEY CONCEPT: [val, cox] pairs")
            print("-" * 40)
            print("For omics data (expression, CNV, miRNA, RPPA, mutations),")
            print("each feature is paired with its cancer-type-specific Cox coefficient.")
            print("Format: [gene1_val, gene1_cox, gene2_val, gene2_cox, ...]")
            print()

            for dtype in self.format_specs.keys():
                self._print_single_format(dtype)
                print("\n" + "-" * 60 + "\n")

            self._print_clinical_encoding()
        else:
            if data_type in self.format_specs:
                print("=" * 80)
                print(f"{self.format_specs[data_type]['name'].upper()} FORMAT REQUIREMENTS")
                print("=" * 80)
                self._print_single_format(data_type)

                if data_type == 'clinical':
                    print()
                    self._print_clinical_encoding()
            else:
                print(f"Unknown data type: {data_type}")
                print(f"Available types: {list(self.format_specs.keys())}")

    def _print_single_format(self, data_type: str) -> None:
        """Print format requirements for single data type"""
        spec = self.format_specs[data_type]

        print(f"  {spec['name']}")
        print(f"  Model Format: {spec['model_format']}")
        print(f"  Expected Format: {spec['expected_format']}")
        print(f"  Feature Names: {spec['feature_names']}")
        print(f"  Data Type: {spec['data_type']}")

        if 'transformation' in spec:
            print(f"  Transformation: {spec['transformation']}")

        if 'expected_range' in spec:
            print(f"  Expected Range: {spec['expected_range']}")

        print(f"  Missing Values: {spec['missing_values']}")

        if 'example_values' in spec:
            print(f"  Example Values: {spec['example_values']}")

        print(f"  Description: {spec['description'].strip()}")

    def _print_clinical_encoding(self) -> None:
        """Print clinical variable encoding details"""
        print("=" * 60)
        print("CLINICAL VARIABLE ENCODING")
        print("=" * 60)
        print("clinical_categories = (10, 2, 6, 5, 4)")
        print()
        for col_name, details in self.CLINICAL_CATEGORIES.items():
            print(f"  {col_name}:")
            print(f"    Categories: {details['n_categories']}")
            print(f"    Encoding: {details['encoding']}")
            print()

    def validate_user_data(self,
                          data_type: str,
                          data: pd.DataFrame,
                          show_plots: bool = False) -> Dict[str, Any]:
        """
        Comprehensive validation of user data for HybridMultiModalModel

        Args:
            data_type: Type of data ('expression', 'cnv', etc.)
            data: User data DataFrame
            show_plots: Whether to display validation plots

        Returns:
            validation_result: Comprehensive validation report
        """

        if data_type not in self.format_specs:
            return {
                'is_valid': False,
                'error': f"Unknown data type: {data_type}",
                'warnings': [],
                'errors': [f"Unknown data type: {data_type}"]
            }

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

        self.logger.info(f"Validating {spec['name']} for HybridMultiModalModel...")
        self.logger.info(f"  Data shape: {data.shape}")

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

        # Update is_valid based on errors
        if result['errors']:
            result['is_valid'] = False

        # Summary logging
        if result['is_valid']:
            self.logger.info("  Validation passed")
        else:
            self.logger.warning(f"  Validation failed: {len(result['errors'])} errors")

        if result['warnings']:
            self.logger.warning(f"  {len(result['warnings'])} warnings")

        return result

    def _validate_expression_data(self, data: pd.DataFrame, spec: Dict) -> Dict:
        """Validate gene expression data"""
        result = {'warnings': [], 'errors': [], 'recommendations': [], 'statistics': {}}

        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            result['errors'].append("No numeric columns found")
            return result

        min_val = numeric_data.min().min()
        max_val = numeric_data.max().max()
        median_val = numeric_data.median().median()

        result['statistics']['value_range'] = (float(min_val), float(max_val))
        result['statistics']['median_value'] = float(median_val)

        # Check if data appears to be log-transformed
        if min_val >= 0 and max_val > 50:
            result['recommendations'].append(
                "Data appears to be raw counts/FPKM. Pipeline will apply log2(x+1) transformation."
            )

        # Check for negative values
        if min_val < -1:
            result['warnings'].append(f"Unusual negative values found (min={min_val:.2f})")

        # Check for extreme values
        if max_val > spec['validation_rules']['max_value']:
            result['warnings'].append(f"Very high values found (max={max_val:.2f})")

        # Zero ratio
        zero_ratio = (numeric_data == 0).sum().sum() / (numeric_data.shape[0] * numeric_data.shape[1])
        result['statistics']['zero_ratio'] = float(zero_ratio)

        if zero_ratio > spec['validation_rules']['zero_ratio_max']:
            result['warnings'].append(f"High zero ratio ({zero_ratio:.1%})")

        return result

    def _validate_cnv_data(self, data: pd.DataFrame, spec: Dict) -> Dict:
        """Validate copy number variation data"""
        result = {'warnings': [], 'errors': [], 'recommendations': [], 'statistics': {}}

        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return result

        min_val = numeric_data.min().min()
        max_val = numeric_data.max().max()
        median_val = numeric_data.median().median()

        result['statistics']['value_range'] = (float(min_val), float(max_val))
        result['statistics']['median_value'] = float(median_val)

        # CNV should be centered around 0
        if abs(median_val) > 0.5:
            result['recommendations'].append(
                "CNV data should be centered around 0 (diploid state)"
            )

        # Extreme values
        if min_val < -5 or max_val > 5:
            result['warnings'].append("Extreme CNV values detected")

        return result

    def _validate_microrna_data(self, data: pd.DataFrame, spec: Dict) -> Dict:
        """Validate microRNA expression data"""
        result = {'warnings': [], 'errors': [], 'recommendations': [], 'statistics': {}}

        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return result

        min_val = numeric_data.min().min()
        max_val = numeric_data.max().max()

        result['statistics']['value_range'] = (float(min_val), float(max_val))

        # Check for negative values
        if min_val < 0:
            result['warnings'].append("Negative values found")

        # Zero ratio
        zero_ratio = (numeric_data == 0).sum().sum() / (numeric_data.shape[0] * numeric_data.shape[1])
        result['statistics']['zero_ratio'] = float(zero_ratio)

        if zero_ratio > 0.8:
            result['warnings'].append(f"Very high zero ratio ({zero_ratio:.1%})")

        return result

    def _validate_rppa_data(self, data: pd.DataFrame, spec: Dict) -> Dict:
        """Validate RPPA protein data"""
        result = {'warnings': [], 'errors': [], 'recommendations': [], 'statistics': {}}

        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return result

        min_val = numeric_data.min().min()
        max_val = numeric_data.max().max()
        median_val = numeric_data.median().median()

        result['statistics']['value_range'] = (float(min_val), float(max_val))
        result['statistics']['median_value'] = float(median_val)

        # RPPA should be centered around 0
        if abs(median_val) > 1:
            result['recommendations'].append("RPPA data should typically be centered around 0")

        # Large range
        if max_val - min_val > 10:
            result['warnings'].append("Very large value range")

        return result

    def _validate_mutations_data(self, data: pd.DataFrame, spec: Dict) -> Dict:
        """Validate mutation data"""
        result = {'warnings': [], 'errors': [], 'recommendations': [], 'statistics': {}}

        # Check if values are in expected range
        allowed_values = set(spec['validation_rules']['allowed_values'])
        actual_values = set()

        for col in data.columns:
            unique_vals = set(data[col].dropna().unique())
            actual_values.update(unique_vals)

        result['statistics']['unique_values'] = sorted([int(v) if pd.notna(v) else None for v in actual_values if pd.notna(v)])

        # Check for invalid values
        invalid_values = actual_values - allowed_values - {np.nan}
        invalid_values = {v for v in invalid_values if pd.notna(v)}

        if invalid_values:
            result['errors'].append(
                f"Invalid mutation values: {invalid_values}. Expected: {allowed_values}"
            )

        # Zero ratio (most genes should have no mutations)
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            zero_ratio = (numeric_data == 0).sum().sum() / (numeric_data.shape[0] * numeric_data.shape[1])
            result['statistics']['zero_ratio'] = float(zero_ratio)

            if zero_ratio < spec['validation_rules']['zero_ratio_min']:
                result['warnings'].append(
                    f"Low zero ratio ({zero_ratio:.1%}). Most genes should have no mutations."
                )

        return result

    def _validate_methylation_data(self, data: pd.DataFrame, spec: Dict) -> Dict:
        """Validate methylation data"""
        result = {'warnings': [], 'errors': [], 'recommendations': [], 'statistics': {}}

        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return result

        min_val = numeric_data.min().min()
        max_val = numeric_data.max().max()
        median_val = numeric_data.median().median()

        result['statistics']['value_range'] = (float(min_val), float(max_val))
        result['statistics']['median_value'] = float(median_val)

        # Beta values must be in [0, 1]
        if min_val < 0 or max_val > 1:
            result['errors'].append(
                f"Methylation values outside [0,1] range: [{min_val:.3f}, {max_val:.3f}]"
            )
            result['recommendations'].append(
                "Ensure data contains beta values, not M-values"
            )

        # Unusual distribution
        if median_val < 0.1 or median_val > 0.9:
            result['warnings'].append("Unusual methylation distribution")

        return result

    def _validate_clinical_data(self, data: pd.DataFrame, spec: Dict) -> Dict:
        """Validate clinical data"""
        result = {'warnings': [], 'errors': [], 'recommendations': [], 'statistics': {}}

        # Check for expected clinical features
        expected_features = ['age', 'sex', 'gender', 'race', 'stage', 'grade']
        found_features = [col for col in data.columns
                        if any(feat in col.lower() for feat in expected_features)]

        result['statistics']['clinical_features_found'] = found_features

        if not found_features:
            result['warnings'].append(
                "No standard clinical features detected. Expected: age, sex, race, stage, grade"
            )

        # Age validation
        age_cols = [col for col in data.columns if 'age' in col.lower()]
        for age_col in age_cols:
            age_data = pd.to_numeric(data[age_col], errors='coerce')
            if age_data.min() < 0 or age_data.max() > 120:
                result['warnings'].append(f"Unusual age values in {age_col}")

        return result

    def create_format_examples(self, data_type: str, output_dir: str = '.') -> None:
        """Create example files showing correct format for HybridMultiModalModel"""

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

        elif data_type == 'clinical':
            example_data = pd.DataFrame({
                'age': [55, 62, 48, 71, 59],
                'sex': ['FEMALE', 'MALE', 'FEMALE', 'MALE', 'FEMALE'],
                'race': ['WHITE', 'BLACK OR AFRICAN AMERICAN', 'WHITE', 'ASIAN', 'WHITE'],
                'stage': ['Stage II', 'Stage III', 'Stage I', 'Stage IV', 'Stage II'],
                'grade': ['G2', 'G3', 'G1', 'G3', 'G2']
            }, index=['Patient_1', 'Patient_2', 'Patient_3', 'Patient_4', 'Patient_5'])

        else:
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
# Example {spec['name']} Data Format for HybridMultiModalModel

## File: {example_file.name}

{spec['description']}

## Format Requirements:
- **Model Format**: {spec.get('model_format', 'See specification')}
- **Transformation**: {spec.get('transformation', 'See specification')}
- **Expected Range**: {spec.get('expected_range', 'See specification')}
- **Data Type**: {spec['data_type']}
- **Feature Names**: {spec['feature_names']}
- **Missing Values**: {spec['missing_values']}

## Usage with HybridMultiModalModel:
```python
from multiomics_model.src.utils.user_data_pipeline import UserDataPipeline

pipeline = UserDataPipeline(cox_coef_path='path/to/cox_coefficients.parquet')

result = pipeline.process_user_data(
    user_files={{'{data_type}': '{example_file.name}'}},
    cancer_type='BRCA'
)
```

## Note on [value, cox] pairs:
For omics data, the pipeline will automatically pair each feature with
its cancer-type-specific Cox regression coefficient. This is handled
internally - users just provide raw feature values.
        """

        readme_file = output_path / f'{data_type}_format_README.md'
        with open(readme_file, 'w') as f:
            f.write(readme_content.strip())

        self.logger.info(f"Created example files:")
        self.logger.info(f"  Data: {example_file}")
        self.logger.info(f"  README: {readme_file}")


def main():
    """Example usage and CLI interface"""
    import argparse

    parser = argparse.ArgumentParser(description='Data Format Guide for HybridMultiModalModel')
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
        result = guide.validate_user_data(args.data_type, data)

        print("\n" + "="*60)
        print("VALIDATION RESULT")
        print("="*60)
        print(f"Valid: {result['is_valid']}")
        if result['warnings']:
            print(f"\nWarnings ({len(result['warnings'])}):")
            for warning in result['warnings']:
                print(f"  - {warning}")
        if result['errors']:
            print(f"\nErrors ({len(result['errors'])}):")
            for error in result['errors']:
                print(f"  - {error}")

    if args.create_examples and args.example_type:
        guide.create_format_examples(args.example_type, args.create_examples)


if __name__ == "__main__":
    main()
