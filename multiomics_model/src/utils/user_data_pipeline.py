#!/usr/bin/env python3
"""
Comprehensive User Data Preprocessing Pipeline
==============================================

This module provides a complete pipeline for preprocessing user data to match
the trained TabTransformer model format. Combines all utilities for seamless
user experience.

Key Features:
- Automatic data type detection
- Feature format conversion
- Protein name mapping
- Data validation and transformation
- Missing value handling
- Ready-to-use model input generation

Usage:
    from src.utils.user_data_pipeline import UserDataPipeline
    
    pipeline = UserDataPipeline('path/to/model_metadata')
    processed_data = pipeline.process_user_data(user_files_dict)
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
    """Complete pipeline for user data preprocessing"""
    
    def __init__(self, 
                 model_metadata_path: str,
                 custom_protein_mappings: Optional[str] = None,
                 logger=None):
        """
        Initialize comprehensive preprocessing pipeline
        
        Args:
            model_metadata_path: Path to trained model metadata directory or JSON file
            custom_protein_mappings: Optional custom protein mappings file
            logger: Optional logger instance
        """
        self.logger = logger or self._setup_logger()
        self.model_metadata_path = Path(model_metadata_path)
        
        # Initialize component utilities
        self._initialize_components(custom_protein_mappings)
        
        # Load model metadata
        self.model_metadata = self._load_model_metadata()
        
        # Expected user data format
        self.expected_formats = self._define_expected_user_formats()
        
        self.logger.info("✅ UserDataPipeline initialized successfully")
        self.logger.info(f"🎯 Model metadata loaded from: {self.model_metadata_path}")
    
    def _setup_logger(self):
        """Setup default logger"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _initialize_components(self, custom_protein_mappings):
        """Initialize component utilities"""
        try:
            # Initialize format guide
            self.format_guide = DataFormatGuide(self.logger)
            
            # Initialize protein mapper
            self.protein_mapper = ProteinMapper(custom_protein_mappings, self.logger)
            
            # Feature converter will be initialized after loading metadata
            self.feature_converter = None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize components: {e}")
            raise
    
    def _load_model_metadata(self) -> Dict:
        """Load model training metadata"""
        metadata = {}
        
        # If it's a directory, look for standard metadata files
        if self.model_metadata_path.is_dir():
            metadata_files = {
                'feature_metadata': 'feature_metadata.json',
                'dataset_info': 'integrated_dataset_summary.json',
                'validation': 'integrated_dataset_validation.json'
            }
            
            for key, filename in metadata_files.items():
                file_path = self.model_metadata_path / filename
                if file_path.exists():
                    try:
                        with open(file_path, 'r') as f:
                            metadata[key] = json.load(f)
                        self.logger.info(f"✅ Loaded {key}: {file_path}")
                    except Exception as e:
                        self.logger.warning(f"⚠️  Failed to load {key}: {e}")
                else:
                    self.logger.warning(f"⚠️  Metadata file not found: {file_path}")
        
        # If it's a single JSON file
        elif self.model_metadata_path.suffix == '.json':
            try:
                with open(self.model_metadata_path, 'r') as f:
                    metadata['feature_metadata'] = json.load(f)
                self.logger.info(f"✅ Loaded metadata from: {self.model_metadata_path}")
            except Exception as e:
                self.logger.error(f"❌ Failed to load metadata: {e}")
        
        if not metadata:
            self.logger.warning("⚠️  No model metadata loaded - some features may be limited")
        
        # Initialize feature converter with metadata
        if 'feature_metadata' in metadata:
            try:
                # Save feature metadata to temp file for FeatureConverter
                temp_metadata_file = Path("/tmp/feature_metadata.json")
                with open(temp_metadata_file, 'w') as f:
                    json.dump(metadata['feature_metadata'], f, indent=2)
                
                self.feature_converter = FeatureConverter(str(temp_metadata_file), self.logger)
                self.logger.info("✅ FeatureConverter initialized with model metadata")
            except Exception as e:
                self.logger.warning(f"⚠️  Could not initialize FeatureConverter: {e}")
        
        return metadata
    
    def _define_expected_user_formats(self) -> Dict:
        """Define expected user input formats for convenience"""
        return {
            'typical_user_format': {
                'orientation': 'samples_as_rows',  # Rows = patients, Columns = features
                'file_types': ['.csv', '.tsv', '.txt', '.xlsx'],
                'index_column': 'patient_id or sample_id',
                'feature_names': 'gene_symbols, protein_names, probe_ids, etc.'
            },
            'data_types': {
                'expression': {
                    'features': 'Gene symbols (TP53, BRCA1, etc.)',
                    'values': 'log2(FPKM/TPM/counts + 1)',
                    'file_example': 'expression_data.csv'
                },
                'cnv': {
                    'features': 'Gene symbols',
                    'values': 'log2 copy number ratios',
                    'file_example': 'cnv_data.csv'
                },
                'microrna': {
                    'features': 'miRNA names (hsa-mir-21, etc.)',
                    'values': 'log2(normalized_reads + 1)',
                    'file_example': 'mirna_data.csv'
                },
                'rppa': {
                    'features': 'Protein names (p53, EGFR, p-AKT-S473, etc.)',
                    'values': 'log2 normalized protein abundance',
                    'file_example': 'rppa_data.csv'
                },
                'mutations': {
                    'features': 'Gene symbols',
                    'values': 'Impact scores (0=none, 1=missense, 2=truncating)',
                    'file_example': 'mutations_data.csv'
                },
                'methylation': {
                    'features': 'CpG probe IDs (cg00000292, etc.)',
                    'values': 'Beta values (0-1)',
                    'file_example': 'methylation_data.csv'
                },
                'clinical': {
                    'features': 'age, gender, race, stage, etc.',
                    'values': 'Mixed numeric and categorical',
                    'file_example': 'clinical_data.csv'
                }
            }
        }
    
    def show_user_format_guide(self):
        """Display comprehensive format guide for users"""
        print("=" * 80)
        print("📋 USER DATA FORMAT GUIDE FOR TABTRANSFORMER MODEL")
        print("=" * 80)
        print()
        print("🎯 GENERAL REQUIREMENTS:")
        print("   • Data format: CSV/TSV files with samples as rows, features as columns")
        print("   • Index: Patient/Sample IDs")
        print("   • Missing values: Allowed (will be handled automatically)")
        print("   • File encoding: UTF-8 preferred")
        print()
        
        print("🔬 DATA TYPE SPECIFIC REQUIREMENTS:")
        print("-" * 50)
        
        for data_type, spec in self.expected_formats['data_types'].items():
            print(f"\n📊 {data_type.upper()}:")
            print(f"   Features: {spec['features']}")
            print(f"   Values: {spec['values']}")
            print(f"   Example: {spec['file_example']}")
        
        print("\n" + "=" * 80)
        print("💡 IMPORTANT TRANSFORMATIONS REQUIRED:")
        print("=" * 80)
        print("   🔄 Expression, CNV, microRNA, RPPA: log2(raw_value + 1)")
        print("   🔄 Mutations: Categorical impact scores (0, 1, 2)")
        print("   🔄 Methylation: Beta values (0-1, no transformation needed)")
        print("   🔄 Clinical: Appropriate encoding for each variable")
        print()
        print("📞 For detailed validation, use: pipeline.validate_user_files()")
    
    def process_user_data(self, 
                         user_files: Dict[str, str],
                         missing_strategy: str = 'special_token',
                         validate_first: bool = True,
                         save_intermediate: bool = False,
                         output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete user data processing pipeline
        
        Args:
            user_files: Dict mapping data_type -> file_path
                      e.g., {'expression': 'expr.csv', 'clinical': 'clin.csv'}
            missing_strategy: How to handle missing values ('special_token', 'zero', 'mean')
            validate_first: Whether to validate data before processing
            save_intermediate: Whether to save intermediate processing results
            output_dir: Directory to save intermediate results
            
        Returns:
            processing_result: Complete processing results and model-ready data
        """
        
        self.logger.info("🚀 Starting comprehensive user data processing...")
        self.logger.info(f"📊 Input files: {list(user_files.keys())}")
        
        start_time = datetime.now()
        
        # Initialize result structure
        result = {
            'success': False,
            'processing_time': None,
            'input_files': user_files.copy(),
            'data_validation': {},
            'feature_mapping': {},
            'processed_data': {},
            'missing_masks': {},
            'final_datasets': {},
            'statistics': {},
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        try:
            # Step 1: Load and validate user data
            self.logger.info("📂 Step 1: Loading and validating user data...")
            loaded_data = {}
            
            for data_type, file_path in user_files.items():
                try:
                    # Load data
                    data = self._load_user_file(file_path)
                    loaded_data[data_type] = data
                    self.logger.info(f"✅ Loaded {data_type}: {data.shape}")
                    
                    # Validate if requested
                    if validate_first:
                        validation_result = self.format_guide.validate_user_data(data_type, data)
                        result['data_validation'][data_type] = validation_result
                        
                        if not validation_result['is_valid']:
                            result['errors'].extend([f"{data_type}: {err}" for err in validation_result['errors']])
                        
                        if validation_result['warnings']:
                            result['warnings'].extend([f"{data_type}: {warn}" for warn in validation_result['warnings']])
                
                except Exception as e:
                    error_msg = f"Failed to load {data_type} from {file_path}: {e}"
                    result['errors'].append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
                    continue
            
            if not loaded_data:
                raise ValueError("No data files were successfully loaded")
            
            # Step 2: Feature conversion and mapping
            self.logger.info("🔄 Step 2: Converting features to model format...")
            
            if not self.feature_converter:
                self.logger.warning("⚠️  No FeatureConverter available - limited feature mapping")
            
            for data_type, data in loaded_data.items():
                try:
                    if self.feature_converter:
                        # Use sophisticated feature conversion
                        converted_data, missing_mask, conversion_report = self.feature_converter.convert_user_data(
                            data, data_type, missing_strategy
                        )
                        
                        result['processed_data'][data_type] = converted_data
                        result['missing_masks'][data_type] = missing_mask
                        result['feature_mapping'][data_type] = conversion_report
                        
                        self.logger.info(f"✅ Converted {data_type}: {conversion_report['mapped_features']} features mapped")
                    
                    else:
                        # Basic processing without feature conversion
                        result['processed_data'][data_type] = data
                        result['missing_masks'][data_type] = pd.DataFrame(
                            False, index=data.index, columns=data.columns
                        )
                        result['feature_mapping'][data_type] = {'mapped_features': len(data.columns)}
                        
                        self.logger.info(f"✅ Basic processing {data_type}: {data.shape[1]} features")
                
                except Exception as e:
                    error_msg = f"Feature conversion failed for {data_type}: {e}"
                    result['errors'].append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
            
            # Step 3: Create model-ready datasets
            self.logger.info("🎯 Step 3: Creating model-ready datasets...")
            
            # Separate Cox and Methylation data
            cox_data_types = ['expression', 'cnv', 'microrna', 'rppa', 'mutations', 'clinical']
            cox_data_parts = []
            methylation_data = None
            
            for data_type in cox_data_types:
                if data_type in result['processed_data']:
                    cox_data_parts.append(result['processed_data'][data_type])
            
            if 'methylation' in result['processed_data']:
                methylation_data = result['processed_data']['methylation']
            
            # Create integrated Cox dataset
            if cox_data_parts:
                try:
                    integrated_cox = pd.concat(cox_data_parts, axis=1)
                    
                    # Remove duplicated columns if any
                    integrated_cox = integrated_cox.loc[:, ~integrated_cox.columns.duplicated()]
                    
                    result['final_datasets']['integrated_cox'] = integrated_cox
                    self.logger.info(f"✅ Integrated Cox dataset: {integrated_cox.shape}")
                    
                except Exception as e:
                    error_msg = f"Failed to create integrated Cox dataset: {e}"
                    result['errors'].append(error_msg)
                    self.logger.error(f"❌ {error_msg}")
            
            # Add methylation dataset
            if methylation_data is not None:
                result['final_datasets']['methylation'] = methylation_data
                self.logger.info(f"✅ Methylation dataset: {methylation_data.shape}")
            
            # Step 4: Generate statistics and recommendations
            self.logger.info("📊 Step 4: Generating statistics and recommendations...")
            result['statistics'] = self._generate_processing_statistics(result)
            result['recommendations'] = self._generate_recommendations(result)
            
            # Step 5: Save intermediate results if requested
            if save_intermediate and output_dir:
                self._save_intermediate_results(result, output_dir)
            
            # Mark as successful
            result['success'] = True
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            self.logger.info("🎉 User data processing completed successfully!")
            self.logger.info(f"⏱️  Processing time: {result['processing_time']:.2f} seconds")
            
        except Exception as e:
            result['errors'].append(f"Pipeline error: {e}")
            result['success'] = False
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"❌ Pipeline failed: {e}")
        
        return result
    
    def _load_user_file(self, file_path: str) -> pd.DataFrame:
        """Load user data file with automatic format detection"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Try different loading methods based on extension
        try:
            if file_path.suffix.lower() == '.csv':
                return pd.read_csv(file_path, index_col=0)
            elif file_path.suffix.lower() in ['.tsv', '.txt']:
                return pd.read_csv(file_path, sep='\t', index_col=0)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                return pd.read_excel(file_path, index_col=0)
            else:
                # Try CSV first, then TSV
                try:
                    return pd.read_csv(file_path, index_col=0)
                except:
                    return pd.read_csv(file_path, sep='\t', index_col=0)
        
        except Exception as e:
            raise ValueError(f"Could not read file {file_path}: {e}")
    
    def _generate_processing_statistics(self, result: Dict) -> Dict:
        """Generate comprehensive processing statistics"""
        stats = {
            'total_files_processed': len(result['processed_data']),
            'successful_conversions': sum(1 for r in result['feature_mapping'].values() 
                                       if r.get('mapped_features', 0) > 0),
            'total_features_mapped': sum(r.get('mapped_features', 0) 
                                       for r in result['feature_mapping'].values()),
            'data_shapes': {},
            'missing_statistics': {},
            'feature_coverage': {}
        }
        
        # Data shapes
        for data_type, data in result['processed_data'].items():
            stats['data_shapes'][data_type] = list(data.shape)
        
        # Missing value statistics
        for data_type, missing_mask in result['missing_masks'].items():
            if not missing_mask.empty:
                total_elements = missing_mask.shape[0] * missing_mask.shape[1]
                missing_elements = missing_mask.sum().sum()
                stats['missing_statistics'][data_type] = {
                    'missing_count': int(missing_elements),
                    'missing_ratio': float(missing_elements / total_elements) if total_elements > 0 else 0
                }
        
        # Feature coverage (how many model features are covered)
        if self.model_metadata and 'feature_metadata' in self.model_metadata:
            model_features = set(self.model_metadata['feature_metadata'].get('feature_columns', []))
            for data_type, data in result['processed_data'].items():
                user_features = set(data.columns)
                coverage = len(user_features.intersection(model_features)) / len(model_features)
                stats['feature_coverage'][data_type] = float(coverage)
        
        return stats
    
    def _generate_recommendations(self, result: Dict) -> List[str]:
        """Generate recommendations based on processing results"""
        recommendations = []
        
        # Check data quality
        for data_type, validation in result.get('data_validation', {}).items():
            if validation.get('warnings'):
                recommendations.append(f"Review {data_type} data quality: {len(validation['warnings'])} warnings")
        
        # Check feature coverage
        stats = result.get('statistics', {})
        for data_type, coverage in stats.get('feature_coverage', {}).items():
            if coverage < 0.5:
                recommendations.append(f"Low feature coverage for {data_type} ({coverage:.1%}). Consider adding more features.")
        
        # Check missing values
        for data_type, missing_stats in stats.get('missing_statistics', {}).items():
            if missing_stats['missing_ratio'] > 0.7:
                recommendations.append(f"High missing ratio for {data_type} ({missing_stats['missing_ratio']:.1%}). Model performance may be affected.")
        
        # General recommendations
        if not result.get('final_datasets'):
            recommendations.append("No final datasets created. Check input data formats and feature mappings.")
        
        if len(result.get('errors', [])) > 0:
            recommendations.append("Address processing errors before using data for model inference.")
        
        return recommendations
    
    def _save_intermediate_results(self, result: Dict, output_dir: str):
        """Save intermediate processing results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save processed datasets
            for data_type, data in result['processed_data'].items():
                file_path = output_path / f"processed_{data_type}_{timestamp}.csv"
                data.to_csv(file_path)
                self.logger.info(f"💾 Saved {data_type}: {file_path}")
            
            # Save final datasets
            for dataset_name, data in result['final_datasets'].items():
                file_path = output_path / f"final_{dataset_name}_{timestamp}.csv"
                data.to_csv(file_path)
                self.logger.info(f"💾 Saved final {dataset_name}: {file_path}")
            
            # Save processing report
            report = {
                'processing_summary': {
                    'success': result['success'],
                    'processing_time': result['processing_time'],
                    'timestamp': timestamp
                },
                'statistics': result['statistics'],
                'warnings': result['warnings'],
                'errors': result['errors'],
                'recommendations': result['recommendations']
            }
            
            report_file = output_path / f"processing_report_{timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"💾 Saved processing report: {report_file}")
            
        except Exception as e:
            self.logger.warning(f"⚠️  Could not save intermediate results: {e}")
    
    def validate_user_files(self, user_files: Dict[str, str]) -> Dict:
        """Quick validation of user files before processing"""
        validation_results = {}
        
        for data_type, file_path in user_files.items():
            try:
                # Load data
                data = self._load_user_file(file_path)
                
                # Validate
                validation = self.format_guide.validate_user_data(data_type, data)
                validation_results[data_type] = validation
                
                self.logger.info(f"📋 {data_type}: {'✅ Valid' if validation['is_valid'] else '❌ Invalid'}")
                
            except Exception as e:
                validation_results[data_type] = {
                    'is_valid': False,
                    'errors': [str(e)],
                    'warnings': []
                }
                self.logger.error(f"❌ {data_type}: {e}")
        
        return validation_results
    
    def create_example_user_data(self, output_dir: str):
        """Create example user data files"""
        self.logger.info("📝 Creating example user data files...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Use format guide to create examples
        for data_type in ['expression', 'cnv', 'microrna', 'rppa', 'mutations', 'methylation', 'clinical']:
            try:
                self.format_guide.create_format_examples(data_type, output_dir)
            except Exception as e:
                self.logger.warning(f"⚠️  Could not create example for {data_type}: {e}")
        
        # Create comprehensive usage guide
        usage_guide = f"""
# User Data Processing Guide

## Quick Start

1. Prepare your data files in the correct format (see examples)
2. Use the pipeline to process your data:

```python
from src.utils.user_data_pipeline import UserDataPipeline

# Initialize pipeline
pipeline = UserDataPipeline('path/to/model/metadata')

# Show format requirements
pipeline.show_user_format_guide()

# Process your data
user_files = {{
    'expression': 'my_expression_data.csv',
    'clinical': 'my_clinical_data.csv'
}}

result = pipeline.process_user_data(user_files)

# Check results
if result['success']:
    print("✅ Processing successful!")
    print(f"Final datasets: {{list(result['final_datasets'].keys())}}")
else:
    print("❌ Processing failed:")
    for error in result['errors']:
        print(f"  • {{error}}")
```

## Data Format Requirements

See individual *_format_README.md files for detailed requirements.

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        """
        
        guide_file = output_path / "USER_GUIDE.md"
        with open(guide_file, 'w') as f:
            f.write(usage_guide.strip())
        
        self.logger.info(f"✅ Created comprehensive user guide: {guide_file}")

def main():
    """CLI interface for user data pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='User Data Processing Pipeline')
    parser.add_argument('--metadata', required=True, help='Model metadata path')
    parser.add_argument('--process', help='Process data files (JSON config)')
    parser.add_argument('--validate', help='Validate data files (JSON config)')
    parser.add_argument('--show-guide', action='store_true', help='Show format guide')
    parser.add_argument('--create-examples', help='Create example files in directory')
    parser.add_argument('--output-dir', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = UserDataPipeline(args.metadata)
    
    if args.show_guide:
        pipeline.show_user_format_guide()
    
    if args.create_examples:
        pipeline.create_example_user_data(args.create_examples)
    
    if args.validate:
        with open(args.validate, 'r') as f:
            user_files = json.load(f)
        
        results = pipeline.validate_user_files(user_files)
        
        print("\n🔍 VALIDATION RESULTS:")
        for data_type, result in results.items():
            status = "✅ Valid" if result['is_valid'] else "❌ Invalid"
            print(f"  {data_type}: {status}")
            if result.get('warnings'):
                for warning in result['warnings']:
                    print(f"    ⚠️  {warning}")
            if result.get('errors'):
                for error in result['errors']:
                    print(f"    ❌ {error}")
    
    if args.process:
        with open(args.process, 'r') as f:
            user_files = json.load(f)
        
        result = pipeline.process_user_data(
            user_files, 
            save_intermediate=bool(args.output_dir),
            output_dir=args.output_dir
        )
        
        print(f"\n🚀 PROCESSING RESULT: {'✅ Success' if result['success'] else '❌ Failed'}")
        print(f"⏱️  Processing time: {result['processing_time']:.2f} seconds")
        
        if result['warnings']:
            print(f"⚠️  Warnings: {len(result['warnings'])}")
            for warning in result['warnings'][:5]:  # Show first 5
                print(f"   • {warning}")
        
        if result['errors']:
            print(f"❌ Errors: {len(result['errors'])}")
            for error in result['errors']:
                print(f"   • {error}")
        
        if result['recommendations']:
            print(f"💡 Recommendations:")
            for rec in result['recommendations']:
                print(f"   • {rec}")

if __name__ == "__main__":
    main()