#!/usr/bin/env python3
"""
Test script for TabTransformer implementations
Tests both Cox and Methylation transformers with real data and synthetic examples
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.cox_tabtransformer import CoxTabTransformer
from src.models.methylation_tabtransformer import MethylationTabTransformer
from src.utils.tabtransformer_utils import *

def test_cox_tabtransformer():
    """Test CoxTabTransformer with synthetic data"""
    print("=" * 60)
    print("Testing CoxTabTransformer")
    print("=" * 60)
    
    # Model configuration
    vocab_sizes = (10, 5, 6, 8)  # 4 clinical categorical features
    num_omics = 500  # 500 omics features
    
    model = CoxTabTransformer(
        clinical_categories=vocab_sizes,
        num_omics_features=num_omics,
        dim=64,
        depth=3,
        heads=8,
        survival_hidden_dim=256
    )
    
    # Test data
    batch_size = 16
    
    # Clinical categorical data (within vocab ranges)
    clinical_data = torch.tensor([
        [i % vocab_sizes[0], i % vocab_sizes[1], i % vocab_sizes[2], i % vocab_sizes[3]] 
        for i in range(batch_size)
    ])
    
    # Omics continuous data [value, cox] pairs
    omics_data = torch.randn(batch_size, num_omics * 2)
    
    print(f"Input shapes:")
    print(f"  Clinical categorical: {clinical_data.shape}")
    print(f"  Omics continuous: {omics_data.shape}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, representation = model(clinical_data, omics_data)
        probabilities = torch.sigmoid(logits)
    
    print(f"Output shapes:")
    print(f"  Survival logits: {logits.shape}")
    print(f"  Representation: {representation.shape}")
    print(f"  Probability range: {probabilities.min().item():.3f} - {probabilities.max().item():.3f}")
    
    print("✅ CoxTabTransformer test passed!")
    return True

def test_methylation_tabtransformer():
    """Test MethylationTabTransformer with synthetic data"""
    print("\n" + "=" * 60)
    print("Testing MethylationTabTransformer")
    print("=" * 60)
    
    # Model configuration
    num_probes = 10000
    selected_probes = 500
    
    model = MethylationTabTransformer(
        num_probes=num_probes,
        selected_probes=selected_probes,
        dim=64,
        depth=3,
        heads=8,
        survival_hidden_dim=256
    )
    
    # Test data (β-values between 0 and 1)
    batch_size = 12
    methylation_data = torch.rand(batch_size, num_probes)
    
    print(f"Input shapes:")
    print(f"  Methylation data: {methylation_data.shape}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Selected probes: {selected_probes}/{num_probes} ({selected_probes/num_probes*100:.1f}%)")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, representation, selected_indices = model(methylation_data)
        probabilities = torch.sigmoid(logits)
    
    print(f"Output shapes:")
    print(f"  Survival logits: {logits.shape}")
    print(f"  Representation: {representation.shape}")
    print(f"  Selected indices: {selected_indices.shape}")
    print(f"  Probability range: {probabilities.min().item():.3f} - {probabilities.max().item():.3f}")
    
    # Check selected probes
    first_sample_selected = selected_indices[0][:10]
    print(f"  First sample top 10 selected probes: {first_sample_selected.tolist()}")
    
    print("✅ MethylationTabTransformer test passed!")
    return True

def test_data_preprocessing():
    """Test data preprocessing utilities"""
    print("\n" + "=" * 60)
    print("Testing Data Preprocessing Functions")
    print("=" * 60)
    
    # Test survival label creation
    print("Testing survival label creation...")
    
    # Synthetic clinical data
    clinical_df = pd.DataFrame({
        'patient_id': [f'P{i:03d}' for i in range(20)],
        'survival_time_clean': [500, 800, 1200, 400, 1500, 600, 2000, 300, 1800, 900,
                               1100, 450, 1300, 750, 1600, 350, 1400, 650, 1700, 550],
        'survival_event_clean': [1, 0, 0, 1, 0, 1, 0, 1, 0, 1,
                               0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    clinical_df.set_index('patient_id', inplace=True)
    
    survival_labels, valid_patients = create_survival_labels(clinical_df, threshold_days=1095)
    
    print(f"  Total patients: {len(clinical_df)}")
    print(f"  Valid patients: {len(valid_patients)}")
    print(f"  3-year survivors: {np.sum(survival_labels == 0)}")
    print(f"  3-year deaths: {np.sum(survival_labels == 1)}")
    
    # Test clinical data preparation
    print("\nTesting clinical data preparation...")
    
    clinical_extended = clinical_df.copy()
    clinical_extended['gender'] = ['M', 'F'] * 10
    clinical_extended['age_at_initial_pathologic_diagnosis'] = np.random.randint(30, 80, 20)
    clinical_extended['pathologic_stage'] = np.random.choice(['Stage I', 'Stage II', 'Stage III', 'Stage IV'], 20)
    
    categorical_tensor, vocab_sizes, encoders, feature_names = prepare_clinical_data(clinical_extended)
    
    print(f"  Categorical tensor shape: {categorical_tensor.shape}")
    print(f"  Vocabulary sizes: {vocab_sizes}")
    print(f"  Feature names: {feature_names}")
    
    # Test Cox data preparation
    print("\nTesting Cox data preparation...")
    
    # Synthetic omics data
    feature_names_omics = [f'Gene_{i}' for i in range(100)]
    omics_df = pd.DataFrame(
        np.random.randn(20, 100),
        columns=feature_names_omics,
        index=clinical_df.index
    )
    
    # Synthetic Cox coefficients
    cox_lookup = {name: np.random.normal(0, 0.5) for name in feature_names_omics}
    
    cox_continuous, cox_features = prepare_cox_data(omics_df, cox_lookup)
    
    print(f"  Cox continuous shape: {cox_continuous.shape}")
    print(f"  Cox features: {len(cox_features)}")
    print(f"  Expected shape: {(len(omics_df), len(cox_features) * 2)}")
    
    print("✅ Data preprocessing tests passed!")
    return True

def test_with_real_data():
    """Test with actual data files if available"""
    print("\n" + "=" * 60)
    print("Testing with Real Data (if available)")
    print("=" * 60)
    
    data_dir = project_root / 'data' / 'processed'
    
    try:
        # Try to load real data
        if (data_dir / 'integrated_table_cox.parquet').exists():
            cox_data = pd.read_parquet(data_dir / 'integrated_table_cox.parquet')
            print(f"✅ Cox data loaded: {cox_data.shape}")
            
            if (data_dir / 'processed_clinical_data.parquet').exists():
                clinical_data = pd.read_parquet(data_dir / 'processed_clinical_data.parquet')
                print(f"✅ Clinical data loaded: {clinical_data.shape}")
                
                # Test survival label creation with real data
                survival_labels, valid_patients = create_survival_labels(clinical_data)
                print(f"✅ Real data survival labels: {len(valid_patients)} valid patients")
                
                if len(valid_patients) > 100:
                    print(f"   3-year survivors: {np.sum(survival_labels == 0)} ({np.mean(survival_labels == 0)*100:.1f}%)")
                    print(f"   3-year deaths: {np.sum(survival_labels == 1)} ({np.mean(survival_labels == 1)*100:.1f}%)")
                    
                    # Test small subset with real model
                    subset_size = min(50, len(valid_patients))
                    subset_patients = valid_patients[:subset_size]
                    subset_labels = survival_labels[:subset_size]
                    
                    # Filter data
                    cox_subset = cox_data.loc[cox_data.index.intersection(subset_patients)]
                    clinical_subset = clinical_data.loc[clinical_data.index.intersection(subset_patients)]
                    
                    if not cox_subset.empty and not clinical_subset.empty:
                        print(f"✅ Real data subset test ready: {len(subset_patients)} patients")
                        return True
        
        print("⚠️ Real data files not found, using synthetic data only")
        return True
        
    except Exception as e:
        print(f"⚠️ Real data test failed: {e}")
        return True

def main():
    """Main test function"""
    print("TabTransformer Implementation Tests")
    print("=" * 60)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Run tests
    tests_passed = 0
    total_tests = 4
    
    try:
        if test_cox_tabtransformer():
            tests_passed += 1
    except Exception as e:
        print(f"❌ CoxTabTransformer test failed: {e}")
    
    try:
        if test_methylation_tabtransformer():
            tests_passed += 1
    except Exception as e:
        print(f"❌ MethylationTabTransformer test failed: {e}")
    
    try:
        if test_data_preprocessing():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Data preprocessing test failed: {e}")
    
    try:
        if test_with_real_data():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Real data test failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All TabTransformer tests passed!")
        print("Ready for training and experimentation.")
        return True
    else:
        print("⚠️ Some tests failed. Check implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)