import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json


class IntegratedCancerDataset(Dataset):
    """
    PyTorch Dataset for integrated cancer multi-omics data

    Supports two model types:
    1. Cox-enhanced TabTransformer: Clinical + Multi-omics with Cox coefficients
    2. Methylation TabTransformer: Clinical + Methylation data
    """

    def __init__(self, 
                 data_file,
                 dataset_type="cox",  # "cox" or "methylation"
                 feature_info_file=None,
                 survival_time_col="survival_time_clean",
                 survival_event_col="survival_event_clean",
                 cancer_type_col="acronym",
                 normalize_features=True,
                 survival_threshold_years=5.0):
        """
        Initialize dataset

        Parameters:
        - data_file: Path to preprocessed data file (parquet)
        - dataset_type: "cox" or "methylation"
        - feature_info_file: Path to feature information JSON
        - survival_time_col: Column name for survival time (days)
        - survival_event_col: Column name for survival event (0/1)
        - cancer_type_col: Column name for cancer type
        - normalize_features: Whether to normalize numerical features
        - survival_threshold_years: Years for binary survival classification
        """

        self.dataset_type = dataset_type
        self.survival_time_col = survival_time_col
        self.survival_event_col = survival_event_col
        self.cancer_type_col = cancer_type_col
        self.survival_threshold_days = survival_threshold_years * 365.25

        # Load data
        print(f"Loading {dataset_type} dataset from {data_file}...")
        self.data = pd.read_parquet(data_file)

        # Load feature information
        if feature_info_file and Path(feature_info_file).exists():
            with open(feature_info_file, 'r') as f:
                self.feature_info = json.load(f)
        else:
            self.feature_info = self._infer_feature_info()

        # Separate features and targets
        self._prepare_features_and_targets()

        # Normalize features if requested
        if normalize_features:
            self._normalize_features()

        print(f"Dataset initialized: {len(self)} samples, {self.n_features} features")
        print(f"Cancer types: {self.data[self.cancer_type_col].nunique()}")
        print(f"Survival events: {self.survival_events.sum()} / {len(self.survival_events)}")

    def _infer_feature_info(self):
        """Infer feature information from data"""

        feature_info = {
            'numerical_features': [],
            'categorical_features': [],
            'target_features': [self.survival_time_col, self.survival_event_col]
        }

        for col in self.data.columns:
            if col in feature_info['target_features'] or col == self.cancer_type_col:
                continue

            if self.data[col].dtype in ['object', 'category']:
                feature_info['categorical_features'].append(col)
            else:
                feature_info['numerical_features'].append(col)

        return feature_info

    def _prepare_features_and_targets(self):
        """Prepare feature matrix and target variables"""

        # Get feature columns (exclude targets and cancer type for features)
        target_cols = [self.survival_time_col, self.survival_event_col]

        if self.cancer_type_col in self.data.columns:
            # Include cancer type as a feature
            feature_cols = [col for col in self.data.columns if col not in target_cols]
        else:
            feature_cols = [col for col in self.data.columns if col not in target_cols]

        self.features = self.data[feature_cols].copy()
        self.n_features = len(feature_cols)

        # Prepare survival targets
        self.survival_times = self.data[self.survival_time_col].values.astype(np.float32)
        self.survival_events = self.data[self.survival_event_col].values.astype(np.int32)

        # Create binary survival labels (survived > threshold)
        self.survival_binary = (
            (self.survival_times > self.survival_threshold_days) | 
            (self.survival_events == 0)
        ).astype(np.int32)

        # Cancer type labels
        if self.cancer_type_col in self.data.columns:
            from sklearn.preprocessing import LabelEncoder
            self.cancer_encoder = LabelEncoder()
            self.cancer_types = self.cancer_encoder.fit_transform(
                self.data[self.cancer_type_col].astype(str)
            ).astype(np.int32)
        else:
            self.cancer_types = np.zeros(len(self.data), dtype=np.int32)
            self.cancer_encoder = None

    def _normalize_features(self):
        """Normalize numerical features"""

        numerical_cols = [col for col in self.features.columns 
                         if self.features[col].dtype in [np.float32, np.float64, np.int32, np.int64]]

        if len(numerical_cols) > 0:
            self.scaler = StandardScaler()
            self.features[numerical_cols] = self.scaler.fit_transform(
                self.features[numerical_cols]
            ).astype(np.float32)
            print(f"Normalized {len(numerical_cols)} numerical features")
        else:
            self.scaler = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single sample"""

        # Get features as tensor
        features = torch.tensor(self.features.iloc[idx].values, dtype=torch.float32)

        # Get targets
        survival_time = torch.tensor(self.survival_times[idx], dtype=torch.float32)
        survival_event = torch.tensor(self.survival_events[idx], dtype=torch.long)
        survival_binary = torch.tensor(self.survival_binary[idx], dtype=torch.long)
        cancer_type = torch.tensor(self.cancer_types[idx], dtype=torch.long)

        return {
            'features': features,
            'survival_time': survival_time,
            'survival_event': survival_event,
            'survival_binary': survival_binary,
            'cancer_type': cancer_type,
            'patient_id': self.data.index[idx]
        }

    def get_feature_names(self):
        """Get feature column names"""
        return list(self.features.columns)

    def get_cancer_type_names(self):
        """Get cancer type names"""
        if self.cancer_encoder:
            return list(self.cancer_encoder.classes_)
        return ['Unknown']

    def create_dataloader(self, batch_size=32, shuffle=True, num_workers=0):
        """Create PyTorch DataLoader"""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )


class MultiModalCancerDataset(Dataset):
    """
    Dataset that combines Cox-enhanced and Methylation models
    for ensemble prediction
    """

    def __init__(self, 
                 cox_data_file,
                 methylation_data_file,
                 cox_feature_info_file=None,
                 methylation_feature_info_file=None,
                 **kwargs):
        """
        Initialize multi-modal dataset

        Parameters:
        - cox_data_file: Path to Cox-enhanced data
        - methylation_data_file: Path to methylation data  
        - cox_feature_info_file: Cox feature info JSON
        - methylation_feature_info_file: Methylation feature info JSON
        """

        # Initialize individual datasets
        self.cox_dataset = IntegratedCancerDataset(
            cox_data_file, 
            dataset_type="cox",
            feature_info_file=cox_feature_info_file,
            **kwargs
        )

        self.methylation_dataset = IntegratedCancerDataset(
            methylation_data_file, 
            dataset_type="methylation",
            feature_info_file=methylation_feature_info_file,
            **kwargs
        )

        # Find common patients
        cox_patients = set(self.cox_dataset.data.index)
        methylation_patients = set(self.methylation_dataset.data.index)

        self.common_patients = sorted(list(cox_patients.intersection(methylation_patients)))

        # Create patient index mappings
        self.cox_idx_map = {patient: idx for idx, patient 
                           in enumerate(self.cox_dataset.data.index) 
                           if patient in self.common_patients}

        self.methylation_idx_map = {patient: idx for idx, patient 
                                   in enumerate(self.methylation_dataset.data.index) 
                                   if patient in self.common_patients}

        print(f"Multi-modal dataset: {len(self.common_patients)} common patients")

    def __len__(self):
        return len(self.common_patients)

    def __getitem__(self, idx):
        """Get sample from both modalities"""

        patient_id = self.common_patients[idx]

        # Get Cox data
        cox_idx = self.cox_idx_map[patient_id]
        cox_sample = self.cox_dataset[cox_idx]

        # Get methylation data
        methylation_idx = self.methylation_idx_map[patient_id]
        methylation_sample = self.methylation_dataset[methylation_idx]

        return {
            'cox_features': cox_sample['features'],
            'methylation_features': methylation_sample['features'],
            'survival_time': cox_sample['survival_time'],
            'survival_event': cox_sample['survival_event'],
            'survival_binary': cox_sample['survival_binary'],
            'cancer_type': cox_sample['cancer_type'],
            'patient_id': patient_id
        }

    def create_dataloader(self, batch_size=32, shuffle=True, num_workers=0):
        """Create PyTorch DataLoader for multi-modal data"""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
