"""
WSI Feature Dataset for MIL Training
====================================

Loads pre-extracted features from HDF5 files for MIL model training.

Features are stored as:
    {slide_id}.h5
    ├── features: (num_patches, feature_dim)
    └── coordinates: (num_patches, 2)
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class WSISample:
    """A single WSI sample with features and label."""
    slide_id: str
    features: torch.Tensor  # (num_patches, feature_dim)
    label: int  # 0 or 1
    coordinates: Optional[torch.Tensor] = None  # (num_patches, 2)
    cancer_type: Optional[str] = None
    patient_id: Optional[str] = None


class WSIFeatureDataset(Dataset):
    """
    PyTorch Dataset for loading pre-extracted WSI features.

    Args:
        features_dir: Directory containing HDF5 feature files
        labels_df: DataFrame with columns ['slide_id', 'label'] and optionally
                   ['patient_id', 'cancer_type']
        max_patches: Maximum number of patches to load per slide (for memory)
        transform: Optional transform to apply to features
    """

    def __init__(
        self,
        features_dir: str,
        labels_df: pd.DataFrame,
        max_patches: Optional[int] = None,
        transform: Optional[Any] = None,
        load_coordinates: bool = False,
    ):
        self.features_dir = Path(features_dir)
        self.max_patches = max_patches
        self.transform = transform
        self.load_coordinates = load_coordinates

        # Validate labels_df
        required_cols = ['slide_id', 'label']
        for col in required_cols:
            if col not in labels_df.columns:
                raise ValueError(f"labels_df must have column '{col}'")

        # Filter to only slides with features
        available_slides = set()
        for f in self.features_dir.glob('*.h5'):
            available_slides.add(f.stem)

        self.labels_df = labels_df[labels_df['slide_id'].isin(available_slides)].reset_index(drop=True)

        if len(self.labels_df) == 0:
            raise ValueError(f"No matching slides found in {features_dir}")

        print(f"WSIFeatureDataset initialized:")
        print(f"  Features directory: {features_dir}")
        print(f"  Total slides: {len(self.labels_df)}")
        print(f"  Label distribution: {self.labels_df['label'].value_counts().to_dict()}")

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int) -> WSISample:
        row = self.labels_df.iloc[idx]
        slide_id = row['slide_id']
        label = int(row['label'])

        # Load features from HDF5
        h5_path = self.features_dir / f"{slide_id}.h5"

        with h5py.File(h5_path, 'r') as f:
            features = f['features'][:]

            if self.load_coordinates and 'coordinates' in f:
                coordinates = f['coordinates'][:]
            else:
                coordinates = None

        # Subsample if too many patches
        if self.max_patches is not None and len(features) > self.max_patches:
            indices = np.random.choice(len(features), self.max_patches, replace=False)
            features = features[indices]
            if coordinates is not None:
                coordinates = coordinates[indices]

        # Convert to tensors
        features = torch.from_numpy(features).float()

        if coordinates is not None:
            coordinates = torch.from_numpy(coordinates).long()

        # Apply transform if provided
        if self.transform is not None:
            features = self.transform(features)

        # Get optional metadata
        cancer_type = row.get('cancer_type', None)
        patient_id = row.get('patient_id', None)

        return WSISample(
            slide_id=slide_id,
            features=features,
            label=label,
            coordinates=coordinates,
            cancer_type=cancer_type,
            patient_id=patient_id,
        )


def collate_features(batch: List[WSISample]) -> Dict[str, Any]:
    """
    Custom collate function for variable-length feature sequences.

    Pads features to the maximum length in the batch and creates a mask.

    Args:
        batch: List of WSISample objects

    Returns:
        Dict with:
            features: (batch, max_patches, feature_dim)
            labels: (batch,)
            mask: (batch, max_patches), True = valid
            slide_ids: List of slide IDs
    """
    # Find max number of patches in batch
    max_patches = max(sample.features.shape[0] for sample in batch)
    feature_dim = batch[0].features.shape[1]

    batch_size = len(batch)

    # Initialize padded tensors
    features = torch.zeros(batch_size, max_patches, feature_dim)
    mask = torch.zeros(batch_size, max_patches, dtype=torch.bool)
    labels = torch.zeros(batch_size, dtype=torch.long)

    slide_ids = []
    cancer_types = []
    patient_ids = []

    for i, sample in enumerate(batch):
        num_patches = sample.features.shape[0]

        features[i, :num_patches] = sample.features
        mask[i, :num_patches] = True
        labels[i] = sample.label

        slide_ids.append(sample.slide_id)
        cancer_types.append(sample.cancer_type)
        patient_ids.append(sample.patient_id)

    return {
        'features': features,
        'labels': labels,
        'mask': mask,
        'slide_ids': slide_ids,
        'cancer_types': cancer_types,
        'patient_ids': patient_ids,
    }


class WSIDataModule:
    """
    Data module for managing train/val/test splits.

    Args:
        features_dir: Directory containing HDF5 feature files
        labels_path: Path to labels CSV or JSON
        splits_path: Optional path to pre-defined splits JSON
        train_ratio: Train ratio if splits not provided
        val_ratio: Validation ratio if splits not provided
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers
        max_patches: Maximum patches per slide
        seed: Random seed for splits
    """

    def __init__(
        self,
        features_dir: str,
        labels_path: str,
        splits_path: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        batch_size: int = 1,
        num_workers: int = 4,
        max_patches: Optional[int] = None,
        seed: int = 42,
    ):
        self.features_dir = features_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_patches = max_patches
        self.seed = seed

        # Load labels
        if labels_path.endswith('.csv'):
            self.labels_df = pd.read_csv(labels_path)
        elif labels_path.endswith('.json'):
            with open(labels_path, 'r') as f:
                labels_data = json.load(f)
            self.labels_df = pd.DataFrame(labels_data)
        else:
            raise ValueError(f"Unsupported labels format: {labels_path}")

        # Load or create splits
        if splits_path and os.path.exists(splits_path):
            with open(splits_path, 'r') as f:
                splits = json.load(f)
            self.train_ids = set(splits['train'])
            self.val_ids = set(splits['val'])
            self.test_ids = set(splits['test'])
        else:
            self._create_splits(train_ratio, val_ratio)

        # Create split DataFrames
        self.train_df = self.labels_df[self.labels_df['slide_id'].isin(self.train_ids)]
        self.val_df = self.labels_df[self.labels_df['slide_id'].isin(self.val_ids)]
        self.test_df = self.labels_df[self.labels_df['slide_id'].isin(self.test_ids)]

        print(f"WSIDataModule initialized:")
        print(f"  Train: {len(self.train_df)} slides")
        print(f"  Val: {len(self.val_df)} slides")
        print(f"  Test: {len(self.test_df)} slides")

    def _create_splits(
        self,
        train_ratio: float,
        val_ratio: float,
    ):
        """Create random train/val/test splits."""
        np.random.seed(self.seed)

        all_ids = self.labels_df['slide_id'].unique().tolist()
        np.random.shuffle(all_ids)

        n = len(all_ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        self.train_ids = set(all_ids[:n_train])
        self.val_ids = set(all_ids[n_train:n_train + n_val])
        self.test_ids = set(all_ids[n_train + n_val:])

    def save_splits(self, output_path: str):
        """Save splits to JSON file."""
        splits = {
            'train': list(self.train_ids),
            'val': list(self.val_ids),
            'test': list(self.test_ids),
        }
        with open(output_path, 'w') as f:
            json.dump(splits, f, indent=2)

    def train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        dataset = WSIFeatureDataset(
            features_dir=self.features_dir,
            labels_df=self.train_df,
            max_patches=self.max_patches,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_features,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        dataset = WSIFeatureDataset(
            features_dir=self.features_dir,
            labels_df=self.val_df,
            max_patches=self.max_patches,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_features,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Get test DataLoader."""
        dataset = WSIFeatureDataset(
            features_dir=self.features_dir,
            labels_df=self.test_df,
            max_patches=self.max_patches,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_features,
            pin_memory=True,
        )


def create_labels_from_clinical(
    clinical_path: str,
    output_path: str,
    survival_threshold_years: float = 3.0,
) -> pd.DataFrame:
    """
    Create labels DataFrame from TCGA clinical data.

    Args:
        clinical_path: Path to clinical data (TSV)
        output_path: Path to save labels CSV
        survival_threshold_years: Threshold for survival (default: 3 years)

    Returns:
        Labels DataFrame with columns: slide_id, label, patient_id, cancer_type
    """
    # Load clinical data
    clinical = pd.read_csv(clinical_path, sep='\t', low_memory=False)

    # Extract relevant columns
    # TCGA clinical data format varies, adjust as needed
    required_cols = ['bcr_patient_barcode', 'days_to_death', 'days_to_last_followup', 'vital_status']

    for col in required_cols:
        if col not in clinical.columns:
            print(f"Warning: Column '{col}' not found in clinical data")

    # Calculate survival
    threshold_days = survival_threshold_years * 365

    labels = []
    for _, row in clinical.iterrows():
        patient_id = row.get('bcr_patient_barcode', '')

        # Determine survival time
        days_to_death = row.get('days_to_death')
        days_to_followup = row.get('days_to_last_followup')
        vital_status = str(row.get('vital_status', '')).lower()

        if pd.notna(days_to_death) and days_to_death != '[Not Available]':
            try:
                survival_days = float(days_to_death)
                # Died before threshold = 1 (poor outcome)
                label = 1 if survival_days < threshold_days else 0
            except ValueError:
                continue
        elif pd.notna(days_to_followup) and days_to_followup != '[Not Available]':
            try:
                survival_days = float(days_to_followup)
                # Alive at last followup
                if survival_days >= threshold_days:
                    label = 0  # Survived past threshold
                elif 'dead' in vital_status:
                    label = 1  # Died before threshold
                else:
                    continue  # Censored before threshold, skip
            except ValueError:
                continue
        else:
            continue

        cancer_type = row.get('acronym', row.get('disease', ''))

        labels.append({
            'patient_id': patient_id,
            'label': label,
            'cancer_type': cancer_type,
        })

    labels_df = pd.DataFrame(labels)

    # Save
    labels_df.to_csv(output_path, index=False)

    return labels_df


if __name__ == "__main__":
    print("WSI Dataset module loaded successfully.")

    # Test with dummy data
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy features
        features_dir = Path(tmpdir) / 'features'
        features_dir.mkdir()

        for i in range(5):
            slide_id = f"slide_{i}"
            num_patches = np.random.randint(50, 200)

            with h5py.File(features_dir / f"{slide_id}.h5", 'w') as f:
                f.create_dataset('features', data=np.random.randn(num_patches, 768).astype(np.float32))
                f.create_dataset('coordinates', data=np.random.randint(0, 1000, (num_patches, 2)))

        # Create labels
        labels_df = pd.DataFrame({
            'slide_id': [f'slide_{i}' for i in range(5)],
            'label': [0, 1, 0, 1, 0],
            'cancer_type': ['BRCA'] * 5,
        })

        # Test dataset
        dataset = WSIFeatureDataset(
            features_dir=str(features_dir),
            labels_df=labels_df,
            max_patches=100,
        )

        print(f"\nDataset length: {len(dataset)}")

        sample = dataset[0]
        print(f"Sample slide_id: {sample.slide_id}")
        print(f"Sample features shape: {sample.features.shape}")
        print(f"Sample label: {sample.label}")

        # Test collate
        batch = [dataset[i] for i in range(3)]
        collated = collate_features(batch)
        print(f"\nCollated features shape: {collated['features'].shape}")
        print(f"Collated mask shape: {collated['mask'].shape}")
        print(f"Collated labels: {collated['labels']}")

        print("\nAll tests passed!")
