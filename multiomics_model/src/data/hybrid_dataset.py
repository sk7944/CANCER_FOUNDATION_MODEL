"""
PyTorch Dataset for Hybrid FC-NN + TabTransformer

Missing Modality Learning (Option 3):
- Cox available (4,504): Clinical + Cox + Methylation
- Cox unavailable (3,720): Clinical + [ZERO] + Methylation
- Total: 8,224 patients
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, List
import json


def preprocess_clinical_data(clinical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Raw TCGA 컬럼을 모델이 기대하는 컬럼으로 매핑

    Mapping:
        gender → sex (0=MALE, 1=FEMALE)
        age_at_initial_pathologic_diagnosis → age_group (10개 bin, 0-9)
        pathologic_stage → ajcc_pathologic_stage (0-4: I, II, III, IV, NA)
        neoplasm_histologic_grade → grade (0-3: G1, G2, G3, G4)
        survival_time_clean → event_time
        survival_event_clean → event_status
    """
    df = clinical_df.copy()

    # 1. sex: gender → numeric
    if 'gender' in df.columns and 'sex' not in df.columns:
        df['sex'] = df['gender'].map({'MALE': 0, 'FEMALE': 1}).fillna(0).astype(int)

    # 2. age_group: age → 10 bins (0-9)
    if 'age_at_initial_pathologic_diagnosis' in df.columns and 'age_group' not in df.columns:
        age = pd.to_numeric(df['age_at_initial_pathologic_diagnosis'], errors='coerce').fillna(60)
        df['age_group'] = pd.cut(age, bins=[0, 30, 40, 50, 55, 60, 65, 70, 75, 80, 120],
                                  labels=range(10), right=False).astype(int)

    # 3. ajcc_pathologic_stage: Stage I-IV → numeric (0-4)
    # 서브스테이지(A, B, C)를 메인 스테이지로 일원화, 누락/부적절은 별도 NA 카테고리
    if 'pathologic_stage' in df.columns and 'ajcc_pathologic_stage' not in df.columns:
        stage_map = {
            # Stage I 계열 → 0
            'Stage I': 0, 'Stage IA': 0, 'Stage IA1': 0, 'Stage IA2': 0, 'Stage IB': 0,
            'Stage IB1': 0, 'Stage IB2': 0, 'Stage IC': 0,
            # Stage II 계열 → 1
            'Stage II': 1, 'Stage IIA': 1, 'Stage IIA1': 1, 'Stage IIA2': 1,
            'Stage IIB': 1, 'Stage IIC': 1, 'Stage IIC1': 1,
            # Stage III 계열 → 2
            'Stage III': 2, 'Stage IIIA': 2, 'Stage IIIB': 2, 'Stage IIIC': 2,
            'Stage IIIC1': 2, 'Stage IIIC2': 2,
            # Stage IV 계열 → 3
            'Stage IV': 3, 'Stage IVA': 3, 'Stage IVB': 3, 'Stage IVC': 3,
            # 누락/부적절/알수없음 → 4 (별도 NA 카테고리)
            '[Not Available]': 4, '[Not Applicable]': 4, '[Discrepancy]': 4,
            '[Unknown]': 4, 'I/II NOS': 4
        }
        df['ajcc_pathologic_stage'] = df['pathologic_stage'].map(stage_map).fillna(4).astype(int)

    # 4. grade: G1-G4 → numeric (0-3)
    if 'neoplasm_histologic_grade' in df.columns and 'grade' not in df.columns:
        grade_map = {
            'G1': 0, 'G2': 1, 'G3': 2, 'G4': 3, 'GX': 2,
            'High Grade': 3, 'Low Grade': 1,
            '[Not Available]': 2, '[Not Applicable]': 2
        }
        df['grade'] = df['neoplasm_histologic_grade'].map(grade_map).fillna(2).astype(int)

    # 5. event_time: survival_time_clean → event_time
    if 'survival_time_clean' in df.columns and 'event_time' not in df.columns:
        df['event_time'] = pd.to_numeric(df['survival_time_clean'], errors='coerce').fillna(0)

    # 6. event_status: survival_event_clean → event_status
    if 'survival_event_clean' in df.columns and 'event_status' not in df.columns:
        df['event_status'] = pd.to_numeric(df['survival_event_clean'], errors='coerce').fillna(0).astype(int)

    # race는 이미 존재, 인코딩 필요
    if 'race' in df.columns:
        race_map = {
            'WHITE': 0, 'BLACK OR AFRICAN AMERICAN': 1, 'ASIAN': 2,
            'AMERICAN INDIAN OR ALASKA NATIVE': 3, 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 4,
            '[Not Available]': 5, '[Not Evaluated]': 5, '[Unknown]': 5
        }
        df['race'] = df['race'].map(race_map).fillna(5).astype(int)

    return df


class HybridMultiOmicsDataset(Dataset):
    """
    Dataset for Hybrid Multi-Modal Learning with Missing Modality Support

    Data files:
    - integrated_table_cox.parquet: 4,504 × 132,100 (cox omics only, 임상 제외)
    - methylation_table.parquet: 8,224 × 396,065 (CG sites)
    """

    def __init__(
        self,
        cox_table_path: str,
        meth_table_path: str,
        clinical_path: str,
        patient_ids: Optional[List[str]] = None,
        split: str = 'train'
    ):
        """
        Args:
            cox_table_path: Path to integrated_table_cox.parquet
            meth_table_path: Path to methylation_table.parquet
            clinical_path: Path to clinical data (for survival labels)
            patient_ids: Patient IDs for this split
            split: 'train', 'val', or 'test'
        """
        self.split = split

        print(f"Loading {split} data...")

        # Load Cox table (4,504 patients)
        cox_df = pd.read_parquet(cox_table_path)
        print(f"  Cox table: {cox_df.shape[0]:,} × {cox_df.shape[1]:,}")

        # Load Methylation table (8,224 patients)
        meth_df = pd.read_parquet(meth_table_path)
        print(f"  Methylation table: {meth_df.shape[0]:,} × {meth_df.shape[1]:,}")

        # Load clinical data for labels and preprocess
        # 두 임상 파일을 병합하여 전체 8,577명 커버
        from pathlib import Path
        clinical_path = Path(clinical_path)
        clinical_dir = clinical_path.parent

        # Primary clinical file (methylation용 - 8,224명)
        meth_clinical_path = clinical_dir / 'processed_clinical_data_for_methylation.parquet'
        # Secondary clinical file (cox용 - 4,504명, 353명 추가)
        cox_clinical_path = clinical_dir / 'processed_clinical_data.parquet'

        clinical_dfs = []
        if meth_clinical_path.exists():
            df1 = pd.read_parquet(meth_clinical_path)
            clinical_dfs.append(df1)
            print(f"  Clinical (methylation): {df1.shape[0]:,} patients")

        if cox_clinical_path.exists():
            df2 = pd.read_parquet(cox_clinical_path)
            # 중복 제거: methylation에 없는 환자만 추가
            if clinical_dfs:
                existing_patients = set(clinical_dfs[0].index)
                new_patients = df2.loc[~df2.index.isin(existing_patients)]
                if len(new_patients) > 0:
                    clinical_dfs.append(new_patients)
                    print(f"  Clinical (cox-only): {len(new_patients):,} patients added")
            else:
                clinical_dfs.append(df2)
                print(f"  Clinical (cox): {df2.shape[0]:,} patients")

        if not clinical_dfs:
            raise FileNotFoundError(f"No clinical data found in {clinical_dir}")

        clinical_df = pd.concat(clinical_dfs, axis=0)
        clinical_df = preprocess_clinical_data(clinical_df)
        print(f"  Clinical data total: {clinical_df.shape[0]:,} patients (preprocessed)")

        # Filter by patient IDs if provided
        if patient_ids is not None:
            cox_df = cox_df.loc[cox_df.index.intersection(patient_ids)]
            meth_df = meth_df.loc[meth_df.index.intersection(patient_ids)]
            clinical_df = clinical_df.loc[clinical_df.index.intersection(patient_ids)]
            print(f"  Filtered to {len(patient_ids):,} patients")

        # Store patient lists
        self.cox_patients = set(cox_df.index)
        self.meth_patients = set(meth_df.index)
        # Use UNION of all patients (Cox + Methylation)
        self.all_patients = sorted(self.cox_patients.union(self.meth_patients))

        print(f"  Cox available: {len(self.cox_patients):,}")
        print(f"  Methylation available: {len(self.meth_patients):,}")
        print(f"  Total patients (union): {len(self.all_patients):,}")

        # Clinical categorical features
        # 항상 clinical_df에서 가져옴 (전체 환자 포함)
        clinical_cat_cols = ['age_group', 'sex', 'race', 'ajcc_pathologic_stage', 'grade']
        self.clinical_cat = clinical_df[clinical_cat_cols].copy()

        # Cox omics features (exclude clinical)
        cox_omics_cols = [col for col in cox_df.columns if col not in clinical_cat_cols]
        self.cox_omics = cox_df[cox_omics_cols].copy()

        # Methylation features
        self.methylation = meth_df.copy()

        # Survival labels
        self.event_times = clinical_df['event_time'].copy()
        self.event_status = clinical_df['event_status'].copy()

        # Cox mask (True if Cox data available)
        self.cox_mask = pd.Series(
            {patient: (patient in self.cox_patients) for patient in self.all_patients}
        )

        # Methylation mask (True if Methylation data available)
        self.meth_mask = pd.Series(
            {patient: (patient in self.meth_patients) for patient in self.all_patients}
        )

        # Convert to numpy arrays
        self._prepare_arrays()

        print(f"✅ {split} dataset ready: {len(self)} patients")
        print(f"   Cox available: {self.cox_mask.sum():,}, unavailable: {(~self.cox_mask).sum():,}")
        print(f"   Meth available: {self.meth_mask.sum():,}, unavailable: {(~self.meth_mask).sum():,}")

    def _prepare_arrays(self):
        """Convert dataframes to numpy arrays for faster access"""
        # Clinical categorical
        self.clinical_cat_array = self.clinical_cat.loc[self.all_patients].values

        # Cox omics (fill missing patients with zeros, fill NaN with 0)
        cox_array_list = []
        for patient in self.all_patients:
            if patient in self.cox_omics.index:
                cox_array_list.append(self.cox_omics.loc[patient].values)
            else:
                cox_array_list.append(np.zeros(len(self.cox_omics.columns)))
        self.cox_omics_array = np.array(cox_array_list, dtype=np.float32)
        # Fill NaN with 0
        self.cox_omics_array = np.nan_to_num(self.cox_omics_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Methylation (fill missing patients with zeros, fill NaN with 0)
        meth_array_list = []
        for patient in self.all_patients:
            if patient in self.methylation.index:
                meth_array_list.append(self.methylation.loc[patient].values)
            else:
                meth_array_list.append(np.zeros(len(self.methylation.columns)))
        self.meth_array = np.array(meth_array_list, dtype=np.float32)
        # Fill NaN with 0
        self.meth_array = np.nan_to_num(self.meth_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Labels
        self.event_times_array = self.event_times.loc[self.all_patients].values.astype(np.float32)
        self.event_status_array = self.event_status.loc[self.all_patients].values.astype(np.float32)

        # Cox mask
        self.cox_mask_array = self.cox_mask.loc[self.all_patients].values

        # Methylation mask
        self.meth_mask_array = self.meth_mask.loc[self.all_patients].values

    def __len__(self) -> int:
        return len(self.all_patients)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            clinical_cat: (5,) categorical features
            cox_omics: (132098,) Cox [val, cox] pairs
            methylation: (396065,) beta values
            cox_mask: (1,) boolean mask - True if Cox available
            meth_mask: (1,) boolean mask - True if Methylation available
            event_time: (1,) survival time
            event_status: (1,) event indicator
            patient_id: string
        """
        return {
            'clinical_cat': torch.from_numpy(self.clinical_cat_array[idx]).long(),
            'cox_omics': torch.from_numpy(self.cox_omics_array[idx]).float(),
            'methylation': torch.from_numpy(self.meth_array[idx]).float(),
            'cox_mask': torch.tensor([self.cox_mask_array[idx]], dtype=torch.bool),
            'meth_mask': torch.tensor([self.meth_mask_array[idx]], dtype=torch.bool),
            'event_time': torch.tensor([self.event_times_array[idx]], dtype=torch.float32),
            'event_status': torch.tensor([self.event_status_array[idx]], dtype=torch.float32),
            'patient_id': self.all_patients[idx]
        }


def create_dataloaders(
    cox_table_path: str,
    meth_table_path: str,
    clinical_path: str,
    splits_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Load splits
    with open(splits_path, 'r') as f:
        splits = json.load(f)

    # Create datasets
    train_dataset = HybridMultiOmicsDataset(
        cox_table_path, meth_table_path, clinical_path,
        patient_ids=splits['train_patients'],
        split='train'
    )

    val_dataset = HybridMultiOmicsDataset(
        cox_table_path, meth_table_path, clinical_path,
        patient_ids=splits['val_patients'],
        split='val'
    )

    test_dataset = HybridMultiOmicsDataset(
        cox_table_path, meth_table_path, clinical_path,
        patient_ids=splits['test_patients'],
        split='test'
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print("\n" + "="*70)
    print("DataLoaders Created")
    print("="*70)
    print(f"Train: {len(train_dataset):,} patients, {len(train_loader):,} batches")
    print(f"Val:   {len(val_dataset):,} patients, {len(val_loader):,} batches")
    print(f"Test:  {len(test_dataset):,} patients, {len(test_loader):,} batches")
    print(f"Batch size: {batch_size}")
    print("="*70)

    return train_loader, val_loader, test_loader
