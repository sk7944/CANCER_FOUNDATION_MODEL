"""
WSI Labels CSV & Splits JSON Generator
=======================================

Generates labels and patient-aligned splits for WSI MIL training.

Steps:
1. Scan H5 feature files, filter Primary Tumor (01) only
2. Load clinical data, compute 3-year survival labels
3. Join slides with labels
4. Align patient splits with existing multi-omics splits
5. Output: wsi_labels.csv + wsi_splits.json

Usage:
    python generate_wsi_labels.py
    python generate_wsi_labels.py --clinical /path/to/clinical.tsv --features_dir /path/to/features
"""

import argparse
import json
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project root (relative to this script)
SCRIPT_DIR = Path(__file__).resolve().parent
WSI_ROOT = SCRIPT_DIR.parent.parent  # wsi_model/
PROJECT_ROOT = WSI_ROOT.parent  # CANCER_FOUNDATION_MODEL/

# Default paths
DEFAULT_CLINICAL = WSI_ROOT / 'data' / 'raw' / 'clinical_PANCAN_patient_with_followup.tsv'
DEFAULT_FEATURES_DIR = WSI_ROOT / 'data' / 'processed' / 'features'
DEFAULT_MULTIOMICS_SPLITS = PROJECT_ROOT / 'multiomics_model' / 'data' / 'processed' / 'train_val_test_splits.json'
DEFAULT_OUTPUT_DIR = WSI_ROOT / 'data' / 'processed'

# 3-year survival threshold (consistent with multi-omics: train_hybrid.py line 56)
THREE_YEARS_DAYS = 3 * 365.25  # 1095.75 days


def parse_tcga_barcode(h5_stem: str) -> dict:
    """
    Parse TCGA barcode from H5 filename stem.

    Example: TCGA-02-0001-01C-01-BS1.0cc8ca55-d024-440c-a4f0-01cf5b3af861
    Returns: {
        'slide_id': full stem,
        'patient_id': 'TCGA-02-0001',
        'sample_type': '01',
        'barcode': 'TCGA-02-0001-01C-01-BS1' (without UUID)
    }
    """
    # Split UUID part
    barcode = h5_stem.split('.')[0] if '.' in h5_stem else h5_stem
    parts = barcode.split('-')

    patient_id = '-'.join(parts[:3]) if len(parts) >= 3 else h5_stem
    sample_type = parts[3][:2] if len(parts) >= 4 else 'XX'

    return {
        'slide_id': h5_stem,
        'patient_id': patient_id,
        'sample_type': sample_type,
        'barcode': barcode,
    }


def scan_features(features_dir: Path) -> pd.DataFrame:
    """Scan H5 feature files and parse TCGA barcodes."""
    h5_files = sorted(features_dir.glob('*.h5'))
    logger.info(f"Found {len(h5_files)} H5 feature files")

    records = [parse_tcga_barcode(f.stem) for f in h5_files]
    df = pd.DataFrame(records)

    # Sample type distribution
    type_counts = df['sample_type'].value_counts()
    SAMPLE_TYPE_NAMES = {
        '01': 'Primary Solid Tumor',
        '02': 'Recurrent Solid Tumor',
        '06': 'Metastatic',
        '11': 'Solid Tissue Normal',
    }
    logger.info("Sample type distribution:")
    for stype, cnt in type_counts.items():
        name = SAMPLE_TYPE_NAMES.get(stype, 'Other')
        logger.info(f"  {stype} ({name}): {cnt}")

    return df


def filter_primary_tumor(slides_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to Primary Solid Tumor (sample_type == '01') only."""
    primary = slides_df[slides_df['sample_type'] == '01'].copy()
    logger.info(
        f"Primary Tumor filter: {len(slides_df)} → {len(primary)} slides "
        f"({primary['patient_id'].nunique()} patients)"
    )
    return primary


def load_clinical_data(clinical_path: Path) -> pd.DataFrame:
    """Load and clean TCGA clinical data."""
    clinical = pd.read_csv(clinical_path, sep='\t', encoding='latin-1', low_memory=False)
    logger.info(f"Clinical data loaded: {len(clinical)} rows, {len(clinical.columns)} columns")

    # Keep relevant columns
    cols = ['bcr_patient_barcode', 'vital_status', 'days_to_death',
            'days_to_last_followup', 'acronym']
    missing_cols = [c for c in cols if c not in clinical.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in clinical data: {missing_cols}")

    df = clinical[cols].copy()

    # Clean: convert survival times to numeric
    df['days_to_death'] = pd.to_numeric(df['days_to_death'], errors='coerce')
    df['days_to_last_followup'] = pd.to_numeric(df['days_to_last_followup'], errors='coerce')

    # Keep only Dead/Alive
    valid_status = df['vital_status'].isin(['Dead', 'Alive'])
    excluded = (~valid_status).sum()
    if excluded > 0:
        logger.info(f"Excluded {excluded} patients with invalid vital_status")
    df = df[valid_status].copy()

    # Deduplicate by patient (keep first occurrence)
    df = df.drop_duplicates(subset='bcr_patient_barcode', keep='first')
    logger.info(f"Clinical data after cleaning: {len(df)} unique patients")

    return df


def compute_3year_labels(clinical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 3-year survival labels consistent with multi-omics model.

    Returns DataFrame with columns: patient_id, label, cancer_type
    Censored patients (followup < 3 years, alive) are excluded.
    """
    labels = []

    for _, row in clinical_df.iterrows():
        patient_id = row['bcr_patient_barcode']
        vital_status = row['vital_status']
        days_to_death = row['days_to_death']
        days_to_followup = row['days_to_last_followup']
        cancer_type = row['acronym']

        if vital_status == 'Dead' and pd.notna(days_to_death):
            # Patient died: check if within 3 years
            label = 1 if days_to_death <= THREE_YEARS_DAYS else 0
        elif vital_status == 'Alive' and pd.notna(days_to_followup):
            if days_to_followup >= THREE_YEARS_DAYS:
                label = 0  # Alive past 3 years
            else:
                continue  # Censored before 3 years → exclude
        elif vital_status == 'Dead' and pd.isna(days_to_death) and pd.notna(days_to_followup):
            # Dead but no days_to_death; use followup
            label = 1 if days_to_followup <= THREE_YEARS_DAYS else 0
        else:
            continue  # No survival time data → exclude

        labels.append({
            'patient_id': patient_id,
            'label': int(label),
            'cancer_type': cancer_type,
        })

    labels_df = pd.DataFrame(labels)
    logger.info(
        f"3-year labels computed: {len(labels_df)} patients "
        f"(label=0: {(labels_df['label'] == 0).sum()}, label=1: {(labels_df['label'] == 1).sum()})"
    )
    return labels_df


def join_slides_labels(slides_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join slide metadata with patient labels.
    Each slide gets its patient's label (multiple slides per patient → multiple rows).
    """
    merged = slides_df.merge(labels_df, on='patient_id', how='inner')

    unmatched_slides = slides_df['patient_id'].nunique() - merged['patient_id'].nunique()
    unmatched_labels = labels_df['patient_id'].nunique() - merged['patient_id'].nunique()

    logger.info(
        f"Slide-label join: {len(merged)} slides from {merged['patient_id'].nunique()} patients"
    )
    logger.info(f"  Unmatched WSI patients (no clinical label): {unmatched_slides}")
    logger.info(f"  Unmatched clinical patients (no WSI): {unmatched_labels}")

    return merged[['slide_id', 'label', 'patient_id', 'cancer_type']].copy()


def assign_splits(
    labeled_df: pd.DataFrame,
    multiomics_splits_path: Path,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign train/val/test splits aligned with existing multi-omics splits.

    - Patients in multi-omics splits → same split
    - WSI-only patients → stratified random assignment (70/15/15)
    """
    # Load multi-omics splits
    with open(multiomics_splits_path, 'r') as f:
        omics_splits = json.load(f)

    omics_train = set(omics_splits['train_patients'])
    omics_val = set(omics_splits['val_patients'])
    omics_test = set(omics_splits['test_patients'])
    omics_all = omics_train | omics_val | omics_test

    # Get unique patients in our labeled data
    all_patients = labeled_df[['patient_id', 'cancer_type', 'label']].drop_duplicates(subset='patient_id')

    # Separate matched vs WSI-only
    matched_mask = all_patients['patient_id'].isin(omics_all)
    matched = all_patients[matched_mask]
    wsi_only = all_patients[~matched_mask]

    logger.info(f"Split alignment:")
    logger.info(f"  Matched with multi-omics splits: {len(matched)} patients")
    logger.info(f"  WSI-only (new assignment): {len(wsi_only)} patients")

    # Assign matched patients
    patient_split = {}
    for pid in matched['patient_id']:
        if pid in omics_train:
            patient_split[pid] = 'train'
        elif pid in omics_val:
            patient_split[pid] = 'val'
        elif pid in omics_test:
            patient_split[pid] = 'test'

    # Assign WSI-only patients with stratified random split
    if len(wsi_only) > 0:
        rng = np.random.RandomState(seed)

        # Group by cancer_type for stratification
        for cancer_type, group in wsi_only.groupby('cancer_type'):
            pids = group['patient_id'].values.tolist()
            rng.shuffle(pids)

            n = len(pids)
            n_train = max(1, int(n * 0.7))
            n_val = max(1, int(n * 0.15)) if n > 2 else 0
            # rest goes to test

            for i, pid in enumerate(pids):
                if i < n_train:
                    patient_split[pid] = 'train'
                elif i < n_train + n_val:
                    patient_split[pid] = 'val'
                else:
                    patient_split[pid] = 'test'

    # Map splits to slides
    labeled_df = labeled_df.copy()
    labeled_df['split'] = labeled_df['patient_id'].map(patient_split)

    # Verify no missing splits
    missing = labeled_df['split'].isna().sum()
    if missing > 0:
        logger.warning(f"{missing} slides have no split assignment (dropping)")
        labeled_df = labeled_df.dropna(subset=['split'])

    return labeled_df


def validate_and_log(df: pd.DataFrame):
    """Validate output and print summary statistics."""
    logger.info("=" * 60)
    logger.info("FINAL OUTPUT SUMMARY")
    logger.info("=" * 60)

    logger.info(f"Total slides: {len(df)}")
    logger.info(f"Unique patients: {df['patient_id'].nunique()}")
    logger.info(f"Cancer types: {df['cancer_type'].nunique()}")

    # Label distribution
    label_dist = df['label'].value_counts().to_dict()
    logger.info(f"Label distribution: {label_dist}")

    # Split distribution
    logger.info("\nPer-split statistics:")
    for split_name in ['train', 'val', 'test']:
        split_df = df[df['split'] == split_name]
        n_slides = len(split_df)
        n_patients = split_df['patient_id'].nunique()
        n_label0 = (split_df['label'] == 0).sum()
        n_label1 = (split_df['label'] == 1).sum()
        ratio = n_label1 / n_slides * 100 if n_slides > 0 else 0
        logger.info(
            f"  {split_name:>5}: {n_slides:>6} slides, {n_patients:>5} patients | "
            f"label=0: {n_label0}, label=1: {n_label1} ({ratio:.1f}% positive)"
        )

    # Multi-slide patients
    slides_per_patient = df.groupby('patient_id').size()
    multi = (slides_per_patient > 1).sum()
    logger.info(f"\nMulti-slide patients: {multi} ({multi / df['patient_id'].nunique() * 100:.1f}%)")

    # Validate: each split has both labels
    for split_name in ['train', 'val', 'test']:
        split_labels = df[df['split'] == split_name]['label'].unique()
        assert len(split_labels) >= 2, (
            f"Split '{split_name}' has only labels {split_labels}! "
            "Need both 0 and 1 for training."
        )
    logger.info("\nValidation passed: all splits have both labels (0 and 1)")

    # Validate: all sample types are 01
    barcodes = df['slide_id'].str.split('.').str[0]
    sample_types = barcodes.str.split('-').str[3].str[:2]
    assert (sample_types == '01').all(), "Non-primary-tumor slides found!"
    logger.info("Validation passed: all slides are Primary Solid Tumor (01)")

    # Patient leak check: no patient in multiple splits
    patient_splits = df[['patient_id', 'split']].drop_duplicates()
    patients_in_multiple = patient_splits.groupby('patient_id').size()
    leaked = (patients_in_multiple > 1).sum()
    assert leaked == 0, f"Patient data leak detected: {leaked} patients in multiple splits!"
    logger.info("Validation passed: no patient appears in multiple splits")


def save_outputs(df: pd.DataFrame, output_dir: Path):
    """Save labels CSV and splits JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Labels CSV
    csv_path = output_dir / 'wsi_labels.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved labels CSV: {csv_path}")

    # 2. Splits JSON
    splits = {
        'train_patients': sorted(df[df['split'] == 'train']['patient_id'].unique().tolist()),
        'val_patients': sorted(df[df['split'] == 'val']['patient_id'].unique().tolist()),
        'test_patients': sorted(df[df['split'] == 'test']['patient_id'].unique().tolist()),
        'train_slides': sorted(df[df['split'] == 'train']['slide_id'].tolist()),
        'val_slides': sorted(df[df['split'] == 'val']['slide_id'].tolist()),
        'test_slides': sorted(df[df['split'] == 'test']['slide_id'].tolist()),
        'metadata': {
            'total_slides': int(len(df)),
            'total_patients': int(df['patient_id'].nunique()),
            'train_slides': int((df['split'] == 'train').sum()),
            'val_slides': int((df['split'] == 'val').sum()),
            'test_slides': int((df['split'] == 'test').sum()),
            'train_patients': int(df[df['split'] == 'train']['patient_id'].nunique()),
            'val_patients': int(df[df['split'] == 'val']['patient_id'].nunique()),
            'test_patients': int(df[df['split'] == 'test']['patient_id'].nunique()),
            'label_distribution': {int(k): int(v) for k, v in df['label'].value_counts().items()},
            'survival_threshold_days': float(THREE_YEARS_DAYS),
            'sample_type_filter': '01 (Primary Solid Tumor)',
        },
    }

    json_path = output_dir / 'wsi_splits.json'
    with open(json_path, 'w') as f:
        json.dump(splits, f, indent=2)
    logger.info(f"Saved splits JSON: {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate WSI labels and splits for MIL training')
    parser.add_argument('--clinical', type=str, default=str(DEFAULT_CLINICAL),
                        help='Path to clinical TSV file')
    parser.add_argument('--features_dir', type=str, default=str(DEFAULT_FEATURES_DIR),
                        help='Path to H5 features directory')
    parser.add_argument('--multiomics_splits', type=str, default=str(DEFAULT_MULTIOMICS_SPLITS),
                        help='Path to multi-omics splits JSON')
    parser.add_argument('--output_dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help='Output directory')
    parser.add_argument('--survival_years', type=float, default=3.0,
                        help='Survival threshold in years (default: 3.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for WSI-only patient split assignment')

    args = parser.parse_args()

    # Update threshold if custom
    global THREE_YEARS_DAYS
    THREE_YEARS_DAYS = args.survival_years * 365.25

    features_dir = Path(args.features_dir)
    clinical_path = Path(args.clinical)
    multiomics_splits_path = Path(args.multiomics_splits)
    output_dir = Path(args.output_dir)

    logger.info("WSI Labels Generator")
    logger.info(f"  Clinical data: {clinical_path}")
    logger.info(f"  Features dir: {features_dir}")
    logger.info(f"  Multi-omics splits: {multiomics_splits_path}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Survival threshold: {THREE_YEARS_DAYS} days ({args.survival_years} years)")

    # Step 1: Scan H5 files and filter Primary Tumor
    slides_df = scan_features(features_dir)
    slides_df = filter_primary_tumor(slides_df)

    # Step 2: Load clinical data and compute labels
    clinical_df = load_clinical_data(clinical_path)
    labels_df = compute_3year_labels(clinical_df)

    # Step 3: Join slides with labels
    labeled_df = join_slides_labels(slides_df, labels_df)

    # Step 4: Assign splits
    labeled_df = assign_splits(labeled_df, multiomics_splits_path, seed=args.seed)

    # Step 5: Validate and save
    validate_and_log(labeled_df)
    save_outputs(labeled_df, output_dir)

    logger.info("\nDone!")


if __name__ == '__main__':
    main()
