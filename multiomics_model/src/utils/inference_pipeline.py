#!/usr/bin/env python3
"""
Inference Pipeline for HybridMultiModalModel
=============================================

새로운 환자 데이터로 3년 생존 예측을 수행하는 추론 파이프라인.

Usage:
    from multiomics_model.src.utils.inference_pipeline import InferencePipeline

    pipeline = InferencePipeline(
        model_checkpoint='path/to/best_model.pth',
        cox_coef_path='path/to/cox_coefficients.parquet'
    )

    result = pipeline.predict(
        expression_df=expr_df,
        methylation_df=meth_df,
        clinical_df=clinical_df,
        cancer_type='BRCA'
    )
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import logging


class InferencePipeline:
    """
    HybridMultiModalModel 추론 파이프라인

    새로운 환자 데이터를 모델 입력 형식으로 변환하고 예측 수행
    """

    # Clinical categories (age_group, sex, race, stage, grade)
    CLINICAL_CATEGORIES = (10, 2, 6, 5, 4)

    # Feature column order (must match training)
    CLINICAL_COLS = ['age_group', 'sex', 'race', 'ajcc_pathologic_stage', 'grade']

    def __init__(
        self,
        model_checkpoint: str,
        cox_coef_path: str,
        feature_order_path: Optional[str] = None,
        device: str = 'cuda',
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            model_checkpoint: Path to trained model checkpoint (.pth)
            cox_coef_path: Path to Cox coefficients parquet file
            feature_order_path: Path to feature order JSON (optional)
            device: 'cuda' or 'cpu'
            logger: Optional logger instance
        """
        self.logger = logger or self._setup_logger()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load Cox coefficients
        self.cox_coef_path = Path(cox_coef_path)
        self.cox_coefficients = self._load_cox_coefficients()

        # Load model
        self.model_checkpoint = Path(model_checkpoint)
        self.model = self._load_model()

        # Feature order (from training)
        self.feature_order = self._load_feature_order(feature_order_path)

        self.logger.info(f"InferencePipeline initialized on {self.device}")

    def _setup_logger(self) -> logging.Logger:
        """Setup default logger"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _load_cox_coefficients(self) -> pd.DataFrame:
        """Load Cox coefficients lookup table"""
        if not self.cox_coef_path.exists():
            raise FileNotFoundError(f"Cox coefficients not found: {self.cox_coef_path}")

        cox_coef = pd.read_parquet(self.cox_coef_path)
        self.logger.info(f"Loaded Cox coefficients: {cox_coef.shape}")
        return cox_coef

    def _load_model(self):
        """Load trained model from checkpoint"""
        from ..models.hybrid_fc_tabtransformer import HybridMultiModalModel

        if not self.model_checkpoint.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_checkpoint}")

        # Load checkpoint to CPU first
        checkpoint = torch.load(self.model_checkpoint, map_location='cpu', weights_only=False)

        # Get model dimensions from checkpoint
        state_dict = checkpoint['model_state_dict']
        cox_input_dim = state_dict['cox_encoder.encoder.0.weight'].shape[1]
        meth_input_dim = state_dict['meth_encoder.encoder.0.weight'].shape[1]

        # Create model
        model = HybridMultiModalModel(
            clinical_categories=self.CLINICAL_CATEGORIES,
            cox_input_dim=cox_input_dim,
            cox_hidden_dims=(2048, 512, 256),
            meth_input_dim=meth_input_dim,
            meth_hidden_dims=(4096, 1024, 256),
            dim=128,
            depth=6,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            encoder_dropout=0.3,
            dim_out=1
        )

        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        self.cox_input_dim = cox_input_dim
        self.meth_input_dim = meth_input_dim

        self.logger.info(f"Model loaded: Cox dim={cox_input_dim}, Meth dim={meth_input_dim}")
        return model

    def _load_feature_order(self, feature_order_path: Optional[str]) -> Optional[Dict]:
        """Load feature order from training"""
        if feature_order_path and Path(feature_order_path).exists():
            with open(feature_order_path, 'r') as f:
                return json.load(f)
        return None

    def preprocess_clinical(self, clinical_df: pd.DataFrame) -> torch.Tensor:
        """
        Preprocess clinical data to categorical tensor

        Args:
            clinical_df: DataFrame with columns [age, sex, race, stage, grade] or raw TCGA columns

        Returns:
            clinical_cat: (n_patients, 5) tensor
        """
        df = clinical_df.copy()

        # Age group encoding (10 bins: 0-9)
        if 'age' in df.columns and 'age_group' not in df.columns:
            age = pd.to_numeric(df['age'], errors='coerce').fillna(60)
            df['age_group'] = pd.cut(
                age,
                bins=[0, 30, 40, 50, 55, 60, 65, 70, 75, 80, 120],
                labels=range(10),
                right=False
            ).astype(int)
        elif 'age_at_initial_pathologic_diagnosis' in df.columns:
            age = pd.to_numeric(df['age_at_initial_pathologic_diagnosis'], errors='coerce').fillna(60)
            df['age_group'] = pd.cut(
                age,
                bins=[0, 30, 40, 50, 55, 60, 65, 70, 75, 80, 120],
                labels=range(10),
                right=False
            ).astype(int)

        # Sex encoding (0=MALE, 1=FEMALE)
        if 'gender' in df.columns and 'sex' not in df.columns:
            df['sex'] = df['gender'].map({'MALE': 0, 'FEMALE': 1, 'M': 0, 'F': 1}).fillna(0).astype(int)
        elif 'sex' in df.columns:
            if df['sex'].dtype == object:
                df['sex'] = df['sex'].map({'MALE': 0, 'FEMALE': 1, 'M': 0, 'F': 1}).fillna(0).astype(int)

        # Race encoding (0-5)
        if 'race' in df.columns:
            if df['race'].dtype == object:
                race_map = {
                    'WHITE': 0, 'BLACK OR AFRICAN AMERICAN': 1, 'ASIAN': 2,
                    'AMERICAN INDIAN OR ALASKA NATIVE': 3,
                    'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER': 4,
                    '[Not Available]': 5, '[Not Evaluated]': 5, '[Unknown]': 5
                }
                df['race'] = df['race'].map(race_map).fillna(5).astype(int)

        # Stage encoding (0-4: I, II, III, IV, NA)
        if 'pathologic_stage' in df.columns and 'ajcc_pathologic_stage' not in df.columns:
            stage_map = {
                'Stage I': 0, 'Stage IA': 0, 'Stage IB': 0, 'I': 0,
                'Stage II': 1, 'Stage IIA': 1, 'Stage IIB': 1, 'II': 1,
                'Stage III': 2, 'Stage IIIA': 2, 'Stage IIIB': 2, 'Stage IIIC': 2, 'III': 2,
                'Stage IV': 3, 'Stage IVA': 3, 'Stage IVB': 3, 'IV': 3,
            }
            df['ajcc_pathologic_stage'] = df['pathologic_stage'].map(stage_map).fillna(4).astype(int)
        elif 'stage' in df.columns:
            stage_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3}
            df['ajcc_pathologic_stage'] = df['stage'].map(stage_map).fillna(4).astype(int)

        # Grade encoding (0-3: G1, G2, G3, G4)
        if 'neoplasm_histologic_grade' in df.columns and 'grade' not in df.columns:
            grade_map = {'G1': 0, 'G2': 1, 'G3': 2, 'G4': 3, 'GX': 2}
            df['grade'] = df['neoplasm_histologic_grade'].map(grade_map).fillna(2).astype(int)
        elif 'grade' in df.columns:
            if df['grade'].dtype == object:
                grade_map = {'G1': 0, 'G2': 1, 'G3': 2, 'G4': 3, '1': 0, '2': 1, '3': 2, '4': 3}
                df['grade'] = df['grade'].map(grade_map).fillna(2).astype(int)

        # Ensure all columns exist with defaults
        for col in self.CLINICAL_COLS:
            if col not in df.columns:
                self.logger.warning(f"Missing clinical column: {col}, using default")
                if col == 'age_group':
                    df[col] = 5  # 60-65 age group
                elif col == 'sex':
                    df[col] = 0  # MALE
                elif col == 'race':
                    df[col] = 5  # Unknown
                elif col == 'ajcc_pathologic_stage':
                    df[col] = 4  # NA
                elif col == 'grade':
                    df[col] = 2  # G3

        # Extract and validate
        clinical_cat = df[self.CLINICAL_COLS].values.astype(np.int64)

        # Clip to valid ranges
        for i, (col, max_val) in enumerate(zip(self.CLINICAL_COLS, self.CLINICAL_CATEGORIES)):
            clinical_cat[:, i] = np.clip(clinical_cat[:, i], 0, max_val - 1)

        return torch.from_numpy(clinical_cat).long()

    def create_cox_omics_tensor(
        self,
        omics_data: Dict[str, pd.DataFrame],
        cancer_type: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create Cox omics tensor with [value, cox] pairs

        Args:
            omics_data: Dict with keys 'expression', 'cnv', 'microrna', 'rppa', 'mutations'
            cancer_type: Cancer type acronym (e.g., 'BRCA', 'LUAD')

        Returns:
            cox_omics: (n_patients, cox_input_dim) tensor
            cox_mask: (n_patients,) boolean tensor
        """
        n_patients = None
        patient_ids = None

        # Determine patient count and IDs
        for data_type, df in omics_data.items():
            if df is not None and len(df) > 0:
                if n_patients is None:
                    n_patients = len(df)
                    patient_ids = df.index.tolist()
                break

        if n_patients is None:
            # No omics data available
            return (
                torch.zeros(1, self.cox_input_dim),
                torch.tensor([False])
            )

        # Get Cox coefficients for this cancer type
        if cancer_type not in self.cox_coefficients.columns:
            available = list(self.cox_coefficients.columns)
            self.logger.warning(f"Cancer type {cancer_type} not found. Available: {available[:5]}...")
            # Use mean across all cancer types
            cox_coef_series = self.cox_coefficients.mean(axis=1)
        else:
            cox_coef_series = self.cox_coefficients[cancer_type]

        # Build [value, cox] pairs for each omics type
        all_features = []

        for data_type in ['expression', 'cnv', 'microrna', 'rppa', 'mutations']:
            if data_type not in omics_data or omics_data[data_type] is None:
                continue

            df = omics_data[data_type]

            # Apply log2 transformation where needed
            if data_type in ['expression', 'microrna']:
                df = np.log2(df + 1)
            elif data_type in ['cnv', 'rppa']:
                df = np.log2(df - df.min().min() + 1)
            # mutations: no transformation (impact scores 0-2)

            # Create [value, cox] pairs
            for col in df.columns:
                # Feature name for Cox lookup
                feature_name = f"{data_type}_{col}"

                # Get Cox coefficient
                if feature_name in cox_coef_series.index:
                    cox_coef = cox_coef_series[feature_name]
                else:
                    cox_coef = 0.0  # Default if not found

                # Add value and cox coefficient
                values = df[col].values.astype(np.float32)
                cox_values = np.full_like(values, cox_coef)

                all_features.append(values)
                all_features.append(cox_values)

        if not all_features:
            return (
                torch.zeros(n_patients, self.cox_input_dim),
                torch.tensor([False] * n_patients)
            )

        # Stack features
        cox_omics = np.column_stack(all_features)

        # Pad or truncate to match expected dimension
        if cox_omics.shape[1] < self.cox_input_dim:
            padding = np.zeros((n_patients, self.cox_input_dim - cox_omics.shape[1]))
            cox_omics = np.hstack([cox_omics, padding])
        elif cox_omics.shape[1] > self.cox_input_dim:
            cox_omics = cox_omics[:, :self.cox_input_dim]

        # Handle NaN/Inf
        cox_omics = np.nan_to_num(cox_omics, nan=0.0, posinf=0.0, neginf=0.0)

        cox_mask = torch.tensor([True] * n_patients, dtype=torch.bool)

        return torch.from_numpy(cox_omics).float(), cox_mask

    def create_methylation_tensor(
        self,
        methylation_df: Optional[pd.DataFrame]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create methylation tensor

        Args:
            methylation_df: DataFrame with CpG sites as columns, patients as rows

        Returns:
            methylation: (n_patients, meth_input_dim) tensor
            meth_mask: (n_patients,) boolean tensor
        """
        if methylation_df is None or len(methylation_df) == 0:
            return (
                torch.zeros(1, self.meth_input_dim),
                torch.tensor([False])
            )

        n_patients = len(methylation_df)
        meth_values = methylation_df.values.astype(np.float32)

        # Pad or truncate to match expected dimension
        if meth_values.shape[1] < self.meth_input_dim:
            padding = np.zeros((n_patients, self.meth_input_dim - meth_values.shape[1]))
            meth_values = np.hstack([meth_values, padding])
        elif meth_values.shape[1] > self.meth_input_dim:
            meth_values = meth_values[:, :self.meth_input_dim]

        # Handle NaN/Inf (beta values should be 0-1)
        meth_values = np.nan_to_num(meth_values, nan=0.5, posinf=1.0, neginf=0.0)
        meth_values = np.clip(meth_values, 0, 1)

        meth_mask = torch.tensor([True] * n_patients, dtype=torch.bool)

        return torch.from_numpy(meth_values).float(), meth_mask

    @torch.no_grad()
    def predict(
        self,
        clinical_df: pd.DataFrame,
        cancer_type: str,
        expression_df: Optional[pd.DataFrame] = None,
        cnv_df: Optional[pd.DataFrame] = None,
        microrna_df: Optional[pd.DataFrame] = None,
        rppa_df: Optional[pd.DataFrame] = None,
        mutations_df: Optional[pd.DataFrame] = None,
        methylation_df: Optional[pd.DataFrame] = None,
        return_features: bool = False
    ) -> Dict:
        """
        Predict 3-year survival probability

        Args:
            clinical_df: Clinical data DataFrame
            cancer_type: Cancer type acronym (e.g., 'BRCA')
            expression_df: Gene expression data (optional)
            cnv_df: Copy number variation data (optional)
            microrna_df: microRNA expression data (optional)
            rppa_df: Protein expression data (optional)
            mutations_df: Mutation data (optional)
            methylation_df: Methylation beta values (optional)
            return_features: Whether to return intermediate features

        Returns:
            result: Dict with predictions and metadata
        """
        self.model.eval()

        # Preprocess clinical data
        clinical_cat = self.preprocess_clinical(clinical_df).to(self.device)
        n_patients = len(clinical_df)

        # Create Cox omics tensor
        omics_data = {
            'expression': expression_df,
            'cnv': cnv_df,
            'microrna': microrna_df,
            'rppa': rppa_df,
            'mutations': mutations_df
        }
        cox_omics, cox_mask = self.create_cox_omics_tensor(omics_data, cancer_type)
        cox_omics = cox_omics.to(self.device)
        cox_mask = cox_mask.to(self.device)

        # Adjust dimensions if needed
        if len(cox_omics) != n_patients:
            if len(cox_omics) == 1:
                cox_omics = cox_omics.expand(n_patients, -1)
                cox_mask = cox_mask.expand(n_patients)

        # Create methylation tensor
        methylation, meth_mask = self.create_methylation_tensor(methylation_df)
        methylation = methylation.to(self.device)
        meth_mask = meth_mask.to(self.device)

        # Adjust dimensions if needed
        if len(methylation) != n_patients:
            if len(methylation) == 1:
                methylation = methylation.expand(n_patients, -1)
                meth_mask = meth_mask.expand(n_patients)

        # Forward pass
        logits, features = self.model(
            clinical_cat, cox_omics, methylation, cox_mask, meth_mask
        )

        # Convert to probabilities
        probabilities = torch.sigmoid(logits).cpu().numpy().flatten()

        # Build result
        result = {
            'survival_probability': 1 - probabilities,  # P(survive 3 years)
            'death_probability': probabilities,  # P(death within 3 years)
            'risk_score': logits.cpu().numpy().flatten(),
            'n_patients': n_patients,
            'cancer_type': cancer_type,
            'cox_available': cox_mask.cpu().numpy().tolist(),
            'meth_available': meth_mask.cpu().numpy().tolist()
        }

        if return_features:
            result['features'] = {
                'cox_encoded': features['cox_encoded'].cpu().numpy(),
                'meth_encoded': features['meth_encoded'].cpu().numpy(),
                'continuous': features['continuous'].cpu().numpy()
            }

        return result

    def predict_single(
        self,
        age: int,
        sex: str,
        race: str,
        stage: str,
        grade: str,
        cancer_type: str,
        expression: Optional[Dict[str, float]] = None,
        methylation: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Predict for a single patient (convenience method)

        Args:
            age: Patient age
            sex: 'MALE' or 'FEMALE'
            race: Race category
            stage: Cancer stage ('I', 'II', 'III', 'IV')
            grade: Tumor grade ('G1', 'G2', 'G3', 'G4')
            cancer_type: Cancer type acronym
            expression: Dict of gene -> expression value (optional)
            methylation: Dict of CpG site -> beta value (optional)

        Returns:
            result: Prediction result
        """
        # Create clinical DataFrame
        clinical_df = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'race': race,
            'stage': stage,
            'grade': grade
        }])

        # Create expression DataFrame if provided
        expression_df = None
        if expression:
            expression_df = pd.DataFrame([expression])

        # Create methylation DataFrame if provided
        methylation_df = None
        if methylation:
            methylation_df = pd.DataFrame([methylation])

        return self.predict(
            clinical_df=clinical_df,
            cancer_type=cancer_type,
            expression_df=expression_df,
            methylation_df=methylation_df
        )


def main():
    """Test inference pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description='HybridMultiModalModel Inference')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    parser.add_argument('--cox-coef', required=True, help='Cox coefficients path')
    parser.add_argument('--test', action='store_true', help='Run test prediction')

    args = parser.parse_args()

    pipeline = InferencePipeline(
        model_checkpoint=args.checkpoint,
        cox_coef_path=args.cox_coef
    )

    if args.test:
        # Test with dummy data
        result = pipeline.predict_single(
            age=55,
            sex='FEMALE',
            race='WHITE',
            stage='II',
            grade='G2',
            cancer_type='BRCA'
        )

        print("\nTest Prediction Result:")
        print(f"  Survival probability: {result['survival_probability'][0]:.2%}")
        print(f"  Death probability: {result['death_probability'][0]:.2%}")
        print(f"  Risk score: {result['risk_score'][0]:.4f}")


if __name__ == "__main__":
    main()
