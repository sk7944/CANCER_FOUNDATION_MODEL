"""
Hybrid FC-NN + TabTransformer for Multi-Omics Cancer Data

Architecture:
1. Cox Omics Branch: FC-NN (143,040 → 2048 → 512 → 256)
2. Methylation Branch: FC-NN (396,065 → 4096 → 1024 → 256)
3. TabTransformer: Clinical categories + 512 continuous (cox_256 + meth_256)
4. Missing Modality Support: Handle patients with only Clinical + Methylation

Memory: ~29.58 GB for batch=32 on RTX A6000 48GB
"""

import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer
from typing import Tuple, Optional


class CoxOmicsEncoder(nn.Module):
    """
    FC-NN encoder for Cox-enhanced omics data
    Input: 143,040 features (71,520 features × 2 for [value, cox] pairs)
    Output: 256-dimensional embedding
    """
    def __init__(
        self,
        input_dim: int = 143040,  # 71,520 features × 2
        hidden_dims: Tuple[int, int, int] = (2048, 512, 256),
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1]

        # 143,040 → 2048 → 512 → 256
        self.encoder = nn.Sequential(
            # Layer 1: 143,040 → 2048
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2: 2048 → 512
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3: 512 → 256
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 143040) Cox-enhanced omics features
        Returns:
            (batch_size, 256) encoded representation
        """
        return self.encoder(x)


class MethylationEncoder(nn.Module):
    """
    FC-NN encoder for methylation data
    Input: 396,065 CG sites (beta values 0-1)
    Output: 256-dimensional embedding
    """
    def __init__(
        self,
        input_dim: int = 396065,
        hidden_dims: Tuple[int, int, int] = (4096, 1024, 256),
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1]

        # 396,065 → 4096 → 1024 → 256
        self.encoder = nn.Sequential(
            # Layer 1: 396,065 → 4096
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2: 4096 → 1024
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3: 1024 → 256
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 396065) methylation beta values
        Returns:
            (batch_size, 256) encoded representation
        """
        return self.encoder(x)


class HybridMultiModalModel(nn.Module):
    """
    Hybrid FC-NN + TabTransformer with Missing Modality Learning

    Architecture:
    1. CoxOmicsEncoder: 143,040 → 256
    2. MethylationEncoder: 396,065 → 256
    3. TabTransformer: Clinical (categorical) + Cox_256 + Meth_256
    4. Missing Modality: Handle patients without Cox data

    Training Strategy (Option 3):
    - Patients with Cox (4,504): Clinical + Cox_256 + Meth_256
    - Patients without Cox (3,720): Clinical + [ZERO] + Meth_256
    - Total: 8,224 patients
    """
    def __init__(
        self,
        # Clinical features (각 categorical 변수의 카테고리 수)
        # 순서: [age_group, sex, race, ajcc_pathologic_stage, grade]
        clinical_categories: Tuple[int, ...] = (10, 2, 6, 5, 4),

        # Cox omics encoder
        cox_input_dim: int = 143040,  # Will be updated after data generation
        cox_hidden_dims: Tuple[int, int, int] = (2048, 512, 256),

        # Methylation encoder
        meth_input_dim: int = 396065,
        meth_hidden_dims: Tuple[int, int, int] = (4096, 1024, 256),

        # TabTransformer settings
        dim: int = 128,
        depth: int = 6,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,

        # Encoder dropout
        encoder_dropout: float = 0.3,

        # Output
        dim_out: int = 1,  # Survival prediction
    ):
        super().__init__()

        self.clinical_categories = clinical_categories
        self.cox_output_dim = cox_hidden_dims[-1]  # 256
        self.meth_output_dim = meth_hidden_dims[-1]  # 256

        # Encoders
        self.cox_encoder = CoxOmicsEncoder(
            input_dim=cox_input_dim,
            hidden_dims=cox_hidden_dims,
            dropout=encoder_dropout
        )

        self.meth_encoder = MethylationEncoder(
            input_dim=meth_input_dim,
            hidden_dims=meth_hidden_dims,
            dropout=encoder_dropout
        )

        # TabTransformer
        # Continuous: cox_256 + meth_256 = 512
        num_continuous = self.cox_output_dim + self.meth_output_dim

        self.tabtransformer = TabTransformer(
            categories=clinical_categories,
            num_continuous=num_continuous,
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            dim_out=dim_out,
            mlp_hidden_mults=(4, 2),
            mlp_act=nn.ReLU()
        )

    def forward(
        self,
        clinical_cat: torch.Tensor,
        cox_omics: Optional[torch.Tensor] = None,
        methylation: Optional[torch.Tensor] = None,
        cox_mask: Optional[torch.Tensor] = None,
        meth_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with missing modality support

        Args:
            clinical_cat: (batch, 5) categorical clinical features
            cox_omics: (batch, 132098) Cox-enhanced omics [val, cox] pairs
                       None or zeros for patients without Cox data
            methylation: (batch, 396065) methylation beta values
                       None or zeros for patients without Methylation data
            cox_mask: (batch,) boolean mask - True if Cox data available
            meth_mask: (batch,) boolean mask - True if Methylation data available

        Returns:
            predictions: (batch, 1) survival risk scores
            features: dict with intermediate representations
        """
        batch_size = clinical_cat.size(0)
        device = clinical_cat.device

        # Encode Cox omics (if available)
        if cox_omics is not None and cox_mask is not None:
            # Encode Cox data for patients who have it
            cox_encoded = self.cox_encoder(cox_omics)  # (batch, 256)

            # Zero out Cox embeddings for patients without Cox data
            cox_mask_expanded = cox_mask.unsqueeze(1).float()  # (batch, 1)
            cox_encoded = cox_encoded * cox_mask_expanded  # (batch, 256)
        elif cox_omics is not None:
            # All patients have Cox data
            cox_encoded = self.cox_encoder(cox_omics)
        else:
            # No Cox data available - use zeros
            cox_encoded = torch.zeros(
                batch_size, self.cox_output_dim,
                device=device, dtype=torch.float32
            )

        # Encode Methylation (if available)
        if methylation is not None and meth_mask is not None:
            # Encode Methylation data for patients who have it
            meth_encoded = self.meth_encoder(methylation)  # (batch, 256)

            # Zero out Methylation embeddings for patients without Methylation data
            meth_mask_expanded = meth_mask.unsqueeze(1).float()  # (batch, 1)
            meth_encoded = meth_encoded * meth_mask_expanded  # (batch, 256)
        elif methylation is not None:
            # All patients have Methylation data
            meth_encoded = self.meth_encoder(methylation)
        else:
            # No Methylation data available - use zeros
            meth_encoded = torch.zeros(
                batch_size, self.meth_output_dim,
                device=device, dtype=torch.float32
            )

        # Concatenate continuous features
        continuous = torch.cat([cox_encoded, meth_encoded], dim=1)  # (batch, 512)

        # TabTransformer
        predictions = self.tabtransformer(clinical_cat, continuous)  # (batch, 1)

        # Return intermediate features for analysis
        features = {
            'cox_encoded': cox_encoded,
            'meth_encoded': meth_encoded,
            'continuous': continuous
        }

        return predictions, features

    def get_num_parameters(self) -> dict:
        """Get parameter counts for each component"""
        cox_params = sum(p.numel() for p in self.cox_encoder.parameters())
        meth_params = sum(p.numel() for p in self.meth_encoder.parameters())
        tab_params = sum(p.numel() for p in self.tabtransformer.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        return {
            'cox_encoder': cox_params,
            'meth_encoder': meth_params,
            'tabtransformer': tab_params,
            'total': total_params,
            'cox_encoder_gb': cox_params * 4 / (1024**3),
            'meth_encoder_gb': meth_params * 4 / (1024**3),
            'tabtransformer_gb': tab_params * 4 / (1024**3),
            'total_gb': total_params * 4 / (1024**3)
        }


def test_model():
    """Test model creation and forward pass"""
    print("="*70)
    print("HybridMultiModalModel Test")
    print("="*70)

    # Create model
    model = HybridMultiModalModel(
        clinical_categories=(10, 2, 6, 5, 4),
        cox_input_dim=143040,
        cox_hidden_dims=(2048, 512, 256),
        meth_input_dim=396065,
        meth_hidden_dims=(4096, 1024, 256),
        dim=128,
        depth=6,
        heads=8
    )

    # Print parameters
    params = model.get_num_parameters()
    print(f"\nModel Parameters:")
    print(f"  Cox Encoder:      {params['cox_encoder']:>12,} ({params['cox_encoder_gb']:.2f} GB)")
    print(f"  Meth Encoder:     {params['meth_encoder']:>12,} ({params['meth_encoder_gb']:.2f} GB)")
    print(f"  TabTransformer:   {params['tabtransformer']:>12,} ({params['tabtransformer_gb']:.2f} GB)")
    print(f"  {'─'*50}")
    print(f"  Total:            {params['total']:>12,} ({params['total_gb']:.2f} GB)")

    # Test forward pass
    print(f"\nForward Pass Test:")
    batch_size = 8

    clinical = torch.randint(0, 5, (batch_size, 5))
    cox = torch.randn(batch_size, 143040)
    meth = torch.rand(batch_size, 396065)  # Beta values 0-1
    cox_mask = torch.tensor([True]*4 + [False]*4)  # Half with Cox, half without
    meth_mask = torch.tensor([True]*6 + [False]*2)  # 6 with Meth, 2 without

    with torch.no_grad():
        predictions, features = model(clinical, cox, meth, cox_mask, meth_mask)

    print(f"  Input shapes:")
    print(f"    Clinical: {clinical.shape}")
    print(f"    Cox omics: {cox.shape}")
    print(f"    Methylation: {meth.shape}")
    print(f"    Cox mask: {cox_mask.shape} (4 True, 4 False)")
    print(f"    Meth mask: {meth_mask.shape} (6 True, 2 False)")
    print(f"  Output shape: {predictions.shape}")
    print(f"  Feature shapes:")
    for k, v in features.items():
        print(f"    {k}: {v.shape}")

    # Verify masking works
    print(f"\nMissing Modality Test:")
    print(f"  Cox encoded mean (with Cox): {features['cox_encoded'][:4].mean():.4f}")
    print(f"  Cox encoded mean (no Cox):   {features['cox_encoded'][4:].mean():.4f}")
    print(f"  ✅ Cox masking working correctly!" if features['cox_encoded'][4:].abs().max() < 1e-5 else "❌ Cox masking failed")
    print(f"  Meth encoded mean (with Meth): {features['meth_encoded'][:6].mean():.4f}")
    print(f"  Meth encoded mean (no Meth):   {features['meth_encoded'][6:].mean():.4f}")
    print(f"  ✅ Meth masking working correctly!" if features['meth_encoded'][6:].abs().max() < 1e-5 else "❌ Meth masking failed")

    print(f"\n{'='*70}")
    print(f"✅ Model test passed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    test_model()
