"""
Multiple Instance Learning Models for WSI Classification
=========================================================

Implements:
- ABMIL: Attention-based MIL (Ilse et al., 2018)
- TransMIL: Transformer-based MIL (Shao et al., 2021)

Reference:
- Ilse et al. "Attention-based Deep Multiple Instance Learning" ICML 2018
- Shao et al. "TransMIL: Transformer based Correlated Multiple Instance Learning
               for Whole Slide Image Classification" NeurIPS 2021
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MILModelConfig:
    """Configuration for MIL models."""
    # Input
    feature_dim: int = 768  # Swin-T output dimension

    # Model architecture
    hidden_dim: int = 256
    num_classes: int = 1  # Binary classification (survival)
    dropout: float = 0.25

    # ABMIL specific
    attention_dim: int = 128
    use_gated_attention: bool = True

    # TransMIL specific
    num_heads: int = 8
    num_layers: int = 2

    # Output
    return_attention: bool = True


class GatedAttention(nn.Module):
    """
    Gated Attention mechanism for ABMIL.

    Computes attention weights using:
    a = softmax(V * tanh(U * h) ⊙ sigmoid(W * h))

    Args:
        feature_dim: Input feature dimension
        hidden_dim: Hidden dimension for attention
        dropout: Dropout rate
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.attention_U = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.attention_W = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gated attention weights.

        Args:
            features: (batch, num_patches, feature_dim)
            mask: (batch, num_patches), True = valid, False = padding

        Returns:
            aggregated: (batch, feature_dim) - weighted sum of features
            attention: (batch, num_patches) - attention weights
        """
        # Compute attention scores
        U = self.attention_U(features)  # (batch, num_patches, hidden_dim)
        V = self.attention_V(features)  # (batch, num_patches, hidden_dim)

        scores = self.attention_W(U * V)  # (batch, num_patches, 1)
        scores = scores.squeeze(-1)  # (batch, num_patches)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax to get attention weights
        attention = F.softmax(scores, dim=1)  # (batch, num_patches)
        attention = self.dropout(attention)

        # Weighted sum
        aggregated = torch.bmm(
            attention.unsqueeze(1),  # (batch, 1, num_patches)
            features  # (batch, num_patches, feature_dim)
        ).squeeze(1)  # (batch, feature_dim)

        return aggregated, attention


class ABMIL(nn.Module):
    """
    Attention-based Multiple Instance Learning.

    Architecture:
        features -> FC -> Attention -> Weighted Sum -> FC -> Output

    Args:
        config: MILModelConfig
    """

    def __init__(self, config: MILModelConfig):
        super().__init__()
        self.config = config

        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Attention mechanism
        if config.use_gated_attention:
            self.attention = GatedAttention(
                feature_dim=config.hidden_dim,
                hidden_dim=config.attention_dim,
                dropout=config.dropout,
            )
        else:
            self.attention = nn.Sequential(
                nn.Linear(config.hidden_dim, config.attention_dim),
                nn.Tanh(),
                nn.Linear(config.attention_dim, 1),
            )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: (batch, num_patches, feature_dim)
            mask: (batch, num_patches), True = valid

        Returns:
            dict with:
                logits: (batch, num_classes)
                attention: (batch, num_patches) if return_attention
        """
        # Project features
        h = self.feature_projection(features)  # (batch, num_patches, hidden_dim)

        # Compute attention and aggregate
        if isinstance(self.attention, GatedAttention):
            aggregated, attention = self.attention(h, mask)
        else:
            # Simple attention
            scores = self.attention(h).squeeze(-1)  # (batch, num_patches)
            if mask is not None:
                scores = scores.masked_fill(~mask, float('-inf'))
            attention = F.softmax(scores, dim=1)
            aggregated = torch.bmm(attention.unsqueeze(1), h).squeeze(1)

        # Classify
        logits = self.classifier(aggregated)  # (batch, num_classes)

        output = {'logits': logits}
        if self.config.return_attention:
            output['attention'] = attention

        return output


class TransformerLayer(nn.Module):
    """
    Transformer layer for TransMIL.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            mask: (batch, seq_len), True = valid
        """
        # Convert mask to attention mask format
        attn_mask = None
        if mask is not None:
            # For MultiheadAttention: (batch * num_heads, seq_len, seq_len)
            # We use key_padding_mask instead: (batch, seq_len), True = ignore
            key_padding_mask = ~mask
        else:
            key_padding_mask = None

        # Self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class PPEG(nn.Module):
    """
    Pyramid Position Encoding Generator for TransMIL.
    Provides spatial inductive bias without explicit position information.
    """

    def __init__(self, dim: int):
        super().__init__()

        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            H, W: spatial dimensions (sqrt(seq_len))
        """
        B, N, C = x.shape

        # Skip class token if present
        cls_token = None
        if N != H * W:
            cls_token = x[:, :1]
            x = x[:, 1:]

        # Reshape to 2D
        x = x.transpose(1, 2).reshape(B, C, H, W)

        # Multi-scale convolution
        x = self.proj(x) + self.proj1(x) + self.proj2(x)

        # Reshape back
        x = x.flatten(2).transpose(1, 2)

        # Add back class token
        if cls_token is not None:
            x = torch.cat([cls_token, x], dim=1)

        return x


class TransMIL(nn.Module):
    """
    Transformer-based Multiple Instance Learning.

    Architecture:
        features -> Projection -> Transformer Layers -> CLS Token -> Output

    Reference:
        Shao et al. "TransMIL: Transformer based Correlated Multiple Instance Learning
        for Whole Slide Image Classification" NeurIPS 2021

    Args:
        config: MILModelConfig
    """

    def __init__(self, config: MILModelConfig):
        super().__init__()
        self.config = config

        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Position encoding (optional, can use PPEG instead)
        self.use_ppeg = True
        if self.use_ppeg:
            self.ppeg = PPEG(config.hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: (batch, num_patches, feature_dim)
            mask: (batch, num_patches), True = valid

        Returns:
            dict with:
                logits: (batch, num_classes)
                attention: (batch, num_patches) if return_attention
        """
        B, N, _ = features.shape

        # Project features
        x = self.feature_projection(features)  # (batch, num_patches, hidden_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1 + num_patches, hidden_dim)

        # Update mask for class token
        if mask is not None:
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)

        # Apply PPEG after first layer
        H = W = int(math.sqrt(N))
        if H * W != N:
            # Pad to square
            H = W = int(math.ceil(math.sqrt(N)))

        for i, layer in enumerate(self.layers):
            x = layer(x, mask)

            # Apply PPEG after first layer
            if i == 0 and self.use_ppeg:
                # Only apply to patch tokens, not class token
                patch_tokens = x[:, 1:]

                # Pad if necessary
                if patch_tokens.shape[1] < H * W:
                    padding = torch.zeros(
                        B, H * W - patch_tokens.shape[1], patch_tokens.shape[2],
                        device=patch_tokens.device
                    )
                    patch_tokens_padded = torch.cat([patch_tokens, padding], dim=1)
                else:
                    patch_tokens_padded = patch_tokens[:, :H*W]

                patch_tokens_ppeg = self.ppeg(patch_tokens_padded, H, W)

                # Unpad
                if patch_tokens.shape[1] < H * W:
                    patch_tokens_ppeg = patch_tokens_ppeg[:, :patch_tokens.shape[1]]

                x = torch.cat([x[:, :1], patch_tokens + patch_tokens_ppeg], dim=1)

        x = self.norm(x)

        # Use class token for classification
        cls_output = x[:, 0]  # (batch, hidden_dim)
        logits = self.classifier(cls_output)  # (batch, num_classes)

        output = {'logits': logits}

        if self.config.return_attention:
            # Compute attention from class token to patches
            # This is a simplified version - full attention would require saving attention weights
            with torch.no_grad():
                # Use last layer's attention pattern approximation
                patch_features = x[:, 1:]  # (batch, num_patches, hidden_dim)
                attention_scores = torch.bmm(
                    cls_output.unsqueeze(1),  # (batch, 1, hidden_dim)
                    patch_features.transpose(1, 2)  # (batch, hidden_dim, num_patches)
                ).squeeze(1)  # (batch, num_patches)

                if mask is not None:
                    attention_scores = attention_scores.masked_fill(~mask[:, 1:], float('-inf'))

                attention = F.softmax(attention_scores, dim=1)
                output['attention'] = attention

        return output


def create_mil_model(
    model_type: str = 'abmil',
    config: Optional[MILModelConfig] = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create MIL models.

    Args:
        model_type: 'abmil' or 'transmil'
        config: MILModelConfig (if None, created from kwargs)
        **kwargs: Arguments for MILModelConfig

    Returns:
        MIL model
    """
    if config is None:
        config = MILModelConfig(**kwargs)

    if model_type.lower() == 'abmil':
        return ABMIL(config)
    elif model_type.lower() == 'transmil':
        return TransMIL(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    print("Testing MIL models...")

    config = MILModelConfig(
        feature_dim=768,
        hidden_dim=256,
        num_classes=1,
    )

    # Test ABMIL
    model = ABMIL(config)
    x = torch.randn(2, 100, 768)  # 2 bags, 100 patches each
    mask = torch.ones(2, 100, dtype=torch.bool)
    mask[1, 50:] = False  # Second bag has only 50 valid patches

    output = model(x, mask)
    print(f"ABMIL output logits: {output['logits'].shape}")
    print(f"ABMIL attention: {output['attention'].shape}")

    # Test TransMIL
    model = TransMIL(config)
    output = model(x, mask)
    print(f"TransMIL output logits: {output['logits'].shape}")
    print(f"TransMIL attention: {output['attention'].shape}")

    print("All tests passed!")
