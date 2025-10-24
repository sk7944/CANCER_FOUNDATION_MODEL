"""
Sharded Methylation TabTransformer

396,065 probes를 10개 샤드로 분할하여 메모리 문제 해결
각 샤드별 독립 모델 훈련 후 Fusion layer로 통합
"""

import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer


class SingleShardModel(nn.Module):
    """
    단일 샤드를 처리하는 TabTransformer 모델

    각 샤드는 ~40K probes를 입력으로 받아 256-dim representation 출력
    """

    def __init__(self,
                 num_probes,          # 이 샤드의 probe 개수 (~40K)
                 dim=64,              # 임베딩 차원
                 depth=4,             # Transformer layers
                 heads=8,             # Attention heads
                 attn_dropout=0.3,
                 ff_dropout=0.3,
                 output_dim=256):     # 출력 representation 차원
        super().__init__()

        self.num_probes = num_probes
        self.output_dim = output_dim

        # TabTransformer (methylation은 모두 연속형)
        self.transformer = TabTransformer(
            categories=(),  # 범주형 없음
            num_continuous=num_probes,  # 이 샤드의 probes
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            dim_out=output_dim  # 256-dim representation
        )

    def forward(self, shard_data):
        """
        Args:
            shard_data: (batch_size, num_probes) - 이 샤드의 methylation 데이터

        Returns:
            representation: (batch_size, output_dim) - 256-dim
        """
        batch_size = shard_data.size(0)

        # TabTransformer는 categorical과 continuous 모두 필요
        # Methylation은 모두 연속형이므로 빈 categorical tensor 전달
        empty_categorical = torch.empty(batch_size, 0, dtype=torch.long, device=shard_data.device)

        # Representation 추출
        representation = self.transformer(empty_categorical, shard_data)

        return representation


class FusionLayer(nn.Module):
    """
    10개 샤드 모델의 representations를 결합하는 Fusion Layer

    Input: [rep_0, rep_1, ..., rep_9] → 2,560-dim
    Output: 256-dim fused representation
    """

    def __init__(self,
                 num_shards=10,
                 shard_dim=256,
                 fusion_dim=256,
                 dropout=0.3):
        super().__init__()

        self.num_shards = num_shards
        self.shard_dim = shard_dim
        self.fusion_dim = fusion_dim

        input_dim = num_shards * shard_dim  # 10 * 256 = 2,560

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, fusion_dim * 4),  # 2,560 → 1,024
            nn.LayerNorm(fusion_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(fusion_dim * 4, fusion_dim * 2),  # 1,024 → 512
            nn.LayerNorm(fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(fusion_dim * 2, fusion_dim),  # 512 → 256
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2)
        )

    def forward(self, shard_representations):
        """
        Args:
            shard_representations: List of 10 tensors, each (batch_size, 256)

        Returns:
            fused_representation: (batch_size, 256)
        """
        # Concatenate all shard representations
        # [rep_0, rep_1, ..., rep_9] → (batch_size, 2560)
        concatenated = torch.cat(shard_representations, dim=1)

        # Fusion
        fused = self.fusion(concatenated)

        return fused


class ShardedMethylationTabTransformer(nn.Module):
    """
    Sharded Methylation TabTransformer

    Architecture:
        1. 10개 샤드별 독립 TabTransformer 모델
        2. 각 모델 → 256-dim representation
        3. Fusion layer → 2,560-dim → 256-dim
        4. Survival prediction head → 1-dim (logit)

    Training Strategy:
        - Phase 1: 각 샤드 모델 독립 훈련
        - Phase 2: 샤드 모델 freeze, Fusion layer만 훈련
        - Phase 3 (optional): End-to-end fine-tuning
    """

    def __init__(self,
                 shard_configs,      # List of dicts with 'num_probes' for each shard
                 dim=64,
                 depth=4,
                 heads=8,
                 attn_dropout=0.3,
                 ff_dropout=0.3,
                 shard_output_dim=256,
                 fusion_dim=256,
                 fusion_dropout=0.3):
        super().__init__()

        self.num_shards = len(shard_configs)
        self.shard_output_dim = shard_output_dim

        # 각 샤드별 독립 모델 생성
        self.shard_models = nn.ModuleList([
            SingleShardModel(
                num_probes=config['num_probes'],
                dim=dim,
                depth=depth,
                heads=heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                output_dim=shard_output_dim
            )
            for config in shard_configs
        ])

        # Fusion layer
        self.fusion_layer = FusionLayer(
            num_shards=self.num_shards,
            shard_dim=shard_output_dim,
            fusion_dim=fusion_dim,
            dropout=fusion_dropout
        )

        # Survival prediction head
        self.survival_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 4, 1)
        )

    def forward(self, shard_data_list, return_representations=False):
        """
        Forward pass

        Args:
            shard_data_list: List of 10 tensors, each (batch_size, num_probes_i)
            return_representations: Whether to return intermediate representations

        Returns:
            survival_logit: (batch_size, 1) - 3년 생존 예측 logit
            representations (optional): Dict of intermediate representations
        """
        # 각 샤드별로 representation 추출
        shard_representations = []
        for shard_id, (model, shard_data) in enumerate(zip(self.shard_models, shard_data_list)):
            rep = model(shard_data)  # (batch_size, 256)
            shard_representations.append(rep)

        # Fusion
        fused_representation = self.fusion_layer(shard_representations)  # (batch_size, 256)

        # Survival prediction
        survival_logit = self.survival_head(fused_representation)  # (batch_size, 1)

        if return_representations:
            representations = {
                'shard_representations': shard_representations,  # List of 10 tensors
                'fused_representation': fused_representation,     # (batch_size, 256)
            }
            return survival_logit, representations
        else:
            return survival_logit

    def freeze_shard_models(self):
        """샤드 모델들을 freeze (Fusion layer 훈련 시 사용)"""
        for model in self.shard_models:
            for param in model.parameters():
                param.requires_grad = False
        print("✓ All shard models frozen")

    def unfreeze_shard_models(self):
        """샤드 모델들을 unfreeze (End-to-end fine-tuning 시 사용)"""
        for model in self.shard_models:
            for param in model.parameters():
                param.requires_grad = True
        print("✓ All shard models unfrozen")

    def freeze_fusion_layer(self):
        """Fusion layer를 freeze"""
        for param in self.fusion_layer.parameters():
            param.requires_grad = False
        for param in self.survival_head.parameters():
            param.requires_grad = False
        print("✓ Fusion layer and survival head frozen")

    def unfreeze_fusion_layer(self):
        """Fusion layer를 unfreeze"""
        for param in self.fusion_layer.parameters():
            param.requires_grad = True
        for param in self.survival_head.parameters():
            param.requires_grad = True
        print("✓ Fusion layer and survival head unfrozen")

    def get_num_parameters(self):
        """모델 파라미터 수 계산"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        shard_params = sum(p.numel() for model in self.shard_models for p in model.parameters())
        fusion_params = sum(p.numel() for p in self.fusion_layer.parameters())
        head_params = sum(p.numel() for p in self.survival_head.parameters())

        return {
            'total': total_params,
            'trainable': trainable_params,
            'shard_models': shard_params,
            'fusion_layer': fusion_params,
            'survival_head': head_params
        }

    def load_shard_model(self, shard_id, checkpoint_path):
        """특정 샤드 모델만 로드 (개별 훈련 후 통합 시 사용)"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        self.shard_models[shard_id].load_state_dict(state_dict)
        print(f"✓ Loaded shard {shard_id} model from {checkpoint_path}")

    def save_shard_model(self, shard_id, save_path):
        """특정 샤드 모델만 저장"""
        torch.save({
            'model_state_dict': self.shard_models[shard_id].state_dict(),
            'shard_id': shard_id
        }, save_path)
        print(f"✓ Saved shard {shard_id} model to {save_path}")
