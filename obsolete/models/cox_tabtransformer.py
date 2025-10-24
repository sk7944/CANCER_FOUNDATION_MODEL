import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer

class CoxTabTransformer(nn.Module):
    """
    Cox 계수를 활용한 멀티오믹스 데이터용 TabTransformer
    
    기존 검증된 TabTransformer 구현체를 활용하여 우리 도메인에 맞게 수정
    - Clinical categorical features → TabTransformer
    - Omics continuous features [measured_value, cox_coefficient] pairs
    """
    
    def __init__(self, 
                 clinical_categories,        # 임상 범주형 특성 vocab sizes tuple
                 num_omics_features,         # 오믹스 특성 개수
                 dim=64,                     # 임베딩 차원
                 depth=6,                    # Transformer layers
                 heads=8,                    # Attention heads
                 attn_dropout=0.1,
                 ff_dropout=0.1,
                 survival_hidden_dim=256):
        super().__init__()
        
        self.num_omics_features = num_omics_features
        self.survival_hidden_dim = survival_hidden_dim
        
        # 기존 TabTransformer를 베이스로 사용
        self.base_transformer = TabTransformer(
            categories=clinical_categories,
            num_continuous=num_omics_features * 2,  # [측정값, Cox계수] 쌍
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            dim_out=survival_hidden_dim  # 중간 representation
        )
        
        # 생존 예측용 최종 헤드
        self.survival_head = nn.Sequential(
            nn.Linear(survival_hidden_dim, survival_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(survival_hidden_dim // 2, survival_hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(survival_hidden_dim // 4, 1)
        )

    def forward(self, clinical_categorical, omics_continuous):
        """
        Forward pass
        
        Args:
            clinical_categorical: 범주형 임상 변수 (batch_size, num_clinical_features)
            omics_continuous: [측정값, Cox계수] 쌍들 flatten된 형태 (batch_size, num_omics_features * 2)
        
        Returns:
            survival_logit: 3년 생존 예측 logit
            representation: TabTransformer의 중간 representation (interpretability용)
        """
        # TabTransformer로 representation 생성
        representation = self.base_transformer(clinical_categorical, omics_continuous)
        
        # 생존 예측
        survival_logit = self.survival_head(representation)
        
        return survival_logit, representation
    
    def get_attention_weights(self, clinical_categorical, omics_continuous):
        """
        Attention weights 추출 (interpretability용)
        
        Returns:
            attention_weights: List of attention weights from each transformer layer
        """
        # TabTransformer 내부 transformer에 hook 등록하여 attention weights 추출
        attention_weights = []
        
        def hook_fn(module, input, output):
            if hasattr(output, 'attention_weights'):
                attention_weights.append(output.attention_weights.detach())
        
        # Hook 등록
        hooks = []
        for layer in self.base_transformer.transformer.layers:
            for sublayer in layer:
                if hasattr(sublayer, 'branch') and hasattr(sublayer.branch, 'fn'):
                    if hasattr(sublayer.branch.fn, 'to_qkv'):  # Attention layer
                        hook = sublayer.register_forward_hook(hook_fn)
                        hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.forward(clinical_categorical, omics_continuous)
        
        # Hook 제거
        for hook in hooks:
            hook.remove()
        
        return attention_weights
    
    def extract_feature_importance(self, clinical_categorical, omics_continuous):
        """
        Feature importance 추출
        
        Returns:
            clinical_importance: 임상 변수 importance scores
            omics_importance: 오믹스 변수 importance scores
        """
        # Gradient-based feature importance
        self.eval()
        
        clinical_categorical.requires_grad_(True)
        omics_continuous.requires_grad_(True)
        
        survival_logit, _ = self.forward(clinical_categorical, omics_continuous)
        
        # Gradient 계산
        survival_logit.backward(torch.ones_like(survival_logit))
        
        clinical_importance = clinical_categorical.grad.abs().mean(dim=0)
        omics_importance = omics_continuous.grad.abs().mean(dim=0)
        
        # Omics importance를 [value, cox] 쌍으로 reshape
        omics_importance = omics_importance.view(self.num_omics_features, 2)
        
        return clinical_importance, omics_importance