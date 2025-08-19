import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer

class MethylationTabTransformer(nn.Module):
    """
    메틸레이션 데이터 전용 TabTransformer
    
    고차원 메틸레이션 데이터(450K+ probes)를 효율적으로 처리하기 위해
    feature selection과 TabTransformer를 결합
    """
    
    def __init__(self, 
                 num_probes=450000,          # 전체 methylation probe 수
                 selected_probes=5000,       # 선택할 probe 수 (약 1% 선택)
                 dim=64,                     # 임베딩 차원
                 depth=4,                    # Transformer layers (methylation용으로 축소)
                 heads=8,                    # Attention heads
                 attn_dropout=0.1,
                 ff_dropout=0.1,
                 survival_hidden_dim=256):
        super().__init__()
        
        self.num_probes = num_probes
        self.selected_probes = selected_probes
        self.survival_hidden_dim = survival_hidden_dim
        
        # Learnable feature selection layer
        # 메틸레이션 데이터의 중요한 probe들을 학습으로 선택
        self.feature_selector = nn.Sequential(
            nn.Linear(num_probes, selected_probes * 4),  # Bottleneck
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(selected_probes * 4, selected_probes),
            nn.Sigmoid()  # 0-1 범위로 정규화
        )
        
        # 선택된 probe들을 "연속형" 변수로 취급하여 TabTransformer에 입력
        self.methylation_transformer = TabTransformer(
            categories=(),  # 범주형 없음 (메틸레이션은 모두 연속형)
            num_continuous=selected_probes,
            dim=dim,
            depth=depth, 
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            dim_out=survival_hidden_dim  # 다른 모달리티와 동일한 출력 차원
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

    def forward(self, methylation_data):
        """
        Forward pass
        
        Args:
            methylation_data: 메틸레이션 β-value (batch_size, num_probes)
        
        Returns:
            survival_logit: 3년 생존 예측 logit
            representation: TabTransformer의 중간 representation
            selected_indices: 선택된 probe들의 인덱스 (interpretability용)
        """
        batch_size = methylation_data.size(0)
        
        # Feature selection - 학습 가능한 가중치로 중요한 probe 선별
        selection_weights = self.feature_selector(methylation_data)
        
        # Top-k selection: 가장 높은 가중치를 가진 probe들 선택
        selected_features, selected_indices = torch.topk(
            selection_weights, 
            k=self.selected_probes, 
            dim=1
        )
        
        # 원본 데이터에서 선택된 feature들 추출
        # Gather operation을 사용하여 동적으로 feature 선택
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, self.selected_probes)
        selected_methylation = methylation_data[batch_indices, selected_indices]
        
        # 선택된 특성에 가중치 적용
        weighted_methylation = selected_methylation * selected_features
        
        # TabTransformer 처리 (범주형은 빈 텐서)
        empty_categorical = torch.empty(batch_size, 0, dtype=torch.long, device=methylation_data.device)
        representation = self.methylation_transformer(empty_categorical, weighted_methylation)
        
        # 생존 예측
        survival_logit = self.survival_head(representation)
        
        return survival_logit, representation, selected_indices

    def get_selected_probes_info(self, methylation_data, probe_names=None):
        """
        선택된 probe들의 정보 반환 (interpretability용)
        
        Args:
            methylation_data: 입력 메틸레이션 데이터
            probe_names: probe 이름 리스트 (선택사항)
        
        Returns:
            selected_info: 선택된 probe들의 정보 딕셔너리
        """
        with torch.no_grad():
            _, _, selected_indices = self.forward(methylation_data)
        
        # 배치의 첫 번째 샘플 기준으로 선택된 probe 정보 반환
        first_sample_indices = selected_indices[0].cpu().numpy()
        
        selected_info = {
            'selected_indices': first_sample_indices,
            'num_selected': len(first_sample_indices)
        }
        
        if probe_names is not None:
            selected_info['selected_probe_names'] = [probe_names[i] for i in first_sample_indices]
        
        return selected_info
    
    def extract_feature_importance(self, methylation_data):
        """
        Feature selection weights를 기반으로 feature importance 추출
        
        Returns:
            importance_scores: 모든 probe에 대한 importance scores
            top_probes_indices: 가장 중요한 probe들의 인덱스
        """
        self.eval()
        
        with torch.no_grad():
            # Feature selector에서 중요도 점수 추출
            importance_scores = self.feature_selector(methylation_data)
            
            # 배치 평균 계산
            mean_importance = importance_scores.mean(dim=0)
            
            # Top probes 인덱스
            _, top_probes_indices = torch.topk(mean_importance, k=self.selected_probes)
        
        return mean_importance, top_probes_indices
    
    def get_attention_patterns(self, methylation_data):
        """
        선택된 메틸레이션 특성들 간의 attention pattern 분석
        
        Returns:
            attention_weights: Transformer layer들의 attention weights
        """
        # TabTransformer의 attention 추출은 더 복잡하므로 
        # 현재는 기본 구조만 제공하고 필요시 확장
        
        attention_weights = []
        
        def hook_fn(module, input, output):
            if hasattr(output, 'attention_weights'):
                attention_weights.append(output.attention_weights.detach())
        
        # Forward pass with hooks
        with torch.no_grad():
            survival_logit, representation, selected_indices = self.forward(methylation_data)
        
        return attention_weights, selected_indices