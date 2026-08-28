"""
Physics-guided convolutional layer (PGCL).

Manuscript role: topic-conditioned gated fusion that concatenates CLS features
with supervised topic activations and applies a Sigmoid gate (not a physics solver).
"""

import torch
import torch.nn as nn


class PhysicsGuidedConvLayer(nn.Module):
    """
    PGCL: Concat(f_cls, t) -> FC fusion -> Sigmoid topic gate -> enhancement.
    Output width is h=128 (manuscript Table II).
    """

    def __init__(self, embed_dim=64, num_topics=15, num_classes=9, hidden_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_topics = num_topics
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        fusion_input_dim = embed_dim + num_topics
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # Topic-conditioned Sigmoid gate in [0, 1]
        self.physics_guidance = nn.Sequential(
            nn.Linear(num_topics, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.feature_enhancement = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, original_features, topic_representation):
        """
        Args:
            original_features: (B, d) CLS features
            topic_representation: (B, K) non-negative topic activations
        Returns:
            enhanced_features: (B, h)
        """
        fused = torch.cat([original_features, topic_representation], dim=1)
        fused_out = self.fusion_layer(fused)
        gate = self.physics_guidance(topic_representation)
        guided = fused_out * gate
        return self.feature_enhancement(guided)

    def get_enhanced_features(self, original_features, topic_representation):
        self.eval()
        with torch.no_grad():
            return self.forward(original_features, topic_representation)
