"""
Supervised topic model used by PGCL (manuscript Table II / Table IV).

Maps BoW (or CLS) features to a non-negative K-topic mixture with
reconstruction, class-prototype, and diversity losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedTopicModelForPGCL(nn.Module):
    """Topic MLP: FC(64 -> 128 -> 128 -> K)/ReLU with non-negative topics."""

    def __init__(self, embed_dim=64, num_topics=15, num_classes=9):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_topics = num_topics
        self.num_classes = num_classes

        # Non-negative topic basis (K, d)
        self.topic_basis = nn.Parameter(torch.randn(num_topics, embed_dim))
        nn.init.xavier_uniform_(self.topic_basis)
        with torch.no_grad():
            self.topic_basis.data = torch.abs(self.topic_basis.data) + 0.1

        # Manuscript: FC(64 -> 128 -> 128 -> K)/ReLU
        self.feature_to_topic = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_topics),
            nn.ReLU(),
        )

        self.class_prototypes = nn.Parameter(torch.randn(num_classes, num_topics))
        nn.init.xavier_uniform_(self.class_prototypes)
        with torch.no_grad():
            self.class_prototypes.data = torch.abs(self.class_prototypes.data) + 0.1

    def forward(self, features):
        """
        Args:
            features: (B, d) BoW-projected or CLS features
        Returns:
            doc_topic: (B, K) normalized non-negative topic weights
            reconstructed_features: (B, d)
        """
        doc_topic = F.relu(self.feature_to_topic(features))
        doc_topic = doc_topic / (doc_topic.sum(dim=1, keepdim=True) + 1e-10)
        reconstructed = torch.matmul(doc_topic, self.topic_basis)
        return doc_topic, reconstructed

    def compute_supervised_loss(self, features, labels):
        doc_topic, reconstructed = self.forward(features)
        recon_loss = F.mse_loss(reconstructed, features)

        proto_loss = features.new_tensor(0.0)
        for class_idx in range(self.num_classes):
            mask = labels == class_idx
            if mask.sum() > 0:
                proto_loss = proto_loss + F.mse_loss(
                    doc_topic[mask].mean(dim=0, keepdim=True),
                    self.class_prototypes[class_idx : class_idx + 1],
                )
        proto_loss = proto_loss / self.num_classes

        sim = torch.matmul(self.topic_basis, self.topic_basis.t())
        eye = torch.eye(self.num_topics, device=sim.device, dtype=torch.bool)
        diversity_loss = torch.mean(torch.abs(sim.masked_fill(eye, 0)))

        return recon_loss + 0.1 * proto_loss + 0.01 * diversity_loss

    def get_topic_representation(self, features):
        self.eval()
        with torch.no_grad():
            doc_topic, _ = self.forward(features)
        return doc_topic
