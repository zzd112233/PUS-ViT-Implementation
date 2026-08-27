"""
监督主题模型（用于物理可引导卷积层）
从Vision Transformer提取的特征中构建监督主题模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SupervisedTopicModelForPGCL(nn.Module):
    """
    监督主题模型（用于物理可引导卷积层）
    将CLS特征分解为主题表示，用于后续的物理可引导卷积
    """
    
    def __init__(self, embed_dim=64, num_topics=175, num_classes=9):
        """
        初始化监督主题模型
        
        Args:
            embed_dim: 输入特征维度（CLS特征维度）
            num_topics: 主题数量
            num_classes: 类别数量（用于监督学习）
        """
        super(SupervisedTopicModelForPGCL, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_topics = num_topics
        self.num_classes = num_classes
        
        # 主题基矩阵 (num_topics, embed_dim) - 非负约束
        self.topic_basis = nn.Parameter(torch.randn(num_topics, embed_dim))
        nn.init.xavier_uniform_(self.topic_basis)
        # 确保初始值为正
        with torch.no_grad():
            self.topic_basis.data = torch.abs(self.topic_basis.data) + 0.1
        
        # 特征到主题映射层（将CLS特征映射到主题空间）- 4层结构
        self.feature_to_topic = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_topics),
            nn.ReLU()  # 确保非负
        )
        
        # 类别原型（每个类别的主题表示原型，用于监督学习）
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, num_topics))
        nn.init.xavier_uniform_(self.class_prototypes)
        with torch.no_grad():
            self.class_prototypes.data = torch.abs(self.class_prototypes.data) + 0.1
    
    def forward(self, cls_features):
        """
        前向传播
        
        Args:
            cls_features: CLS特征，形状为 (B, embed_dim)
            
        Returns:
            topic_representation: 主题表示 (B, num_topics)
            reconstructed_features: 重构特征 (B, embed_dim)
        """
        # 1. 将特征映射到主题空间
        doc_topic = self.feature_to_topic(cls_features)  # (B, num_topics)
        # 确保非负
        doc_topic = F.relu(doc_topic)
        # 归一化（使主题分布更合理）
        doc_topic = doc_topic / (doc_topic.sum(dim=1, keepdim=True) + 1e-10)
        
        # 2. 通过主题基重构特征
        reconstructed_features = torch.matmul(doc_topic, self.topic_basis)  # (B, embed_dim)
        
        return doc_topic, reconstructed_features
    
    def compute_supervised_loss(self, cls_features, labels):
        """
        计算监督损失（用于训练）
        
        Args:
            cls_features: CLS特征 (B, embed_dim)
            labels: 真实标签 (B,)
            
        Returns:
            loss: 监督损失
        """
        doc_topic, reconstructed_features = self.forward(cls_features)
        
        # 1. 重构损失（确保主题表示能重构原始特征）
        recon_loss = F.mse_loss(reconstructed_features, cls_features)
        
        # 2. 类别原型损失（同类样本的主题表示应该接近该类别的原型）
        proto_loss = 0.0
        for class_idx in range(self.num_classes):
            class_mask = (labels == class_idx)
            if class_mask.sum() > 0:
                class_topics = doc_topic[class_mask]  # (n_class_samples, num_topics)
                class_proto = self.class_prototypes[class_idx:class_idx+1]  # (1, num_topics)
                # 计算该类样本与类别原型的距离
                proto_loss += F.mse_loss(class_topics.mean(dim=0, keepdim=True), class_proto)
        
        proto_loss = proto_loss / self.num_classes
        
        # 3. 主题多样性损失（鼓励不同主题之间的差异）
        topic_similarity = torch.matmul(self.topic_basis, self.topic_basis.t())  # (num_topics, num_topics)
        mask = torch.eye(self.num_topics, device=topic_similarity.device).bool()
        topic_similarity = topic_similarity.masked_fill(mask, 0)
        diversity_loss = torch.mean(torch.abs(topic_similarity))
        
        # 组合损失
        total_loss = recon_loss + 0.1 * proto_loss + 0.01 * diversity_loss
        
        return total_loss
    
    def get_topic_representation(self, cls_features):
        """
        获取主题表示（用于物理可引导卷积层）
        
        Args:
            cls_features: CLS特征
            
        Returns:
            doc_topic: 文档-主题分布 (B, num_topics)
        """
        self.eval()
        with torch.no_grad():
            doc_topic, _ = self.forward(cls_features)
        return doc_topic


if __name__ == "__main__":
    # 测试代码
    batch_size = 32
    embed_dim = 64
    num_topics = 175
    num_classes = 9
    
    model = SupervisedTopicModelForPGCL(
        embed_dim=embed_dim,
        num_topics=num_topics,
        num_classes=num_classes
    )
    
    # 模拟输入
    cls_features = torch.randn(batch_size, embed_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 前向传播
    doc_topic, recon_features = model(cls_features)
    
    print(f"输入特征形状: {cls_features.shape}")
    print(f"主题表示形状: {doc_topic.shape}")
    print(f"重构特征形状: {recon_features.shape}")
    
    # 计算损失
    loss = model.compute_supervised_loss(cls_features, labels)
    print(f"\n监督损失: {loss.item():.4f}")

