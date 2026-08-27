"""
物理可引导卷积层（Physics-Guided Convolutional Layer）
将原始特征和主题表示结合，用于地物分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsGuidedConvLayer(nn.Module):
    """
    物理可引导卷积层
    将Vision Transformer的原始特征和主题模型的主题表示结合
    通过物理约束引导特征学习，用于地物分类
    """
    
    def __init__(self, embed_dim=64, num_topics=175, num_classes=9, hidden_dim=128):
        """
        初始化物理可引导卷积层
        
        Args:
            embed_dim: 原始特征维度（CLS特征维度）
            num_topics: 主题数量
            num_classes: 类别数量
            hidden_dim: 隐藏层维度
        """
        super(PhysicsGuidedConvLayer, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_topics = num_topics
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # 特征融合层：将原始特征和主题表示融合
        # 输入: embed_dim (原始特征) + num_topics (主题表示)
        fusion_input_dim = embed_dim + num_topics
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 物理约束层：基于主题表示的物理先验
        # 主题表示可以看作物理过程的抽象
        self.physics_guidance = nn.Sequential(
            nn.Linear(num_topics, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()  # 作为门控机制
        )
        
        # 特征增强层：应用物理引导后再增强
        self.feature_enhancement = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    
    def forward(self, original_features, topic_representation):
        """
        前向传播
        
        Args:
            original_features: 原始CLS特征 (B, embed_dim)
            topic_representation: 主题表示 (B, num_topics)
            
        Returns:
            enhanced_features: 增强后的特征 (B, hidden_dim)
        """
        # 1. 特征融合：拼接原始特征和主题表示
        fused_features = torch.cat([original_features, topic_representation], dim=1)  # (B, embed_dim + num_topics)
        
        # 2. 通过融合层
        fused_output = self.fusion_layer(fused_features)  # (B, hidden_dim)
        
        # 3. 物理引导：基于主题表示生成引导信号
        physics_guidance_signal = self.physics_guidance(topic_representation)  # (B, hidden_dim)
        
        # 4. 应用物理引导（门控机制）
        guided_features = fused_output * physics_guidance_signal  # (B, hidden_dim)
        
        # 5. 特征增强
        enhanced_features = self.feature_enhancement(guided_features)  # (B, hidden_dim)
        
        # 6. 分类
        return enhanced_features
    
    def get_enhanced_features(self, original_features, topic_representation):
        """
        获取增强后的特征（用于分析）
        
        Args:
            original_features: 原始特征
            topic_representation: 主题表示
            
        Returns:
            enhanced_features: 增强后的特征
        """
        self.eval()
        with torch.no_grad():
            enhanced_features = self.forward(original_features, topic_representation)
        return enhanced_features


class DualBranchFusionPGCL(nn.Module):
    """
    双分支融合PGCL层
    在RSD训练时，两个分支处理RSD数据并融合形成此层
    在T数据训练时，使用此层接受CLS特征并输出增强特征
    """
    
    def __init__(self, embed_dim=64, num_topics=175, num_classes=9, hidden_dim=128, 
                 dropout=0.1, in_channels=12, img_size=7):
        """
        初始化双分支融合PGCL层
        
        Args:
            embed_dim: 嵌入维度（CLS特征维度）
            num_topics: 主题数量
            num_classes: 类别数量
            hidden_dim: 隐藏层维度
            dropout: Dropout率
            in_channels: 输入通道数（用于RSD训练时的分支1投影）
            img_size: 图像大小（用于RSD训练时的分支1投影）
        """
        super(DualBranchFusionPGCL, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_topics = num_topics
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.img_size = img_size
        
        # 延迟导入避免循环导入
        from SupervisedTopicModelForPGCL import SupervisedTopicModelForPGCL
        
        # ========== 分支1：直接主题建模路径 ==========
        # 分支1的投影层（用于RSD训练时，将原始RSD数据投影到embed_dim）
        self.branch1_projection = nn.Sequential(
            nn.Flatten(),  # (B, C, H, W) -> (B, C*H*W)
            nn.Linear(in_channels * img_size * img_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 分支1的主题模型（模拟RSD数据直接与主题模型结合）
        self.branch1_topic_model = SupervisedTopicModelForPGCL(
            embed_dim=embed_dim,
            num_topics=num_topics,
            num_classes=num_classes
        )
        
        # 分支1的PGCL
        self.branch1_pgcl = PhysicsGuidedConvLayer(
            embed_dim=embed_dim,
            num_topics=num_topics,
            num_classes=num_classes,
            hidden_dim=hidden_dim
        )
        
        # ========== 分支2：高阶特征路径 ==========
        # 分支2的主题模型（模拟ViT提取的CLS特征与主题模型结合）
        self.branch2_topic_model = SupervisedTopicModelForPGCL(
            embed_dim=embed_dim,
            num_topics=num_topics,
            num_classes=num_classes
        )
        
        # 分支2的PGCL
        self.branch2_pgcl = PhysicsGuidedConvLayer(
            embed_dim=embed_dim,
            num_topics=num_topics,
            num_classes=num_classes,
            hidden_dim=hidden_dim
        )
        
        # ========== 融合层：融合两个分支的增强特征 ==========
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    
    def forward(self, cls_features, raw_data=None):
        """
        前向传播
        
        Args:
            cls_features: CLS特征 (B, embed_dim)
                         在RSD训练时，这是从ViT encoder提取的CLS特征
                         在T训练时，这是从T数据的ViT encoder提取的CLS特征
            raw_data: 原始RSD数据 (B, C, H, W)，可选
                      如果提供，分支1使用原始数据（RSD训练模式）
                      如果不提供，分支1也使用CLS特征（T训练模式）
        
        Returns:
            enhanced_features: 融合后的增强特征 (B, hidden_dim)
            branch1_topic: 分支1的主题表示 (B, num_topics)
            branch2_topic: 分支2的主题表示 (B, num_topics)
        """
        # ========== 分支1：直接主题建模路径 ==========
        if raw_data is not None:
            # RSD训练模式：使用原始RSD数据
            branch1_features = self.branch1_projection(raw_data)  # (B, embed_dim)
        else:
            # T训练模式：使用CLS特征（模拟直接主题建模）
            branch1_features = cls_features  # (B, embed_dim)
        
        branch1_topic, _ = self.branch1_topic_model(branch1_features)  # (B, num_topics)
        branch1_enhanced = self.branch1_pgcl(branch1_features, branch1_topic)  # (B, hidden_dim)
        
        # ========== 分支2：高阶特征路径 ==========
        # 将CLS特征通过主题模型（模拟ViT提取的CLS特征与主题模型结合）
        branch2_topic, _ = self.branch2_topic_model(cls_features)  # (B, num_topics)
        branch2_enhanced = self.branch2_pgcl(cls_features, branch2_topic)  # (B, hidden_dim)
        
        # ========== 融合两个分支的增强特征 ==========
        fused_features = torch.cat([branch1_enhanced, branch2_enhanced], dim=1)  # (B, hidden_dim * 2)
        enhanced_features = self.fusion_layer(fused_features)  # (B, hidden_dim)
        
        return enhanced_features, branch1_topic, branch2_topic


# if __name__ == "__main__":
#     # 简单测试（仅在直接运行本文件时执行，不在import时执行）
#     batch_size = 32
#     embed_dim = 64
#     num_topics = 175
#     num_classes = 9

#     model = PhysicsGuidedConvLayer(
#         embed_dim=embed_dim,
#         num_topics=num_topics,
#         num_classes=num_classes
#     )

#     # 模拟输入
#     original_features = torch.randn(batch_size, embed_dim)
#     topic_representation = torch.randn(batch_size, num_topics)

#     # 前向传播
#     enhanced_features = model(original_features, topic_representation)

#     print(f"原始特征形状: {original_features.shape}")
#     print(f"主题表示形状: {topic_representation.shape}")
#     print(f"增强特征形状: {enhanced_features.shape}")
#     print(f"增强特征形状: {enhanced_features.shape}")
