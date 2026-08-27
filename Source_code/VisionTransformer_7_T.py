import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=10, embed_dim=64, img_size=7, patch_size=1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Conv2d 实现 patch embedding
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, mlp_dim=108, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 自注意力
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # 前馈网络
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=7, patch_size=1, in_channels=10, num_classes=9,
                 embed_dim=64, depth=4, num_heads=4, mlp_dim=108, dropout=0.1,
                 num_topics=175, use_pgcl=False, rsd_model_path=None):
        """
        初始化Vision Transformer
        
        Args:
            img_size: 图像大小
            patch_size: patch大小
            in_channels: 输入通道数
            num_classes: 类别数
            embed_dim: 嵌入维度
            depth: Transformer深度
            num_heads: 注意力头数
            mlp_dim: MLP维度
            dropout: Dropout率
            num_topics: 主题数量（用于监督主题模型）
            use_pgcl: 是否使用物理可引导卷积层
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_pgcl = use_pgcl
        self.depth = depth
        
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, img_size, patch_size)
        num_patches = self.patch_embed.num_patches

        # 分类 token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer 编码层（4个）
        self.blocks = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # 从RSD模型加载的物理可引导层（用于第1、2、3、4个Encoder之后）
        self.rsd_pgcl_layers = nn.ModuleList()
        if rsd_model_path is not None:
            self._load_rsd_pgcl_layers(rsd_model_path, embed_dim, num_topics, num_classes)

        # 原始分类头
        self.head = nn.Linear(embed_dim, num_classes)

        # 参数初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
    
    def _load_rsd_pgcl_layers(self, rsd_model_path, embed_dim, num_topics, num_classes):
        """
        从RSD模型加载物理可引导层
        
        Args:
            rsd_model_path: RSD模型权重文件路径
            embed_dim: 嵌入维度
            num_topics: 主题数量
            num_classes: 类别数
        """
        try:
            # 加载RSD模型权重
            rsd_state_dict = torch.load(rsd_model_path, map_location='cpu')
            
            # 导入必要的模块
            from SupervisedTopicModelForPGCL import SupervisedTopicModelForPGCL
            from PhysicsGuidedConvLayer import PhysicsGuidedConvLayer
            
            # 尝试加载融合PGCL层（双分支架构）
            from PhysicsGuidedConvLayer import DualBranchFusionPGCL
            fusion_pgcl_keys = {k.replace('fusion_pgcl.', ''): v 
                                for k, v in rsd_state_dict.items() 
                                if 'fusion_pgcl' in k}
            
            if fusion_pgcl_keys:
                # 使用融合PGCL层（双分支架构）
                fusion_pgcl = DualBranchFusionPGCL(
                    embed_dim=embed_dim,
                    num_topics=num_topics,
                    num_classes=num_classes,
                    hidden_dim=128,
                    dropout=0.1,
                    in_channels=10,  # T数据的通道数
                    img_size=7
                )
                
                # 过滤掉形状不匹配的键（特别是 branch1_projection，因为RSD和T的通道数不同）
                fusion_pgcl_dict = fusion_pgcl.state_dict()
                filtered_fusion_keys = {}
                skipped_keys = []
                
                for k, v in fusion_pgcl_keys.items():
                    if k in fusion_pgcl_dict:
                        # 检查形状是否匹配
                        if fusion_pgcl_dict[k].shape == v.shape:
                            filtered_fusion_keys[k] = v
                        else:
                            skipped_keys.append(f"{k}: checkpoint shape {v.shape} != model shape {fusion_pgcl_dict[k].shape}")
                    else:
                        skipped_keys.append(f"{k}: key not in model")
                
                if skipped_keys:
                    print(f"  警告: 以下键被跳过（形状不匹配或不存在）:")
                    for key in skipped_keys[:5]:  # 只显示前5个
                        print(f"    - {key}")
                    if len(skipped_keys) > 5:
                        print(f"    ... 还有 {len(skipped_keys) - 5} 个键被跳过")
                    print(f"  注意: branch1_projection 层因通道数不匹配（RSD:12 vs T:10）被跳过，将使用随机初始化")
                
                # 加载过滤后的权重
                missing_keys, unexpected_keys = fusion_pgcl.load_state_dict(filtered_fusion_keys, strict=False)
                if missing_keys:
                    missing_important = [k for k in missing_keys if 'branch1_projection' not in k]
                    if missing_important:
                        print(f"  警告: 部分键缺失: {len(missing_important)} 个（branch1_projection相关键缺失是正常的）")
                
                print(f"  成功加载融合PGCL层（双分支架构）")
                
                # 为第1、2、3、4个Encoder创建PGCL层（索引为0、1、2、3）
                # 每个encoder使用同一个融合PGCL层（共享权重）
                for i in [0, 1, 2, 3]:
                    projection = nn.Linear(128, embed_dim)
                    pgcl_module = nn.ModuleDict({
                        'fusion_pgcl': fusion_pgcl,  # 使用融合PGCL层
                        'projection': projection
                    })
                    self.rsd_pgcl_layers.append(pgcl_module)
            else:
                # 回退到单分支架构的加载方式
                print(f"  未找到融合PGCL层，使用单分支架构")
                for i in [0, 1, 2, 3]:
                    # 创建主题模型
                    topic_model = SupervisedTopicModelForPGCL(
                        embed_dim=embed_dim,
                        num_topics=num_topics,
                        num_classes=num_classes
                    )
                    
                    # 创建物理可引导卷积层
                    pgcl = PhysicsGuidedConvLayer(
                        embed_dim=embed_dim,
                        num_topics=num_topics,
                        num_classes=num_classes,
                        hidden_dim=128
                    )
                    
                    projection = nn.Linear(128, embed_dim)
                    
                    # 尝试加载单分支架构的权重
                    topic_model_keys = {k.replace('topic_model.', ''): v 
                                       for k, v in rsd_state_dict.items() 
                                       if 'topic_model' in k and 'fusion_pgcl' not in k}
                    pgcl_keys = {k.replace('pgcl.', ''): v 
                                 for k, v in rsd_state_dict.items() 
                                 if 'pgcl' in k and 'fusion_pgcl' not in k}
                    
                    if topic_model_keys:
                        topic_model.load_state_dict(topic_model_keys, strict=False)
                    if pgcl_keys:
                        pgcl.load_state_dict(pgcl_keys, strict=False)
                    
                    # 将topic_model、pgcl和projection组合成一个模块
                    pgcl_module = nn.ModuleDict({
                        'topic_model': topic_model,
                        'pgcl': pgcl,
                        'projection': projection
                    })
                    
                    self.rsd_pgcl_layers.append(pgcl_module)
            
            print(f"Successfully loaded RSD PGCL layers from {rsd_model_path}")
        except Exception as e:
            print(f"Warning: Failed to load RSD PGCL layers: {e}")
            print("Creating new PGCL layers instead...")
            # 如果加载失败，创建新的层
            from SupervisedTopicModelForPGCL import SupervisedTopicModelForPGCL
            from PhysicsGuidedConvLayer import PhysicsGuidedConvLayer
            
            for i in [0, 1, 2, 3]:
                topic_model = SupervisedTopicModelForPGCL(
                    embed_dim=embed_dim,
                    num_topics=num_topics,
                    num_classes=num_classes
                )
                pgcl = PhysicsGuidedConvLayer(
                    embed_dim=embed_dim,
                    num_topics=num_topics,
                    num_classes=num_classes,
                    hidden_dim=128
                )
                projection = nn.Linear(128, embed_dim)
                pgcl_module = nn.ModuleDict({
                    'topic_model': topic_model,
                    'pgcl': pgcl,
                    'projection': projection
                })
                self.rsd_pgcl_layers.append(pgcl_module)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+num_patches, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 通过Transformer Encoder，在第1、2、3、4个Encoder后插入RSD的PGCL层
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            
            # 在第1、2、3、4个Encoder之后（索引为0、1、2、3）应用RSD的PGCL层
            if i in [0, 1, 2, 3] and len(self.rsd_pgcl_layers) > 0:
                # 获取当前CLS token的特征
                current_cls = x[:, 0]  # (B, embed_dim)
                
                # 获取对应的RSD PGCL层（索引为0、1、2、3对应第1、2、3、4个Encoder）
                pgcl_idx = i
                if pgcl_idx < len(self.rsd_pgcl_layers):
                    rsd_module = self.rsd_pgcl_layers[pgcl_idx]
                    
                    # 检查是否使用融合PGCL层（双分支架构）
                    if 'fusion_pgcl' in rsd_module:
                        # 使用融合PGCL层（双分支架构）
                        # 融合PGCL层接受CLS特征，内部通过两个分支处理并融合
                        enhanced_features, _, _ = rsd_module['fusion_pgcl'](
                            current_cls, raw_data=None  # T训练时只使用CLS特征
                        )  # (B, hidden_dim)
                    else:
                        # 使用单分支架构
                        # 通过主题模型获取主题表示
                        topic_representation, _ = rsd_module['topic_model'](current_cls)  # (B, num_topics)
                        
                        # 通过物理可引导卷积层获得增强特征
                        enhanced_features = rsd_module['pgcl'](current_cls, topic_representation)  # (B, hidden_dim)
                    
                    # 投影回embed_dim
                    projected_features = rsd_module['projection'](enhanced_features)  # (B, embed_dim)
                    
                    # 避免对view结果做in-place操作：重新拼接CLS与patch token
                    updated_cls = current_cls + projected_features  # (B, embed_dim)
                    x = torch.cat([updated_cls.unsqueeze(1), x[:, 1:]], dim=1)

        x = self.norm(x)
        cls_out = x[:, 0]  # 取 [CLS] token
        
        out = self.head(cls_out)
        return out, cls_out
    
    def extract_features(self, x):
        """
        提取CLS token的特征
        
        Args:
            x: 输入张量，形状为 (B, C, H, W)
            
        Returns:
            features: CLS token特征，形状为 (B, embed_dim)
        """
        B = x.size(0)
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1+num_patches, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # 取 [CLS] token
        return cls_out
    
    def get_topic_representation(self, x):
        """
        获取主题表示（用于分析）
        
        Args:
            x: 输入张量，形状为 (B, C, H, W)
            
        Returns:
            topic_representation: 主题表示，形状为 (B, num_topics)
        """
        if not self.use_pgcl:
            raise ValueError("模型未启用物理可引导卷积层，无法获取主题表示")
        
        cls_features = self.extract_features(x)
        topic_representation = self.topic_model.get_topic_representation(cls_features)
        return topic_representation


# if __name__ == "__main__":
#     batch_size = 32
#     in_channels = 10
#     H, W = 7, 7
#     num_classes = 7
# 
#     model = VisionTransformer(img_size=H, in_channels=in_channels, num_classes=num_classes)
#     dummy_input = torch.randn(batch_size, in_channels, H, W)
#     output = model(dummy_input)
# 
#     print("输入形状:", dummy_input.shape)   # (32,10,7,7)
#     print("输出形状:", output.shape)      # (32,7)
