"""
Stage-1 PUS-ViT on RSD (+ Span) inputs.

Manuscript Table II:
  - 12 channels, patch Conv(12->64, 7x7)/BN/ReLU -> N=1 image token
  - 3 encoder blocks, d=64, H=4, MLP 64->256->64
  - BoW -> topic MLP; CLS -> PGCL; classifier FC(128->M)
"""

import torch
import torch.nn as nn

from BagOfWordsEncoder import BagOfWordsEncoder
from PhysicsGuidedConvLayer import PhysicsGuidedConvLayer
from SupervisedTopicModelForPGCL import SupervisedTopicModelForPGCL


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=12, embed_dim=64, img_size=7, patch_size=7):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Single-path Stage-1: BoW->topic; ViT CLS->PGCL; head on 128-D PGCL output."""

    def __init__(
        self,
        img_size=7,
        patch_size=7,
        in_channels=12,
        num_classes=9,
        embed_dim=64,
        depth=3,
        num_heads=4,
        mlp_dim=256,
        dropout=0.1,
        num_topics=15,
        use_pgcl=True,
        use_bow=True,
        num_words=64,
        **kwargs,
    ):
        super().__init__()
        # Ignore legacy kwargs (e.g., use_dual_branch) for API stability
        _ = kwargs
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_pgcl = use_pgcl
        self.use_bow = use_bow and use_pgcl
        self.in_channels = in_channels
        self.pgcl_hidden_dim = 128

        self.patch_embed = PatchEmbedding(in_channels, embed_dim, img_size, patch_size)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        self.blocks = nn.ModuleList(
            [TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        if use_pgcl:
            if self.use_bow:
                self.bow_encoder = BagOfWordsEncoder(
                    in_channels=in_channels, num_words=num_words, embed_dim=embed_dim
                )
            self.topic_model = SupervisedTopicModelForPGCL(
                embed_dim=embed_dim, num_topics=num_topics, num_classes=num_classes
            )
            self.pgcl = PhysicsGuidedConvLayer(
                embed_dim=embed_dim,
                num_topics=num_topics,
                num_classes=num_classes,
                hidden_dim=self.pgcl_hidden_dim,
            )
            # Stage-1 head: FC(128 -> M) on PGCL readout
            self.pgcl_classifier = nn.Sequential(
                nn.LayerNorm(self.pgcl_hidden_dim),
                nn.GELU(),
                nn.Linear(self.pgcl_hidden_dim, num_classes),
            )
        else:
            self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def _encode_vit(self, x):
        b = x.size(0)
        tokens = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        tokens = self.pos_drop(torch.cat((cls_tokens, tokens), dim=1) + self.pos_embed)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 0]

    def forward(self, x):
        cls_features = self._encode_vit(x)
        if not self.use_pgcl:
            return self.head(cls_features), cls_features

        topic_input = self.bow_encoder(x)[0] if self.use_bow else cls_features
        topic_representation, _ = self.topic_model(topic_input)
        enhanced = self.pgcl(cls_features, topic_representation)
        logits = self.pgcl_classifier(enhanced)
        return logits, cls_features, topic_representation

    def extract_features(self, x):
        return self._encode_vit(x)

    def get_topic_representation(self, x):
        if not self.use_pgcl:
            raise ValueError("PGCL/topic branch is disabled.")
        if self.use_bow:
            bow_feat, _ = self.bow_encoder(x)
            return self.topic_model.get_topic_representation(bow_feat)
        return self.topic_model.get_topic_representation(self.extract_features(x))
