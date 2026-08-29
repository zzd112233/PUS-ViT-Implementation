"""
Stage-2 PUS-ViT backbone on T-matrix (+ Span) inputs.

Manuscript Table II:
  - 10 channels, Conv(10->64, WxW)/BN/ReLU -> N=1 (W=7 Barnaul / W=11 Ober)
  - 4 encoders; after each encoder: inserted PGCL + FC(128->64) residual on CLS only
  - Topic activations t come from the *frozen* Stage-1 guidance topic network
  - Patch tokens are forwarded unchanged
"""

import torch
import torch.nn as nn

from PhysicsGuidedConvLayer import PhysicsGuidedConvLayer


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=10, embed_dim=64, img_size=7, patch_size=7):
        super().__init__()
        if img_size != patch_size:
            raise ValueError(
                f"Manuscript uses a single image token (N=1): require img_size==patch_size, "
                f"got {img_size} vs {patch_size}."
            )
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
    def __init__(
        self,
        img_size=7,
        patch_size=7,
        in_channels=10,
        num_classes=9,
        embed_dim=64,
        depth=4,
        num_heads=4,
        mlp_dim=256,
        dropout=0.1,
        num_topics=15,
        use_pgcl=False,
        rsd_model_path=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_pgcl = use_pgcl
        self.depth = depth
        self.num_topics = num_topics
        self.frozen_topic = None  # set via set_frozen_guidance_topic()

        self.patch_embed = PatchEmbedding(in_channels, embed_dim, img_size, patch_size)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        self.blocks = nn.ModuleList(
            [TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        # Auxiliary KD reference head: FC(64 -> M)
        self.head = nn.Linear(embed_dim, num_classes)

        # Inserted PGCL + 128->64 adapter after each encoder (trainable in Stage-2)
        self.rsd_pgcl_layers = nn.ModuleList()
        self._build_inserted_pgcl(rsd_model_path, embed_dim, num_topics, num_classes, depth)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

    def set_frozen_guidance_topic(self, topic_model):
        """Attach Stage-1 topic network as a frozen guidance copy supplying t."""
        self.frozen_topic = topic_model
        self.frozen_topic.eval()
        for p in self.frozen_topic.parameters():
            p.requires_grad = False

    def _build_inserted_pgcl(self, rsd_model_path, embed_dim, num_topics, num_classes, depth):
        pgcl_sd = {}
        if rsd_model_path is not None:
            try:
                state = torch.load(rsd_model_path, map_location="cpu")
                pgcl_sd = {
                    k.replace("pgcl.", ""): v
                    for k, v in state.items()
                    if k.startswith("pgcl.")
                }
                if pgcl_sd:
                    print(f"  Initializing inserted PGCL from Stage-1 checkpoint: {rsd_model_path}")
            except Exception as exc:
                print(f"  Warning: could not load Stage-1 PGCL weights ({exc}); using random init.")

        for _ in range(depth):
            pgcl = PhysicsGuidedConvLayer(
                embed_dim=embed_dim,
                num_topics=num_topics,
                num_classes=num_classes,
                hidden_dim=128,
            )
            if pgcl_sd:
                pgcl.load_state_dict(pgcl_sd, strict=False)
            projection = nn.Linear(128, embed_dim)  # FC(128 -> 64)
            # Manuscript: only inserted PGCL + adapter are trained; topic is frozen guidance
            self.rsd_pgcl_layers.append(nn.ModuleDict({"pgcl": pgcl, "projection": projection}))

    def forward(self, x):
        if self.frozen_topic is None:
            raise RuntimeError(
                "Call set_frozen_guidance_topic(...) before forward "
                "(manuscript: frozen Stage-1 topic supplies t)."
            )
        b = x.size(0)
        tokens = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        tokens = self.pos_drop(torch.cat((cls_tokens, tokens), dim=1) + self.pos_embed)

        for i, blk in enumerate(self.blocks):
            tokens = blk(tokens)
            if i < len(self.rsd_pgcl_layers):
                current_cls = tokens[:, 0]
                module = self.rsd_pgcl_layers[i]
                # Frozen guidance topic supplies t (no gradient into Stage-1 topic copy)
                with torch.no_grad():
                    topic, _ = self.frozen_topic(current_cls)
                topic = topic.detach()
                enhanced = module["pgcl"](current_cls, topic)
                # Residual update on CLS only; patch tokens unchanged
                updated_cls = current_cls + module["projection"](enhanced)
                tokens = torch.cat([updated_cls.unsqueeze(1), tokens[:, 1:]], dim=1)

        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]
        return self.head(cls_out), cls_out

    def extract_features(self, x):
        _, cls_out = self.forward(x)
        return cls_out
