"""
Bag-of-visual-words (BoW) encoder for Stage-1 RSD patches.

Local descriptors at each pixel -> soft codeword assignment -> histogram w in R^V
-> linear projection to d for the topic branch (manuscript Table II).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BagOfWordsEncoder(nn.Module):
    def __init__(self, in_channels=12, num_words=64, embed_dim=64, temperature=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.num_words = num_words
        self.embed_dim = embed_dim
        self.temperature = temperature

        self.codebook = nn.Parameter(torch.randn(num_words, in_channels))
        nn.init.xavier_uniform_(self.codebook)

        self.hist_proj = nn.Sequential(
            nn.Linear(num_words, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) RSD patch (C=12 by default)
        Returns:
            bow_feat: (B, d)
            hist: (B, V) normalized BoW histogram
        """
        b, c, h, w = x.shape
        desc = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        dist = torch.cdist(desc, self.codebook.unsqueeze(0).expand(b, -1, -1))
        assign = F.softmax(-dist / max(self.temperature, 1e-6), dim=-1)
        hist = assign.sum(dim=1)
        hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-8)
        return self.hist_proj(hist), hist
