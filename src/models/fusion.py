import torch
import torch.nn as nn


class AttentionGatedFusion(nn.Module):
    """
    Projects F_CNN and F_ViT into a shared 512-D space, then learns a
    per-sample scalar gate alpha in [0,1] that adaptively weights the two
    branches:

        F_fused = alpha * F_CNN' + (1 - alpha) * F_ViT'
        alpha   = sigmoid(W_a . [F_CNN'; F_ViT'] + b_a)
    """

    def __init__(self, cnn_dim: int, vit_dim: int, fusion_dim: int = 512):
        super().__init__()
        self.proj_cnn = nn.Linear(cnn_dim, fusion_dim)
        self.proj_vit = nn.Linear(vit_dim, fusion_dim)
        self.gate = nn.Linear(fusion_dim * 2, 1)

    def forward(self, f_cnn: torch.Tensor, f_vit: torch.Tensor):
        f_cnn_p = self.proj_cnn(f_cnn)
        f_vit_p = self.proj_vit(f_vit)
        concat = torch.cat([f_cnn_p, f_vit_p], dim=1)
        alpha = torch.sigmoid(self.gate(concat))  # [B, 1]
        fused = alpha * f_cnn_p + (1 - alpha) * f_vit_p
        return fused, alpha


class AttentionGatedFusion3(nn.Module):
    """
    Three-way generalization of AttentionGatedFusion, used when
    `model.use_metadata: true` adds a clinical-metadata branch alongside
    CNN and ViT. Each branch is projected into the shared fusion space,
    and a softmax gate (instead of a single sigmoid) learns a per-sample
    3-way weighting so metadata can contribute without being forced to
    compete 50/50 with the image branches the way a naive concatenation
    would:

        F_fused = w_cnn * F_CNN' + w_vit * F_ViT' + w_meta * F_Meta'
        [w_cnn, w_vit, w_meta] = softmax(W_a . [F_CNN'; F_ViT'; F_Meta'] + b_a)
    """

    def __init__(self, cnn_dim: int, vit_dim: int, meta_dim: int, fusion_dim: int = 512):
        super().__init__()
        self.proj_cnn = nn.Linear(cnn_dim, fusion_dim)
        self.proj_vit = nn.Linear(vit_dim, fusion_dim)
        self.proj_meta = nn.Linear(meta_dim, fusion_dim)
        self.gate = nn.Linear(fusion_dim * 3, 3)

    def forward(self, f_cnn: torch.Tensor, f_vit: torch.Tensor, f_meta: torch.Tensor):
        c = self.proj_cnn(f_cnn)
        v = self.proj_vit(f_vit)
        m = self.proj_meta(f_meta)
        concat = torch.cat([c, v, m], dim=1)
        weights = torch.softmax(self.gate(concat), dim=1)  # [B, 3]
        fused = weights[:, 0:1] * c + weights[:, 1:2] * v + weights[:, 2:3] * m
        return fused, weights
