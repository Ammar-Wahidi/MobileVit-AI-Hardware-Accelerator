## Pytorch model

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Dict
import numpy as np

# PyTorch Implementation
class MV2(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1, expansion_ratio=1):
        super().__init__()
        hidden_dim = int(in_channels * expansion_ratio)
        self.use_residual = strides == 1 and in_channels == out_channels

        # 1. Point-wise conv (expand)
        self.pw_expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )

        # 2. Depth-wise conv
        self.dw_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=strides, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )

        # 3. Point-wise linear conv (project)
        self.pw_project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        skip = x
        x = self.pw_expand(x)
        x = self.dw_conv(x)
        x = self.pw_project(x)
        if self.use_residual:
            x = x + skip
        return x


class LocalRepresentation(nn.Module):
    def __init__(self, in_channels, num_filters, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(num_filters, out_dim, kernel_size=1, stride=1, padding= 0,bias=False )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp_dim = mlp_dim
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        skip = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = skip + attn_out

        skip2 = x
        x_norm = self.norm2(x)

        x = self.fc1(x_norm)
        x = self.act(x)

        y1 = self.fc2(x)

        x = skip2 + y1

        return x

class Fusion(nn.Module):
    def __init__(self, in_channels, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(2*in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.act2 = nn.SiLU()

    def forward(self, x, x_fusion):
        x_f = self.conv1(x_fusion)
        x_f = self.bn1(x_f)
        x_f = self.act1(x_f)

        x_concat = torch.cat([x, x_f], dim=1)

        x_out = self.conv2(x_concat)
        x_out = self.bn2(x_out)
        x_out = self.act2(x_out)

        return x_out

import numpy as np

class MobileViTBlock(nn.Module):
    def __init__(self, in_channels, num_filters, dim, num_heads, patch_size=2, num_layers=1):
        super().__init__()
        self.patch_h = patch_size
        self.patch_w = patch_size
        self.patch_area = patch_size * patch_size

        self.local_rep = LocalRepresentation(in_channels, num_filters, dim)
        self.transformers = nn.ModuleList([
            TransformerEncoder(dim, num_heads=num_heads, mlp_dim=dim*2) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.fusion = Fusion(in_channels, dim)

    def unfolding(self, x: Tensor) -> Tuple[Tensor, Dict]:
        b, c, h, w = x.shape

        # Calculate new dimensions (must be divisible by patch_size)
        new_h = int(np.ceil(h / self.patch_h) * self.patch_h)
        new_w = int(np.ceil(w / self.patch_w) * self.patch_w)
        interpolate = False

        if new_h != h or new_w != w:
            # Use bilinear interpolation to pad feature map
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        num_patch_h = new_h // self.patch_h
        num_patch_w = new_w // self.patch_w
        num_patches = num_patch_h * num_patch_w

        # [B, C, H, W] --> [B*C*n_h, p_h, n_w, p_w]
        x = x.reshape(b * c * num_patch_h, self.patch_h, num_patch_w, self.patch_w)
        # [B*C*n_h, p_h, n_w, p_w] --> [B*C*n_h, n_w, p_h, p_w]
        x = x.permute(0, 2, 1, 3)
        # [B*C*n_h, n_w, p_h, p_w] --> [B, C, N, P]
        x = x.reshape(b, c, num_patches, self.patch_area)
        # [B, C, N, P] --> [B, P, N, C]
        x = x.permute(0, 3, 2, 1)
        # [B, P, N, C] --> [BP, N, C]
        patches = x.reshape(b * self.patch_area, num_patches, c)

        info = {
            "orig_size": (h, w),
            "batch_size": b,
            "interpolate": interpolate,
            "num_patches_h": num_patch_h,
            "num_patches_w": num_patch_w,
            "total_patches": num_patches
        }
        return patches, info

    def folding(self, x: Tensor, info: Dict) -> Tensor:
        b = info["batch_size"]
        # Use consistent naming: num_patches_h and num_patches_w
        num_patches_h = info["num_patches_h"]
        num_patches_w = info["num_patches_w"]

        h_new = num_patches_h * self.patch_h
        w_new = num_patches_w * self.patch_w
        channels = x.shape[-1]

        # 1. [BP, N, C] --> [B, P, N, C]
        x = x.reshape(b, self.patch_area, info["total_patches"], channels)

        # 2. [B, P, N, C] --> [B, C, N, P]
        x = x.permute(0, 3, 2, 1)

        # 3. [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        # FIXED: Changed num_patch_h/w to num_patches_h/w
        x = x.reshape(b * channels * num_patches_h, num_patches_w, self.patch_h, self.patch_w)

        # 4. [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        x = x.permute(0, 2, 1, 3)

        # 5. [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        x = x.reshape(b, channels, h_new, w_new)

        # 6. Revert interpolation if it was applied
        if info["interpolate"]:
            x = F.interpolate(x, size=info["orig_size"], mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        res = x
        x = self.local_rep(x)

        # Global Representation (Transformer)
        patches, info = self.unfolding(x)
        for transformer in self.transformers:
            patches = transformer(patches)
        patches = self.norm(patches)

        x_global = self.folding(patches, info)

        # Fusion
        out = self.fusion(res, x_global)
        return out


class MobileViT_XXS(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU()
        )

        # Stage 1
        self.mv2_1 = MV2(16, 16, expansion_ratio=2)

        # Stage 2
        self.mv2_2a = MV2(16, 24, strides=2, expansion_ratio=2)
        self.mv2_2b = MV2(24, 24, expansion_ratio=2)
        self.mv2_2c = MV2(24, 24, expansion_ratio=2)

        # Stage 3
        self.mv2_3a = MV2(24, 48, strides=2, expansion_ratio=2)
        self.mvit_3b = MobileViTBlock(48, 48, 64,num_heads = 2, patch_size=2, num_layers=2)

        # Stage 4
        self.mv2_4a = MV2(48, 64, strides=2, expansion_ratio=2)
        self.mvit_4b = MobileViTBlock(64, 64, 80,num_heads = 4 ,patch_size=2, num_layers=4)

        # Stage 5
        self.mv2_5a = MV2(64, 80, strides=2, expansion_ratio=2)
        self.mvit_5b = MobileViTBlock(80, 80, 96,num_heads = 4, patch_size=2, num_layers=3)

        # Head
        self.head_conv = nn.Sequential(
            nn.Conv2d(80, 320, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(320),
            nn.SiLU()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(320, num_classes)

    def forward(self, x):
        x = self.stem(x)


        x = self.mv2_1(x)


        x = self.mv2_2a(x)
        x = self.mv2_2b(x)
        x = self.mv2_2c(x)

        x = self.mv2_3a(x)

        x = self.mvit_3b(x)

        x = self.mv2_4a(x)
        x = self.mvit_4b(x)

        x = self.mv2_5a(x)
        x = self.mvit_5b(x)

        x = self.head_conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

my_model = MobileViT_XXS(in_channels=3, num_classes=1000)
x = torch.randn(2, 3, 224, 224)  # batch of 2 images
y = my_model(x)
print(y.shape)