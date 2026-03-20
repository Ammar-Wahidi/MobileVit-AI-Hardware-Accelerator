"""
MobileViT-XXS  --  Static PTQ, maximum quantization coverage
=============================================================
Goal: quantize Linear (fc1/fc2) and MultiheadAttention statically.
      Only LayerNorm stays FP32 (no INT8 kernel in any eager namespace).

Key architectural change vs previous versions:
  OLD: entire transformer block wrapped in one DeQuant...Quant boundary
       -> fc1, fc2, MHA all fell inside the FP32 island
  NEW: DeQuant/Quant placed ONLY around each individual LayerNorm
       -> fc1, fc2 get static qconfig -> nnq.Linear after convert()
       -> MHA keeps static qconfig -> nnq.Linear (its internal proj weights)
       -> only LayerNorm uses qconfig=None and is surrounded by micro-stubs

Data flow inside TransformerEncoder (NEW):
  INT8 patches
    -> DeQuant -> LayerNorm(FP32) -> Quant       [norm1 island]
    -> MHA (INT8 static)                          [attn]
    -> residual add via FloatFunctional           [skip_add_attn]
    -> DeQuant -> LayerNorm(FP32) -> Quant       [norm2 island]
    -> fc1 (INT8 static)
    -> Hardswish
    -> fc2 (INT8 static)
    -> residual add via FloatFunctional           [skip_add_mlp]
  INT8 patches
"""

import os, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict
import numpy as np

import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd

from torch.ao.quantization import (
    QuantStub, DeQuantStub,
    get_default_qconfig, prepare, convert,
)


# ---------------------------------------------------------------------------
# MV2 Block
# ---------------------------------------------------------------------------
class MV2(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1, expansion_ratio=1):
        super().__init__()
        hidden_dim = int(in_channels * expansion_ratio)
        self.use_residual = (strides == 1 and in_channels == out_channels)
        self.pw_expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.Hardswish()
        )
        self.dw_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, strides, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.Hardswish()
        )
        self.pw_project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if self.use_residual:
            self.skip_add = nnq.FloatFunctional()

    def forward(self, x):
        skip = x
        x = self.pw_expand(x)
        x = self.dw_conv(x)
        x = self.pw_project(x)
        if self.use_residual:
            x = self.skip_add.add(x, skip)
        return x


# ---------------------------------------------------------------------------
# LocalRepresentation
# ---------------------------------------------------------------------------
class LocalRepresentation(nn.Module):
    def __init__(self, in_channels, num_filters, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(num_filters)
        self.act1  = nn.Hardswish()
        self.conv2 = nn.Conv2d(num_filters, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        return self.conv2(self.act1(self.bn1(self.conv1(x))))


# ---------------------------------------------------------------------------
# LayerNormIsland
# A tiny wrapper: DeQuant -> LayerNorm -> Quant
# This lets LayerNorm run in FP32 while everything around it stays INT8.
# The DeQuant/Quant stubs are registered as proper nn.Module children so
# prepare() inserts observers on them and convert() replaces them correctly.
# ---------------------------------------------------------------------------
class LayerNormIsland(nn.Module):
    """
    Wraps a single LayerNorm with dequant/quant boundary stubs so that
    LayerNorm runs in FP32 while the surrounding graph stays INT8.

         INT8 input
           |
        DeQuantStub  (INT8 -> FP32)
           |
        LayerNorm    (FP32 - no INT8 kernel exists)
           |
        QuantStub    (FP32 -> INT8)
           |
         INT8 output
    """
    def __init__(self, normalized_shape):
        super().__init__()
        self.dequant = DeQuantStub()
        self.norm    = nn.LayerNorm(normalized_shape)
        self.quant   = QuantStub()

    def forward(self, x):
        x = self.dequant(x)   # INT8 -> FP32
        x = self.norm(x)      # FP32 LayerNorm
        x = self.quant(x)     # FP32 -> INT8
        return x


# ---------------------------------------------------------------------------
# TransformerEncoder  --  fully quantized (except LayerNorm islands)
#
# All residual '+' ops use FloatFunctional because they operate on INT8
# tensors that have been output by quantized ops; plain '+' would fail.
#
# MHA: nn.MultiheadAttention with static qconfig -> convert() replaces its
#      internal in_proj / out_proj Linear layers with nnq.Linear.
#      The MHA module itself stays as nn.MultiheadAttention wrapper but
#      its weight Linear submodules become quantized.
#
# fc1, fc2: nn.Linear with static qconfig -> nnq.Linear after convert().
# ---------------------------------------------------------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim):
        super().__init__()
        # norm1/norm2: each wrapped in a LayerNormIsland (FP32 micro-island)
        self.norm1 = LayerNormIsland(dim)
        self.attn  = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                           batch_first=True, dropout=0.0)
        self.norm2 = LayerNormIsland(dim)
        self.fc1   = nn.Linear(dim, mlp_dim)
        self.act   = nn.Hardswish()
        self.fc2   = nn.Linear(mlp_dim, dim)

        # FloatFunctional for the two residual additions
        # (inputs are INT8 quantized tensors; plain '+' not observer-traceable)
        self.skip_add_attn = nnq.FloatFunctional()
        self.skip_add_mlp  = nnq.FloatFunctional()

    def forward(self, x):
        # --- Attention sub-block ---
        skip = x
        xn = self.norm1(x)                          # INT8 -> FP32 -> INT8
        attn_out, _ = self.attn(xn, xn, xn)        # INT8
        x = self.skip_add_attn.add(attn_out, skip)  # INT8 + INT8

        # --- MLP sub-block ---
        skip2 = x
        xn = self.norm2(x)                          # INT8 -> FP32 -> INT8
        x = self.fc1(xn)                            # INT8
        x = self.act(x)
        x = self.fc2(x)                             # INT8
        x = self.skip_add_mlp.add(x, skip2)         # INT8 + INT8
        return x


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
class Fusion(nn.Module):
    def __init__(self, in_channels, dim):
        super().__init__()
        self.conv1    = nn.Conv2d(dim, in_channels, 1, 1, 0, bias=False)
        self.bn1      = nn.BatchNorm2d(in_channels)
        self.act1     = nn.Hardswish()
        self.conv2    = nn.Conv2d(2 * in_channels, in_channels, 3, 1, 1, bias=False)
        self.bn2      = nn.BatchNorm2d(in_channels)
        self.act2     = nn.Hardswish()
        self.cat_func = nnq.FloatFunctional()

    def forward(self, x, x_fusion):
        x_f = self.act1(self.bn1(self.conv1(x_fusion)))
        return self.act2(self.bn2(self.conv2(self.cat_func.cat([x, x_f], dim=1))))


# ---------------------------------------------------------------------------
# MobileViTBlock
# Now the transformer runs fully inside the INT8 quantized graph.
# No outer DeQuant/Quant boundary needed around the whole transformer.
# Each LayerNorm has its own micro DeQuant/Quant island inside TransformerEncoder.
# ---------------------------------------------------------------------------
class MobileViTBlock(nn.Module):
    def __init__(self, in_channels, num_filters, dim, num_heads,
                 patch_size=2, num_layers=1):
        super().__init__()
        self.patch_h    = patch_size
        self.patch_w    = patch_size
        self.patch_area = patch_size * patch_size
        self.local_rep  = LocalRepresentation(in_channels, num_filters, dim)
        self.transformers = nn.ModuleList([
            TransformerEncoder(dim, num_heads=num_heads, mlp_dim=dim * 2)
            for _ in range(num_layers)
        ])
        # Final LayerNorm also wrapped in an island
        self.norm   = LayerNormIsland(dim)
        self.fusion = Fusion(in_channels, dim)
        # No outer dequant/quant stubs needed anymore

    def unfolding(self, x: Tensor) -> Tuple[Tensor, Dict]:
        b, c, h, w = x.shape
        new_h = int(np.ceil(h / self.patch_h) * self.patch_h)
        new_w = int(np.ceil(w / self.patch_w) * self.patch_w)
        interpolate = new_h != h or new_w != w
        if interpolate:
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        nph = new_h // self.patch_h
        npw = new_w // self.patch_w
        np_ = nph * npw
        x = x.reshape(b*c*nph, self.patch_h, npw, self.patch_w).permute(0,2,1,3)
        patches = x.reshape(b, c, np_, self.patch_area).permute(0,3,2,1).reshape(b*self.patch_area, np_, c)
        return patches, {"orig_size":(h,w),"batch_size":b,"interpolate":interpolate,
                         "num_patches_h":nph,"num_patches_w":npw,"total_patches":np_}

    def folding(self, x: Tensor, info: Dict) -> Tensor:
        b, nph, npw, c = info["batch_size"], info["num_patches_h"], info["num_patches_w"], x.shape[-1]
        x = x.reshape(b, self.patch_area, info["total_patches"], c).permute(0,3,2,1)
        x = x.reshape(b*c*nph, npw, self.patch_h, self.patch_w).permute(0,2,1,3)
        x = x.reshape(b, c, nph*self.patch_h, npw*self.patch_w)
        if info["interpolate"]:
            x = F.interpolate(x, size=info["orig_size"], mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        res = x
        x   = self.local_rep(x)
        patches, info = self.unfolding(x)
        for t in self.transformers:
            patches = t(patches)
        patches = self.norm(patches)
        return self.fusion(res, self.folding(patches, info))


# ---------------------------------------------------------------------------
# MobileViT-XXS
# ---------------------------------------------------------------------------
class MobileViT_XXS(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        self.quant   = QuantStub()
        self.dequant = DeQuantStub()
        self.stem    = nn.Sequential(nn.Conv2d(in_channels, 16, 3, 2, 1, bias=False),
                                     nn.BatchNorm2d(16), nn.Hardswish())
        self.mv2_1   = MV2(16, 16, expansion_ratio=2)
        self.mv2_2a  = MV2(16, 24, strides=2, expansion_ratio=2)
        self.mv2_2b  = MV2(24, 24, expansion_ratio=2)
        self.mv2_2c  = MV2(24, 24, expansion_ratio=2)
        self.mv2_3a  = MV2(24, 48, strides=2, expansion_ratio=2)
        self.mvit_3b = MobileViTBlock(48, 48, 64, num_heads=2, patch_size=2, num_layers=2)
        self.mv2_4a  = MV2(48, 64, strides=2, expansion_ratio=2)
        self.mvit_4b = MobileViTBlock(64, 64, 80, num_heads=4, patch_size=2, num_layers=4)
        self.mv2_5a  = MV2(64, 80, strides=2, expansion_ratio=2)
        self.mvit_5b = MobileViTBlock(80, 80, 96, num_heads=4, patch_size=2, num_layers=3)
        self.head_conv = nn.Sequential(nn.Conv2d(80, 320, 1, 1, 0, bias=False),
                                       nn.BatchNorm2d(320), nn.Hardswish())
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(320, num_classes)

    def forward(self, x):
        x = self.quant(x)
        x = self.stem(x); x = self.mv2_1(x)
        x = self.mv2_2a(x); x = self.mv2_2b(x); x = self.mv2_2c(x)
        x = self.mv2_3a(x); x = self.mvit_3b(x)
        x = self.mv2_4a(x); x = self.mvit_4b(x)
        x = self.mv2_5a(x); x = self.mvit_5b(x)
        x = self.head_conv(x); x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


# ---------------------------------------------------------------------------
# QConfig assignment
# Only LayerNorm gets qconfig=None.
# Everything else (Conv, Linear, MHA internals) gets static qconfig.
# ---------------------------------------------------------------------------
def assign_qconfigs(model: nn.Module, backend: str = 'fbgemm'):
    static_qconfig = get_default_qconfig(backend)
    model.qconfig  = static_qconfig

    # Only disable LayerNorm -- it has no INT8 kernel in any eager namespace
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            module.qconfig = None

    # Classifier: post-dequant, receives float32 -> must not be quantized
    #model.classifier.qconfig = None


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
QUANTIZED_TYPES = {
    nnq.Conv2d:      "INT8 static | nnq.Conv2d",
    nnq.BatchNorm2d: "INT8 static | fused into Conv",
    nnq.Hardswish:   "INT8 static | nnq.Hardswish",
    nnq.Linear:      "INT8 static | nnq.Linear",
    nnqd.Linear:     "INT8 dynamic| nnqd.Linear",
}
FLOAT_TYPES = {
    nn.LayerNorm: "FP32 | no INT8 kernel (absent from entire quantized API)",
    nn.Linear:    "FP32 | qconfig=None (post-dequant classifier)",
    nn.Dropout:   "FP32 | identity at eval",
}
INFRA_TYPES = {
    nnq.Quantize:        "INFRA | Quantize  (INT8 entry)",
    nnq.DeQuantize:      "INFRA | DeQuantize(FP32 exit)",
    nnq.FloatFunctional: "INFRA | FloatFunctional (add/cat)",
}


def print_quantization_summary(model: nn.Module):
    print(); print("=" * 82)
    print(f"  {'LAYER NAME':<52}  STATUS")
    print("=" * 82)
    n_int8 = n_fp32 = n_infra = 0
    for name, module in model.named_modules():
        t = type(module)
        if t in QUANTIZED_TYPES:
            print(f"  {name:<52}  {QUANTIZED_TYPES[t]}"); n_int8 += 1
        elif t in INFRA_TYPES:
            print(f"  {name:<52}  {INFRA_TYPES[t]}");     n_infra += 1
        elif t in FLOAT_TYPES:
            print(f"  {name:<52}  {FLOAT_TYPES[t]}");     n_fp32 += 1
    print("=" * 82)
    print(f"  INT8: {n_int8}  |  FP32: {n_fp32}  |  Infra: {n_infra}")
    print("=" * 82)
    print()
    print("QUANTIZATION POLICY:")
    print("  LayerNorm    -> FP32 (qconfig=None): absent from all quantized namespaces")
    print("  Linear fc1/fc2  -> INT8 static (nnq.Linear): quantized activation+weight")
    print("  MultiheadAttention -> INT8 static: internal proj Linears become nnq.Linear")
    print("  LayerNormIsland: DeQuant->LayerNorm(FP32)->Quant micro-boundary per norm")
    print()


# ---------------------------------------------------------------------------
# Main PTQ
# ---------------------------------------------------------------------------
def run_static_ptq():
    print("=" * 60)
    print("Building MobileViT-XXS ...")
    model = MobileViT_XXS()
    model.eval()

    assign_qconfigs(model, backend='fbgemm')
    print("QConfigs assigned (static everywhere, LayerNorm=None, classifier=None).")

    print("Preparing ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_prepared = prepare(model)

    print("Calibrating (10 random batches) ...")
    with torch.no_grad():
        for i in range(10):
            model_prepared(torch.randn(2, 3, 224, 224))
            if (i+1) % 5 == 0: print(f"  Batch {i+1}/10")
    print("Calibration done.")

    print("Converting ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_q = convert(model_prepared)
    print("Conversion done.")

    print("Inference test ...")
    with torch.no_grad():
        out = model_q(torch.randn(2, 3, 224, 224))
    print(f"Output shape: {out.shape}")

    print_quantization_summary(model_q)

    def size_mb(m):
        torch.save(m.state_dict(), "/tmp/_tmp.pt")
        return os.path.getsize("/tmp/_tmp.pt") / 1e6

    fp32 = size_mb(MobileViT_XXS())
    int8 = size_mb(model_q)
    print("=" * 60)
    print(f"FP32: {fp32:.2f} MB  |  INT8: {int8:.2f} MB  |  Ratio: {fp32/int8:.2f}x")
    print("=" * 60)
    print("Static PTQ completed successfully!")
    return model_q


if __name__ == "__main__":
    quantized_model = run_static_ptq()