"""
MVT-accV3_fused.py
==================
MobileViT-XXS NumPy Inference Accelerator — BN-Fused Version

Architecture : MobileViT-XXS (Apple, ICLR 2022)
Fusion       : Conv + BatchNorm fused at runtime into a single linear op
               W_fused = W * (gamma/std),  b_fused = beta - mean*(gamma/std)
               This removes all BN division from the inference path.
Activation   : Swish / SiLU  (x * sigmoid(x))
Precision    : float32 throughout (golden reference model)
Purpose      : Software golden model for FPGA RTL verification.
               Every output here should match the RTL to within floating-point
               rounding tolerance before taping out.

File layout
-----------
  Section 1  — Primitive helpers       (padding, conv, BN, fuse)
  Section 2  — Activations             (sigmoid, swish, softmax)
  Section 3  — MV2 block               (expand → depthwise → project)
  Section 4  — Transformer pieces      (MHA, MLP, encoder)
  Section 5  — MobileViT block         (LocalRep, unfold, transformer, fold, fusion)
  Section 6  — Head + classifier       (AvgPool, Linear)
  Section 7  — Full MobileViT_XXS_()   (top-level inference function)

Usage
-----
    from Mapping_PretraindModel_NOapi import extract_all_weights
    weights, bn_prams = extract_all_weights(model)
    logits = MobileViT_XXS_(image_np, bn_prams, **weights)
"""

import numpy as np
import math


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PRIMITIVE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def zero_pad(X, pad):
    """
    Zero-pad the spatial dimensions (H, W) of a feature map.

    Parameters
    ----------
    X   : ndarray, shape (m, C, H, W)
    pad : int — number of zeros added on each side of H and W

    Returns
    -------
    X_pad : ndarray, shape (m, C, H + 2*pad, W + 2*pad)
    """
    return np.pad(X, ((0,0), (0,0), (pad,pad), (pad,pad)),
                  mode='constant', constant_values=0)


def conv_single_step(a_slice_prev, W, b):
    """
    Compute one output element of a convolution: dot(a_slice, W) + b.

    Parameters
    ----------
    a_slice_prev : ndarray, shape (C_in, f, f)  — input window
    W            : ndarray, shape (C_in, f, f)  — one filter
    b            : ndarray, shape (1, 1, 1)      — scalar bias

    Returns
    -------
    float — single convolution output value
    """
    return float(np.sum(a_slice_prev * W) + b.item())


def conv_forward(A_prev, W, b, stride, pad):
    """
    Standard 2-D convolution forward pass (all channels, all spatial positions).

    Parameters
    ----------
    A_prev : ndarray, shape (m, C_in, H_in, W_in)
    W      : ndarray, shape (C_out, C_in, f, f)
    b      : ndarray, shape (C_out, 1, 1, 1)
    stride : int
    pad    : int

    Returns
    -------
    Z : ndarray, shape (m, C_out, H_out, W_out)
        where H_out = (H_in - f + 2*pad) / stride + 1
    """
    m, n_C_prev, n_H_prev, n_W_prev = A_prev.shape
    f   = W.shape[2]
    n_C = W.shape[0]
    n_H = int((n_H_prev - f + 2*pad) / stride + 1)
    n_W = int((n_W_prev - f + 2*pad) / stride + 1)

    Z = np.zeros((m, n_C, n_H, n_W))
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            vs, ve = h*stride, h*stride + f
            for w in range(n_W):
                hs, he = w*stride, w*stride + f
                for c in range(n_C):
                    Z[i, c, h, w] = conv_single_step(
                        a_prev_pad[:, vs:ve, hs:he], W[c], b[c])
    return Z


def depthwise_conv(A_prev, W, b, stride, pad):
    """
    Depthwise convolution: each input channel is convolved with its own kernel.
    No cross-channel mixing — groups = C_in = C_out.

    Parameters
    ----------
    A_prev : ndarray, shape (m, C, H_in, W_in)
    W      : ndarray, shape (C, f, f)   — one kernel per channel
    b      : ndarray, shape (C,)        — one bias per channel
    stride : int
    pad    : int

    Returns
    -------
    A : ndarray, shape (m, C, H_out, W_out)
    """
    m, n_C_prev, n_H_prev, n_W_prev = A_prev.shape
    f   = W.shape[2]
    n_H = int((n_H_prev - f + 2*pad) / stride + 1)
    n_W = int((n_W_prev - f + 2*pad) / stride + 1)
    A   = np.zeros((m, n_C_prev, n_H, n_W))

    A_prev_p = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev = A_prev_p[i]
        for h in range(n_H):
            vs, ve = h*stride, h*stride + f
            for w in range(n_W):
                hs, he = w*stride, w*stride + f
                for c in range(n_C_prev):
                    A[i, c, h, w] = np.sum(a_prev[c, vs:ve, hs:he] * W[c]) + b[c]
    return A


def batch_norm_forward(Z, gamma, beta, running_mean, running_var, eps=1e-5):
    """
    BatchNorm inference pass (uses frozen running statistics, NOT batch stats).
    Kept for compatibility — in the fused path this is bypassed.

    Parameters
    ----------
    Z            : ndarray, shape (m, C, H, W)
    gamma        : ndarray, shape (C,)   — learned scale
    beta         : ndarray, shape (C,)   — learned shift
    running_mean : ndarray, shape (C,)
    running_var  : ndarray, shape (C,)
    eps          : float — numerical stability constant

    Returns
    -------
    Z_norm : ndarray, same shape as Z
    """
    gamma = gamma.reshape(1, -1, 1, 1)
    beta  = beta.reshape(1, -1, 1, 1)
    mean  = running_mean.reshape(1, -1, 1, 1)
    var   = running_var.reshape(1, -1, 1, 1)
    return gamma * (Z - mean) / np.sqrt(var + eps) + beta


# ── BN Fusion helpers ────────────────────────────────────────────────────────

def batch_norm_to_linear(gamma, beta, running_mean, running_var, eps=1e-5):
    """
    Convert BN parameters into a per-channel affine transform (A, B) such that:

        BN(Z) = A * Z + B

    where A = gamma / std  and  B = beta - gamma * mean / std.

    This lets us absorb BN into the preceding conv/linear weight:
        fused_output = A * (W*x + b) + B = (A*W)*x + (A*b + B)

    Parameters
    ----------
    gamma, beta         : ndarray, shape (C,)
    running_mean/var    : ndarray, shape (C,)

    Returns
    -------
    A : ndarray, shape (1, C, 1, 1)  — per-channel scale
    B : ndarray, shape (1, C, 1, 1)  — per-channel shift
    """
    gamma = gamma.reshape(1, -1, 1, 1)
    beta  = beta.reshape(1, -1, 1, 1)
    mean  = running_mean.reshape(1, -1, 1, 1)
    var   = running_var.reshape(1, -1, 1, 1)

    std = np.sqrt(var + eps)
    A   = gamma / std
    B   = beta - (gamma * mean / std)
    return A, B


def fuse_layer_bn(W, b, A, B):
    """
    Absorb BN affine (A, B) into a preceding conv or linear layer weight.

    Works for any weight shape:
        Linear   : W is (C_out, C_in)
        Conv     : W is (C_out, C_in, f, f)
        Depthwise: W is (C, f, f)

    Math
    ----
        W_fused = A * W          (A broadcast along all dims after C_out)
        b_fused = A * b + B

    After fusion, the layer computes:
        A*(W*x + b) + B  ==  W_fused*x + b_fused
    and BN is completely gone from the forward path.

    Parameters
    ----------
    W : ndarray — weight tensor (any shape, first dim = output channels)
    b : ndarray — bias tensor (same shape as original bias)
    A : ndarray — BN scale,  shape (1, C, 1, 1) or broadcastable
    B : ndarray — BN shift,  shape (1, C, 1, 1) or broadcastable

    Returns
    -------
    W_fused : ndarray, same shape as W
    b_fused : ndarray, same shape as b
    """
    A_flat = A.flatten()   # (C_out,)
    B_flat = B.flatten()   # (C_out,)
    b_flat = b.flatten()
    b_shape = b.shape

    # Reshape A to (C_out, 1, 1, ...) to broadcast over W's trailing dims
    w_broadcast_shape = (-1,) + (1,) * (W.ndim - 1)
    W_fused = W * A_flat.reshape(w_broadcast_shape)

    # b_fused = A * b + B
    b_fused = (A_flat * b_flat + B_flat).reshape(b_shape)

    return W_fused, b_fused


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ACTIVATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def sigmoid(x):
    """
    Numerically stable sigmoid: avoids overflow for large negative x.
    Uses two branches:  1/(1+e^-x)  for x>=0,  e^x/(1+e^x)  for x<0.
    """
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def swish(x):
    """
    Swish / SiLU activation: f(x) = x * sigmoid(x).
    Used after every BN-fused conv in MobileViT-XXS.
    Note: pw_project in MV2 has NO activation (linear projection).
    """
    return x * sigmoid(x)


def softmax(x, axis=-1):
    """
    Numerically stable softmax along the given axis.
    Subtracts the max before exp to prevent overflow — does not change output.

    Parameters
    ----------
    x    : ndarray
    axis : int — axis along which probabilities sum to 1 (default: last)

    Returns
    -------
    ndarray, same shape as x, values in (0, 1) summing to 1 along axis
    """
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def layer_norm_forward(X, gamma, beta, eps=1e-5):
    """
    LayerNorm: normalize over the last dimension (d_model), then scale+shift.
    Used inside transformer blocks — NOT fusible because it normalizes
    over runtime activations, not fixed training statistics.

    Parameters
    ----------
    X     : ndarray, shape (..., d_model)
    gamma : ndarray, shape (d_model,)
    beta  : ndarray, shape (d_model,)

    Returns
    -------
    out : ndarray, same shape as X
    """
    mean = np.mean(X, axis=-1, keepdims=True)
    var  = np.var(X,  axis=-1, keepdims=True)
    return gamma * (X - mean) / np.sqrt(var + eps) + beta


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MV2 BLOCK  (MobileNetV2 inverted residual)
# ═══════════════════════════════════════════════════════════════════════════════

def MobileNet_v2(Input,
                 W_expan, W_depth, W_point,
                 b_expan, b_depth, b_point,
                 stride_depth,
                 W_gamma_expan, W_gamma_depth, W_gamma_point,
                 b_gamma_expan, b_gamma_depth, b_gamma_point,
                 running_mean1, running_var1,
                 running_mean2, running_var2,
                 running_mean3, running_var3):
    """
    MobileNetV2 inverted residual block with BN fused at runtime.

    Data flow
    ---------
        Input
          │
          ├──[save for residual]
          │
          ▼
        1×1 conv (expand channels by ratio 2)
          └─ BN fused → Swish
          ▼
        3×3 depthwise conv (stride=stride_depth)
          └─ BN fused → Swish
          ▼
        1×1 conv (project back, NO activation)
          └─ BN fused
          ▼
        (+) residual add  ← only if stride=1 AND in_ch == out_ch
          ▼
        Output

    BN fusion is done per-layer via batch_norm_to_linear + fuse_layer_bn.
    The BN parameters are consumed here — nothing is stored.

    Parameters
    ----------
    Input        : ndarray, shape (m, C_in, H, W)
    W_expan      : ndarray, shape (C_hid, C_in, 1, 1)
    W_depth      : ndarray, shape (C_hid, f, f)         — squeezed depthwise
    W_point      : ndarray, shape (C_out, C_hid, 1, 1)
    b_*          : bias tensors (matching shapes)
    stride_depth : int — 1 keeps spatial size, 2 halves it
    W_gamma_*/b_gamma_* : BN gamma/beta for each sub-layer
    running_mean*/var*  : BN running stats (3 pairs: expand, depth, point)

    Returns
    -------
    ndarray, shape (m, C_out, H_out, W_out)
    """
    # ── Fuse BN into each conv layer at runtime ──────────────────────────────
    A_expan, B_expan = batch_norm_to_linear(W_gamma_expan, b_gamma_expan, running_mean1, running_var1)
    A_depth, B_depth = batch_norm_to_linear(W_gamma_depth, b_gamma_depth, running_mean2, running_var2)
    A_point, B_point = batch_norm_to_linear(W_gamma_point, b_gamma_point, running_mean3, running_var3)

    W_expan_f, b_expan_f = fuse_layer_bn(W_expan, b_expan, A_expan, B_expan)
    W_depth_f, b_depth_f = fuse_layer_bn(W_depth, b_depth, A_depth, B_depth)
    W_point_f, b_point_f = fuse_layer_bn(W_point, b_point, A_point, B_point)

    # ── Forward pass (no BN calls — already absorbed) ────────────────────────
    # 1×1 expand: increase channel depth
    Z1 = conv_forward(Input, W_expan_f, b_expan_f, 1, 0)
    A1 = swish(Z1)

    # 3×3 depthwise: spatial filtering, one kernel per channel
    Z2 = depthwise_conv(A1, W_depth_f, b_depth_f, stride_depth, 1)
    A2 = swish(Z2)

    # 1×1 project: reduce back to output channels (NO activation on this layer)
    Z3 = conv_forward(A2, W_point_f, b_point_f, 1, 0)

    # Residual add: only when spatial size and channel count are unchanged
    if stride_depth == 1 and Input.shape == Z3.shape:
        return Z3 + Input
    return Z3


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRANSFORMER PIECES
# ═══════════════════════════════════════════════════════════════════════════════

def MultiHeadAttention(X, W, W_o, Bias, Bias_o, num_heads):
    """
    Multi-Head Self-Attention (MHSA) for the MobileViT transformer block.

    The QKV projection is fused into a single matrix multiply using the
    concatenated in_proj_weight from PyTorch's MultiheadAttention:
        W  shape: (3*d_model, d_model)  → splits into W_Q, W_K, W_V
        W_o shape: (d_model, d_model)   → output projection

    Attention formula (per head):
        Attention(Q, K, V) = softmax(Q K^T / sqrt(head_dim)) V

    Parameters
    ----------
    X        : ndarray, shape (B, N, d_model)   B=batch*P, N=num_patches
    W        : ndarray, shape (3*d_model, d_model)
    W_o      : ndarray, shape (d_model, d_model)
    Bias     : ndarray, shape (3*d_model,)
    Bias_o   : ndarray, shape (d_model,)
    num_heads: int

    Returns
    -------
    ndarray, shape (B, N, d_model)
    """
    batch_size, seq_len, d_model = X.shape
    head_dim = d_model // num_heads

    # ── QKV projection (single matmul, then split) ───────────────────────────
    QKV = np.matmul(X, W.T) + Bias.reshape(1, 1, -1)
    Q = QKV[:, :,          0 :   d_model]
    K = QKV[:, :,   d_model : 2*d_model]
    V = QKV[:, :, 2*d_model : 3*d_model]

    # ── Reshape to (B, heads, N, head_dim) ───────────────────────────────────
    Q = Q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    # ── Scaled dot-product attention ─────────────────────────────────────────
    Q      = Q / np.sqrt(head_dim)                    # scale before matmul
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2))   # (B, heads, N, N)
    attn   = softmax(scores, axis=-1)                 # attention weights

    # ── Weighted sum of values ────────────────────────────────────────────────
    context = np.matmul(attn, V)                                          # (B, heads, N, head_dim)
    context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    # ── Output projection ─────────────────────────────────────────────────────
    return np.matmul(context, W_o.T) + Bias_o.reshape(1, 1, -1)


def MLP(X, W1_fc1, b1_fc1, W2_fc2, b2_fc2):
    """
    Transformer MLP (Feed-Forward Network): two linear layers with Swish.

        fc1: d_model → mlp_dim  (mlp_dim = 2 * d_model in MobileViT-XXS)
        fc2: mlp_dim → d_model

    Parameters
    ----------
    X       : ndarray, shape (B, N, d_model)
    W1_fc1  : ndarray, shape (mlp_dim, d_model)
    b1_fc1  : ndarray, shape (mlp_dim,)
    W2_fc2  : ndarray, shape (d_model, mlp_dim)
    b2_fc2  : ndarray, shape (d_model,)

    Returns
    -------
    ndarray, shape (B, N, d_model)
    """
    Z1 = np.matmul(X, W1_fc1.T) + b1_fc1    # expand
    A1 = swish(Z1)                            # non-linearity
    return np.matmul(A1, W2_fc2.T) + b2_fc2  # project back


def transformer_encoder(X,
                         W_QKV_atten, W_o_atten, Bias_QKV_atten, Bias_o_atten,
                         W1_fc1, b1_fc1, W2_fc2, b2_fc2,
                         W_gamma1, b_beta1, W_gamma2, b_beta2,
                         num_heads):
    """
    Single transformer encoder layer (Pre-LN variant as used in MobileViT).

    Data flow
    ---------
        x
        │
        ├── skip1 = x
        ▼
       LayerNorm1
        ▼
       MultiHeadAttention
        ▼
      (+) skip1        ← attention residual
        │
        ├── skip2 = x
        ▼
       LayerNorm2
        ▼
       MLP  (fc1 → Swish → fc2)
        ▼
      (+) skip2        ← MLP residual
        ▼
       output

    Note: LayerNorm is NOT fusible (normalizes over runtime activations).

    Parameters
    ----------
    X              : ndarray, shape (B, N, d_model)
    W_QKV_atten    : ndarray, shape (3*d_model, d_model)
    W_o_atten      : ndarray, shape (d_model, d_model)
    Bias_QKV_atten : ndarray, shape (3*d_model,)
    Bias_o_atten   : ndarray, shape (d_model,)
    W1_fc1, W2_fc2 : MLP weights
    b1_fc1, b2_fc2 : MLP biases
    W_gamma1/2     : LayerNorm scale (gamma) for sub-layer 1 and 2
    b_beta1/2      : LayerNorm shift (beta)  for sub-layer 1 and 2
    num_heads      : int

    Returns
    -------
    ndarray, shape (B, N, d_model)
    """
    # ── Attention sub-block ───────────────────────────────────────────────────
    skip1 = X
    X = layer_norm_forward(X, W_gamma1, b_beta1)
    X = MultiHeadAttention(X, W_QKV_atten, W_o_atten,
                           Bias_QKV_atten, Bias_o_atten, num_heads)
    X = X + skip1                    # attention residual add

    # ── MLP sub-block ─────────────────────────────────────────────────────────
    skip2 = X
    X = layer_norm_forward(X, W_gamma2, b_beta2)
    X = MLP(X, W1_fc1, b1_fc1, W2_fc2, b2_fc2)
    return X + skip2                 # MLP residual add


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MOBILEVIT BLOCK
# Sub-sections: LocalRep │ Unfold │ Transformer │ Fold │ Fusion
# ═══════════════════════════════════════════════════════════════════════════════

# ── 5a. Local Representation ─────────────────────────────────────────────────

def Local_representations(input,
                           W_loacal_3x3, b_local_3x3,
                           W_loacal_1x1, b_local_1x1,
                           W_gamma_local_3x3, W_beta_local_3x3,
                           running_mean1, running_var1):
    """
    Local feature extraction stage of the MobileViT block.

    Data flow
    ---------
        input
          ▼
        3×3 conv → BN fused → Swish    (spatial feature extraction)
          ▼
        1×1 conv                        (channel projection to d_model, no act)
          ▼
        output  shape: (m, d_model, H, W)

    Note: conv2 (1×1) has NO BatchNorm and NO activation in the original model.

    Parameters
    ----------
    input              : ndarray, shape (m, C_in, H, W)
    W_loacal_3x3       : ndarray, shape (C_out, C_in, 3, 3)
    b_local_3x3        : ndarray, shape (C_out, 1, 1, 1)
    W_loacal_1x1       : ndarray, shape (d_model, C_out, 1, 1)
    b_local_1x1        : ndarray, shape (d_model, 1, 1, 1)
    W_gamma_local_3x3  : BN gamma for conv1
    W_beta_local_3x3   : BN beta  for conv1
    running_mean/var1  : BN running stats for conv1

    Returns
    -------
    ndarray, shape (m, d_model, H, W)
    """
    # Fuse BN into the 3×3 conv weight and bias
    A_local, B_local = batch_norm_to_linear(
        W_gamma_local_3x3, W_beta_local_3x3, running_mean1, running_var1)
    W_fused, b_fused = fuse_layer_bn(W_loacal_3x3, b_local_3x3, A_local, B_local)

    # 3×3 conv (fused) → Swish
    x = conv_forward(input, W_fused, b_fused, 1, 1)
    x = swish(x)

    # 1×1 projection: no BN, no activation
    x = conv_forward(x, W_loacal_1x1, b_local_1x1, 1, 0)
    return x


# ── 5b. Bilinear resize (for unfolding when H/W not divisible by patch_size) ──

def bilinear_resize(x, new_h, new_w):
    """
    Bilinear interpolation matching PyTorch's F.interpolate(align_corners=False).
    Used only when feature map dimensions are not divisible by patch_size.

    Parameters
    ----------
    x      : ndarray, shape (m, C, H, W)
    new_h  : int — target height
    new_w  : int — target width

    Returns
    -------
    ndarray, shape (m, C, new_h, new_w)
    """
    b, c, h, w = x.shape
    # Compute sampling coordinates (align_corners=False: pixel centers at 0.5)
    row_idx = (np.arange(new_h) + 0.5) * (h / new_h) - 0.5
    col_idx = (np.arange(new_w) + 0.5) * (w / new_w) - 0.5
    row_idx = np.clip(row_idx, 0, h - 1)
    col_idx = np.clip(col_idx, 0, w - 1)

    r0 = np.floor(row_idx).astype(int)
    r1 = np.minimum(r0 + 1, h - 1)
    c0 = np.floor(col_idx).astype(int)
    c1 = np.minimum(c0 + 1, w - 1)

    # Fractional offsets for bilinear weights
    dr = (row_idx - r0)[:, None]   # (new_h, 1)
    dc = (col_idx - c0)[None, :]   # (1, new_w)

    # Four-corner bilinear interpolation
    top_left     = x[:, :, r0, :][:, :, :, c0]
    top_right    = x[:, :, r0, :][:, :, :, c1]
    bottom_left  = x[:, :, r1, :][:, :, :, c0]
    bottom_right = x[:, :, r1, :][:, :, :, c1]

    out = (top_left     * (1 - dr) * (1 - dc) +
           top_right    * (1 - dr) *      dc  +
           bottom_left  *      dr  * (1 - dc) +
           bottom_right *      dr  *      dc)
    return out.astype(x.dtype)


# ── 5c. Unfold ────────────────────────────────────────────────────────────────

def unfold(x, patch_size=2):
    """
    Rearrange a feature map from spatial [C, H, W] layout into patch-sequence
    [B*P, N, C] layout for the transformer.

    Each patch covers a (patch_h × patch_w) = (2×2) pixel region.
    The transformer sees N patches, each as a C-dimensional token.

    Reshaping sequence
    ------------------
        [B, C, H, W]
          → [B*C*n_ph, ph, n_pw, pw]     split H into n_ph blocks of size ph
          → [B*C*n_ph, n_pw, ph, pw]     transpose
          → [B, C, N, P]                 N = n_ph*n_pw patches, P = ph*pw pixels
          → [B, P, N, C]                 transpose: pixel-first grouping
          → [B*P, N, C]                  flatten B and P for transformer

    Parameters
    ----------
    x          : ndarray, shape (B, C, H, W)
    patch_size : int — spatial size of each square patch (default 2)

    Returns
    -------
    patches : ndarray, shape (B*P, N, C)
    info    : dict — metadata needed to reverse the operation in fold()
    """
    b, c, h, w = x.shape
    ph = pw = patch_size

    # Pad to nearest multiple of patch_size (matches MVT.py ceil logic)
    new_h = math.ceil(h / ph) * ph
    new_w = math.ceil(w / pw) * pw
    interpolate = False
    if new_h != h or new_w != w:
        x = bilinear_resize(x, new_h, new_w)
        interpolate = True

    patch_area  = ph * pw
    num_patch_h = new_h // ph
    num_patch_w = new_w // pw
    num_patches = num_patch_h * num_patch_w

    # Reshape into patch layout
    x       = x.reshape(b * c * num_patch_h, ph, num_patch_w, pw)
    x       = x.transpose(0, 2, 1, 3)
    x       = x.reshape(b, c, num_patches, patch_area)
    x       = x.transpose(0, 3, 2, 1)
    patches = x.reshape(b * patch_area, num_patches, c)

    info = dict(b=b, c=c, orig_h=h, orig_w=w,
                num_patch_h=num_patch_h, num_patch_w=num_patch_w,
                num_patches=num_patches, patch_area=patch_area,
                ph=ph, pw=pw, interpolate=interpolate)
    return patches, info


# ── 5d. Fold ──────────────────────────────────────────────────────────────────

def fold(patches, info):
    """
    Reverse of unfold(): rearrange transformer output [B*P, N, C] back to
    spatial feature map [B, C, H, W].

    Reshaping sequence (exact inverse of unfold)
    ------
        [B*P, N, C]
          → [B, P, N, C]
          → [B, C, N, P]
          → [B*C*n_ph, n_pw, ph, pw]
          → [B*C*n_ph, ph, n_pw, pw]
          → [B, C, H_new, W_new]
          → bilinear resize back to (H, W)  if interpolation was applied

    Parameters
    ----------
    patches : ndarray, shape (B*P, N, C)
    info    : dict — must be the exact dict returned by unfold()

    Returns
    -------
    ndarray, shape (B, C, H, W)
    """
    b           = info['b']
    num_patch_h = info['num_patch_h']
    num_patch_w = info['num_patch_w']
    patch_area  = info['patch_area']
    ph          = info['ph']
    pw          = info['pw']
    channels    = patches.shape[-1]

    x = patches.reshape(b, patch_area, info['num_patches'], channels)
    x = x.transpose(0, 3, 2, 1)
    x = x.reshape(b * channels * num_patch_h, num_patch_w, ph, pw)
    x = x.transpose(0, 2, 1, 3)
    x = x.reshape(b, channels, num_patch_h * ph, num_patch_w * pw)

    # Undo padding interpolation if it was applied during unfold
    if info['interpolate']:
        x = bilinear_resize(x, info['orig_h'], info['orig_w'])
    return x


# ── 5e. Fusion ────────────────────────────────────────────────────────────────

def fusion(x_global, input_feat,
           W_fusion_1x1, b_fusion_1x1,
           W_fusion_3x3, b_fusion_3x3,
           W_gamma_f_1x1, b_beta_f_1x1,
           W_gamma_f_3x3, b_beta_f_3x3,
           running_mean1, running_var1,
           running_mean2, running_var2):
    """
    Fusion block: merge transformer global features with the original local
    feature map saved at the start of the MobileViT block.

    Data flow
    ---------
        x_global  (from fold)     input_feat  (saved residual)
             │                          │
        1×1 conv                        │
        BN fused → Swish                │
             │                          │
             └──── concat(input, x) ───┘   channel-wise concat
                         │
                    3×3 conv
                    BN fused → Swish
                         │
                       output

    Parameters
    ----------
    x_global      : ndarray, shape (B, d_model, H, W) — transformer output
    input_feat    : ndarray, shape (B, C, H, W)       — original input residual
    W_fusion_1x1  : ndarray, shape (C, d_model, 1, 1)
    b_fusion_1x1  : ndarray, shape (C, 1, 1, 1)
    W_fusion_3x3  : ndarray, shape (C, 2*C, 3, 3)
    b_fusion_3x3  : ndarray, shape (C, 1, 1, 1)
    W_gamma/b_beta/running_*: BN params for 1×1 and 3×3 convs

    Returns
    -------
    ndarray, shape (B, C, H, W)
    """
    # Fuse BN into both conv layers
    A_1x1, B_1x1 = batch_norm_to_linear(W_gamma_f_1x1, b_beta_f_1x1, running_mean1, running_var1)
    A_3x3, B_3x3 = batch_norm_to_linear(W_gamma_f_3x3, b_beta_f_3x3, running_mean2, running_var2)
    W_1x1_f, b_1x1_f = fuse_layer_bn(W_fusion_1x1, b_fusion_1x1, A_1x1, B_1x1)
    W_3x3_f, b_3x3_f = fuse_layer_bn(W_fusion_3x3, b_fusion_3x3, A_3x3, B_3x3)

    # Project global features from d_model → C channels
    x = conv_forward(x_global, W_1x1_f, b_1x1_f, 1, 0)
    x = swish(x)

    # Spatial dimensions must match for concat — raise early if not
    if x.shape[2:] != input_feat.shape[2:]:
        raise ValueError(
            f"Spatial mismatch in fusion: global {x.shape[2:]} vs local {input_feat.shape[2:]}")

    # Concatenate local + global along channel axis → [B, 2C, H, W]
    concat = np.concatenate([input_feat, x], axis=1)

    # Mix with 3×3 conv → back to C channels
    x = conv_forward(concat, W_3x3_f, b_3x3_f, 1, 1)
    x = swish(x)
    return x


# ── 5f. Full MobileViT Block ──────────────────────────────────────────────────

def MobileViTBlock_(input,
                    # Local representation weights
                    W_loacal_3x3, b_local_3x3,
                    W_loacal_1x1, b_local_1x1,
                    W_gamma_local_3x3, W_beta_local_3x3,
                    # Transformer layer weights (lists of length L)
                    W_QKV_atten, W_o_atten, Bias_QKV_atten, Bias_o_atten,
                    W1_fc1, b1_fc1, W2_fc2, b2_fc2,
                    W_gamma1, b_beta1, W_gamma2, b_beta2,
                    # Final LayerNorm (applied after all transformer layers)
                    W_norm_gamma, b_norm_beta,
                    # Fusion weights
                    W_fusion_1x1, b_fusion_1x1,
                    W_fusion_3x3, b_fusion_3x3,
                    W_gamma_f_1x1, b_beta_f_1x1,
                    W_gamma_f_3x3, b_beta_f_3x3,
                    # Block configuration
                    L, patch_size, num_heads,
                    # BN running statistics (3 pairs: local-3x3, fusion-1x1, fusion-3x3)
                    running_mean1, running_var1,    # local 3×3 conv BN
                    running_mean2, running_var2,    # fusion 1×1 conv BN
                    running_mean3, running_var3):   # fusion 3×3 conv BN
    """
    Full MobileViT Block combining local CNN features with global transformer
    context via the unfold → transformer → fold pipeline.

    Data flow
    ---------
        input  (B, C, H, W)
          │
          ├──[save input as fusion residual]
          │
          ▼
        LocalRep   conv3x3(BN-fused)+Swish → conv1x1
          ▼
        Unfold     [B,C,H,W] → [B*P, N, C]   (address remap, no compute)
          ▼
        Transformer × L layers
            (LayerNorm → MHA → residual) + (LayerNorm → MLP → residual)
          ▼
        Final LayerNorm
          ▼
        Fold       [B*P, N, C] → [B, C, H, W]   (inverse of unfold)
          ▼
        Fusion     concat(input, folded) → conv1x1 → conv3x3
          ▼
        output  (B, C, H, W)

    Parameters
    ----------
    L          : int — number of transformer encoder layers (2, 4, or 3)
    patch_size : int — patch height = width (always 2 in MobileViT-XXS)
    num_heads  : int — number of attention heads (2 or 4)
    """
    # Step 1 — Local CNN feature extraction
    x_local = Local_representations(
        input,
        W_loacal_3x3, b_local_3x3,
        W_loacal_1x1, b_local_1x1,
        W_gamma_local_3x3, W_beta_local_3x3,
        running_mean1, running_var1)

    # Step 2 — Unfold: rearrange spatial map into patch tokens
    patches, info = unfold(x_local, patch_size)

    # Step 3 — L transformer encoder layers (each with MHA + MLP + residuals)
    for i in range(L):
        patches = transformer_encoder(
            patches,
            W_QKV_atten[i], W_o_atten[i],
            Bias_QKV_atten[i], Bias_o_atten[i],
            W1_fc1[i], b1_fc1[i], W2_fc2[i], b2_fc2[i],
            W_gamma1[i], b_beta1[i], W_gamma2[i], b_beta2[i],
            num_heads)

    # Step 4 — Final LayerNorm on the full patch sequence
    patches = layer_norm_forward(patches, W_norm_gamma, b_norm_beta)

    # Step 5 — Fold: reassemble patches back into spatial feature map
    x_global = fold(patches, info)

    # Step 6 — Fusion: merge global transformer output with original local input
    x_out = fusion(
        x_global, input,
        W_fusion_1x1, b_fusion_1x1,
        W_fusion_3x3, b_fusion_3x3,
        W_gamma_f_1x1, b_beta_f_1x1,
        W_gamma_f_3x3, b_beta_f_3x3,
        running_mean2, running_var2,
        running_mean3, running_var3)

    return x_out


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — HEAD + CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def global_avg_pool(x):
    """
    Global average pooling: reduce spatial (H, W) → scalar per channel.
    Equivalent to PyTorch AdaptiveAvgPool2d(1) + flatten.

    Parameters
    ----------
    x : ndarray, shape (m, C, H, W)

    Returns
    -------
    ndarray, shape (m, C)
    """
    return np.mean(x, axis=(2, 3))


def linear_classifier(x, W_cls, b_cls):
    """
    Linear (fully connected) classification head.
    Computes: logits = x @ W_cls + b_cls

    Parameters
    ----------
    x     : ndarray, shape (m, C)              — pooled features
    W_cls : ndarray, shape (C, num_classes)    — weight (already transposed by caller)
    b_cls : ndarray, shape (num_classes,)

    Returns
    -------
    logits : ndarray, shape (m, num_classes)
    """
    return np.matmul(x, W_cls) + b_cls


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — FULL MobileViT_XXS INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def MobileViT_XXS_(Inputs, bn_prams,
                   # ── Stem ──────────────────────────────────────────────────
                   W_stem, b_stem, W_stem_gamma, b_beta_stem,
                   # ── Stage 1: MV2 (16→16, stride=1, residual) ─────────────
                   W_expan_1,  W_depth_1,  W_point_1,
                   b_expan_1,  b_depth_1,  b_point_1,
                   W_gamma_expan_1, W_gamma_depth_1, W_gamma_point_1,
                   b_gamma_expan_1, b_gamma_depth_1, b_gamma_point_1,
                   # ── Stage 2a: MV2 (16→24, stride=2, no residual) ─────────
                   W_expan_2a, W_depth_2a, W_point_2a,
                   b_expan_2a, b_depth_2a, b_point_2a,
                   W_gamma_expan_2a, W_gamma_depth_2a, W_gamma_point_2a,
                   b_gamma_expan_2a, b_gamma_depth_2a, b_gamma_point_2a,
                   # ── Stage 2b: MV2 (24→24, stride=1, residual) ────────────
                   W_expan_2b, W_depth_2b, W_point_2b,
                   b_expan_2b, b_depth_2b, b_point_2b,
                   W_gamma_expan_2b, W_gamma_depth_2b, W_gamma_point_2b,
                   b_gamma_expan_2b, b_gamma_depth_2b, b_gamma_point_2b,
                   # ── Stage 2c: MV2 (24→24, stride=1, residual) ────────────
                   W_expan_2c, W_depth_2c, W_point_2c,
                   b_expan_2c, b_depth_2c, b_point_2c,
                   W_gamma_expan_2c, W_gamma_depth_2c, W_gamma_point_2c,
                   b_gamma_expan_2c, b_gamma_depth_2c, b_gamma_point_2c,
                   # ── Stage 3a: MV2 (24→48, stride=2, no residual) ─────────
                   W_expan_3a, W_depth_3a, W_point_3a,
                   b_expan_3a, b_depth_3a, b_point_3a,
                   W_gamma_expan_3a, W_gamma_depth_3a, W_gamma_point_3a,
                   b_gamma_expan_3a, b_gamma_depth_3a, b_gamma_point_3a,
                   # ── Stage 3b: MobileViTBlock (48ch, dim=64, L=2, heads=2) ─
                   W_loacal_3x3_3b, b_local_3x3_3b,
                   W_loacal_1x1_3b, b_local_1x1_3b,
                   W_gamma_local_3x3_3b, W_beta_local_3x3_3b,
                   W_QKV_atten_3b, W_o_atten_3b, Bias_QKV_atten_3b, Bias_o_atten_3b,
                   W1_fc1_3b, b1_fc1_3b, W2_fc2_3b, b2_fc2_3b,
                   W_gamma1_3b, b_beta1_3b, W_gamma2_3b, b_beta2_3b,
                   W_norm_gamma_3b, b_norm_beta_3b,
                   W_fusion_1x1_3b, b_fusion_1x1_3b,
                   W_fusion_3x3_3b, b_fusion_3x3_3b,
                   W_gamma_f_1x1_3b, b_beta_f_1x1_3b,
                   W_gamma_f_3x3_3b, b_beta_f_3x3_3b,
                   # ── Stage 4a: MV2 (48→64, stride=2, no residual) ─────────
                   W_expan_4a, W_depth_4a, W_point_4a,
                   b_expan_4a, b_depth_4a, b_point_4a,
                   W_gamma_expan_4a, W_gamma_depth_4a, W_gamma_point_4a,
                   b_gamma_expan_4a, b_gamma_depth_4a, b_gamma_point_4a,
                   # ── Stage 4b: MobileViTBlock (64ch, dim=80, L=4, heads=4) ─
                   W_loacal_3x3_4b, b_local_3x3_4b,
                   W_loacal_1x1_4b, b_local_1x1_4b,
                   W_gamma_local_3x3_4b, W_beta_local_3x3_4b,
                   W_QKV_atten_4b, W_o_atten_4b, Bias_QKV_atten_4b, Bias_o_atten_4b,
                   W1_fc1_4b, b1_fc1_4b, W2_fc2_4b, b2_fc2_4b,
                   W_gamma1_4b, b_beta1_4b, W_gamma2_4b, b_beta2_4b,
                   W_norm_gamma_4b, b_norm_beta_4b,
                   W_fusion_1x1_4b, b_fusion_1x1_4b,
                   W_fusion_3x3_4b, b_fusion_3x3_4b,
                   W_gamma_f_1x1_4b, b_beta_f_1x1_4b,
                   W_gamma_f_3x3_4b, b_beta_f_3x3_4b,
                   # ── Stage 5a: MV2 (64→80, stride=2, no residual) ─────────
                   W_expan_5a, W_depth_5a, W_point_5a,
                   b_expan_5a, b_depth_5a, b_point_5a,
                   W_gamma_expan_5a, W_gamma_depth_5a, W_gamma_point_5a,
                   b_gamma_expan_5a, b_gamma_depth_5a, b_gamma_point_5a,
                   # ── Stage 5b: MobileViTBlock (80ch, dim=96, L=3, heads=4) ─
                   W_loacal_3x3_5b, b_local_3x3_5b,
                   W_loacal_1x1_5b, b_local_1x1_5b,
                   W_gamma_local_3x3_5b, W_beta_local_3x3_5b,
                   W_QKV_atten_5b, W_o_atten_5b, Bias_QKV_atten_5b, Bias_o_atten_5b,
                   W1_fc1_5b, b1_fc1_5b, W2_fc2_5b, b2_fc2_5b,
                   W_gamma1_5b, b_beta1_5b, W_gamma2_5b, b_beta2_5b,
                   W_norm_gamma_5b, b_norm_beta_5b,
                   W_fusion_1x1_5b, b_fusion_1x1_5b,
                   W_fusion_3x3_5b, b_fusion_3x3_5b,
                   W_gamma_f_1x1_5b, b_beta_f_1x1_5b,
                   W_gamma_f_3x3_5b, b_beta_f_3x3_5b,
                   # ── Head: 1×1 conv (80→320) + BN + Swish ─────────────────
                   W_head, b_head, W_gamma_head, b_beta_head,
                   # ── Classifier: Linear (320→1000) ─────────────────────────
                   W_cls, b_cls):
    """
    Full MobileViT-XXS inference — BN-fused version.

    Network structure
    -----------------
        Input  [1, 3, 224, 224]
          │
        Stem   3×3 conv, stride=2  →  [1, 16, 112, 112]
          │
        Stage 1  MV2(16→16, s=1)  →  [1, 16, 112, 112]
          │
        Stage 2  MV2(16→24, s=2)  →  [1, 24,  56,  56]
                 MV2(24→24, s=1)     (residual)
                 MV2(24→24, s=1)     (residual)
          │
        Stage 3  MV2(24→48, s=2)  →  [1, 48,  28,  28]
                 MobileViTBlock(48, dim=64, L=2, heads=2)
          │
        Stage 4  MV2(48→64, s=2)  →  [1, 64,  14,  14]
                 MobileViTBlock(64, dim=80, L=4, heads=4)
          │
        Stage 5  MV2(64→80, s=2)  →  [1, 80,   7,   7]
                 MobileViTBlock(80, dim=96, L=3, heads=4)
          │
        Head   1×1 conv (80→320) → BN fused → Swish
          │
        AvgPool  [1, 320,  7,  7]  →  [1, 320]
          │
        Classifier  Linear(320→1000)
          │
        Logits  [1, 1000]

    BN fusion
    ---------
    BN is fused at the call site of each sub-function (stem, MV2 layers, head).
    bn_prams[i] holds the frozen running statistics needed for fusion.
    After fusion the BN operation is a no-op — only the fused weight matters.

    Parameters
    ----------
    Inputs    : ndarray, shape (m, 3, 224, 224) — normalized image batch
    bn_prams  : list of 64 ndarray — (mean, var) pairs in order:
                  [0,1] stem | [2..7] mv2_1 | [8..13] mv2_2a | [14..19] mv2_2b
                  [20..25] mv2_2c | [26..31] mv2_3a | [32..37] mvit_3b
                  [38..43] mv2_4a | [44..49] mvit_4b | [50..55] mv2_5a
                  [56..61] mvit_5b | [62..63] head

    Returns
    -------
    logits : ndarray, shape (m, 1000)
    """

    # ── Stem: 3×3 conv, stride=2  [3,224,224] → [16,112,112] ─────────────────
    A_stem, B_stem = batch_norm_to_linear(W_stem_gamma, b_beta_stem, bn_prams[0], bn_prams[1])
    W_stem_f, b_stem_f = fuse_layer_bn(W_stem, b_stem, A_stem, B_stem)
    x = conv_forward(Inputs, W_stem_f, b_stem_f, 2, 1)
    x = swish(x)

    # ── Stage 1: MV2(16→16, stride=1, residual)  [16,112,112] ────────────────
    x = MobileNet_v2(x, W_expan_1, W_depth_1, W_point_1,
                     b_expan_1, b_depth_1, b_point_1, 1,
                     W_gamma_expan_1, W_gamma_depth_1, W_gamma_point_1,
                     b_gamma_expan_1, b_gamma_depth_1, b_gamma_point_1,
                     *bn_prams[2:8])

    # ── Stage 2: 3× MV2  [16,112,112] → [24,56,56] ───────────────────────────
    x = MobileNet_v2(x, W_expan_2a, W_depth_2a, W_point_2a,        # stride=2, no residual
                     b_expan_2a, b_depth_2a, b_point_2a, 2,
                     W_gamma_expan_2a, W_gamma_depth_2a, W_gamma_point_2a,
                     b_gamma_expan_2a, b_gamma_depth_2a, b_gamma_point_2a,
                     *bn_prams[8:14])
    x = MobileNet_v2(x, W_expan_2b, W_depth_2b, W_point_2b,        # stride=1, residual
                     b_expan_2b, b_depth_2b, b_point_2b, 1,
                     W_gamma_expan_2b, W_gamma_depth_2b, W_gamma_point_2b,
                     b_gamma_expan_2b, b_gamma_depth_2b, b_gamma_point_2b,
                     *bn_prams[14:20])
    x = MobileNet_v2(x, W_expan_2c, W_depth_2c, W_point_2c,        # stride=1, residual
                     b_expan_2c, b_depth_2c, b_point_2c, 1,
                     W_gamma_expan_2c, W_gamma_depth_2c, W_gamma_point_2c,
                     b_gamma_expan_2c, b_gamma_depth_2c, b_gamma_point_2c,
                     *bn_prams[20:26])

    # ── Stage 3: MV2 + MobileViTBlock  [24,56,56] → [48,28,28] ──────────────
    x = MobileNet_v2(x, W_expan_3a, W_depth_3a, W_point_3a,        # stride=2
                     b_expan_3a, b_depth_3a, b_point_3a, 2,
                     W_gamma_expan_3a, W_gamma_depth_3a, W_gamma_point_3a,
                     b_gamma_expan_3a, b_gamma_depth_3a, b_gamma_point_3a,
                     *bn_prams[26:32])
    x = MobileViTBlock_(x,                                          # L=2, heads=2, dim=64
                        W_loacal_3x3_3b, b_local_3x3_3b,
                        W_loacal_1x1_3b, b_local_1x1_3b,
                        W_gamma_local_3x3_3b, W_beta_local_3x3_3b,
                        W_QKV_atten_3b, W_o_atten_3b, Bias_QKV_atten_3b, Bias_o_atten_3b,
                        W1_fc1_3b, b1_fc1_3b, W2_fc2_3b, b2_fc2_3b,
                        W_gamma1_3b, b_beta1_3b, W_gamma2_3b, b_beta2_3b,
                        W_norm_gamma_3b, b_norm_beta_3b,
                        W_fusion_1x1_3b, b_fusion_1x1_3b,
                        W_fusion_3x3_3b, b_fusion_3x3_3b,
                        W_gamma_f_1x1_3b, b_beta_f_1x1_3b,
                        W_gamma_f_3x3_3b, b_beta_f_3x3_3b,
                        2, 2, 2,                                    # L, patch_size, num_heads
                        *bn_prams[32:38])

    # ── Stage 4: MV2 + MobileViTBlock  [48,28,28] → [64,14,14] ──────────────
    x = MobileNet_v2(x, W_expan_4a, W_depth_4a, W_point_4a,        # stride=2
                     b_expan_4a, b_depth_4a, b_point_4a, 2,
                     W_gamma_expan_4a, W_gamma_depth_4a, W_gamma_point_4a,
                     b_gamma_expan_4a, b_gamma_depth_4a, b_gamma_point_4a,
                     *bn_prams[38:44])
    x = MobileViTBlock_(x,                                          # L=4, heads=4, dim=80
                        W_loacal_3x3_4b, b_local_3x3_4b,
                        W_loacal_1x1_4b, b_local_1x1_4b,
                        W_gamma_local_3x3_4b, W_beta_local_3x3_4b,
                        W_QKV_atten_4b, W_o_atten_4b, Bias_QKV_atten_4b, Bias_o_atten_4b,
                        W1_fc1_4b, b1_fc1_4b, W2_fc2_4b, b2_fc2_4b,
                        W_gamma1_4b, b_beta1_4b, W_gamma2_4b, b_beta2_4b,
                        W_norm_gamma_4b, b_norm_beta_4b,
                        W_fusion_1x1_4b, b_fusion_1x1_4b,
                        W_fusion_3x3_4b, b_fusion_3x3_4b,
                        W_gamma_f_1x1_4b, b_beta_f_1x1_4b,
                        W_gamma_f_3x3_4b, b_beta_f_3x3_4b,
                        4, 2, 4,                                    # L, patch_size, num_heads
                        *bn_prams[44:50])

    # ── Stage 5: MV2 + MobileViTBlock  [64,14,14] → [80,7,7] ────────────────
    x = MobileNet_v2(x, W_expan_5a, W_depth_5a, W_point_5a,        # stride=2
                     b_expan_5a, b_depth_5a, b_point_5a, 2,
                     W_gamma_expan_5a, W_gamma_depth_5a, W_gamma_point_5a,
                     b_gamma_expan_5a, b_gamma_depth_5a, b_gamma_point_5a,
                     *bn_prams[50:56])
    x = MobileViTBlock_(x,                                          # L=3, heads=4, dim=96
                        W_loacal_3x3_5b, b_local_3x3_5b,
                        W_loacal_1x1_5b, b_local_1x1_5b,
                        W_gamma_local_3x3_5b, W_beta_local_3x3_5b,
                        W_QKV_atten_5b, W_o_atten_5b, Bias_QKV_atten_5b, Bias_o_atten_5b,
                        W1_fc1_5b, b1_fc1_5b, W2_fc2_5b, b2_fc2_5b,
                        W_gamma1_5b, b_beta1_5b, W_gamma2_5b, b_beta2_5b,
                        W_norm_gamma_5b, b_norm_beta_5b,
                        W_fusion_1x1_5b, b_fusion_1x1_5b,
                        W_fusion_3x3_5b, b_fusion_3x3_5b,
                        W_gamma_f_1x1_5b, b_beta_f_1x1_5b,
                        W_gamma_f_3x3_5b, b_beta_f_3x3_5b,
                        3, 2, 4,                                    # L, patch_size, num_heads
                        *bn_prams[56:62])

    # ── Head: 1×1 conv (80→320) + BN fused + Swish  [80,7,7] → [320,7,7] ────
    A_head, B_head = batch_norm_to_linear(W_gamma_head, b_beta_head, bn_prams[62], bn_prams[63])
    W_head_f, b_head_f = fuse_layer_bn(W_head, b_head, A_head, B_head)
    x = conv_forward(x, W_head_f, b_head_f, 1, 0)
    x = swish(x)

    # ── Global Average Pool  [320,7,7] → [320] ────────────────────────────────
    x = global_avg_pool(x)

    # ── Classifier  [320] → [1000] ────────────────────────────────────────────
    # W_cls from extract_all_weights() is (1000, 320) — transpose to (320, 1000)
    # so that  x @ W_cls.T  ==  x @ (1000,320).T  ==  x @ (320,1000)
    W_cls = W_cls.T
    return linear_classifier(x, W_cls, b_cls)