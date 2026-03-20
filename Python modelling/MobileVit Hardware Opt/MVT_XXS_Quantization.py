# ═══════════════════════════════════════════════════════════════════════════════
# Quantized MobileViT-XXS — Pure NumPy Inference Engine
#
# Quantization scheme summary:
#   Weights      : symmetric  int8,  per-channel  (Z_w = 0, no online overhead)
#   Activations  : asymmetric uint8, per-tensor   (expand / project / stem / head)
#   Activations  : asymmetric uint8, per-channel  (depthwise ONLY — mathematically
#                  valid because channel c output depends only on channel c input)
#   Transformer  : symmetric  int8  weights, asymmetric uint8 activations (per-tensor)
#   Bias         : int32, quantized to accumulator scale S_w * S_x
#   Zero-point   : folded into bias offline — inner MAC loop stays clean
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import math


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — QUANTIZATION PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

def quantize_symmetric(W_fp32):
    """
    Per-tensor symmetric int8 quantization for weights.
    Maps fp32 values to [-127, 127] using a single scale factor.

    Formula:  S_w = max(|W|) / 127
              W_q = clip( round(W / S_w), -127, 127 )

    Z_w = 0 by definition — eliminates online zero-point overhead in MACs.
    Used for: transformer QKV weights, attention output, MLP weights, classifier.

    Parameters:  W_fp32 : ndarray float32 (any shape)
    Returns:     W_q : int8 (same shape),  S_w : float scalar
    """
    abs_max = np.max(np.abs(W_fp32))
    S_w     = float(abs_max / 127.0) if abs_max > 1e-10 else 1.0
    W_q     = np.clip(np.round(W_fp32 / S_w), -127, 127).astype(np.int8)
    return W_q, S_w


def quantize_symmetric_per_channel(W_fp32):
    """
    Per-channel symmetric int8 quantization for weights.
    Each output channel c gets its own scale S_w[c] = max(|W[c]|) / 127.

    Advantage over per-tensor: channels with very different ranges each use
    their full 254-level resolution instead of sharing one global scale
    dominated by the largest channel. Improves SNR by ~3-5 dB in practice.

    Works for any weight shape — the broadcast shape is inferred from W.ndim:
        conv:      (C_out, C_in, f, f)  →  S_w shape (C_out, 1, 1, 1)
        depthwise: (C, f, f)            →  S_w shape (C, 1, 1)
        pointwise: (C_out, C_in, 1, 1)  →  S_w shape (C_out, 1, 1, 1)

    Used for: all conv layers (stem, expand, depthwise, project, fusion, head).

    Parameters:  W_fp32 : ndarray float32, first dim = C_out
    Returns:     W_q : int8 (same shape),  S_w : float array (C_out,)
    """
    n_C     = W_fp32.shape[0]
    W_flat  = W_fp32.reshape(n_C, -1)
    abs_max = np.max(np.abs(W_flat), axis=1)                        # (C_out,)
    S_w     = np.where(abs_max > 1e-10, abs_max / 127.0, 1.0)      # (C_out,)
    S_w_bc  = S_w.reshape((n_C,) + (1,) * (W_fp32.ndim - 1))       # broadcast shape
    W_q     = np.clip(np.round(W_fp32 / S_w_bc), -127, 127).astype(np.int8)
    return W_q, S_w


def quantize_asymmetric(X_fp32):
    """
    Per-tensor asymmetric uint8 quantization for activations and inputs.
    Maps fp32 values to [0, 255] using scale S_x and zero-point Z_x.

    Formula:  S_x = (max - min) / 255
              Z_x = clip( round(-min / S_x), 0, 255 )
              X_q = clip( round(X / S_x) + Z_x, 0, 255 )

    Asymmetric is preferred for activations because post-ReLU/Swish outputs
    are non-negative — symmetric would waste half the range on negative values.
    Z_x ≠ 0 introduces a correction term that is folded into bias offline
    (see fold_zx_into_bias / fold_zx_into_bias_per_channel).

    Used for: stem input, expand input, project input, fusion inputs, head input,
              transformer inputs, classifier input.

    Parameters:  X_fp32 : ndarray float32 (any shape)
    Returns:     X_q : uint8 (same shape),  S_x : float scalar,  Z_x : int scalar
    """
    x_min = float(X_fp32.min())
    x_max = float(X_fp32.max())
    r     = x_max - x_min
    S_x   = r / 255.0 if r > 1e-10 else 1e-10
    Z_x   = int(np.clip(np.round(-x_min / S_x), 0, 255))
    X_q   = np.clip(np.round(X_fp32 / S_x) + Z_x, 0, 255).astype(np.uint8)
    return X_q, S_x, Z_x


def quantize_asymmetric_per_channel_act(X_fp32):
    """
    Per-channel asymmetric uint8 quantization for activations.

    WHY per-channel here? After BN fusion, different channels can have
    wildly different ranges (e.g. ch12 max=153 vs ch31 max=0.21 — a 700×
    difference). Per-tensor quantization calibrates to the largest channel,
    wasting resolution on all others. Per-channel gives each channel its
    own S_x[c] / Z_x[c], recovering that lost resolution.

    WHY only for depthwise? Per-channel activation quantization is only
    mathematically valid when each output element depends on exactly ONE
    input channel — i.e. depthwise conv. For regular conv, output channel
    c_out mixes ALL input channels (each on a different scale), making the
    accumulator scale ambiguous. Depthwise output[c] = W[c] * X[c] only,
    so the scale factors out cleanly: Y = S_w[c] * S_x[c] * acc[c].

    Parameters:  X_fp32 : ndarray float32, shape (m, C, H, W)
    Returns:     X_q : uint8 (same shape),
                 S_x : float array (C,),
                 Z_x : int   array (C,)
    """
    m, C, H, W = X_fp32.shape
    S_x = np.zeros(C, dtype=np.float32)
    Z_x = np.zeros(C, dtype=np.int32)
    X_q = np.zeros_like(X_fp32, dtype=np.uint8)
    for c in range(C):
        ch     = X_fp32[:, c, :, :]
        x_min  = float(ch.min())
        x_max  = float(ch.max())
        r      = x_max - x_min
        S_x[c] = r / 255.0 if r > 1e-10 else 1e-10
        Z_x[c] = int(np.clip(np.round(-x_min / S_x[c]), 0, 255))
        X_q[:, c, :, :] = np.clip(
            np.round(ch / S_x[c]) + Z_x[c], 0, 255).astype(np.uint8)
    return X_q, S_x, Z_x


# ───────────────────────────────────────────────────────────────────────────────
# BIAS QUANTIZATION
# ───────────────────────────────────────────────────────────────────────────────

def quantize_bias_int32(b_fp32, S_w, S_x):
    """
    Quantize bias to int32 on the accumulator scale S_w * S_x (per-tensor).

    The bias must be on the same integer scale as the MAC accumulator so it
    can be added directly without any conversion:
        b_q = round( b_fp32 / (S_w * S_x) )

    Used for: transformer matmuls, MLP layers, classifier — all per-tensor.

    Parameters:  b_fp32 : float32 (C_out,),  S_w / S_x : float scalars
    Returns:     ndarray int32 (C_out,)
    """
    return np.round(b_fp32.flatten() / (S_w * S_x)).astype(np.int32)


def quantize_bias_int32_per_channel(b_fp32, S_w, S_x):
    """
    Quantize bias to int32 — per-channel weights, scalar activation scale.

    Each output channel c has its own weight scale S_w[c], so:
        b_q[c] = round( b_fp32[c] / (S_w[c] * S_x) )

    Used for: all conv layers with per-channel weight quantization
              (stem, expand, project, fusion, local_rep, head).

    Parameters:  b_fp32 : float32 (C_out,)
                 S_w    : float array (C_out,)  — per-channel weight scales
                 S_x    : float scalar           — single activation scale
    Returns:     ndarray int32 (C_out,)
    """
    return np.round(b_fp32.flatten() / (S_w * S_x)).astype(np.int32)


def quantize_bias_int32_per_channel_act(b_fp32, S_w, S_x):
    """
    Quantize bias to int32 — per-channel weights AND per-channel activations.

    For depthwise with per-channel activation:
        b_q[c] = round( b_fp32[c] / (S_w[c] * S_x[c]) )

    Used for: depthwise conv only (the only layer using per-channel activations).

    Parameters:  b_fp32 : float32 (C,)
                 S_w    : float array (C,)  — per-channel weight scales
                 S_x    : float array (C,)  — per-channel activation scales
    Returns:     ndarray int32 (C,)
    """
    return np.round(b_fp32.flatten() / (S_w * S_x)).astype(np.int32)


# ───────────────────────────────────────────────────────────────────────────────
# ZERO-POINT FOLDING INTO BIAS
# ───────────────────────────────────────────────────────────────────────────────

def fold_zx_into_bias(b_int32, W_q, Z_x):
    """
    Fold asymmetric activation zero-point correction into bias (OFFLINE step).

    From the quantized matmul equation:
        Y ≈ S_w * S_x * (W_q · X_q) - S_w * S_x * Z_x * Σ W_q
                                        └─────────────────────┘
                                         correction term → folded here

    After folding:
        b_adjusted[c] = b_int32[c] - Z_x * sum( W_q[c, ...] )

    The inner MAC loop then computes only W_q · X_q — no Z_x subtraction
    at runtime, keeping the hot path clean and hardware-friendly.

    Used for: transformer matmuls, MLP, classifier (all per-tensor Z_x).

    Parameters:  b_int32 : int32 (C_out,)
                 W_q     : int8, first dim = C_out
                 Z_x     : int scalar
    Returns:     int32 (C_out,)
    """
    if Z_x == 0:
        return b_int32                                      # symmetric input — nothing to fold
    sum_axes = tuple(range(1, W_q.ndim))                   # all dims except C_out
    W_sum    = W_q.astype(np.int32).sum(axis=sum_axes)     # (C_out,)
    return b_int32 - Z_x * W_sum


def fold_zx_into_bias_per_channel(b_int32, W_q, Z_x):
    """
    Fold scalar activation zero-point into bias for per-channel weight layers.

    Identical math to fold_zx_into_bias — works for any weight shape because
    the sum is over all dims except the first (C_out):
        b_adjusted[c] = b_int32[c] - Z_x * sum( W_q[c, ...] )

    Used for: stem, expand, project, fusion 1×1 and 3×3, local_rep, head.

    Parameters:  b_int32 : int32 (C_out,)
                 W_q     : int8, first dim = C_out  (any number of trailing dims)
                 Z_x     : int scalar
    Returns:     int32 (C_out,)
    """
    if Z_x == 0:
        return b_int32
    sum_axes = tuple(range(1, W_q.ndim))
    W_sum    = W_q.astype(np.int32).sum(axis=sum_axes)     # (C_out,)
    return b_int32 - Z_x * W_sum


def fold_zx_per_channel_act(b_int32, W_q, Z_x):
    """
    Fold per-channel activation zero-points into bias for depthwise conv.

    Each depthwise channel c has its own zero-point Z_x[c]:
        b_adjusted[c] = b_int32[c] - Z_x[c] * sum( W_q[c, f, f] )

    Vectorized: W_sum[c] = sum of all 9 (f×f) weights in channel c.

    Used for: depthwise conv only.

    Parameters:  b_int32 : int32 (C,)
                 W_q     : int8  (C, f, f)
                 Z_x     : int   array (C,)
    Returns:     int32 (C,)
    """
    W_sum = W_q.astype(np.int32).reshape(W_q.shape[0], -1).sum(axis=1)  # (C,)
    return b_int32 - Z_x * W_sum


# ───────────────────────────────────────────────────────────────────────────────
# DEQUANTIZATION
# ───────────────────────────────────────────────────────────────────────────────

def dequantize(acc_int32, S_w, S_x):
    """
    Dequantize int32 accumulator → fp32 (per-tensor).

    After fold_zx_into_bias the Z_x correction is already inside the
    accumulator, so dequantization is a single scale multiply:
        Y_fp32 = S_w * S_x * acc_int32

    One fp multiply per output element — done ONCE after all MACs finish,
    before any nonlinear activation.

    Used for: transformer matmuls, MLP, classifier.

    Parameters:  acc_int32 : ndarray int32 (any shape)
                 S_w / S_x : float scalars
    Returns:     ndarray float32 (same shape)
    """
    return acc_int32.astype(np.float32) * float(S_w * S_x)


def dequantize_per_channel(acc_int32, S_w, S_x):
    """
    Dequantize int32 accumulator → fp32, per-channel weight scale.

    Each output channel c scales by its own S_w[c]:
        Y_fp32[c] = S_w[c] * S_x * acc_int32[c]

    The scale vector is broadcast over (m, C, H, W) by reshaping to (1, C, 1, 1).

    Used for: all conv layers (stem, expand, project, fusion, local_rep, head).

    Parameters:  acc_int32 : int32  (m, C, H, W)
                 S_w       : float  array (C,)
                 S_x       : float  scalar
    Returns:     float32 (m, C, H, W)
    """
    scale = (S_w * S_x).reshape(1, -1, 1, 1)
    return acc_int32.astype(np.float32) * scale.astype(np.float32)


def dequantize_depthwise_perch_act(acc_int32, S_w, S_x):
    """
    Dequantize int32 accumulator → fp32, per-channel weight AND activation scale.

    For depthwise with per-channel activations:
        Y_fp32[c] = S_w[c] * S_x[c] * acc_int32[c]

    Both S_w and S_x are (C,) vectors — element-wise multiply, then broadcast.

    Used for: depthwise conv only.

    Parameters:  acc_int32 : int32  (m, C, H, W)
                 S_w       : float  array (C,)
                 S_x       : float  array (C,)
    Returns:     float32 (m, C, H, W)
    """
    scale = (S_w * S_x).reshape(1, -1, 1, 1)
    return acc_int32.astype(np.float32) * scale.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PADDING
# ═══════════════════════════════════════════════════════════════════════════════

def zero_pad(X, pad):
    """
    Zero-pad spatial dimensions (H, W) of a float32 feature map.
    Used by fp32 conv and depthwise conv reference implementations.

    Parameters:  X   : float32 (m, C, H, W)
                 pad : int — pixels to add on each side
    Returns:     float32 (m, C, H+2*pad, W+2*pad)
    """
    return np.pad(X, ((0,0),(0,0),(pad,pad),(pad,pad)),
                  mode='constant', constant_values=0)


def zero_pad_q(X, pad, Z_x=0):
    """
    Pad spatial dimensions of a quantized feature map with the activation zero-point.

    WHY Z_x instead of 0? In quantized space, the integer Z_x represents the
    real value 0.0. Padding with literal 0 would inject non-zero real values,
    corrupting border convolution results.

    Used by conv_forward_q (scalar Z_x) and depthwise_conv_q (scalar Z_x).
    depthwise_conv_q_perch_act handles per-channel padding internally.

    Parameters:  X   : uint8 (m, C, H, W)
                 pad : int
                 Z_x : int scalar (default 0 for symmetric activations)
    Returns:     uint8 (m, C, H+2*pad, W+2*pad)
    """
    return np.pad(X, ((0,0),(0,0),(pad,pad),(pad,pad)),
                  mode='constant', constant_values=Z_x)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CONVOLUTION KERNELS
# ═══════════════════════════════════════════════════════════════════════════════

def conv_single_step(a_slice_prev, W, b):
    """
    FP32 single-step convolution — applies one filter to one spatial patch.

    Computes the dot product between a_slice_prev and W, adds bias scalar.
    Called inside conv_forward's innermost loop.

    Parameters:  a_slice_prev : float32 (C_in, f, f)
                 W            : float32 (C_in, f, f)
                 b            : float32 scalar (1,1,1)
    Returns:     float scalar
    """
    return float(np.sum(a_slice_prev * W) + b.item())


def conv_forward(A_prev, W, b, stride, pad):
    """
    FP32 standard convolution — reference implementation.

    Used as the ground-truth baseline for SNR comparisons and in
    Local_representations / fusion (not yet quantized paths).

    Parameters:  A_prev : float32 (m, C_in, H, W)
                 W      : float32 (C_out, C_in, f, f)
                 b      : float32 (C_out, 1, 1, 1)
                 stride, pad : int
    Returns:     float32 (m, C_out, H_out, W_out)
    """
    m, n_C_prev, n_H_prev, n_W_prev = A_prev.shape
    n_C = W.shape[0];  f = W.shape[2]
    n_H = int((n_H_prev - f + 2*pad) / stride + 1)
    n_W = int((n_W_prev - f + 2*pad) / stride + 1)
    Z          = np.zeros((m, n_C, n_H, n_W))
    A_prev_pad = zero_pad(A_prev, pad)
    for i in range(m):
        a = A_prev_pad[i]
        for h in range(n_H):
            vs, ve = h*stride, h*stride+f
            for w in range(n_W):
                hs, he = w*stride, w*stride+f
                for c in range(n_C):
                    Z[i,c,h,w] = conv_single_step(a[:, vs:ve, hs:he], W[c], b[c])
    return Z


def depthwise_conv(A_prev, W, b, stride, pad):
    """
    FP32 depthwise convolution — reference implementation.

    Each input channel c is convolved with its own filter W[c] independently.
    No cross-channel mixing — output channel count equals input channel count.

    Parameters:  A_prev : float32 (m, C, H, W)
                 W      : float32 (C, f, f)   — one filter per channel
                 b      : float32 (C,)
                 stride, pad : int
    Returns:     float32 (m, C, H_out, W_out)
    """
    m, n_C, n_H_prev, n_W_prev = A_prev.shape
    f   = W.shape[1]
    n_H = int((n_H_prev - f + 2*pad) / stride + 1)
    n_W = int((n_W_prev - f + 2*pad) / stride + 1)
    A          = np.zeros((m, n_C, n_H, n_W))
    A_prev_pad = zero_pad(A_prev, pad)
    for i in range(m):
        a = A_prev_pad[i]
        for h in range(n_H):
            vs, ve = h*stride, h*stride+f
            for w in range(n_W):
                hs, he = w*stride, w*stride+f
                for c in range(n_C):
                    A[i,c,h,w] = np.sum(a[c, vs:ve, hs:he] * W[c]) + b[c]
    return A


def conv_single_step_q(a_slice_prev, W, b):
    """
    Quantized single-step convolution — integer arithmetic only.

    Computes int8 * int8 element-wise products, accumulates in int32,
    adds int32 bias (which already has Z_x correction folded in).

    Parameters:  a_slice_prev : uint8 (C_in, f, f)
                 W            : int8  (C_in, f, f)
                 b            : int32 scalar
    Returns:     int32 scalar
    """
    s = np.sum(a_slice_prev.astype(np.int32) * W.astype(np.int32))
    return s + int(b)


def conv_forward_q(A_prev, W, b, stride, pad, z_x):
    """
    Quantized standard convolution — integer accumulation.

    Pads input with scalar zero-point z_x, then accumulates int8*int8
    products into int32. Bias is pre-quantized and Z_x-folded.
    Dequantization happens outside this function (caller's responsibility).

    Parameters:  A_prev : uint8 (m, C_in, H, W)
                 W      : int8  (C_out, C_in, f, f)
                 b      : int32 (C_out,) — already Z_x folded
                 stride, pad : int
                 z_x    : int scalar — activation zero-point for padding
    Returns:     int32 (m, C_out, H_out, W_out)
    """
    m, n_C_prev, n_H_prev, n_W_prev = A_prev.shape
    n_C, _, f, _ = W.shape
    n_H = int((n_H_prev - f + 2*pad) / stride) + 1
    n_W = int((n_W_prev - f + 2*pad) / stride) + 1
    Z          = np.zeros((m, n_C, n_H, n_W), dtype=np.int32)
    A_prev_pad = zero_pad_q(A_prev, pad, z_x)
    for i in range(m):
        a = A_prev_pad[i]
        for h in range(n_H):
            vs, ve = h*stride, h*stride+f
            for w in range(n_W):
                hs, he = w*stride, w*stride+f
                patch  = a[:, vs:ve, hs:he].astype(np.int32)
                for c in range(n_C):
                    Z[i,c,h,w] = int(np.sum(patch * W[c].astype(np.int32))) + int(b[c])
    return Z


def depthwise_conv_q(A_prev, W, b, stride, pad, Z_x):
    """
    Quantized depthwise convolution — scalar zero-point padding.

    Per-tensor activation version (single Z_x for all channels).
    Kept for reference / fallback. Production path uses depthwise_conv_q_perch_act.

    Parameters:  A_prev : uint8 (m, C, H, W)
                 W      : int8  (C, f, f)
                 b      : int32 (C,) — Z_x folded
                 stride, pad : int
                 Z_x    : int scalar
    Returns:     int32 (m, C, H_out, W_out)
    """
    m, n_C, n_H_prev, n_W_prev = A_prev.shape
    f   = W.shape[1]
    n_H = int((n_H_prev - f + 2*pad) / stride + 1)
    n_W = int((n_W_prev - f + 2*pad) / stride + 1)
    Z          = np.zeros((m, n_C, n_H, n_W), dtype=np.int32)
    A_prev_pad = zero_pad_q(A_prev, pad, Z_x)
    for i in range(m):
        a = A_prev_pad[i]
        for h in range(n_H):
            vs, ve = h*stride, h*stride+f
            for w in range(n_W):
                hs, he = w*stride, w*stride+f
                for c in range(n_C):
                    Z[i,c,h,w] = (
                        int(np.sum(a[c, vs:ve, hs:he].astype(np.int32)
                                   * W[c].astype(np.int32))) + int(b[c]))
    return Z


def depthwise_conv_q_perch_act(A_prev, W, b, stride, pad, Z_x_vec):
    """
    Quantized depthwise convolution — per-channel activation zero-points.

    Each channel c is padded with its own Z_x_vec[c] to correctly represent
    real-value 0 in that channel's quantized domain. This is the production
    path used in MobileNet_v2.

    The per-channel padding is done inside the loop rather than via zero_pad_q
    because each channel needs a different constant_values argument.

    Parameters:  A_prev   : uint8 (m, C, H, W)
                 W        : int8  (C, f, f)
                 b        : int32 (C,) — per-channel Z_x folded via fold_zx_per_channel_act
                 stride, pad : int
                 Z_x_vec  : int array (C,) — per-channel zero-points
    Returns:     int32 (m, C, H_out, W_out)
    """
    m, n_C, n_H_prev, n_W_prev = A_prev.shape
    f   = W.shape[1]
    n_H = int((n_H_prev - f + 2*pad) / stride + 1)
    n_W = int((n_W_prev - f + 2*pad) / stride + 1)
    Z   = np.zeros((m, n_C, n_H, n_W), dtype=np.int32)
    for i in range(m):
        for c in range(n_C):
            ch_pad = np.pad(A_prev[i, c],
                            ((pad,pad),(pad,pad)),
                            mode='constant',
                            constant_values=int(Z_x_vec[c]))
            w_ch = W[c].astype(np.int32)
            for h in range(n_H):
                vs, ve = h*stride, h*stride+f
                for w_i in range(n_W):
                    hs, he = w_i*stride, w_i*stride+f
                    Z[i,c,h,w_i] = (
                        int(np.sum(ch_pad[vs:ve, hs:he].astype(np.int32) * w_ch))
                        + int(b[c]))
    return Z


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BATCH NORM & FUSION
# ═══════════════════════════════════════════════════════════════════════════════

def batch_norm_forward(Z, gamma, beta, running_mean, running_var, eps=1e-5):
    """
    Batch normalization forward pass — inference mode (uses running statistics).

    BN formula:  Z_norm = gamma * (Z - mean) / sqrt(var + eps) + beta

    NOTE: This is only used in layers that are NOT quantized (e.g. the fp32
    head path if needed). For quantized layers, BN is fused offline into
    the conv weights via batch_norm_to_linear + fuse_layer_bn.

    Parameters:  Z            : float32 (m, C, H, W)
                 gamma, beta  : float32 (C,)
                 running_mean, running_var : float32 (C,)
    Returns:     float32 (m, C, H, W)
    """
    gamma = gamma.reshape(1,-1,1,1);  beta = beta.reshape(1,-1,1,1)
    mean  = running_mean.reshape(1,-1,1,1)
    var   = running_var.reshape(1,-1,1,1)
    return gamma * (Z - mean) / np.sqrt(var + eps) + beta


def batch_norm_to_linear(gamma, beta, running_mean, running_var, eps=1e-5):
    """
    Convert BN parameters to an affine transform: Output = A * Z + B.

    At inference, BN is: gamma * (Z - mean) / std + beta
    This can be rewritten as:  A * Z + B  where:
        A = gamma / std          (per-channel scale)
        B = beta - gamma*mean/std (per-channel shift)

    This A and B are then absorbed into the preceding conv layer's W and b
    via fuse_layer_bn, eliminating the BN computation entirely at runtime.

    Parameters:  gamma, beta : float32 (C,)
                 running_mean, running_var : float32 (C,)
    Returns:     A : float32 (1, C, 1, 1),  B : float32 (1, C, 1, 1)
    """
    gamma = gamma.reshape(1,-1,1,1);  beta = beta.reshape(1,-1,1,1)
    mean  = running_mean.reshape(1,-1,1,1)
    var   = running_var.reshape(1,-1,1,1)
    std   = np.sqrt(var + eps)
    return gamma / std, beta - (gamma * mean / std)


def fuse_layer_bn(W, b, A, B):
    """
    Fuse BN affine transform into conv/depthwise/linear weights and bias.

    Given the BN transform Output = A * (W·x + b) + B, we can pre-compute:
        W_fused = A * W      (scale each output channel's kernel)
        b_fused = A * b + B  (scale and shift the bias)

    After fusion the layer computes W_fused·x + b_fused, which is identical
    to conv + BN but requires no separate BN step at inference.

    Works for any weight shape:
        (C_out, C_in, f, f) — regular conv
        (C, f, f)           — depthwise conv
        (C_out, C_in)       — linear

    Parameters:  W : float32 (C_out, ...)
                 b : float32 matching W's first dim, any trailing shape
                 A : float32 (1, C_out, 1, 1) — BN scale from batch_norm_to_linear
                 B : float32 (1, C_out, 1, 1) — BN shift from batch_norm_to_linear
    Returns:     W_fused : float32 (same shape as W)
                 b_fused : float32 (same shape as b)
    """
    A_flat = A.flatten();  B_flat = B.flatten();  b_flat = b.flatten()
    W_fused = W * A_flat.reshape((-1,) + (1,) * (W.ndim - 1))
    b_fused = (A_flat * b_flat + B_flat).reshape(b.shape)
    return W_fused, b_fused


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ACTIVATIONS & NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def sigmoid(x):
    """
    Numerically stable sigmoid function.

    Naive implementation: 1 / (1 + exp(-x)) overflows for large negative x,
    and exp(x) / (1 + exp(x)) overflows for large positive x.

    np.where evaluates BOTH branches for every element before selecting,
    causing overflow warnings even when the result would be discarded.

    Fix: use boolean indexing to evaluate each branch only where it's valid:
        x >= 0 : use 1 / (1 + exp(-x))     — safe, exp of non-positive
        x <  0 : use exp(x) / (1 + exp(x)) — safe, exp of negative

    Parameters:  x : ndarray float32 (any shape)
    Returns:     ndarray float32 (same shape), values in (0, 1)
    """
    out       = np.empty_like(x)
    pos       = x >= 0
    out[pos]  = 1 / (1 + np.exp(-x[pos]))
    out[~pos] = np.exp(x[~pos]) / (1 + np.exp(x[~pos]))
    return out


def swish(x):
    """
    Swish activation: f(x) = x * sigmoid(x).

    Swish is smooth, non-monotonic, and empirically outperforms ReLU on
    deep networks. Used throughout MobileViT-XXS after every conv layer.

    Parameters:  x : ndarray float32 (any shape)
    Returns:     ndarray float32 (same shape)
    """
    return x * sigmoid(x)


def softmax(x, axis=-1):
    """
    Numerically stable softmax.

    Subtracts max before exp to prevent overflow — mathematically equivalent
    since the constant cancels in numerator/denominator.

    Used for: attention score normalization in MultiHeadAttention.

    Parameters:  x    : ndarray float32 (any shape)
                 axis : int — axis along which to normalize (default: last)
    Returns:     ndarray float32 (same shape), sums to 1 along axis
    """
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def layer_norm_forward(X, gamma, beta, eps=1e-5):
    """
    Layer normalization — normalizes over the last dimension (d_model).

    Unlike BN which normalizes over the batch dimension, LayerNorm normalizes
    each token independently, making it suitable for variable-length sequences
    and small batch sizes common in transformers.

    LN formula:  X_norm = gamma * (X - mean) / sqrt(var + eps) + beta
    where mean and var are computed over the last dim for each token.

    Used for: transformer encoder pre-norm (before attention and MLP),
              and the final LayerNorm before fold in MobileViTBlock.

    Parameters:  X     : float32 (..., d_model)
                 gamma : float32 (d_model,)
                 beta  : float32 (d_model,)
    Returns:     float32 (same shape as X)
    """
    mean = np.mean(X, axis=-1, keepdims=True)
    var  = np.var(X,  axis=-1, keepdims=True)
    return gamma * (X - mean) / np.sqrt(var + eps) + beta


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MobileNetV2 INVERTED RESIDUAL BLOCK
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
    MobileNetV2 inverted residual block — fully quantized with BN fused.

    Data flow:
        Input ──┬──────────────────────────────────────────────┐
                ▼                                              │ (residual)
        1×1 expand conv  [per-tensor act, per-channel weights] │
                └─ BN fused → Swish                           │
                ▼                                              │
        3×3 depthwise conv [per-channel act, per-channel W]   │
                └─ BN fused → Swish                           │
                ▼                                              │
        1×1 project conv [per-tensor act, per-channel weights]│
                └─ BN fused (no activation)                   │
                ▼                                              │
        (+) residual ←─────────────────────────────────────────┘
              (only when stride=1 AND input.shape == output.shape)

    BN fusion strategy: batch_norm_to_linear converts each BN into an
    affine A*z+B, then fuse_layer_bn absorbs it into the conv weights and
    bias. This eliminates all BN arithmetic at inference.

    Quantization per sub-layer:
        expand  : per-tensor activation  (input may have mixed channel ranges,
                  correct approach when channels mix in regular conv)
        depth   : per-channel activation (each channel independent — valid for
                  depthwise, and necessary because BN fusion creates 700× channel
                  range differences that devastate per-tensor SNR)
        project : per-tensor activation  (same reasoning as expand)

    Parameters:
        Input        : float32 (m, C_in, H, W)
        W_expan      : float32 (C_hid, C_in, 1, 1)
        W_depth      : float32 (C_hid, f, f)  — squeezed from (C_hid, 1, f, f)
        W_point      : float32 (C_out, C_hid, 1, 1)
        b_*          : bias tensors (matching conv shapes)
        stride_depth : int — 1 keeps spatial dims, 2 halves them
        W_gamma_*/b_gamma_* : BN gamma/beta for each sub-layer
        running_mean*/var*  : BN running stats (3 pairs)
    Returns:
        float32 (m, C_out, H_out, W_out)
    """
    # ── Step 1: Fuse BN into weights offline ─────────────────────────────────
    A_expan, B_expan = batch_norm_to_linear(W_gamma_expan, b_gamma_expan, running_mean1, running_var1)
    A_depth, B_depth = batch_norm_to_linear(W_gamma_depth, b_gamma_depth, running_mean2, running_var2)
    A_point, B_point = batch_norm_to_linear(W_gamma_point, b_gamma_point, running_mean3, running_var3)
    W_expan_f, b_expan_f = fuse_layer_bn(W_expan, b_expan, A_expan, B_expan)
    W_depth_f, b_depth_f = fuse_layer_bn(W_depth, b_depth, A_depth, B_depth)
    W_point_f, b_point_f = fuse_layer_bn(W_point, b_point, A_point, B_point)

    # ── Step 2: 1×1 Expand — per-tensor activation ───────────────────────────
    W_e_q, S_we       = quantize_symmetric_per_channel(W_expan_f)
    X_q,   S_xe, Z_xe = quantize_asymmetric(Input)
    b_e_q             = quantize_bias_int32_per_channel(b_expan_f, S_we, S_xe)
    b_e_q             = fold_zx_into_bias_per_channel(b_e_q, W_e_q, Z_xe)
    acc               = conv_forward_q(X_q, W_e_q, b_e_q, 1, 0, Z_xe)
    A1                = swish(dequantize_per_channel(acc, S_we, S_xe))

    # ── Step 3: 3×3 Depthwise — per-channel activation ───────────────────────
    # Per-channel activation valid here: output[c] depends ONLY on input[c]
    W_d_q, S_wd       = quantize_symmetric_per_channel(W_depth_f)
    X_q2,  S_xd, Z_xd = quantize_asymmetric_per_channel_act(A1)
    b_d_q             = quantize_bias_int32_per_channel_act(b_depth_f, S_wd, S_xd)
    b_d_q             = fold_zx_per_channel_act(b_d_q, W_d_q, Z_xd)
    acc2              = depthwise_conv_q_perch_act(X_q2, W_d_q, b_d_q, stride_depth, 1, Z_xd)
    A2                = swish(dequantize_depthwise_perch_act(acc2, S_wd, S_xd))

    # ── Step 4: 1×1 Project — per-tensor activation ──────────────────────────
    W_p_q, S_wp       = quantize_symmetric_per_channel(W_point_f)
    X_q3,  S_xp, Z_xp = quantize_asymmetric(A2)
    b_p_q             = quantize_bias_int32_per_channel(b_point_f, S_wp, S_xp)
    b_p_q             = fold_zx_into_bias_per_channel(b_p_q, W_p_q, Z_xp)
    acc3              = conv_forward_q(X_q3, W_p_q, b_p_q, 1, 0, Z_xp)
    Z3                = dequantize_per_channel(acc3, S_wp, S_xp)

    # ── Step 5: Residual shortcut ─────────────────────────────────────────────
    if stride_depth == 1 and Input.shape == Z3.shape:
        return Z3 + Input
    return Z3


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — TRANSFORMER COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def MultiHeadAttention(X, W, W_o, Bias, Bias_o, num_heads):
    """
    Multi-head self-attention — quantized matmuls, fp32 softmax.

    Quantization strategy:
        QKV projection : symmetric W_q, asymmetric X_q → fold Z_x into bias
        Q·K^T scores   : symmetric Q_q × symmetric K_q → dequantize before softmax
        softmax        : fp32 (probabilities are non-negative, well-behaved)
        attn·V         : asymmetric attn_q × symmetric V_q
                         Z_attn correction applied after dequantization
        output proj    : symmetric W_o_q, asymmetric context_q → fold Z_x into bias

    Data flow:
        X (B, S, D) → QKV proj → split Q,K,V → reshape to heads
        → Q/√d · K^T → softmax → attn · V → reshape → out proj → output

    Parameters:
        X        : float32 (batch, seq_len, d_model)
        W        : float32 (3*d_model, d_model)  — packed QKV weight
        W_o      : float32 (d_model, d_model)    — output projection
        Bias     : float32 (3*d_model,)
        Bias_o   : float32 (d_model,)
        num_heads: int
    Returns:
        float32 (batch, seq_len, d_model)
    """
    batch_size, seq_len, d_model = X.shape
    head_dim = d_model // num_heads

    # ── QKV projection ────────────────────────────────────────────────────────
    W_q, S_w     = quantize_symmetric(W)
    X_q, S_x, Z_x = quantize_asymmetric(X)
    Bias_q       = quantize_bias_int32(Bias, S_w, S_x)
    Bias_q       = fold_zx_into_bias(Bias_q, W_q, Z_x)
    QKV_q        = np.matmul(X_q.astype(np.int32), W_q.astype(np.int32).T) + Bias_q.reshape(1,1,-1)
    QKV          = dequantize(QKV_q, S_w, S_x)

    # ── Split and reshape to multi-head format ────────────────────────────────
    Q = QKV[:,:,          :   d_model].reshape(batch_size, seq_len, num_heads, head_dim).transpose(0,2,1,3)
    K = QKV[:,:,   d_model: 2*d_model].reshape(batch_size, seq_len, num_heads, head_dim).transpose(0,2,1,3)
    V = QKV[:,:, 2*d_model: 3*d_model].reshape(batch_size, seq_len, num_heads, head_dim).transpose(0,2,1,3)
    Q = Q / np.sqrt(head_dim)          # scale before quantizing scores

    # ── Attention scores Q·K^T ────────────────────────────────────────────────
    Q_q, S_q = quantize_symmetric(Q)
    K_q, S_k = quantize_symmetric(K)
    scores_q  = np.matmul(Q_q.astype(np.int32), K_q.astype(np.int32).transpose(0,1,3,2))
    scores    = dequantize(scores_q, S_q, S_k)
    attn      = softmax(scores, axis=-1)    # fp32 softmax — probabilities in [0,1]

    # ── Weighted sum attn·V ───────────────────────────────────────────────────
    V_q,    S_v            = quantize_symmetric(V)
    attn_q, S_attn, Z_attn = quantize_asymmetric(attn)
    context_q = np.matmul(attn_q.astype(np.int32), V_q.astype(np.int32))
    context   = dequantize(context_q, S_v, S_attn)
    # Subtract Z_attn correction (asymmetric activation zero-point for attn)
    # Equivalent to the fold_zx step but applied in fp32 after dequant here
    V_sum   = V_q.astype(np.float32).sum(axis=2, keepdims=True)   # sum over seq_len
    context = context - (S_v * S_attn * V_sum * float(Z_attn))

    # ── Output projection ─────────────────────────────────────────────────────
    context  = context.transpose(0,2,1,3).reshape(batch_size, seq_len, d_model)
    W_o_q, S_W_o         = quantize_symmetric(W_o)
    context_q, S_ctx, Z_ctx = quantize_asymmetric(context)
    Bias_o_q = quantize_bias_int32(Bias_o, S_W_o, S_ctx)
    Bias_o_q = fold_zx_into_bias(Bias_o_q, W_o_q, Z_ctx)
    out_q    = np.matmul(context_q.astype(np.int32), W_o_q.astype(np.int32).T) + Bias_o_q.reshape(1,1,-1)
    return dequantize(out_q, S_W_o, S_ctx)


def MLP(X, W1_fc1, b1_fc1, W2_fc2, b2_fc2):
    """
    Two-layer feed-forward network used inside each transformer encoder block.

    Structure: Linear → Swish → Linear
    Typically expands d_model → 4*d_model → d_model (hidden expansion).

    Both linear layers are quantized:
        Layer 1: symmetric W, asymmetric X → fold Z_x → int matmul → dequant → Swish
        Layer 2: symmetric W, asymmetric A1 → fold Z_x → int matmul → dequant

    Parameters:
        X       : float32 (batch, seq_len, d_model)
        W1_fc1  : float32 (d_ff, d_model)    — expansion weight
        b1_fc1  : float32 (d_ff,)
        W2_fc2  : float32 (d_model, d_ff)    — projection weight
        b2_fc2  : float32 (d_model,)
    Returns:
        float32 (batch, seq_len, d_model)
    """
    # ── Layer 1: expand ───────────────────────────────────────────────────────
    W1_q, S_W1       = quantize_symmetric(W1_fc1)
    X_q, S_x1, Z_x1  = quantize_asymmetric(X)
    b1_q             = quantize_bias_int32(b1_fc1, S_W1, S_x1)
    b1_q             = fold_zx_into_bias(b1_q, W1_q, Z_x1)
    Z1_q             = np.matmul(X_q.astype(np.int32), W1_q.astype(np.int32).T) + b1_q
    A1               = swish(dequantize(Z1_q, S_W1, S_x1))

    # ── Layer 2: project ──────────────────────────────────────────────────────
    W2_q, S_W2       = quantize_symmetric(W2_fc2)
    A1_q, S_x2, Z_x2 = quantize_asymmetric(A1)
    b2_q             = quantize_bias_int32(b2_fc2, S_W2, S_x2)
    b2_q             = fold_zx_into_bias(b2_q, W2_q, Z_x2)
    out_q            = np.matmul(A1_q.astype(np.int32), W2_q.astype(np.int32).T) + b2_q
    return dequantize(out_q, S_W2, S_x2)


def transformer_encoder(X,
                         W_QKV_atten, W_o_atten, Bias_QKV_atten, Bias_o_atten,
                         W1_fc1, b1_fc1, W2_fc2, b2_fc2,
                         W_gamma1, b_beta1, W_gamma2, b_beta2,
                         num_heads):
    """
    Single transformer encoder block with pre-norm architecture.

    Structure (pre-norm = LayerNorm before the sub-layer):
        X → LayerNorm → MultiHeadAttention → (+) skip → LayerNorm → MLP → (+) skip

    Pre-norm is more stable than post-norm for training and is used in MobileViT.
    The residual connections allow gradients to flow and features to persist.

    Parameters:
        X                  : float32 (batch, seq_len, d_model)
        W_QKV_atten        : float32 (3*d_model, d_model)
        W_o_atten          : float32 (d_model, d_model)
        Bias_QKV_atten     : float32 (3*d_model,)
        Bias_o_atten       : float32 (d_model,)
        W1_fc1, W2_fc2     : MLP weights
        b1_fc1, b2_fc2     : MLP biases
        W_gamma1/2, b_beta1/2 : LayerNorm scale/shift for attn and MLP
        num_heads          : int
    Returns:
        float32 (batch, seq_len, d_model)
    """
    # ── Attention sub-block ───────────────────────────────────────────────────
    skip1 = X
    X     = layer_norm_forward(X, W_gamma1, b_beta1)
    X     = MultiHeadAttention(X, W_QKV_atten, W_o_atten,
                               Bias_QKV_atten, Bias_o_atten, num_heads)
    X     = X + skip1

    # ── MLP sub-block ─────────────────────────────────────────────────────────
    skip2 = X
    X     = layer_norm_forward(X, W_gamma2, b_beta2)
    X     = MLP(X, W1_fc1, b1_fc1, W2_fc2, b2_fc2)
    return X + skip2


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MobileViT BLOCK COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def bilinear_resize(x, new_h, new_w):
    """
    Bilinear interpolation — matches PyTorch F.interpolate(align_corners=False).

    Used to pad feature maps to the next multiple of patch_size before unfolding,
    and to revert back after folding if interpolation was applied.

    align_corners=False: pixel centers at (i+0.5)/N * old_size - 0.5,
    which is what PyTorch uses by default.

    Parameters:  x     : float32 (b, c, h, w)
                 new_h, new_w : int — target spatial dimensions
    Returns:     float32 (b, c, new_h, new_w)
    """
    b, c, h, w = x.shape
    row_idx = np.clip((np.arange(new_h)+0.5)*(h/new_h)-0.5, 0, h-1)
    col_idx = np.clip((np.arange(new_w)+0.5)*(w/new_w)-0.5, 0, w-1)
    r0 = np.floor(row_idx).astype(int);  r1 = np.minimum(r0+1, h-1)
    c0 = np.floor(col_idx).astype(int);  c1 = np.minimum(c0+1, w-1)
    dr = (row_idx - r0)[:, None];        dc = (col_idx - c0)[None, :]
    return (x[:,:,r0,:][:,:,:,c0] * (1-dr)*(1-dc) +
            x[:,:,r0,:][:,:,:,c1] * (1-dr)*   dc  +
            x[:,:,r1,:][:,:,:,c0] *    dr *(1-dc) +
            x[:,:,r1,:][:,:,:,c1] *    dr *   dc).astype(x.dtype)


def unfold(x, patch_size=2):
    """
    Unfold a spatial feature map into non-overlapping patches for transformer input.

    MobileViT's key insight: instead of processing the full spatial map with a
    transformer (O(H*W)² attention), it divides the map into patches and processes
    pixels at the same position across patches together, giving global context
    at manageable sequence length.

    Reshaping sequence:
        (b, c, H, W) → (b*patch_area, num_patches, c)
    where num_patches = (H//ph) * (W//pw) and patch_area = ph*pw.

    If H or W is not a multiple of patch_size, bilinear_resize pads first,
    and the info dict records this so fold() can revert it.

    Parameters:  x          : float32 (b, c, H, W)
                 patch_size  : int (ph = pw = patch_size)
    Returns:     patches : float32 (b*patch_area, num_patches, c)
                 info    : dict with shape metadata for fold()
    """
    b, c, h, w = x.shape
    ph = pw = patch_size
    new_h = math.ceil(h/ph)*ph;  new_w = math.ceil(w/pw)*pw
    interpolate = (new_h != h or new_w != w)
    if interpolate:
        x = bilinear_resize(x, new_h, new_w)
    num_patch_h = new_h//ph;  num_patch_w = new_w//pw
    num_patches = num_patch_h * num_patch_w;  patch_area = ph * pw
    x = x.reshape(b*c*num_patch_h, ph, num_patch_w, pw).transpose(0,2,1,3)
    x = x.reshape(b, c, num_patches, patch_area).transpose(0,3,2,1)
    patches = x.reshape(b*patch_area, num_patches, c)
    return patches, dict(b=b, c=c, orig_h=h, orig_w=w,
                         num_patch_h=num_patch_h, num_patch_w=num_patch_w,
                         num_patches=num_patches, patch_area=patch_area,
                         ph=ph, pw=pw, interpolate=interpolate)


def fold(patches, info):
    """
    Fold transformer-processed patches back into a spatial feature map.

    Exact inverse of unfold(). Reverts the reshape sequence and, if
    bilinear_resize was applied during unfold, resizes back to original dims.

    Parameters:  patches : float32 (b*patch_area, num_patches, c)
                 info    : dict returned by unfold()
    Returns:     float32 (b, c, orig_h, orig_w)
    """
    b, ph, pw   = info['b'], info['ph'], info['pw']
    nph, npw    = info['num_patch_h'], info['num_patch_w']
    num_patches = info['num_patches'];  patch_area = info['patch_area']
    C           = patches.shape[-1]
    x = patches.reshape(b, patch_area, num_patches, C).transpose(0,3,2,1)
    x = x.reshape(b*C*nph, npw, ph, pw).transpose(0,2,1,3)
    x = x.reshape(b, C, nph*ph, npw*pw)
    if info['interpolate']:
        x = bilinear_resize(x, info['orig_h'], info['orig_w'])
    return x


def Local_representations(input,
                           W_loacal_3x3, b_local_3x3,
                           W_loacal_1x1, b_local_1x1,
                           W_gamma_local_3x3, W_beta_local_3x3,
                           running_mean1, running_var1):
    """
    Local representation extractor for MobileViTBlock.

    Extracts local features before the global transformer encoding.
    Structure: 3×3 conv (BN fused) → Swish → 1×1 conv (no BN, no activation)

    The 3×3 conv captures local spatial context within each channel.
    The 1×1 conv projects to the transformer's d_model dimension.
    BN is fused offline into the 3×3 conv weights for efficiency.

    Quantization: both convs use per-channel weight quant + per-tensor activation.

    Parameters:
        input             : float32 (m, C_in, H, W)
        W_loacal_3x3      : float32 (C_out, C_in, 3, 3)
        b_local_3x3       : float32 (C_out, 1, 1, 1)
        W_loacal_1x1      : float32 (d_model, C_out, 1, 1)
        b_local_1x1       : float32 (d_model, 1, 1, 1)
        W_gamma_local_3x3 : float32 (C_out,) — BN scale
        W_beta_local_3x3  : float32 (C_out,) — BN shift
        running_mean1/var1 : float32 (C_out,) — BN running stats
    Returns:
        float32 (m, d_model, H, W)
    """
    # Fuse BN into 3×3 conv offline
    A_local, B_local = batch_norm_to_linear(
        W_gamma_local_3x3, W_beta_local_3x3, running_mean1, running_var1)
    W_fused, b_fused = fuse_layer_bn(W_loacal_3x3, b_local_3x3, A_local, B_local)

    # 3×3 conv (BN fused) → Swish
    W_l3_q, S_wl3       = quantize_symmetric_per_channel(W_fused)
    X_q,    S_xl3, Z_xl3 = quantize_asymmetric(input)
    b_l3_q              = quantize_bias_int32_per_channel(b_fused, S_wl3, S_xl3)
    b_l3_q              = fold_zx_into_bias_per_channel(b_l3_q, W_l3_q, Z_xl3)
    acc                 = conv_forward_q(X_q, W_l3_q, b_l3_q, 1, 1, Z_xl3)
    x                   = swish(dequantize_per_channel(acc, S_wl3, S_xl3))

    # 1×1 conv only (no BN, no activation) — projects to d_model
    W_l1_q, S_wl1       = quantize_symmetric_per_channel(W_loacal_1x1)
    X_q,    S_xl1, Z_xl1 = quantize_asymmetric(x)
    b_l1_q              = quantize_bias_int32_per_channel(b_local_1x1, S_wl1, S_xl1)
    b_l1_q              = fold_zx_into_bias_per_channel(b_l1_q, W_l1_q, Z_xl1)
    acc2                = conv_forward_q(X_q, W_l1_q, b_l1_q, 1, 0, Z_xl1)
    return dequantize_per_channel(acc2, S_wl1, S_xl1)


def fusion(x_global, input_feat,
           W_fusion_1x1, b_fusion_1x1,
           W_fusion_3x3, b_fusion_3x3,
           W_gamma_f_1x1, b_beta_f_1x1,
           W_gamma_f_3x3, b_beta_f_3x3,
           running_mean1, running_var1,
           running_mean2, running_var2):
    """
    Fusion module — combines global transformer features with local conv features.

    After the transformer processes patches globally, this module merges
    the global context back with the original local features:
        1. Project global features (d_model → C) with 1×1 conv → Swish
        2. Concatenate with original input along channel axis → [B, 2C, H, W]
        3. Mix with 3×3 conv (C_in=2C → C_out=C) → Swish

    Both conv layers have BN fused offline. Both are quantized with
    per-channel weights and per-tensor activation.

    Parameters:
        x_global    : float32 (B, d_model, H, W) — from transformer fold
        input_feat  : float32 (B, C, H, W)       — original block input
        W_fusion_1x1 : float32 (C, d_model, 1, 1)
        W_fusion_3x3 : float32 (C, 2*C, 3, 3)
        b_fusion_*   : biases (matching shapes)
        W_gamma_f_*/b_beta_f_* : BN params for 1×1 and 3×3
        running_mean*/var*      : BN running stats (2 pairs)
    Returns:
        float32 (B, C, H, W)
    """
    # Fuse BN into both conv layers offline
    A_1x1, B_1x1 = batch_norm_to_linear(W_gamma_f_1x1, b_beta_f_1x1, running_mean1, running_var1)
    A_3x3, B_3x3 = batch_norm_to_linear(W_gamma_f_3x3, b_beta_f_3x3, running_mean2, running_var2)
    W_1x1_f, b_1x1_f = fuse_layer_bn(W_fusion_1x1, b_fusion_1x1, A_1x1, B_1x1)
    W_3x3_f, b_3x3_f = fuse_layer_bn(W_fusion_3x3, b_fusion_3x3, A_3x3, B_3x3)

    # 1×1 conv: project global d_model → C channels → Swish
    W_g1_q, S_wg1        = quantize_symmetric_per_channel(W_1x1_f)
    X_q,    S_xg1, Z_xg1 = quantize_asymmetric(x_global)
    b_g1_q               = quantize_bias_int32_per_channel(b_1x1_f, S_wg1, S_xg1)
    b_g1_q               = fold_zx_into_bias_per_channel(b_g1_q, W_g1_q, Z_xg1)
    acc                  = conv_forward_q(X_q, W_g1_q, b_g1_q, 1, 0, Z_xg1)
    x                    = swish(dequantize_per_channel(acc, S_wg1, S_xg1))

    if x.shape[2:] != input_feat.shape[2:]:
        raise ValueError(
            f"Spatial mismatch in fusion: global {x.shape[2:]} vs local {input_feat.shape[2:]}")

    # Concatenate local + projected global → [B, 2C, H, W]
    concat = np.concatenate([input_feat, x], axis=1)

    # 3×3 conv: mix 2C → C channels → Swish
    W_g3_q, S_wg3        = quantize_symmetric_per_channel(W_3x3_f)
    concat_q, S_xg3, Z_xg3 = quantize_asymmetric(concat)
    b_g3_q               = quantize_bias_int32_per_channel(b_3x3_f, S_wg3, S_xg3)
    b_g3_q               = fold_zx_into_bias_per_channel(b_g3_q, W_g3_q, Z_xg3)
    acc2                 = conv_forward_q(concat_q, W_g3_q, b_g3_q, 1, 1, Z_xg3)
    return swish(dequantize_per_channel(acc2, S_wg3, S_xg3))


def MobileViTBlock_(input,
                    W_loacal_3x3, b_local_3x3,
                    W_loacal_1x1, b_local_1x1,
                    W_gamma_local_3x3, W_beta_local_3x3,
                    W_QKV_atten, W_o_atten, Bias_QKV_atten, Bias_o_atten,
                    W1_fc1, b1_fc1, W2_fc2, b2_fc2,
                    W_gamma1, b_beta1, W_gamma2, b_beta2,
                    W_norm_gamma, b_norm_beta,
                    W_fusion_1x1, b_fusion_1x1,
                    W_fusion_3x3, b_fusion_3x3,
                    W_gamma_f_1x1, b_beta_f_1x1,
                    W_gamma_f_3x3, b_beta_f_3x3,
                    L, patch_size, num_heads,
                    running_mean1, running_var1,
                    running_mean2, running_var2,
                    running_mean3, running_var3):
    """
    MobileViT Block — combines local conv features with global transformer context.

    This is MobileViT's core innovation: it applies a transformer to non-overlapping
    patches of the feature map, enabling global receptive field without the quadratic
    cost of full spatial self-attention.

    Pipeline:
        1. Local representations : 3×3 conv + 1×1 conv → local features (m, d, H, W)
        2. Unfold                : reshape to patches   → (b*ph*pw, num_patches, d)
        3. Transformer (L layers): global attention     → same shape
        4. Final LayerNorm       : normalize patches
        5. Fold                  : reshape back         → (m, d, H, W)
        6. Fusion                : merge global + local → (m, C, H, W)

    Parameters:
        input           : float32 (m, C, H, W)
        W_loacal_*      : local representation weights/biases
        W_QKV_atten etc : transformer weight lists, each of length L
        W_norm_gamma/beta : final LayerNorm params
        W_fusion_*      : fusion module weights/biases
        L               : int — number of transformer layers
        patch_size      : int — spatial patch size (ph = pw)
        num_heads       : int — attention heads
        running_mean*/var* : BN stats (local 3×3, fusion 1×1, fusion 3×3)
    Returns:
        float32 (m, C, H, W)
    """
    # Step 1: Extract local features
    x_local = Local_representations(
        input,
        W_loacal_3x3, b_local_3x3, W_loacal_1x1, b_local_1x1,
        W_gamma_local_3x3, W_beta_local_3x3,
        running_mean1, running_var1)

    # Step 2: Unfold into patches for transformer
    patches, info = unfold(x_local, patch_size)

    # Step 3: Apply L transformer encoder blocks
    for i in range(L):
        patches = transformer_encoder(
            patches,
            W_QKV_atten[i], W_o_atten[i], Bias_QKV_atten[i], Bias_o_atten[i],
            W1_fc1[i], b1_fc1[i], W2_fc2[i], b2_fc2[i],
            W_gamma1[i], b_beta1[i], W_gamma2[i], b_beta2[i],
            num_heads)

    # Step 4: Final LayerNorm before folding
    patches = layer_norm_forward(patches, W_norm_gamma, b_norm_beta)

    # Step 5: Fold patches back to spatial feature map
    x_global = fold(patches, info)

    # Step 6: Fuse global context with original local features
    return fusion(
        x_global, input,
        W_fusion_1x1, b_fusion_1x1, W_fusion_3x3, b_fusion_3x3,
        W_gamma_f_1x1, b_beta_f_1x1, W_gamma_f_3x3, b_beta_f_3x3,
        running_mean2, running_var2, running_mean3, running_var3)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — HEAD & CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def global_avg_pool(x):
    """
    Global average pooling over spatial dimensions.

    Reduces (m, C, H, W) → (m, C) by averaging all spatial positions.
    Provides spatial-dimension-invariant feature vectors for classification.

    Parameters:  x : float32 (m, C, H, W)
    Returns:     float32 (m, C)
    """
    return np.mean(x, axis=(2, 3))


def linear_classifier(x, W_cls, b_cls):
    """
    FP32 linear classifier — kept as reference / fallback.

    Parameters:  x     : float32 (m, C)
                 W_cls : float32 (C, num_classes)  — already transposed
                 b_cls : float32 (num_classes,)
    Returns:     float32 (m, num_classes)
    """
    return np.matmul(x, W_cls) + b_cls


def linear_classifier_q(x_q, W_cls_q, b_cls_q):
    """
    Quantized linear classifier — integer matmul only.

    Performs int8 × int8 matmul accumulated in int32.
    W_cls_q must already be transposed to (C, num_classes) by the caller.
    b_cls_q must already be quantized and Z_x folded by the caller.
    Dequantization is applied by the caller after this function returns.

    Parameters:  x_q     : uint8 (m, C)
                 W_cls_q : int8  (C, num_classes) — transposed
                 b_cls_q : int32 (num_classes,)   — Z_x folded
    Returns:     int32 (m, num_classes) — accumulator scale, not yet dequantized
    """
    return np.matmul(x_q.astype(np.int32), W_cls_q.astype(np.int32)) + b_cls_q


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — FULL MODEL: MobileViT-XXS
# ═══════════════════════════════════════════════════════════════════════════════

def MobileViT_XXS_(Inputs, bn_prams,
                   W_stem, b_stem, W_stem_gamma, b_beta_stem,
                   W_expan_1,  W_depth_1,  W_point_1,
                   b_expan_1,  b_depth_1,  b_point_1,
                   W_gamma_expan_1, W_gamma_depth_1, W_gamma_point_1,
                   b_gamma_expan_1, b_gamma_depth_1, b_gamma_point_1,
                   W_expan_2a, W_depth_2a, W_point_2a,
                   b_expan_2a, b_depth_2a, b_point_2a,
                   W_gamma_expan_2a, W_gamma_depth_2a, W_gamma_point_2a,
                   b_gamma_expan_2a, b_gamma_depth_2a, b_gamma_point_2a,
                   W_expan_2b, W_depth_2b, W_point_2b,
                   b_expan_2b, b_depth_2b, b_point_2b,
                   W_gamma_expan_2b, W_gamma_depth_2b, W_gamma_point_2b,
                   b_gamma_expan_2b, b_gamma_depth_2b, b_gamma_point_2b,
                   W_expan_2c, W_depth_2c, W_point_2c,
                   b_expan_2c, b_depth_2c, b_point_2c,
                   W_gamma_expan_2c, W_gamma_depth_2c, W_gamma_point_2c,
                   b_gamma_expan_2c, b_gamma_depth_2c, b_gamma_point_2c,
                   W_expan_3a, W_depth_3a, W_point_3a,
                   b_expan_3a, b_depth_3a, b_point_3a,
                   W_gamma_expan_3a, W_gamma_depth_3a, W_gamma_point_3a,
                   b_gamma_expan_3a, b_gamma_depth_3a, b_gamma_point_3a,
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
                   W_expan_4a, W_depth_4a, W_point_4a,
                   b_expan_4a, b_depth_4a, b_point_4a,
                   W_gamma_expan_4a, W_gamma_depth_4a, W_gamma_point_4a,
                   b_gamma_expan_4a, b_gamma_depth_4a, b_gamma_point_4a,
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
                   W_expan_5a, W_depth_5a, W_point_5a,
                   b_expan_5a, b_depth_5a, b_point_5a,
                   W_gamma_expan_5a, W_gamma_depth_5a, W_gamma_point_5a,
                   b_gamma_expan_5a, b_gamma_depth_5a, b_gamma_point_5a,
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
                   W_head, b_head, W_gamma_head, b_beta_head,
                   W_cls, b_cls):
    """
    MobileViT-XXS full forward pass — quantized inference.

    Architecture overview:
        Stem          : 3×3 conv, stride=2, BN fused → Swish        [3→16 ch]
        Stage 1       : MV2 block, stride=1                          [16→16 ch]
        Stage 2       : MV2×3 (stride 2,1,1)                        [16→24 ch]
        Stage 3       : MV2 (stride 2) + MobileViTBlock (L=2, d=64) [24→48 ch]
        Stage 4       : MV2 (stride 2) + MobileViTBlock (L=4, d=80) [48→64 ch]
        Stage 5       : MV2 (stride 2) + MobileViTBlock (L=3, d=96) [64→80 ch]
        Head          : 1×1 conv, BN fused → Swish                  [80→320 ch]
        Classifier    : GlobalAvgPool → quantized linear → logits

    bn_prams layout (64 entries = 32 running_mean/var pairs):
        [0,1]   stem
        [2-7]   mv2_1
        [8-13]  mv2_2a,  [14-19] mv2_2b,  [20-25] mv2_2c
        [26-31] mv2_3a,  [32-37] mvit_3b
        [38-43] mv2_4a,  [44-49] mvit_4b
        [50-55] mv2_5a,  [56-61] mvit_5b
        [62-63] head

    Parameters:
        Inputs    : float32 (m, 3, 224, 224)
        bn_prams  : list of 64 float32 arrays — BN running stats
        W_*/b_*   : model weights (see extract_weights_MVT_to_numpy.py)
    Returns:
        float32 (m, num_classes) — raw logits
    """
    # ── Stem: 3×3 conv stride=2, BN fused ────────────────────────────────────
    A_stem, B_stem     = batch_norm_to_linear(W_stem_gamma, b_beta_stem, bn_prams[0], bn_prams[1])
    W_stem_f, b_stem_f = fuse_layer_bn(W_stem, b_stem, A_stem, B_stem)
    W_s_q, S_ws        = quantize_symmetric_per_channel(W_stem_f)
    In_q, S_xi, Z_xi   = quantize_asymmetric(Inputs)
    b_s_q              = quantize_bias_int32_per_channel(b_stem_f, S_ws, S_xi)
    b_s_q              = fold_zx_into_bias_per_channel(b_s_q, W_s_q, Z_xi)
    acc_stem           = conv_forward_q(In_q, W_s_q, b_s_q, 2, 1, Z_xi)
    x                  = swish(dequantize_per_channel(acc_stem, S_ws, S_xi))

    # ── Stage 1: MV2, stride=1 ────────────────────────────────────────────────
    x = MobileNet_v2(x, W_expan_1, W_depth_1, W_point_1,
                     b_expan_1, b_depth_1, b_point_1, 1,
                     W_gamma_expan_1, W_gamma_depth_1, W_gamma_point_1,
                     b_gamma_expan_1, b_gamma_depth_1, b_gamma_point_1,
                     *bn_prams[2:8])

    # ── Stage 2: MV2×3 (stride 2, 1, 1) ──────────────────────────────────────
    x = MobileNet_v2(x, W_expan_2a, W_depth_2a, W_point_2a,
                     b_expan_2a, b_depth_2a, b_point_2a, 2,
                     W_gamma_expan_2a, W_gamma_depth_2a, W_gamma_point_2a,
                     b_gamma_expan_2a, b_gamma_depth_2a, b_gamma_point_2a,
                     *bn_prams[8:14])
    x = MobileNet_v2(x, W_expan_2b, W_depth_2b, W_point_2b,
                     b_expan_2b, b_depth_2b, b_point_2b, 1,
                     W_gamma_expan_2b, W_gamma_depth_2b, W_gamma_point_2b,
                     b_gamma_expan_2b, b_gamma_depth_2b, b_gamma_point_2b,
                     *bn_prams[14:20])
    x = MobileNet_v2(x, W_expan_2c, W_depth_2c, W_point_2c,
                     b_expan_2c, b_depth_2c, b_point_2c, 1,
                     W_gamma_expan_2c, W_gamma_depth_2c, W_gamma_point_2c,
                     b_gamma_expan_2c, b_gamma_depth_2c, b_gamma_point_2c,
                     *bn_prams[20:26])

    # ── Stage 3: MV2 stride=2 + MobileViTBlock (L=2, patch=2, heads=2) ───────
    x = MobileNet_v2(x, W_expan_3a, W_depth_3a, W_point_3a,
                     b_expan_3a, b_depth_3a, b_point_3a, 2,
                     W_gamma_expan_3a, W_gamma_depth_3a, W_gamma_point_3a,
                     b_gamma_expan_3a, b_gamma_depth_3a, b_gamma_point_3a,
                     *bn_prams[26:32])
    x = MobileViTBlock_(x,
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
                        2, 2, 2, *bn_prams[32:38])

    # ── Stage 4: MV2 stride=2 + MobileViTBlock (L=4, patch=2, heads=4) ───────
    x = MobileNet_v2(x, W_expan_4a, W_depth_4a, W_point_4a,
                     b_expan_4a, b_depth_4a, b_point_4a, 2,
                     W_gamma_expan_4a, W_gamma_depth_4a, W_gamma_point_4a,
                     b_gamma_expan_4a, b_gamma_depth_4a, b_gamma_point_4a,
                     *bn_prams[38:44])
    x = MobileViTBlock_(x,
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
                        4, 2, 4, *bn_prams[44:50])

    # ── Stage 5: MV2 stride=2 + MobileViTBlock (L=3, patch=2, heads=4) ───────
    x = MobileNet_v2(x, W_expan_5a, W_depth_5a, W_point_5a,
                     b_expan_5a, b_depth_5a, b_point_5a, 2,
                     W_gamma_expan_5a, W_gamma_depth_5a, W_gamma_point_5a,
                     b_gamma_expan_5a, b_gamma_depth_5a, b_gamma_point_5a,
                     *bn_prams[50:56])
    x = MobileViTBlock_(x,
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
                        3, 2, 4, *bn_prams[56:62])

    # ── Head: 1×1 conv, BN fused → Swish → GlobalAvgPool ────────────────────
    A_head, B_head     = batch_norm_to_linear(W_gamma_head, b_beta_head, bn_prams[62], bn_prams[63])
    W_head_f, b_head_f = fuse_layer_bn(W_head, b_head, A_head, B_head)
    W_h_q, S_wh        = quantize_symmetric_per_channel(W_head_f)
    Xh_q, S_xh, Z_xh   = quantize_asymmetric(x)
    b_h_q              = quantize_bias_int32_per_channel(b_head_f, S_wh, S_xh)
    b_h_q              = fold_zx_into_bias_per_channel(b_h_q, W_h_q, Z_xh)
    acc_head           = conv_forward_q(Xh_q, W_h_q, b_h_q, 1, 0, Z_xh)
    x                  = swish(dequantize_per_channel(acc_head, S_wh, S_xh))
    x                  = global_avg_pool(x)     # (m, 320)

    # ── Classifier: quantized linear → logits ────────────────────────────────
    # W_cls is (num_classes, 320) from extractor; transpose to (320, num_classes)
    # for x @ W_cls_q where x is (m, 320)
    W_cls_q, S_Wcls      = quantize_symmetric(W_cls)          # per-tensor for classifier
    x_q, S_xcls, Z_xcls  = quantize_asymmetric(x)
    b_cls_q              = quantize_bias_int32(b_cls, S_Wcls, S_xcls)
    b_cls_q              = fold_zx_into_bias(b_cls_q, W_cls_q, Z_xcls)
    W_cls_q              = W_cls_q.T                           # (320, num_classes)
    cls_q                = linear_classifier_q(x_q, W_cls_q, b_cls_q)
    return dequantize(cls_q, S_Wcls, S_xcls)                   # fp32 logits (m, num_classes)