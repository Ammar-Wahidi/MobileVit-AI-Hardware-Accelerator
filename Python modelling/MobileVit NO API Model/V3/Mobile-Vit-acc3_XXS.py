
# No API Model

import numpy as np
import math

# ─────────────────────────────────────────────────────────────
# Primitive helpers
# ─────────────────────────────────────────────────────────────

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
    as illustrated in Figure 1.

    Argument:
    X -- python numpy array of shape (m, n_c, n_H, n_W) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_C, n_H + 2 * pad, n_W + 2 * pad)
    """

    return np.pad(X, ((0,0),(0,0),(pad,pad),(pad,pad)),
                  mode='constant', constant_values=0)


def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    of the previous layer.

    Arguments:
    a_slice_prev -- slice of input data of shape (n_C_prev, f, f)
    W -- Weight parameters contained in a window - matrix of shape (n_C_prev, f, f)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """

    return float(np.sum(a_slice_prev * W) + b.item())


def conv_forward(A_prev, W, b, stride, pad):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    A_prev -- output activations of the previous layer,
        numpy array of shape (m, n_C_prev, n_H_prev, n_W_prev)
    W -- Weights, numpy array of shape (n_C, n_C_prev, f, f)
    b -- Biases, numpy array of shape (n_C, 1, 1, 1)
    stride
    pad

    Returns:
    Z -- conv output, numpy array of shape (m, n_C, n_H, n_W)
    """

    m, n_C_prev, n_H_prev, n_W_prev = A_prev.shape
    f  = W.shape[2]
    n_C = W.shape[0]
    n_H = int((n_H_prev - f + 2*pad) / stride + 1)
    n_W = int((n_W_prev - f + 2*pad) / stride + 1)
    Z = np.zeros((m, n_C, n_H, n_W))
    A_prev_pad = zero_pad(A_prev, pad)
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            vs, ve = h*stride, h*stride+f
            for w in range(n_W):
                hs, he = w*stride, w*stride+f
                for c in range(n_C):
                    Z[i,c,h,w] = conv_single_step(
                        a_prev_pad[:, vs:ve, hs:he], W[c], b[c])
    return Z


def depthwise_conv(A_prev, W, b, stride, pad):
    """
    Depthwise Conv, every slice conv with one channel

    Arguments:
     A_prev -- output activations of the previous layer,
             numpy array of shape (m, n_C_prev, n_H_prev, n_W_prev)
    W -- Weights, numpy array of shape (n_C_prev, f, f)
    b -- Biases, numpy array of shape (n_C_prev, 1, 1)
    stride

    return
    A output, numpy array of shape (m, n_C_prev, n_H_prev, n_W_prev)
    """
    # Retrieve dimensions from the input shape
    m, n_C_prev, n_H_prev, n_W_prev = A_prev.shape
    f  = W.shape[2]
    n_H = int((n_H_prev - f + 2*pad) / stride + 1)
    n_W = int((n_W_prev - f + 2*pad) / stride + 1)
    A = np.zeros((m, n_C_prev, n_H, n_W))

    # Apply padding
    A_prev_p = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev = A_prev_p[i]
        for h in range(n_H):
            vs, ve = h*stride, h*stride+f
            for w in range(n_W):
                hs, he = w*stride, w*stride+f
                for c in range(n_C_prev):
                    A[i,c,h,w] = np.sum(a_prev[c, vs:ve, hs:he] * W[c]) + b[c]
    return A


def batch_norm_forward(Z, gamma, beta, running_mean, running_var, eps=1e-5):
    """
    Batch Normalization forward pass (inference-only).

    Arguments:
    Z            -- numpy array of shape (m, n_C, n_H, n_W) (conv output)
    gamma        -- scale parameter, shape (n_C,)
    beta         -- shift parameter, shape (n_C,)
    running_mean -- numpy array of shape (n_C,) (learned during training)
    running_var  -- numpy array of shape (n_C,) (learned during training)
    eps          -- small constant to avoid division by zero

    Returns:
    Z_norm       -- normalized and scaled output (same shape as Z)
    """
    # Reshape gamma, beta, mean, var for broadcasting
    gamma = gamma.reshape(1,-1,1,1)
    beta  = beta.reshape(1,-1,1,1)
    mean  = running_mean.reshape(1,-1,1,1)
    var   = running_var.reshape(1,-1,1,1)
    # Normalize using stored running statistics, Scale and shift
    return gamma * (Z - mean) / np.sqrt(var + eps) + beta


def layer_norm_forward(X, gamma, beta, eps=1e-5):
    """
    Layer Normalization for Transformer inputs (inference-only).

    Arguments:
    X     -- numpy array of shape (batch_size, seq_len, d_model)
    gamma -- scale parameter, shape (d_model,) or broadcastable
    beta  -- shift parameter, shape (d_model,) or broadcastable
    eps   -- small constant for numerical stability

    Returns:
    out   -- normalized tensor, same shape as X
    """
    """LayerNorm over last dim. X: (..., d_model)."""

    # Mean and variance across the last dimension (d_model)
    mean = np.mean(X, axis=-1, keepdims=True)
    var  = np.var(X,  axis=-1, keepdims=True)

    # Normalize , Scale + shift (broadcast along batch and seq_len)
    return gamma * (X - mean) / np.sqrt(var + eps) + beta


def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def swish(x):
    """
    Swish activation function.

    Argument:
    Z -- numpy array

    Returns:
    A -- activated output, same shape as Z
    """
    return x * sigmoid(x)


def softmax(x, axis=-1):
    """
    Stable softmax function for transformer attention.

    Arguments:
    x    -- numpy array of any shape
    axis -- axis along which to apply softmax (default: last axis)

    Returns:
    out  -- softmax probabilities, same shape as x
    """
    # Subtract max for numerical stability
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))

    # Normalize to get probabilities
    return e / np.sum(e, axis=axis, keepdims=True)


# ─────────────────────────────────────────────────────────────
# MobileNetV2 block
# ─────────────────────────────────────────────────────────────

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
    Simplified MobileNetV2 block:
    1. 1x1 conv (expansion)
    2. BN + Swish
    3. Depthwise conv
    4. BN + Swish
    5. 1x1 conv (projection)
    6. BN + Swish
    7. Residual connection (if stride=1 and shapes match)

    Arguments:
    Input       -- input tensor of shape (m, n_C_prev, n_H_prev, n_W_prev)
    W           -- list/array of weight tensors:
                     W_expan -> standard conv weights  (n_C,n_C_prev,f,f) , f=1
                     W_depth -> depthwise conv weights (n_C,n_C_prev,f,f) , f=3
                     W_point -> pointwise conv weights (n_C,n_C_prev,f,f) , f=1
    bias        -- list/array of bias tensors:
                     b_expan -> standard conv bias  (n_C,1,1,1)
                     b_depth -> depthwise conv bias (n_C,1,1,1)
                     b_point -> pointwise conv bias (n_C,1,1,1)
    stride      -- stride for depthwise conv
    W_gamma       -- BN scale, shape (1,1,1,n_C)
    b_beta        -- BN shift, shape (1,1,1,n_C)

    Returns:
    A_out
    """
    # 1×1 expand
    Z1 = conv_forward(Input, W_expan, b_expan, 1, 0)
    Z1 = batch_norm_forward(Z1, W_gamma_expan, b_gamma_expan, running_mean1, running_var1)
    A1 = swish(Z1)
    # 3×3 depthwise
    Z2 = depthwise_conv(A1, W_depth, b_depth, stride_depth, 1)
    Z2 = batch_norm_forward(Z2, W_gamma_depth, b_gamma_depth, running_mean2, running_var2)
    A2 = swish(Z2)
    # 1×1 project
    Z3 = conv_forward(A2, W_point, b_point, 1, 0)
    Z3 = batch_norm_forward(Z3, W_gamma_point, b_gamma_point, running_mean3, running_var3)
    if stride_depth == 1 and Input.shape == Z3.shape:
        return Z3 + Input
    return Z3


# ─────────────────────────────────────────────────────────────
# Transformer pieces
# ─────────────────────────────────────────────────────────────

def MultiHeadAttention(X, W, W_o, Bias, Bias_o, num_heads):
    """
    Multi-Head Attention for transformer (MobileViT style)

    Arguments:
    X        -- input tensor, shape (batch_size, seq_len, d_model)
    W        -- weights for Q,K,V, shape (3*d_model, d_model)
    W_o      -- output projection weight, shape (d_model, d_model)
    Bias     -- bias for Q,K,V, shape (3*d_model,)
    Bias_o   -- bias for output projection, shape (d_model,)
    num_heads-- number of attention heads

    Returns:
    Out_linear  -- output of Multi-Head Attention, shape: (batch_size, seq_len, d_model)
    """

    batch_size, seq_len, d_model = X.shape
    head_dim = d_model // num_heads
    QKV = np.matmul(X, W.T) + Bias.reshape(1, 1, -1)
    Q = QKV[:,:, 0         : d_model  ]
    K = QKV[:,:, d_model   : 2*d_model]
    V = QKV[:,:, 2*d_model : 3*d_model]
    Q = Q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0,2,1,3)
    K = K.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0,2,1,3)
    V = V.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0,2,1,3)
    Q = Q / np.sqrt(head_dim)
    scores  = np.matmul(Q, K.transpose(0,1,3,2))
    attn    = softmax(scores, axis=-1)
    context = np.matmul(attn, V)
    context = context.transpose(0,2,1,3).reshape(batch_size, seq_len, d_model)
    return np.matmul(context, W_o.T) + Bias_o.reshape(1, 1, -1)


def MLP(X, W1_fc1, b1_fc1, W2_fc2, b2_fc2):
    """
    Simple feed-forward network (MLP) for transformer block.

    Arguments:
    X       -- input (1, seq_len, d_model)
    W1, W2  -- weights for two linear layers
               W1: (d_ff, d_model), W2: (d_model, d_ff)
    b1, b2  -- biases: b1 (d_ff,), b2 (d_model,)
    activation -- 'swish'

    Returns:
    out     -- output, same shape as X
    """

    Z1 = np.matmul(X, W1_fc1.T) + b1_fc1
    A1 = swish(Z1)
    return np.matmul(A1, W2_fc2.T) + b2_fc2


def transformer_encoder(X,
                         W_QKV_atten, W_o_atten, Bias_QKV_atten, Bias_o_atten,
                         W1_fc1, b1_fc1, W2_fc2, b2_fc2,
                         W_gamma1, b_beta1, W_gamma2, b_beta2,
                         num_heads):
    """
    Single transformer encoder block (MobileViT style).

    Arguments:
    X                  -- input tensor, shape: (batch, seq_len, d_model)
    W_QKV_atten         -- attention weights for QKV, shape: (3*d_model, d_model)
    W_o_atten           -- attention output linear weights, shape: (d_model, d_model)
    Bias_QKV_atten      -- bias for QKV, shape: (3*d_model,)
    Bias_o_atten        -- bias for output projection, shape: (d_model,)
    W1_fc1, W2_fc2      -- MLP weights, W1: (d_model, d_ff), W2: (d_ff, d_model)
    b1_fc1, b2_fc2      -- MLP biases, b1: (d_ff,), b2: (d_model,)
    W_gamma1, W_gamma2  -- scale parameters for layer normalization, shape: (1, 1, d_model)
    b_beta1, b_beta2    -- shift parameters for layer normalization, shape: (1, 1, d_model)
    num_heads           -- number of attention heads

    Returns:
    Z -- output tensor, shape: (batch, seq_len, d_model)
    """
    # Attention sub-block
    skip1 = X
    X = layer_norm_forward(X, W_gamma1, b_beta1)
    X = MultiHeadAttention(X, W_QKV_atten, W_o_atten,
                           Bias_QKV_atten, Bias_o_atten, num_heads)
    X = X + skip1
    # MLP sub-block
    skip2 = X
    X = layer_norm_forward(X, W_gamma2, b_beta2)
    X = MLP(X, W1_fc1, b1_fc1, W2_fc2, b2_fc2)
    return X + skip2


# ─────────────────────────────────────────────────────────────
# Local_representations
# ─────────────────────────────────────────────────────────────

def Local_representations(input,
                           W_loacal_3x3, b_local_3x3,
                           W_loacal_1x1, b_local_1x1,
                           W_gamma_local_3x3, W_beta_local_3x3,
                           running_mean1, running_var1):
    """
    Local Representation for MobileVIt Block

    Arguments:
    input              -- input tensor, shape (m, C_in, H, W)
    W_local_3x3        -- weights for 3x3 conv, shape (C_Out, C_in, 3, 3)
    b_local_3x3        -- bias for 3x3 conv, shape (C_Out, 1, 1, 1)
    W_local_1x1        -- weights for 1x1 conv, shape (d_model, C_Out, 1, 1)
    b_local_1x1        -- bias for 1x1 conv, shape (d_model, 1, 1, 1)
    W_gamma_local_3x3  -- BN scale for 3x3 conv, shape (1, C_Out, 1, 1)
    W_beta_local_3x3   -- BN shift for 3x3 conv, shape (1, C_Out, 1, 1)
    W_gamma_local_1x1  -- BN scale for 1x1 conv, shape (1, d_model, 1, 1)
    W_beta_local_1x1   -- BN shift for 1x1 conv, shape (1, d_model, 1, 1)

    Returns:
    out -- output tensor after conv → BN → Swish, shape (m, d_model, H, W)
    """
    # 3×3 conv → BN → Swish
    x = conv_forward(input, W_loacal_3x3, b_local_3x3, 1, 1)
    x = batch_norm_forward(x, W_gamma_local_3x3, W_beta_local_3x3,
                           running_mean1, running_var1)
    x = swish(x)
    # 1×1 conv only (no BN, no activation)
    x = conv_forward(x, W_loacal_1x1, b_local_1x1, 1, 0)
    return x


# ─────────────────────────────────────────────────────────────
# unfold
# ─────────────────────────────────────────────────────────────

def bilinear_resize(x, new_h, new_w):
    """Matches PyTorch F.interpolate(..., mode='bilinear', align_corners=False)."""
    b, c, h, w = x.shape
    row_idx = (np.arange(new_h) + 0.5) * (h / new_h) - 0.5
    col_idx = (np.arange(new_w) + 0.5) * (w / new_w) - 0.5
    row_idx = np.clip(row_idx, 0, h - 1)
    col_idx = np.clip(col_idx, 0, w - 1)
    r0 = np.floor(row_idx).astype(int)
    r1 = np.minimum(r0 + 1, h - 1)
    c0 = np.floor(col_idx).astype(int)
    c1 = np.minimum(c0 + 1, w - 1)
    dr = (row_idx - r0)[:, None]
    dc = (col_idx - c0)[None, :]
    top_left     = x[:, :, r0, :][:, :, :, c0]
    top_right    = x[:, :, r0, :][:, :, :, c1]
    bottom_left  = x[:, :, r1, :][:, :, :, c0]
    bottom_right = x[:, :, r1, :][:, :, :, c1]
    out = (top_left     * (1 - dr) * (1 - dc) +
           top_right    * (1 - dr) *      dc  +
           bottom_left  *      dr  * (1 - dc) +
           bottom_right *      dr  *      dc)
    return out.astype(x.dtype)

def unfold(x, patch_size=2):
    b, c, h, w = x.shape
    ph = pw = patch_size

    # Pad to next multiple of patch_size — matches MVT.py ceil logic
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

    x = x.reshape(b * c * num_patch_h, ph, num_patch_w, pw)
    x = x.transpose(0, 2, 1, 3)
    x = x.reshape(b, c, num_patches, patch_area)
    x = x.transpose(0, 3, 2, 1)
    patches = x.reshape(b * patch_area, num_patches, c)

    info = dict(b=b, c=c, orig_h=h, orig_w=w,
                num_patch_h=num_patch_h, num_patch_w=num_patch_w,
                num_patches=num_patches, patch_area=patch_area,
                ph=ph, pw=pw, interpolate=interpolate)
    return patches, info



# ─────────────────────────────────────────────────────────────
# fold
# ─────────────────────────────────────────────────────────────

def fold(patches, info):
    b           = info['b']
    num_patch_h = info['num_patch_h']
    num_patch_w = info['num_patch_w']
    num_patches = info['num_patches']
    patch_area  = info['patch_area']
    ph = info['ph']
    pw = info['pw']
    channels    = patches.shape[-1]

    x = patches.reshape(b, patch_area, num_patches, channels)
    x = x.transpose(0, 3, 2, 1)
    x = x.reshape(b * channels * num_patch_h, num_patch_w, ph, pw)
    x = x.transpose(0, 2, 1, 3)
    x = x.reshape(b, channels, num_patch_h * ph, num_patch_w * pw)

    # Revert interpolation if applied — matches MVT.py folding() step 6
    if info['interpolate']:
        x = bilinear_resize(x, info['orig_h'], info['orig_w'])

    return x


# ─────────────────────────────────────────────────────────────
# Fusion  
# ─────────────────────────────────────────────────────────────

def fusion(x_global, input_feat,
           W_fusion_1x1, b_fusion_1x1,
           W_fusion_3x3, b_fusion_3x3,
           W_gamma_f_1x1, b_beta_f_1x1,
           W_gamma_f_3x3, b_beta_f_3x3,
           running_mean1, running_var1,
           running_mean2, running_var2):
    """
    Fuse global features from transformer with local features.

    Arguments:
    x_global -- (B, d_model, H, W)
    input    -- (B, C, H, W)
    W_fusion_1x1 -- (C, d_model, 1, 1)
    b_fusion_1x1 -- (C, 1, 1, 1)
    W_gamma_f_1x1 -- (C, 1, 1)
    b_beta_f_1x1  -- (C, 1, 1)
    W_fusion_3x3  -- (C, 2*C, 3, 3)
    b_fusion_3x3  -- (C, 1, 1, 1)
    W_gamma_f_3x3 -- (C, 1, 1)
    b_beta_f_3x3  -- (C, 1, 1)

    Returns:
    x_fusion -- fused features (B, C_out, H, W)
    """
    # 1×1 → BN → Swish
    x = conv_forward(x_global, W_fusion_1x1, b_fusion_1x1, 1, 0)
    x = batch_norm_forward(x, W_gamma_f_1x1, b_beta_f_1x1, running_mean1, running_var1)
    x = swish(x)
    if x.shape[2:] != input_feat.shape[2:]:
        raise ValueError("Spatial dim mismatch in fusion.")
    # concat original input with projected global features
    concat = np.concatenate([input_feat, x], axis=1)
    # 3×3 → BN → Swish
    x = conv_forward(concat, W_fusion_3x3, b_fusion_3x3, 1, 1)
    x = batch_norm_forward(x, W_gamma_f_3x3, b_beta_f_3x3, running_mean2, running_var2)
    return swish(x)


# ─────────────────────────────────────────────────────────────
# MobileViTBlock
# ─────────────────────────────────────────────────────────────

def MobileViTBlock_(input,
                   # Local representation
                   W_loacal_3x3, b_local_3x3,
                   W_loacal_1x1, b_local_1x1,
                   W_gamma_local_3x3, W_beta_local_3x3,
                   # Transformer layers (lists, length L)
                   W_QKV_atten, W_o_atten, Bias_QKV_atten, Bias_o_atten,
                   W1_fc1, b1_fc1, W2_fc2, b2_fc2,
                   W_gamma1, b_beta1, W_gamma2, b_beta2,
                   # Final LayerNorm (NEW – matches MVT.py self.norm)
                   W_norm_gamma, b_norm_beta,
                   # Fusion
                   W_fusion_1x1, b_fusion_1x1,
                   W_fusion_3x3, b_fusion_3x3,
                   W_gamma_f_1x1, b_beta_f_1x1,
                   W_gamma_f_3x3, b_beta_f_3x3,
                   # Config
                   L, patch_size, num_heads,
                   # BN running stats: local-3x3, fusion-1x1, fusion-3x3
                   running_mean1, running_var1,   # local 3×3 BN
                   running_mean2, running_var2,   # fusion 1×1 BN
                   running_mean3, running_var3):  # fusion 3×3 BN

    # 1. Local feature extraction
    x_local = Local_representations(
        input,
        W_loacal_3x3, b_local_3x3,
        W_loacal_1x1, b_local_1x1,
        W_gamma_local_3x3, W_beta_local_3x3,
        running_mean1, running_var1)

    # 2. Unfold (FIX 2)
    patches, info = unfold(x_local, patch_size)

    # 3. Transformer encoding
    for i in range(L):
        patches = transformer_encoder(
            patches,
            W_QKV_atten[i], W_o_atten[i],
            Bias_QKV_atten[i], Bias_o_atten[i],
            W1_fc1[i], b1_fc1[i], W2_fc2[i], b2_fc2[i],
            W_gamma1[i], b_beta1[i], W_gamma2[i], b_beta2[i],
            num_heads)

    # 4. Final LayerNorm (FIX 4 – matches MVT.py self.norm)
    patches = layer_norm_forward(patches, W_norm_gamma, b_norm_beta)

    # 5. Fold (FIX 3)
    x_global = fold(patches, info)

    # 6. Fusion
    x_out = fusion(
        x_global, input,
        W_fusion_1x1, b_fusion_1x1,
        W_fusion_3x3, b_fusion_3x3,
        W_gamma_f_1x1, b_beta_f_1x1,
        W_gamma_f_3x3, b_beta_f_3x3,
        running_mean2, running_var2,
        running_mean3, running_var3)

    return x_out


# ─────────────────────────────────────────────────────────────
# Global average pool + linear classifier
# ─────────────────────────────────────────────────────────────

def global_avg_pool(x):
    """
    Global average pooling over spatial dimensions (H, W) for input shape (m, C, H, W).

    Arguments:
    x -- input tensor, shape (m, C, H, W)

    Returns:
    pooled -- tensor of shape (m, C)
    """
    return np.mean(x, axis=(2, 3))


def linear_classifier(x, W_cls, b_cls):
    """
    Linear classifier for input shape (m, C).

    Arguments:
    x     -- input tensor, shape (m, C)
    W_cls -- weights of shape (C, num_classes)
    b_cls -- biases of shape (num_classes,)

    Returns:
    logits -- output tensor, shape (m, num_classes)
    """
    return np.matmul(x, W_cls) + b_cls


# ─────────────────────────────────────────────────────────────
# MobileViT_XXS_  
# ─────────────────────────────────────────────────────────────

def MobileViT_XXS_(Inputs, bn_prams,
                   # Stem
                   W_stem, b_stem, W_stem_gamma, b_beta_stem,
                   # Stage 1
                   W_expan_1,  W_depth_1,  W_point_1,
                   b_expan_1,  b_depth_1,  b_point_1,
                   W_gamma_expan_1, W_gamma_depth_1, W_gamma_point_1,
                   b_gamma_expan_1, b_gamma_depth_1, b_gamma_point_1,
                   # Stage 2a
                   W_expan_2a, W_depth_2a, W_point_2a,
                   b_expan_2a, b_depth_2a, b_point_2a,
                   W_gamma_expan_2a, W_gamma_depth_2a, W_gamma_point_2a,
                   b_gamma_expan_2a, b_gamma_depth_2a, b_gamma_point_2a,
                   # Stage 2b
                   W_expan_2b, W_depth_2b, W_point_2b,
                   b_expan_2b, b_depth_2b, b_point_2b,
                   W_gamma_expan_2b, W_gamma_depth_2b, W_gamma_point_2b,
                   b_gamma_expan_2b, b_gamma_depth_2b, b_gamma_point_2b,
                   # Stage 2c
                   W_expan_2c, W_depth_2c, W_point_2c,
                   b_expan_2c, b_depth_2c, b_point_2c,
                   W_gamma_expan_2c, W_gamma_depth_2c, W_gamma_point_2c,
                   b_gamma_expan_2c, b_gamma_depth_2c, b_gamma_point_2c,
                   # Stage 3a
                   W_expan_3a, W_depth_3a, W_point_3a,
                   b_expan_3a, b_depth_3a, b_point_3a,
                   W_gamma_expan_3a, W_gamma_depth_3a, W_gamma_point_3a,
                   b_gamma_expan_3a, b_gamma_depth_3a, b_gamma_point_3a,
                   # Stage 3b – MobileViTBlock
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
                   # Stage 4a
                   W_expan_4a, W_depth_4a, W_point_4a,
                   b_expan_4a, b_depth_4a, b_point_4a,
                   W_gamma_expan_4a, W_gamma_depth_4a, W_gamma_point_4a,
                   b_gamma_expan_4a, b_gamma_depth_4a, b_gamma_point_4a,
                   # Stage 4b – MobileViTBlock
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
                   # Stage 5a
                   W_expan_5a, W_depth_5a, W_point_5a,
                   b_expan_5a, b_depth_5a, b_point_5a,
                   W_gamma_expan_5a, W_gamma_depth_5a, W_gamma_point_5a,
                   b_gamma_expan_5a, b_gamma_depth_5a, b_gamma_point_5a,
                   # Stage 5b – MobileViTBlock
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
                   # Head
                   W_head, b_head, W_gamma_head, b_beta_head,
                   # Classifier
                   W_cls, b_cls):

    # ── Stem ─────────────────────────────────────────────────
    x = conv_forward(Inputs, W_stem, b_stem, 2, 1)
    x = batch_norm_forward(x, W_stem_gamma, b_beta_stem, bn_prams[0], bn_prams[1])
    x = swish(x)

    # ── Stage 1 ───────────────────────────────────────────────
    x = MobileNet_v2(x, W_expan_1, W_depth_1, W_point_1,
                     b_expan_1, b_depth_1, b_point_1, 1,
                     W_gamma_expan_1, W_gamma_depth_1, W_gamma_point_1,
                     b_gamma_expan_1, b_gamma_depth_1, b_gamma_point_1,
                     *bn_prams[2:8])

    # ── Stage 2 ───────────────────────────────────────────────
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

    # ── Stage 3 ───────────────────────────────────────────────
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
                       2, 2, 2,                               # L, patch_size, num_heads
                       *bn_prams[32:38])

    # ── Stage 4 ───────────────────────────────────────────────
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
                       4, 2, 4,                               # L, patch_size, num_heads
                       *bn_prams[44:50])

    # ── Stage 5 ───────────────────────────────────────────────
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
                       3, 2, 4,                               # L, patch_size, num_heads
                       *bn_prams[56:62])

    # ── Head ──────────────────────────────────────────────────
    x = conv_forward(x, W_head, b_head, 1, 0)
    x = batch_norm_forward(x, W_gamma_head, b_beta_head, bn_prams[62], bn_prams[63])
    x = swish(x)
    x = global_avg_pool(x)

    # classifier: acc2 does W_cls = W_cls.T then x @ W_cls
    # → same as x @ W_cls_original.T
    # The mapping supplies W_cls as (num_classes, 320); we transpose here.
    W_cls = W_cls.T
    return linear_classifier(x, W_cls, b_cls)