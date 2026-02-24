import os

import numpy as np
from einops import rearrange
import re


# NO API Model
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

    X_pad = np.pad(X, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant', constant_values = (0,0))


    return X_pad

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

    Z = float(np.sum(a_slice_prev * W) + b.item())

    return Z

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

    shape_a = np.array(A_prev.shape)
    m = shape_a[0]
    n_C_prev = shape_a[1]
    n_H_prev = shape_a[2]
    n_W_prev = shape_a[3]

    shape_W = np.array(W.shape)
    f = shape_W[2]
    n_C = shape_W[0]

    stride = stride
    pad = pad


    n_H = int(((n_H_prev-f+(2*pad))/(stride)) + 1)
    n_W = int(((n_W_prev-f+(2*pad))/(stride)) + 1)

    Z = np.zeros((m,n_C,n_H,n_W))

    A_prev_pad = zero_pad(A_prev,pad)

    a_prev_pad = np.zeros ((n_C_prev,n_H,n_W))

    for i in range (m):
        a_prev_pad = A_prev_pad[i]
        for h in range (n_H):
            vert_start = 0 + h*stride
            vert_end = f + h*stride
            for w in range (n_W):
                horz_start = 0 + w*stride
                horz_end = f + w*stride
                for c in range (n_C):
                    a_prev_slice_pad = a_prev_pad[:,vert_start:vert_end,horz_start:horz_end]
                    weights = W[c,:,:,:]
                    biases = b[c]
                    Z[i,c,h,w] = conv_single_step(a_prev_slice_pad,weights,biases)

    return Z

import numpy as np

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
    gamma = gamma.reshape(1, -1, 1, 1)
    beta = beta.reshape(1, -1, 1, 1)
    mean = running_mean.reshape(1, -1, 1, 1)
    var = running_var.reshape(1, -1, 1, 1)

    # Normalize using stored running statistics
    Z_hat = (Z - mean) / np.sqrt(var + eps)

    # Scale and shift
    Z_norm = gamma * Z_hat + beta

    return Z_norm

def sigmoid(x):
    # numerically stable version
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),            # safe when x >= 0
        np.exp(x) / (1 + np.exp(x))      # safe when x < 0
    )

def swish(x):
    """
    Swish activation function.

    Argument:
    Z -- numpy array

    Returns:
    A -- activated output, same shape as Z
    """
    return x * sigmoid(x)


def depthwise_conv (A_prev,W ,b, stride, pad):
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
    (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape
    f = W.shape[2]

    # Apply padding
    A_prev = zero_pad(A_prev,pad)

    # Define the dimensions of the output
    n_H = int(((n_H_prev-f+(2*pad))/(stride)) + 1)
    n_W = int(((n_W_prev-f+(2*pad))/(stride)) + 1)
    n_C = n_C_prev

    # Initialize output matrix A
    A = np.zeros((m, n_C, n_H, n_W))

    for i in range (m):
        a_prev = A_prev[i]
        for h in range (n_H):
            vert_start = 0 + h*stride
            vert_end = f + h*stride
            for w in range (n_W):
                horiz_start = 0+ w*stride
                horiz_end = f+w*stride
                for c in range (n_C):
                    a_prev_slice = a_prev[c,vert_start:vert_end,horiz_start:horiz_end]
                    A[i, c, h, w] = np.sum(a_prev_slice * W[c, :, :]) + b[c]


    return A

def MobileNet_v2 (Input, W_expan ,W_depth ,W_point ,b_expan ,b_depth ,b_point,stride_depth
                  ,W_gamma_expan,W_gamma_depth,W_gamma_point,b_gamma_expan,b_gamma_depth,b_gamma_point
                  ,running_mean1,running_var1,running_mean2,running_var2,running_mean3,running_var3):
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
    Z1_expan = conv_forward(Input,W_expan,b_expan,1,0)
    Z1_norm_expan = batch_norm_forward(Z1_expan,W_gamma_expan,b_gamma_expan,running_mean1,running_var1)
    A1 = swish(Z1_norm_expan)

    Z2_depth = depthwise_conv (A1,W_depth,b_depth, stride_depth,1)
    Z2_norm_depth = batch_norm_forward(Z2_depth,W_gamma_depth,b_gamma_depth,running_mean2,running_var2)
    A2 = swish(Z2_norm_depth)

    Z3_point = conv_forward(A2,W_point,b_point,1,0)
    Z3_norm_point = batch_norm_forward(Z3_point,W_gamma_point,b_gamma_point,running_mean3,running_var3)
    A3 = Z3_norm_point


    if (stride_depth == 1 and Input.shape == A3.shape):
         A_out = A3 + Input
    else:
         A_out = A3

    return A_out


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
    # Mean and variance across the last dimension (d_model)
    mean = np.mean(X, axis=-1, keepdims=True)
    var  = np.var(X, axis=-1, keepdims=True)

    # Normalize
    X_norm = (X - mean) / np.sqrt(var + eps)

    # Scale + shift (broadcast along batch and seq_len)
    out = gamma * X_norm + beta

    return out

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
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)

    # Normalize to get probabilities
    out = e_x / np.sum(e_x, axis=axis, keepdims=True)
    return out

def Local_representations(input,W_loacal_3x3,b_local_3x3,W_loacal_1x1,b_local_1x1,
                          W_gamma_local_3x3,W_beta_local_3x3,running_mean1,running_var1):

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
    # 3x3 Conv
    x = conv_forward(input,W_loacal_3x3,b_local_3x3,1,1)
    x = batch_norm_forward(x,W_gamma_local_3x3,W_beta_local_3x3,running_mean1,running_var1)
    x = swish(x)

    # 1x1 Conv
    x = conv_forward(x,W_loacal_1x1,b_local_1x1,1,0)

    return x

def unfold(x_local,patch_size = 2):
    """
    Convert a feature map into flattened patches for transformer input.

    Arguments:
    x_local    -- input tensor, shape (batch, channels, height, width)
    patch_size -- size of each patch (ph = pw)

    Returns:
    x_patches  -- tensor of shape (batch, num_patches, patch_dim)
                  where num_patches = (height//patch_size)*(width//patch_size)
                  and patch_dim = patch_size*patch_size*channels
    """

    # Patch embedding for transformer
    ph = pw = patch_size
    _, _, h, w = x_local.shape
    x_patches = rearrange(x_local, 'b d (h ph) (w pw) -> b (h w ph pw) d', ph=ph, pw=pw)

    return x_patches


# def MultiHeadAttention(X, W, W_o, Bias, Bias_o, num_heads=4):
#     """
#     Multi-Head Attention for transformer (MobileViT style)

#     Arguments:
#     X        -- input tensor, shape (batch_size, seq_len, d_model)
#     W        -- weights for Q,K,V, shape (3*d_model, d_model)
#     W_o      -- output projection weight, shape (d_model, d_model)
#     Bias     -- bias for Q,K,V, shape (3*d_model,)
#     Bias_o   -- bias for output projection, shape (d_model,)
#     num_heads-- number of attention heads

#     Returns:
#     Out_linear  -- output of Multi-Head Attention, shape: (batch_size, seq_len, d_model)
#     """

#     batch_size, seq_len, d_model = X.shape
#     head_dim = d_model // num_heads

#     # Compute Q, K, V from combined weights
#     # W shape: (3*d_model, d_model), so multiply as X @ W.T
#     QKV = np.matmul(X, W.T) + Bias  # shape: (batch_size, seq_len, 3*d_model)

#     Q = QKV[:, :, 0:d_model]
#     K = QKV[:, :, d_model:2*d_model]
#     V = QKV[:, :, 2*d_model:3*d_model]

#      # Reshape into (batch, num_heads, seq_len, head_dim)
#     Q = Q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
#     K = K.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
#     V = V.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

#     # Attention scores: (batch, num_heads, seq_len, seq_len)
#     scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
#     attn = softmax(scores, axis=-1)

#     # Weighted sum of values
#     context = np.matmul(attn, V)  # (batch, num_heads, seq_len, head_dim)

#     # Concatenate heads: (batch, seq_len, d_model)
#     context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

#     # Final linear projection
#     Out_linear = np.matmul(context, W_o) + Bias_o

#     return Out_linear

def MultiHeadAttention(X, W, W_o, Bias, Bias_o, num_heads=4):
    batch_size, seq_len, d_model = X.shape
    head_dim = d_model // num_heads

    # QKV projection (NO transpose here)
    QKV = np.matmul(X, W.T) + Bias.reshape(1, 1, -1)

    Q = QKV[:, :, 0:d_model]
    K = QKV[:, :, d_model:2*d_model]
    V = QKV[:, :, 2*d_model:3*d_model]

    # Reshape into heads
    Q = Q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    # Scale Q BEFORE matmul (PyTorch convention)
    Q = Q / np.sqrt(head_dim)

    # Attention scores
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2))
    attn = softmax(scores, axis=-1)

    # Weighted sum
    context = np.matmul(attn, V)

    # Concatenate heads
    context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    # Output projection (transpose weight here!)
    Out_linear = np.matmul(context, W_o.T) + Bias_o.reshape(1, 1, -1)

    return Out_linear


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
    W1_fc1 = W1_fc1.T
    # Linear 1: expand
    Z1_expand = np.matmul(X,W1_fc1) + b1_fc1  # shape: (1, seq_len, d_ff)
    A1 = swish(Z1_expand)

    Z2_proj = np.matmul(A1, W2_fc2.T) + b2_fc2

    return Z2_proj


def transformer_encoder (X,W_QKV_atten,W_o_atten,Bias_QKV_atten,Bias_o_atten
                         ,W1_fc1,b1_fc1,W2_fc2,b2_fc2
                         ,W_gamma1,b_beta1,W_gamma2,b_beta2,num_heads=4):
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

    # Attention block
    skip_1 = X

    X = layer_norm_forward(X,W_gamma1,b_beta1)

    out_attn = MultiHeadAttention(X,W_QKV_atten,W_o_atten,Bias_QKV_atten,Bias_o_atten,num_heads)

    X = out_attn + skip_1


    # MLP block
    skip_2 = X
    X_norm = layer_norm_forward(X,W_gamma2,b_beta2)

    X = MLP(X_norm,W1_fc1,b1_fc1,W2_fc2,b2_fc2)

    Z = X + skip_2


    return Z

def fold(x_patches,h,w,patch_size = 2):

    ph = pw = patch_size
    # Reshape back
    x_global = rearrange(x_patches, 'b (h w ph pw) d -> b d (h ph) (w pw)',
    h=h//ph, w=w//pw, ph=ph, pw=pw)

    return x_global

def fusion(x_global,input,W_fusion_1x1,b_fusion_1x1,W_fusion_3x3,b_fusion_3x3
           ,W_gamma_f_1x1,b_beta_f_1x1,W_gamma_f_3x3,b_beta_f_3x3,running_mean1,running_var1,running_mean2,running_var2):
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

    # 1x1 conv to match channels
    x = conv_forward(x_global,W_fusion_1x1,b_fusion_1x1,1,0)
    x_norm = batch_norm_forward(x,W_gamma_f_1x1,b_beta_f_1x1,running_mean1,running_var1)
    x = swish(x_norm)

    # If X and fus_1x1 shapes differ in channels, spatial dims must match.
    if x.shape[2:] != input.shape[2:]:
        raise ValueError("Spatial dims mismatch between fusion features and input for concatenation.")
    # Concatenate along channel axis
    concat = np.concatenate([input, x], axis=1)   # shape (B,2C, H, W)

    # 3x3 conv to fuse
    X_fusion = conv_forward(concat,W_fusion_3x3,b_fusion_3x3,1,1)
    X_fusion_norm = batch_norm_forward(X_fusion,W_gamma_f_3x3,b_beta_f_3x3,running_mean2,running_var2)
    x_fusion = swish(X_fusion_norm)

    return x_fusion

def MobileViTBlock(input,W_loacal_3x3,b_local_3x3,W_loacal_1x1,b_local_1x1,W_gamma_local_3x3,W_beta_local_3x3
                    ,W_QKV_atten,W_o_atten,Bias_QKV_atten,Bias_o_atten,W1_fc1,b1_fc1,W2_fc2,b2_fc2,W_gamma1,b_beta1,W_gamma2,b_beta2
                    ,W_fusion_1x1,b_fusion_1x1,W_fusion_3x3,b_fusion_3x3,W_gamma_f_1x1,b_beta_f_1x1,W_gamma_f_3x3,b_beta_f_3x3
                    ,L,patch_size,num_heads
                    ,running_mean1,running_var1,running_mean3,running_var3,running_mean4,running_var4):

    # -------------------------------
    # 1. Local feature extraction
    # -------------------------------
    # Apply a 3x3 conv followed by BN + swish, then 1x1 conv + BN + swish
    x_local = Local_representations(input,W_loacal_3x3,b_local_3x3,W_loacal_1x1,b_local_1x1,
                                W_gamma_local_3x3,W_beta_local_3x3,running_mean1,running_var1)


    _, _, h, w = x_local.shape

    # -------------------------------
    # 2. Unfold into patches for transformer
    # -------------------------------
    x_unfold = unfold(x_local,patch_size)

    # -------------------------------
    # 3. Transformer encoding
    # -------------------------------
    x_transformer = x_unfold
    for i in range (L):
        x_transformer = transformer_encoder(x_transformer,W_QKV_atten[i],W_o_atten[i],Bias_QKV_atten[i],Bias_o_atten[i]
                                            ,W1_fc1[i],b1_fc1[i],W2_fc2[i],b2_fc2[i]
                                            ,W_gamma1[i],b_beta1[i],W_gamma2[i],b_beta2[i],num_heads)

    # -------------------------------
    # 4. Fold back patches to spatial layout
    # -------------------------------
    x_global = fold(x_transformer,h,w,patch_size)

    # -------------------------------
    # 5. Fusion with input features
    # -------------------------------
    x_fusion = fusion(x_global,input,W_fusion_1x1,b_fusion_1x1,W_fusion_3x3,b_fusion_3x3
                      ,W_gamma_f_1x1,b_beta_f_1x1,W_gamma_f_3x3,b_beta_f_3x3,running_mean3,running_var3,running_mean4,running_var4)

    # Final output of MobileViT block
    x_out = x_fusion  # shape: (batch, channels, H, W)

    return x_out

def global_avg_pool(x):
    """
    Global average pooling over spatial dimensions (H, W) for input shape (m, C, H, W).

    Arguments:
    x -- input tensor, shape (m, C, H, W)

    Returns:
    pooled -- tensor of shape (m, C)
    """
    pooled = np.mean(x, axis=(2, 3))  # average over H and W
    return pooled

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
    logits = np.matmul(x, W_cls) + b_cls
    return logits

def MobileViT_XXS_(Inputs,bn_prams,W_stem,b_stem,W_stem_gamma,b_beta_stem
                  ,W_expan_1 ,W_depth_1 ,W_point_1 ,b_expan_1 ,b_depth_1 ,b_point_1,W_gamma_expan_1,W_gamma_depth_1,W_gamma_point_1,b_gamma_expan_1,b_gamma_depth_1,b_gamma_point_1
                  ,W_expan_2a ,W_depth_2a ,W_point_2a ,b_expan_2a ,b_depth_2a ,b_point_2a,W_gamma_expan_2a,W_gamma_depth_2a,W_gamma_point_2a,b_gamma_expan_2a,b_gamma_depth_2a,b_gamma_point_2a
                  ,W_expan_2b ,W_depth_2b ,W_point_2b ,b_expan_2b ,b_depth_2b ,b_point_2b,W_gamma_expan_2b,W_gamma_depth_2b,W_gamma_point_2b,b_gamma_expan_2b,b_gamma_depth_2b,b_gamma_point_2b
                  ,W_expan_2c ,W_depth_2c ,W_point_2c ,b_expan_2c ,b_depth_2c ,b_point_2c,W_gamma_expan_2c,W_gamma_depth_2c,W_gamma_point_2c,b_gamma_expan_2c,b_gamma_depth_2c,b_gamma_point_2c
                  ,W_expan_3a ,W_depth_3a ,W_point_3a ,b_expan_3a ,b_depth_3a ,b_point_3a,W_gamma_expan_3a,W_gamma_depth_3a,W_gamma_point_3a,b_gamma_expan_3a,b_gamma_depth_3a,b_gamma_point_3a
                  ,W_loacal_3x3_3b,b_local_3x3_3b,W_loacal_1x1_3b,b_local_1x1_3b,W_gamma_local_3x3_3b,W_beta_local_3x3_3b,W_QKV_atten_3b,W_o_atten_3b,Bias_QKV_atten_3b,Bias_o_atten_3b,W1_fc1_3b,b1_fc1_3b,W2_fc2_3b,b2_fc2_3b,W_gamma1_3b,b_beta1_3b,W_gamma2_3b,b_beta2_3b,W_fusion_1x1_3b,b_fusion_1x1_3b,W_fusion_3x3_3b,b_fusion_3x3_3b,W_gamma_f_1x1_3b,b_beta_f_1x1_3b,W_gamma_f_3x3_3b,b_beta_f_3x3_3b
                  ,W_expan_4a ,W_depth_4a ,W_point_4a ,b_expan_4a ,b_depth_4a ,b_point_4a,W_gamma_expan_4a,W_gamma_depth_4a,W_gamma_point_4a,b_gamma_expan_4a,b_gamma_depth_4a,b_gamma_point_4a
                  ,W_loacal_3x3_4b,b_local_3x3_4b,W_loacal_1x1_4b,b_local_1x1_4b,W_gamma_local_3x3_4b,W_beta_local_3x3_4b,W_QKV_atten_4b,W_o_atten_4b,Bias_QKV_atten_4b,Bias_o_atten_4b,W1_fc1_4b,b1_fc1_4b,W2_fc2_4b,b2_fc2_4b,W_gamma1_4b,b_beta1_4b,W_gamma2_4b,b_beta2_4b,W_fusion_1x1_4b,b_fusion_1x1_4b,W_fusion_3x3_4b,b_fusion_3x3_4b,W_gamma_f_1x1_4b,b_beta_f_1x1_4b,W_gamma_f_3x3_4b,b_beta_f_3x3_4b
                  ,W_expan_5a ,W_depth_5a ,W_point_5a ,b_expan_5a ,b_depth_5a ,b_point_5a,W_gamma_expan_5a,W_gamma_depth_5a,W_gamma_point_5a,b_gamma_expan_5a,b_gamma_depth_5a,b_gamma_point_5a
                  ,W_loacal_3x3_5b,b_local_3x3_5b,W_loacal_1x1_5b,b_local_1x1_5b,W_gamma_local_3x3_5b,W_beta_local_3x3_5b,W_QKV_atten_5b,W_o_atten_5b,Bias_QKV_atten_5b,Bias_o_atten_5b,W1_fc1_5b,b1_fc1_5b,W2_fc2_5b,b2_fc2_5b,W_gamma1_5b,b_beta1_5b,W_gamma2_5b,b_beta2_5b,W_fusion_1x1_5b,b_fusion_1x1_5b,W_fusion_3x3_5b,b_fusion_3x3_5b,W_gamma_f_1x1_5b,b_beta_f_1x1_5b,W_gamma_f_3x3_5b,b_beta_f_3x3_5b
                  ,W_head,b_head,W_gamma_head,b_beta_head
                  ,W_cls,b_cls):

    # Stem
    stem = conv_forward(Inputs,W_stem,b_stem,2,1)
    stem_norm = batch_norm_forward(stem,W_stem_gamma,b_beta_stem,bn_prams[0],bn_prams[1])
    stem = swish(stem_norm)

    # Stage 1
    mv2_1 = MobileNet_v2(stem, W_expan_1 ,W_depth_1 ,W_point_1 ,b_expan_1 ,b_depth_1 ,b_point_1,1
                        ,W_gamma_expan_1,W_gamma_depth_1,W_gamma_point_1,b_gamma_expan_1,b_gamma_depth_1,b_gamma_point_1
                        ,*bn_prams[2:8])
    # Stage 2
    mv2_2a = MobileNet_v2(mv2_1, W_expan_2a ,W_depth_2a ,W_point_2a ,b_expan_2a ,b_depth_2a ,b_point_2a,2
                        ,W_gamma_expan_2a,W_gamma_depth_2a,W_gamma_point_2a,b_gamma_expan_2a,b_gamma_depth_2a,b_gamma_point_2a,*bn_prams[8:14])
    mv2_2b = MobileNet_v2(mv2_2a, W_expan_2b ,W_depth_2b ,W_point_2b ,b_expan_2b ,b_depth_2b ,b_point_2b,1
                        ,W_gamma_expan_2b,W_gamma_depth_2b,W_gamma_point_2b,b_gamma_expan_2b,b_gamma_depth_2b,b_gamma_point_2b,*bn_prams[14:20])
    mv2_2c = MobileNet_v2(mv2_2b, W_expan_2c ,W_depth_2c ,W_point_2c ,b_expan_2c ,b_depth_2c ,b_point_2c,1
                        ,W_gamma_expan_2c,W_gamma_depth_2c,W_gamma_point_2c,b_gamma_expan_2c,b_gamma_depth_2c,b_gamma_point_2c,*bn_prams[20:26])

    # Stage 3
    mv2_3a = MobileNet_v2(mv2_2c, W_expan_3a ,W_depth_3a ,W_point_3a ,b_expan_3a ,b_depth_3a ,b_point_3a,2
                        ,W_gamma_expan_3a,W_gamma_depth_3a,W_gamma_point_3a,b_gamma_expan_3a,b_gamma_depth_3a,b_gamma_point_3a,*bn_prams[26:32])


    mvit_3b = MobileViTBlock(mv2_3a,W_loacal_3x3_3b,b_local_3x3_3b,W_loacal_1x1_3b,b_local_1x1_3b,W_gamma_local_3x3_3b,W_beta_local_3x3_3b
                             ,W_QKV_atten_3b,W_o_atten_3b,Bias_QKV_atten_3b,Bias_o_atten_3b,W1_fc1_3b,b1_fc1_3b,W2_fc2_3b,b2_fc2_3b,W_gamma1_3b,b_beta1_3b,W_gamma2_3b,b_beta2_3b
                             ,W_fusion_1x1_3b,b_fusion_1x1_3b,W_fusion_3x3_3b,b_fusion_3x3_3b,W_gamma_f_1x1_3b,b_beta_f_1x1_3b,W_gamma_f_3x3_3b,b_beta_f_3x3_3b
                             ,2 ,2 ,4,*bn_prams[32:38])

    # Stage 4
    mv2_4a = MobileNet_v2(mvit_3b, W_expan_4a ,W_depth_4a ,W_point_4a ,b_expan_4a ,b_depth_4a ,b_point_4a,2
                        ,W_gamma_expan_4a,W_gamma_depth_4a,W_gamma_point_4a,b_gamma_expan_4a,b_gamma_depth_4a,b_gamma_point_4a,*bn_prams[38:44])

    mvit_4b = MobileViTBlock(mv2_4a,W_loacal_3x3_4b,b_local_3x3_4b,W_loacal_1x1_4b,b_local_1x1_4b,W_gamma_local_3x3_4b,W_beta_local_3x3_4b
                            ,W_QKV_atten_4b,W_o_atten_4b,Bias_QKV_atten_4b,Bias_o_atten_4b,W1_fc1_4b,b1_fc1_4b,W2_fc2_4b,b2_fc2_4b,W_gamma1_4b,b_beta1_4b,W_gamma2_4b,b_beta2_4b
                            ,W_fusion_1x1_4b,b_fusion_1x1_4b,W_fusion_3x3_4b,b_fusion_3x3_4b,W_gamma_f_1x1_4b,b_beta_f_1x1_4b,W_gamma_f_3x3_4b,b_beta_f_3x3_4b
                            ,4 , 2 ,4,*bn_prams[44:50])

    # Stage 5
    mv2_5a = MobileNet_v2(mvit_4b, W_expan_5a ,W_depth_5a ,W_point_5a ,b_expan_5a ,b_depth_5a ,b_point_5a,2
                        ,W_gamma_expan_5a,W_gamma_depth_5a,W_gamma_point_5a,b_gamma_expan_5a,b_gamma_depth_5a,b_gamma_point_5a,*bn_prams[50:56])
    mvit_5b = MobileViTBlock(mv2_5a,W_loacal_3x3_5b,b_local_3x3_5b,W_loacal_1x1_5b,b_local_1x1_5b,W_gamma_local_3x3_5b,W_beta_local_3x3_5b
                            ,W_QKV_atten_5b,W_o_atten_5b,Bias_QKV_atten_5b,Bias_o_atten_5b,W1_fc1_5b,b1_fc1_5b,W2_fc2_5b,b2_fc2_5b,W_gamma1_5b,b_beta1_5b,W_gamma2_5b,b_beta2_5b
                            ,W_fusion_1x1_5b,b_fusion_1x1_5b,W_fusion_3x3_5b,b_fusion_3x3_5b,W_gamma_f_1x1_5b,b_beta_f_1x1_5b,W_gamma_f_3x3_5b,b_beta_f_3x3_5b
                            ,3 ,2 ,4,*bn_prams[56:62])

    head_conv_1x1 = conv_forward(mvit_5b,W_head,b_head,1,0)
    head_conv_1x1_norm = batch_norm_forward(head_conv_1x1,W_gamma_head,b_beta_head,*bn_prams[62:64])
    head_conv = swish(head_conv_1x1_norm)

    pool = global_avg_pool(head_conv)

    W_cls = W_cls.T
    classifier = linear_classifier(pool,W_cls,b_cls)

    return classifier



# Read file and store lines into a list
with open("all_parameters.txt", "r") as f:
    lines = f.readlines()

# Remove the newline characters at the end of each line
lines = [line.strip() for line in lines]

# Now you can access like list[0] -> line 1, list[249] -> line 250
print(lines[0])      # first line
print(lines[249])    # 250th line


# Prase Lines
def parse_line(line):
    # Split into parts
    name, rest = line.split(" | dtype: ")
    dtype_str, rest = rest.split(" | shape: ")
    shape_str, values_str = rest.split(" -> ")

    # Parse dtype
    dtype = np.float32 if "float32" in dtype_str else np.float64

    # Parse shape
    shape = tuple(map(int, re.findall(r"\d+", shape_str)))

    # Parse values
    values_list = eval(values_str.strip())  # convert string list to Python list

    values = np.array(values_list, dtype=dtype).reshape(shape)

    return {
        "name": name.strip(),
        "dtype": dtype,
        "shape": shape,
        "values": values
    }

# Example usage
line = lines[20]
parsed = parse_line(line)

print("Name:", parsed["name"])
print("Dtype:", parsed["dtype"])
print("Shape:", parsed["shape"])
print("Values array shape:", parsed["values"].shape)

# take All prams
prams = [None] * 250
for i in range(250):
    line = lines[i]
    parsed = parse_line(line)

    prams[i] = parsed["values"]


# Read file and store lines into a list
with open("bn_running_stats.txt", "r") as f:
    lines_bn = f.readlines()

# Remove the newline characters at the end of each line
lines_bn = [line.strip() for line in lines_bn]

# Now you can access like list[0] -> line 1, list[249] -> line 250
print(lines_bn[0])      # first line
print(lines_bn[60])    # 250th line

# take All BN prams
bn_prams = [None] * 70
for i in range(70):
    line_bn = lines_bn[i]
    bn_prams_parsed = parse_line(line_bn)

    bn_prams[i] = bn_prams_parsed["values"]
print(bn_prams[60])


# Fill Prameters function
def fill_mobilevit_parameters(prams):
    """
    Assign parameters from prams list to MobileViT_XXS arguments, handling transformer parameters as lists.

    Arguments:
    prams -- list of 250 NumPy arrays containing model parameters

    Returns:
    args -- dictionary of arguments for MobileViT_XXS
    """

    # Initialize index for prams
    idx = 0

    # Dictionary to store arguments
    args = {}

    # Stem (4 parameters)
    args['W_stem'] = prams[idx]; idx += 1
    args['b_stem'] = prams[idx]; idx += 1
    args['W_stem_gamma'] = prams[idx]; idx += 1
    args['b_beta_stem'] = prams[idx]; idx += 1

    # Stage 1 (12 parameters)
    args['W_expan_1'] = prams[idx]; idx += 1
    args['b_expan_1'] = prams[idx]; idx += 1
    args['W_gamma_expan_1'] = prams[idx]; idx += 1
    args['b_gamma_expan_1'] = prams[idx]; idx += 1
    args['W_depth_1'] = prams[idx]; idx += 1
    args['b_depth_1'] = prams[idx]; idx += 1
    args['W_gamma_depth_1'] = prams[idx]; idx += 1
    args['b_gamma_depth_1'] = prams[idx]; idx += 1
    args['W_point_1'] = prams[idx]; idx += 1
    args['b_point_1'] = prams[idx]; idx += 1
    args['W_gamma_point_1'] = prams[idx]; idx += 1
    args['b_gamma_point_1'] = prams[idx]; idx += 1

    # Stage 2a (12 parameters)
    args['W_expan_2a'] = prams[idx]; idx += 1
    args['b_expan_2a'] = prams[idx]; idx += 1
    args['W_gamma_expan_2a'] = prams[idx]; idx += 1
    args['b_gamma_expan_2a'] = prams[idx]; idx += 1
    args['W_depth_2a'] = prams[idx]; idx += 1
    args['b_depth_2a'] = prams[idx]; idx += 1
    args['W_gamma_depth_2a'] = prams[idx]; idx += 1
    args['b_gamma_depth_2a'] = prams[idx]; idx += 1
    args['W_point_2a'] = prams[idx]; idx += 1
    args['b_point_2a'] = prams[idx]; idx += 1
    args['W_gamma_point_2a'] = prams[idx]; idx += 1
    args['b_gamma_point_2a'] = prams[idx]; idx += 1

    # Stage 2b (12 parameters)
    args['W_expan_2b'] = prams[idx]; idx += 1
    args['b_expan_2b'] = prams[idx]; idx += 1
    args['W_gamma_expan_2b'] = prams[idx]; idx += 1
    args['b_gamma_expan_2b'] = prams[idx]; idx += 1
    args['W_depth_2b'] = prams[idx]; idx += 1
    args['b_depth_2b'] = prams[idx]; idx += 1
    args['W_gamma_depth_2b'] = prams[idx]; idx += 1
    args['b_gamma_depth_2b'] = prams[idx]; idx += 1
    args['W_point_2b'] = prams[idx]; idx += 1
    args['b_point_2b'] = prams[idx]; idx += 1
    args['W_gamma_point_2b'] = prams[idx]; idx += 1
    args['b_gamma_point_2b'] = prams[idx]; idx += 1

    # Stage 2c (12 parameters)
    args['W_expan_2c'] = prams[idx]; idx += 1
    args['b_expan_2c'] = prams[idx]; idx += 1
    args['W_gamma_expan_2c'] = prams[idx]; idx += 1
    args['b_gamma_expan_2c'] = prams[idx]; idx += 1
    args['W_depth_2c'] = prams[idx]; idx += 1
    args['b_depth_2c'] = prams[idx]; idx += 1
    args['W_gamma_depth_2c'] = prams[idx]; idx += 1
    args['b_gamma_depth_2c'] = prams[idx]; idx += 1
    args['W_point_2c'] = prams[idx]; idx += 1
    args['b_point_2c'] = prams[idx]; idx += 1
    args['W_gamma_point_2c'] = prams[idx]; idx += 1
    args['b_gamma_point_2c'] = prams[idx]; idx += 1

    # Stage 3a (12 parameters)
    args['W_expan_3a'] = prams[idx]; idx += 1
    args['b_expan_3a'] = prams[idx]; idx += 1
    args['W_gamma_expan_3a'] = prams[idx]; idx += 1
    args['b_gamma_expan_3a'] = prams[idx]; idx += 1
    args['W_depth_3a'] = prams[idx]; idx += 1
    args['b_depth_3a'] = prams[idx]; idx += 1
    args['W_gamma_depth_3a'] = prams[idx]; idx += 1
    args['b_gamma_depth_3a'] = prams[idx]; idx += 1
    args['W_point_3a'] = prams[idx]; idx += 1
    args['b_point_3a'] = prams[idx]; idx += 1
    args['W_gamma_point_3a'] = prams[idx]; idx += 1
    args['b_gamma_point_3a'] = prams[idx]; idx += 1

    # Stage 3b (16 non-transformer + 12*2 transformer parameters)
    args['W_loacal_3x3_3b'] = prams[idx]; idx += 1
    args['b_local_3x3_3b'] = prams[idx]; idx += 1
    args['W_gamma_local_3x3_3b'] = prams[idx]; idx += 1
    args['W_beta_local_3x3_3b'] = prams[idx]; idx += 1
    args['W_loacal_1x1_3b'] = prams[idx]; idx += 1
    args['b_local_1x1_3b'] = prams[idx]; idx += 1
    # Transformer parameters (L=2)
    args['W_gamma1_3b'] = [prams[idx], prams[idx + 12]]; idx += 1
    args['b_beta1_3b'] = [prams[idx], prams[idx + 12]]; idx += 1
    args['W_QKV_atten_3b'] = [prams[idx], prams[idx + 12]]; idx += 1
    args['Bias_QKV_atten_3b'] = [prams[idx], prams[idx + 12]]; idx += 1
    args['W_o_atten_3b'] = [prams[idx], prams[idx + 12]]; idx += 1
    args['Bias_o_atten_3b'] = [prams[idx], prams[idx + 12]]; idx += 1
    args['W_gamma2_3b'] = [prams[idx], prams[idx + 12]]; idx += 1
    args['b_beta2_3b'] = [prams[idx], prams[idx + 12]]; idx += 1
    args['W1_fc1_3b'] = [prams[idx], prams[idx + 12]]; idx += 1
    args['b1_fc1_3b'] = [prams[idx], prams[idx + 12]]; idx += 1
    args['W2_fc2_3b'] = [prams[idx], prams[idx + 12]]; idx += 1
    args['b2_fc2_3b'] = [prams[idx], prams[idx + 12]]; idx += 1
    idx += 12  # Skip to the end of the second transformer's parameters
    # Fusion parameters
    args['W_fusion_1x1_3b'] = prams[idx]; idx += 1
    args['b_fusion_1x1_3b'] = prams[idx]; idx += 1
    args['W_gamma_f_1x1_3b'] = prams[idx]; idx += 1
    args['b_beta_f_1x1_3b'] = prams[idx]; idx += 1
    args['W_fusion_3x3_3b'] = prams[idx]; idx += 1
    args['b_fusion_3x3_3b'] = prams[idx]; idx += 1
    args['W_gamma_f_3x3_3b'] = prams[idx]; idx += 1
    args['b_beta_f_3x3_3b'] = prams[idx]; idx += 1

    # Stage 4a (12 parameters)
    args['W_expan_4a'] = prams[idx]; idx += 1
    args['b_expan_4a'] = prams[idx]; idx += 1
    args['W_gamma_expan_4a'] = prams[idx]; idx += 1
    args['b_gamma_expan_4a'] = prams[idx]; idx += 1
    args['W_depth_4a'] = prams[idx]; idx += 1
    args['b_depth_4a'] = prams[idx]; idx += 1
    args['W_gamma_depth_4a'] = prams[idx]; idx += 1
    args['b_gamma_depth_4a'] = prams[idx]; idx += 1
    args['W_point_4a'] = prams[idx]; idx += 1
    args['b_point_4a'] = prams[idx]; idx += 1
    args['W_gamma_point_4a'] = prams[idx]; idx += 1
    args['b_gamma_point_4a'] = prams[idx]; idx += 1

    # Stage 4b (16 non-transformer + 12*4 transformer parameters)
    args['W_loacal_3x3_4b'] = prams[idx]; idx += 1
    args['b_local_3x3_4b'] = prams[idx]; idx += 1
    args['W_gamma_local_3x3_4b'] = prams[idx]; idx += 1
    args['W_beta_local_3x3_4b'] = prams[idx]; idx += 1
    args['W_loacal_1x1_4b'] = prams[idx]; idx += 1
    args['b_local_1x1_4b'] = prams[idx]; idx += 1
    # Transformer parameters (L=4)
    args['W_gamma1_4b'] = [prams[idx], prams[idx + 12], prams[idx + 24], prams[idx + 36]]; idx += 1
    args['b_beta1_4b'] = [prams[idx], prams[idx + 12], prams[idx + 24], prams[idx + 36]]; idx += 1
    args['W_QKV_atten_4b'] = [prams[idx], prams[idx + 12], prams[idx + 24], prams[idx + 36]]; idx += 1
    args['Bias_QKV_atten_4b'] = [prams[idx], prams[idx + 12], prams[idx + 24], prams[idx + 36]]; idx += 1
    args['W_o_atten_4b'] = [prams[idx], prams[idx + 12], prams[idx + 24], prams[idx + 36]]; idx += 1
    args['Bias_o_atten_4b'] = [prams[idx], prams[idx + 12], prams[idx + 24], prams[idx + 36]]; idx += 1
    args['W_gamma2_4b'] = [prams[idx], prams[idx + 12], prams[idx + 24], prams[idx + 36]]; idx += 1
    args['b_beta2_4b'] = [prams[idx], prams[idx + 12], prams[idx + 24], prams[idx + 36]]; idx += 1
    args['W1_fc1_4b'] = [prams[idx], prams[idx + 12], prams[idx + 24], prams[idx + 36]]; idx += 1
    args['b1_fc1_4b'] = [prams[idx], prams[idx + 12], prams[idx + 24], prams[idx + 36]]; idx += 1
    args['W2_fc2_4b'] = [prams[idx], prams[idx + 12], prams[idx + 24], prams[idx + 36]]; idx += 1
    args['b2_fc2_4b'] = [prams[idx], prams[idx + 12], prams[idx + 24], prams[idx + 36]]; idx += 1
    idx += 36  # Skip to the end of the second transformer's parameters
    # Fusion parameters
    args['W_fusion_1x1_4b'] = prams[idx]; idx += 1
    args['b_fusion_1x1_4b'] = prams[idx]; idx += 1
    args['W_gamma_f_1x1_4b'] = prams[idx]; idx += 1
    args['b_beta_f_1x1_4b'] = prams[idx]; idx += 1
    args['W_fusion_3x3_4b'] = prams[idx]; idx += 1
    args['b_fusion_3x3_4b'] = prams[idx]; idx += 1
    args['W_gamma_f_3x3_4b'] = prams[idx]; idx += 1
    args['b_beta_f_3x3_4b'] = prams[idx]; idx += 1

    # Stage 5a (12 parameters)
    args['W_expan_5a'] = prams[idx]; idx += 1
    args['b_expan_5a'] = prams[idx]; idx += 1
    args['W_gamma_expan_5a'] = prams[idx]; idx += 1
    args['b_gamma_expan_5a'] = prams[idx]; idx += 1
    args['W_depth_5a'] = prams[idx]; idx += 1
    args['b_depth_5a'] = prams[idx]; idx += 1
    args['W_gamma_depth_5a'] = prams[idx]; idx += 1
    args['b_gamma_depth_5a'] = prams[idx]; idx += 1
    args['W_point_5a'] = prams[idx]; idx += 1
    args['b_point_5a'] = prams[idx]; idx += 1
    args['W_gamma_point_5a'] = prams[idx]; idx += 1
    args['b_gamma_point_5a'] = prams[idx]; idx += 1

    # Stage 5b (16 non-transformer + 12*3 transformer parameters)
    args['W_loacal_3x3_5b'] = prams[idx]; idx += 1
    args['b_local_3x3_5b'] = prams[idx]; idx += 1
    args['W_gamma_local_3x3_5b'] = prams[idx]; idx += 1
    args['W_beta_local_3x3_5b'] = prams[idx]; idx += 1
    args['W_loacal_1x1_5b'] = prams[idx]; idx += 1
    args['b_local_1x1_5b'] = prams[idx]; idx += 1
    # Transformer parameters (L=3)
    args['W_gamma1_5b'] = [prams[idx], prams[idx + 12], prams[idx + 24]]; idx += 1
    args['b_beta1_5b'] = [prams[idx], prams[idx + 12], prams[idx + 24]]; idx += 1
    args['W_QKV_atten_5b'] = [prams[idx], prams[idx + 12], prams[idx + 24]]; idx += 1
    args['Bias_QKV_atten_5b'] = [prams[idx], prams[idx + 12], prams[idx + 24]]; idx += 1
    args['W_o_atten_5b'] = [prams[idx], prams[idx + 12], prams[idx + 24]]; idx += 1
    args['Bias_o_atten_5b'] = [prams[idx], prams[idx + 12], prams[idx + 24]]; idx += 1
    args['W_gamma2_5b'] = [prams[idx], prams[idx + 12], prams[idx + 24]]; idx += 1
    args['b_beta2_5b'] = [prams[idx], prams[idx + 12], prams[idx + 24]]; idx += 1
    args['W1_fc1_5b'] = [prams[idx], prams[idx + 12], prams[idx + 24]]; idx += 1
    args['b1_fc1_5b'] = [prams[idx], prams[idx + 12], prams[idx + 24]]; idx += 1
    args['W2_fc2_5b'] = [prams[idx], prams[idx + 12], prams[idx + 24]]; idx += 1
    args['b2_fc2_5b'] = [prams[idx], prams[idx + 12], prams[idx + 24]]; idx += 1
    idx += 24  # Skip to the end of the second transformer's parameters
    # Fusion parameters
    args['W_fusion_1x1_5b'] = prams[idx]; idx += 1
    args['b_fusion_1x1_5b'] = prams[idx]; idx += 1
    args['W_gamma_f_1x1_5b'] = prams[idx]; idx += 1
    args['b_beta_f_1x1_5b'] = prams[idx]; idx += 1
    args['W_fusion_3x3_5b'] = prams[idx]; idx += 1
    args['b_fusion_3x3_5b'] = prams[idx]; idx += 1
    args['W_gamma_f_3x3_5b'] = prams[idx]; idx += 1
    args['b_beta_f_3x3_5b'] = prams[idx]; idx += 1

    # Head (4 parameters)
    args['W_head'] = prams[idx]; idx += 1
    args['b_head'] = prams[idx]; idx += 1
    args['W_gamma_head'] = prams[idx]; idx += 1
    args['b_beta_head'] = prams[idx]; idx += 1

    # Classifier (2 parameters)
    args['W_cls'] = prams[idx]; idx += 1
    args['b_cls'] = prams[idx]; idx += 1

    # Verify total parameters used
    print(f"Total parameters used: {idx}")
    if idx < len(prams):
        print(f"Remaining parameters: {prams[idx:]}")

    return args

# Get arguments
args = fill_mobilevit_parameters(prams)

# Call MobileViT_XXS
# Check Output Shape
# Call MobileViT_XXS
Inputs = np.random.rand(1, 3, 256, 256)  # Example input: batch_size=1, channels=3, 224x224
output = MobileViT_XXS_(Inputs,bn_prams ,**args)
print(output.shape)
