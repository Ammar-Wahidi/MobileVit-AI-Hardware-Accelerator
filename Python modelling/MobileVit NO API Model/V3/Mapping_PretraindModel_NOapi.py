"""
extract_weights_MVT_to_numpy.py
================================
Extracts all weights from a trained/initialized MVT.py (PyTorch) MobileViT_XXS
and returns them ready to pass directly into mobile_vit_numpy_fixed.py's
MobileViT_XXS_().

All five mismatches between the original acc2 and MVT.py are now fixed in
mobile_vit_numpy_fixed.py, so this mapping is clean and 1-to-1.

Usage
-----
    from MVT import MobileViT_XXS as PT_Model
    from extract_weights_MVT_to_numpy import extract_all_weights
    from mobile_vit_numpy_fixed import MobileViT_XXS_

    model = PT_Model()
    # model.load_state_dict(torch.load("checkpoint.pth"))
    model.eval()

    weights, bn_prams = extract_all_weights(model)

    import numpy as np
    x_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
    output = MobileViT_XXS_(x_np, bn_prams, **weights)

Weight-shape notes
------------------
Conv weights  : PyTorch (out, in, f, f)  = numpy expects same shape  ✓
Depthwise     : PyTorch (C, 1, f, f)     → squeezed to (C, f, f)     ✓
bias=False    : zero arrays provided in correct shape                  ✓
BN gamma/beta : extracted from .weight / .bias                        ✓
BN running    : extracted from .running_mean / .running_var           ✓
QKV weight    : PyTorch in_proj_weight (3d, d) — numpy expects same   ✓
MLP weights   : PyTorch (out, in); numpy does W.T internally          ✓
LayerNorm     : weight (d,) / bias (d,) — directly usable            ✓
Classifier    : (num_classes, feat); numpy does W_cls.T internally    ✓

bn_prams layout (64 entries total)
-----------------------------------
 [0,  1]   stem          running_mean, running_var
 [2 .. 7]  mv2_1         expand, depthwise, point  (each: mean, var)
 [8 ..13]  mv2_2a
 [14..19]  mv2_2b
 [20..25]  mv2_2c
 [26..31]  mv2_3a
 [32..37]  mvit_3b       local-3x3, fusion-1x1, fusion-3x3
 [38..43]  mv2_4a
 [44..49]  mvit_4b       local-3x3, fusion-1x1, fusion-3x3
 [50..55]  mv2_5a
 [56..61]  mvit_5b       local-3x3, fusion-1x1, fusion-3x3
 [62, 63]  head conv
"""

import numpy as np


# ─────────────────────────────────────────────────────────────
# Tensor conversion helpers
# ─────────────────────────────────────────────────────────────

def t(tensor):
    """PyTorch tensor → numpy float32."""
    return tensor.detach().cpu().numpy().astype(np.float32)


def zero_conv_bias(conv):
    """Zero bias shaped (out_channels, 1, 1, 1) for bias=False conv."""
    return np.zeros((conv.weight.shape[0], 1, 1, 1), dtype=np.float32)


def dw_weight(conv):
    """Squeeze depthwise weight from (C, 1, f, f) → (C, f, f)."""
    return t(conv.weight).squeeze(1)


def zero_dw_bias(conv):
    """Zero bias shaped (C,) for bias=False depthwise conv."""
    return np.zeros(conv.weight.shape[0], dtype=np.float32)


# ─────────────────────────────────────────────────────────────
# MV2 block extractor
# ─────────────────────────────────────────────────────────────

def extract_mv2(block, suffix):
    """
    Extract weights + BN running stats for one MV2 block.

    MVT.py MV2 structure:
        pw_expand  : [Conv2d(bias=False), BN, SiLU]
        dw_conv    : [Conv2d(bias=False, groups=C), BN, SiLU]
        pw_project : [Conv2d(bias=False), BN]          <- no activation

    Returns
    -------
    weights  : dict  (12 entries)
    bn_stats : list  [mean1, var1, mean2, var2, mean3, var3]
    """
    conv_e, bn_e = block.pw_expand[0],  block.pw_expand[1]
    conv_d, bn_d = block.dw_conv[0],    block.dw_conv[1]
    conv_p, bn_p = block.pw_project[0], block.pw_project[1]

    weights = {
        f'W_expan_{suffix}':        t(conv_e.weight),
        f'b_expan_{suffix}':        zero_conv_bias(conv_e),
        f'W_gamma_expan_{suffix}':  t(bn_e.weight),
        f'b_gamma_expan_{suffix}':  t(bn_e.bias),

        f'W_depth_{suffix}':        dw_weight(conv_d),
        f'b_depth_{suffix}':        zero_dw_bias(conv_d),
        f'W_gamma_depth_{suffix}':  t(bn_d.weight),
        f'b_gamma_depth_{suffix}':  t(bn_d.bias),

        f'W_point_{suffix}':        t(conv_p.weight),
        f'b_point_{suffix}':        zero_conv_bias(conv_p),
        f'W_gamma_point_{suffix}':  t(bn_p.weight),
        f'b_gamma_point_{suffix}':  t(bn_p.bias),
    }

    bn_stats = [
        t(bn_e.running_mean), t(bn_e.running_var),
        t(bn_d.running_mean), t(bn_d.running_var),
        t(bn_p.running_mean), t(bn_p.running_var),
    ]

    return weights, bn_stats


# ─────────────────────────────────────────────────────────────
# MobileViTBlock extractor
# ─────────────────────────────────────────────────────────────

def extract_mvit(block, suffix):
    """
    Extract weights + BN running stats for one MobileViTBlock.

    MVT.py MobileViTBlock structure:
        local_rep   : LocalRepresentation
            conv1 (3x3, bias=False) → BN → SiLU
            conv2 (1x1, bias=False) — no BN, no activation
        transformers: ModuleList[TransformerEncoder]
            norm1 (LayerNorm), attn (MHA), norm2 (LayerNorm), fc1, fc2
        norm        : LayerNorm   <- final norm before fold
        fusion      : Fusion
            conv1 (1x1, bias=False) → BN → SiLU
            conv2 (3x3, bias=False) → BN → SiLU

    bn_stats (6 entries per block):
        local-3x3 BN  : mean, var
        fusion-1x1 BN : mean, var
        fusion-3x3 BN : mean, var

    Returns
    -------
    weights  : dict
    bn_stats : list  (6 entries)
    """
    lr  = block.local_rep
    fus = block.fusion

    # Local representation
    conv3 = lr.conv1;  bn3 = lr.bn1
    conv1 = lr.conv2   # bias=False, no BN

    weights = {
        f'W_loacal_3x3_{suffix}':      t(conv3.weight),
        f'b_local_3x3_{suffix}':       zero_conv_bias(conv3),
        f'W_gamma_local_3x3_{suffix}': t(bn3.weight),
        f'W_beta_local_3x3_{suffix}':  t(bn3.bias),
        f'W_loacal_1x1_{suffix}':      t(conv1.weight),
        f'b_local_1x1_{suffix}':       zero_conv_bias(conv1),
    }

    bn_stats = [t(bn3.running_mean), t(bn3.running_var)]  # local-3x3

    # Transformer layers
    lists = {k: [] for k in [
        'W_QKV_atten', 'Bias_QKV_atten',
        'W_o_atten',   'Bias_o_atten',
        'W_gamma1',    'b_beta1',
        'W_gamma2',    'b_beta2',
        'W1_fc1',      'b1_fc1',
        'W2_fc2',      'b2_fc2',
    ]}

    for layer in block.transformers:
        # PyTorch MHA: in_proj_weight (3d, d) — numpy uses X @ W.T, so pass as-is
        lists['W_QKV_atten'].append(t(layer.attn.in_proj_weight))   # (3d, d)
        lists['Bias_QKV_atten'].append(t(layer.attn.in_proj_bias))  # (3d,)
        # out_proj weight (d, d) — numpy does context @ W_o.T, pass as-is
        lists['W_o_atten'].append(t(layer.attn.out_proj.weight))    # (d, d)
        lists['Bias_o_atten'].append(t(layer.attn.out_proj.bias))   # (d,)

        lists['W_gamma1'].append(t(layer.norm1.weight))
        lists['b_beta1'].append(t(layer.norm1.bias))
        lists['W_gamma2'].append(t(layer.norm2.weight))
        lists['b_beta2'].append(t(layer.norm2.bias))

        # MLP: numpy does X @ W.T + b, so pass weight as (out, in) as-is
        lists['W1_fc1'].append(t(layer.fc1.weight))   # (mlp_dim, dim)
        lists['b1_fc1'].append(t(layer.fc1.bias))
        lists['W2_fc2'].append(t(layer.fc2.weight))   # (dim, mlp_dim)
        lists['b2_fc2'].append(t(layer.fc2.bias))

    for key, val in lists.items():
        weights[f'{key}_{suffix}'] = val

    # Final LayerNorm (MVT.py self.norm)
    weights[f'W_norm_gamma_{suffix}'] = t(block.norm.weight)  # (dim,)
    weights[f'b_norm_beta_{suffix}']  = t(block.norm.bias)    # (dim,)

    # Fusion
    fc1, fbn1 = fus.conv1, fus.bn1
    fc2, fbn2 = fus.conv2, fus.bn2

    weights[f'W_fusion_1x1_{suffix}']   = t(fc1.weight)
    weights[f'b_fusion_1x1_{suffix}']   = zero_conv_bias(fc1)
    weights[f'W_gamma_f_1x1_{suffix}']  = t(fbn1.weight)
    weights[f'b_beta_f_1x1_{suffix}']   = t(fbn1.bias)

    weights[f'W_fusion_3x3_{suffix}']   = t(fc2.weight)
    weights[f'b_fusion_3x3_{suffix}']   = zero_conv_bias(fc2)
    weights[f'W_gamma_f_3x3_{suffix}']  = t(fbn2.weight)
    weights[f'b_beta_f_3x3_{suffix}']   = t(fbn2.bias)

    bn_stats += [
        t(fbn1.running_mean), t(fbn1.running_var),   # fusion-1x1
        t(fbn2.running_mean), t(fbn2.running_var),   # fusion-3x3
    ]

    return weights, bn_stats  # bn_stats has exactly 6 entries


# ─────────────────────────────────────────────────────────────
# Main extraction function
# ─────────────────────────────────────────────────────────────

def extract_all_weights(model):
    """
    Extract every parameter from a MVT.py MobileViT_XXS PyTorch model.

    Parameters
    ----------
    model : MobileViT_XXS (from MVT.py), in eval() mode

    Returns
    -------
    weights  : dict   — unpack with **weights into MobileViT_XXS_()
    bn_prams : list   — 64 numpy arrays (running mean/var pairs), passed as
                        the second positional argument to MobileViT_XXS_()
    """
    model.eval()
    weights  = {}
    bn_prams = []

    # Stem
    conv_s, bn_s = model.stem[0], model.stem[1]
    weights['W_stem']       = t(conv_s.weight)
    weights['b_stem']       = zero_conv_bias(conv_s)
    weights['W_stem_gamma'] = t(bn_s.weight)
    weights['b_beta_stem']  = t(bn_s.bias)
    bn_prams += [t(bn_s.running_mean), t(bn_s.running_var)]   # [0, 1]

    # Stage 1
    w, bn = extract_mv2(model.mv2_1, '1')
    weights.update(w); bn_prams += bn                          # [2..7]

    # Stage 2
    for blk, suf in [(model.mv2_2a, '2a'),
                     (model.mv2_2b, '2b'),
                     (model.mv2_2c, '2c')]:
        w, bn = extract_mv2(blk, suf)
        weights.update(w); bn_prams += bn                      # [8..25]

    # Stage 3
    w, bn = extract_mv2(model.mv2_3a, '3a')
    weights.update(w); bn_prams += bn                          # [26..31]

    w, bn = extract_mvit(model.mvit_3b, '3b')
    weights.update(w); bn_prams += bn                          # [32..37]

    # Stage 4
    w, bn = extract_mv2(model.mv2_4a, '4a')
    weights.update(w); bn_prams += bn                          # [38..43]

    w, bn = extract_mvit(model.mvit_4b, '4b')
    weights.update(w); bn_prams += bn                          # [44..49]

    # Stage 5
    w, bn = extract_mv2(model.mv2_5a, '5a')
    weights.update(w); bn_prams += bn                          # [50..55]

    w, bn = extract_mvit(model.mvit_5b, '5b')
    weights.update(w); bn_prams += bn                          # [56..61]

    # Head conv
    conv_h, bn_h = model.head_conv[0], model.head_conv[1]
    weights['W_head']       = t(conv_h.weight)
    weights['b_head']       = zero_conv_bias(conv_h)
    weights['W_gamma_head'] = t(bn_h.weight)
    weights['b_beta_head']  = t(bn_h.bias)
    bn_prams += [t(bn_h.running_mean), t(bn_h.running_var)]   # [62, 63]

    # Classifier
    # numpy does:  W_cls = W_cls.T  then  x @ W_cls
    # so pass the original (num_classes, feat) weight; the transpose happens inside.
    weights['W_cls'] = t(model.classifier.weight)   # (num_classes, 320)
    weights['b_cls'] = t(model.classifier.bias)     # (num_classes,)

    assert len(bn_prams) == 64, \
        f"Expected 64 bn_prams entries, got {len(bn_prams)}"

    return weights, bn_prams


