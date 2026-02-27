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

def load_apple_weights(my_model, apple_state_dict):
    """
    Corrected mapping that includes running_mean and running_var for ALL layers.
    """
    new_state_dict = {}

    # Helper to map weight, bias, running_mean, and running_var for BN
    def map_bn(my_prefix, apple_prefix):
        m = {}
        m[f'{my_prefix}.weight'] = apple_state_dict[f'{apple_prefix}.weight']
        m[f'{my_prefix}.bias']   = apple_state_dict[f'{apple_prefix}.bias']
        m[f'{my_prefix}.running_mean'] = apple_state_dict[f'{apple_prefix}.running_mean']
        m[f'{my_prefix}.running_var']  = apple_state_dict[f'{apple_prefix}.running_var']
        return m

    # 1. Stem
    new_state_dict['stem.0.weight'] = apple_state_dict['mobilevit.conv_stem.convolution.weight']
    new_state_dict.update(map_bn('stem.1', 'mobilevit.conv_stem.normalization'))

    # 2. MV2 Helper
    def map_mv2(my_prefix, apple_prefix):
        m = {}
        # Expand
        m[f'{my_prefix}.pw_expand.0.weight'] = apple_state_dict[f'{apple_prefix}.expand_1x1.convolution.weight']
        m.update(map_bn(f'{my_prefix}.pw_expand.1', f'{apple_prefix}.expand_1x1.normalization'))
        # Depthwise
        m[f'{my_prefix}.dw_conv.0.weight'] = apple_state_dict[f'{apple_prefix}.conv_3x3.convolution.weight']
        m.update(map_bn(f'{my_prefix}.dw_conv.1', f'{apple_prefix}.conv_3x3.normalization'))
        # Project
        m[f'{my_prefix}.pw_project.0.weight'] = apple_state_dict[f'{apple_prefix}.reduce_1x1.convolution.weight']
        m.update(map_bn(f'{my_prefix}.pw_project.1', f'{apple_prefix}.reduce_1x1.normalization'))
        return m

    # Stage 1 & 2
    new_state_dict.update(map_mv2('mv2_1', 'mobilevit.encoder.layer.0.layer.0'))
    new_state_dict.update(map_mv2('mv2_2a', 'mobilevit.encoder.layer.1.layer.0'))
    new_state_dict.update(map_mv2('mv2_2b', 'mobilevit.encoder.layer.1.layer.1'))
    new_state_dict.update(map_mv2('mv2_2c', 'mobilevit.encoder.layer.1.layer.2'))
    # Stage 3-5 Downsampling MV2s
    new_state_dict.update(map_mv2('mv2_3a', 'mobilevit.encoder.layer.2.downsampling_layer'))
    new_state_dict.update(map_mv2('mv2_4a', 'mobilevit.encoder.layer.3.downsampling_layer'))
    new_state_dict.update(map_mv2('mv2_5a', 'mobilevit.encoder.layer.4.downsampling_layer'))

    # 3. MobileViT Block
    def map_mvit_block(my_name, apple_idx):
        m = {}
        apple_base = f'mobilevit.encoder.layer.{apple_idx}'

        # Local Rep
        m[f'{my_name}.local_rep.conv1.weight'] = apple_state_dict[f'{apple_base}.conv_kxk.convolution.weight']
        m.update(map_bn(f'{my_name}.local_rep.bn1', f'{apple_base}.conv_kxk.normalization'))
        m[f'{my_name}.local_rep.conv2.weight'] = apple_state_dict[f'{apple_base}.conv_1x1.convolution.weight']

        # Block final Norm (LayerNorm - no running stats)
        m[f'{my_name}.norm.weight'] = apple_state_dict[f'{apple_base}.layernorm.weight']
        m[f'{my_name}.norm.bias']   = apple_state_dict[f'{apple_base}.layernorm.bias']

        # Fusion - FIXED: Now includes running_mean and running_var
        m[f'{my_name}.fusion.conv1.weight'] = apple_state_dict[f'{apple_base}.conv_projection.convolution.weight']
        m.update(map_bn(f'{my_name}.fusion.bn1', f'{apple_base}.conv_projection.normalization'))

        m[f'{my_name}.fusion.conv2.weight'] = apple_state_dict[f'{apple_base}.fusion.convolution.weight']
        m.update(map_bn(f'{my_name}.fusion.bn2', f'{apple_base}.fusion.normalization'))

        # Transformers
        num_layers = len(getattr(my_model, my_name).transformers)
        for i in range(num_layers):
            my_t = f'{my_name}.transformers.{i}'
            apple_t = f'{apple_base}.transformer.layer.{i}'

            m[f'{my_t}.norm1.weight'] = apple_state_dict[f'{apple_t}.layernorm_before.weight']
            m[f'{my_t}.norm1.bias']   = apple_state_dict[f'{apple_t}.layernorm_before.bias']
            m[f'{my_t}.norm2.weight'] = apple_state_dict[f'{apple_t}.layernorm_after.weight']
            m[f'{my_t}.norm2.bias']   = apple_state_dict[f'{apple_t}.layernorm_after.bias']

            m[f'{my_t}.fc1.weight'] = apple_state_dict[f'{apple_t}.intermediate.dense.weight']
            m[f'{my_t}.fc1.bias']   = apple_state_dict[f'{apple_t}.intermediate.dense.bias']
            m[f'{my_t}.fc2.weight'] = apple_state_dict[f'{apple_t}.output.dense.weight']
            m[f'{my_t}.fc2.bias']   = apple_state_dict[f'{apple_t}.output.dense.bias']

            # MultiheadAttention mapping
            attn_path = f'{apple_t}.attention'
            qw = apple_state_dict[f'{attn_path}.attention.query.weight']
            kw = apple_state_dict[f'{attn_path}.attention.key.weight']
            vw = apple_state_dict[f'{attn_path}.attention.value.weight']
            m[f'{my_t}.attn.in_proj_weight'] = torch.cat([qw, kw, vw], dim=0)

            qb = apple_state_dict[f'{attn_path}.attention.query.bias']
            kb = apple_state_dict[f'{attn_path}.attention.key.bias']
            vb = apple_state_dict[f'{attn_path}.attention.value.bias']
            m[f'{my_t}.attn.in_proj_bias'] = torch.cat([qb, kb, vb], dim=0)

            m[f'{my_t}.attn.out_proj.weight'] = apple_state_dict[f'{attn_path}.output.dense.weight']
            m[f'{my_t}.attn.out_proj.bias']   = apple_state_dict[f'{attn_path}.output.dense.bias']
        return m

    new_state_dict.update(map_mvit_block('mvit_3b', 2))
    new_state_dict.update(map_mvit_block('mvit_4b', 3))
    new_state_dict.update(map_mvit_block('mvit_5b', 4))

    # 4. Final Head - FIXED: Now includes running_mean and running_var
    new_state_dict['head_conv.0.weight'] = apple_state_dict['mobilevit.conv_1x1_exp.convolution.weight']
    new_state_dict.update(map_bn('head_conv.1', 'mobilevit.conv_1x1_exp.normalization'))

    new_state_dict['classifier.weight'] = apple_state_dict['classifier.weight']
    new_state_dict['classifier.bias']   = apple_state_dict['classifier.bias']

    # Load
    my_model.load_state_dict(new_state_dict, strict=False)
    my_model.eval() # CRITICAL for BatchNorm inference
    print("✅ Successfully mapped weights and enabled eval mode.")