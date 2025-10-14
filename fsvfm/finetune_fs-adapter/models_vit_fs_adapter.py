# -*- coding: utf-8 -*-
# Author: Gaojian Wang@ZJUICSR
# --------------------------------------------------------
# This source code is licensed under the Attribution-NonCommercial 4.0 International License.
# You can find the license in the LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm.models.vision_transformer


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.linear1 = nn.Linear(c_in, c_in // reduction, bias=False)
        self.activation1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(c_in // reduction, c_in, bias=False)
        self.activation2 = nn.ReLU(inplace=True)

        # self.fc = nn.Sequential(
        #     nn.Linear(c_in, c_in // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(c_in // reduction, c_in, bias=False),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        x = self.linear1(x)
        x_bottleneck = self.activation1(x)

        x = self.linear2(x_bottleneck)
        x = self.activation2(x)

        return x, x_bottleneck


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, adapter_reduction=4, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        self.adapter = Adapter(kwargs['embed_dim'], adapter_reduction)
        self.projector = nn.Linear(kwargs['embed_dim'] // adapter_reduction,
                                   (kwargs['embed_dim'] // adapter_reduction))

        self.head = nn.Linear(kwargs['embed_dim']*2, kwargs['num_classes']) if kwargs['num_classes'] > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x, training=False):
        x = self.forward_features(x)

        x_adapter, x_bottleneck = self.adapter(x)
        proj_features = self.projector(x_bottleneck)

        x = torch.concatenate([x, x_adapter], dim=-1)
        x = self.head(x)

        if training:
            # Return logits for cls loss, projected features for contrastive loss
            return x, F.normalize(proj_features, dim=-1)
        return x


def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,  # ViT-small config in MOCO_V3
        # patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, qkv_bias=True,  # ViT-small config in timm
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
