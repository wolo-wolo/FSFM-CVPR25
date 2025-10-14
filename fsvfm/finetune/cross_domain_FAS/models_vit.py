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

import timm.models.vision_transformer
from timm.models.vision_transformer import default_cfgs
from timm.models.helpers import load_pretrained


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        # if self.global_pool:
        #     x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        #     outcome = self.fc_norm(x)
        if self.global_pool:
            x_gp = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            x_gp = self.fc_norm(x_gp)
            # x[:, 0, :] = x_gp
            # outcome = x
            x_new = torch.zeros_like(x)
            x_new[:, 0, :] = x_gp
            x_new[:, 1:, :] = x[:, 1:, :]
            outcome = x_new
        else:
            x = self.norm(x)
            # outcome = x[:, 0]
            outcome = x  # for fas code

        return outcome

    # def reset_classifier(self, num_classes, global_pool=''):
    #     self.num_classes = num_classes
    #     self.head = nn.ModuleList([
    #         nn.Linear(self.embed_dim, 512),
    #         nn.Linear(512, num_classes) if num_classes > 0 else nn.Identity()
    #     ])
    #
    # def forward(self, x):
    #     x = self.forward_features(x)
    #     x = self.head(x)
    #     return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


def vit_small_patch16(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)  # ViT-small config in MOCO_V3
    # model = VisionTransformer(
    #     patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, qkv_bias=True,
    #     norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)  # ViT-small config in timm
    model.default_cfg = default_cfgs['vit_small_patch16_224']
    # if pretrained:
    #     load_pretrained(
    #         model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
        # for timm version 0.6.12:
        # pretrained_cfg = resolve_pretrained_cfg('vit_base_patch16_224',
        #                                         pretrained_cfg=kwargs.pop('pretrained_cfg', None))
        # load_pretrained(
        #     model, pretrained_cfg=pretrained_cfg,
        #     num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


def vit_base_patch16(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
        # for timm version 0.6.12:
        # pretrained_cfg = resolve_pretrained_cfg('vit_base_patch16_224',
        #                                         pretrained_cfg=kwargs.pop('pretrained_cfg', None))
        # load_pretrained(
        #     model, pretrained_cfg=pretrained_cfg,
        #     num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


def vit_large_patch16(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch16_224']
    if pretrained:
         load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
        # for timm version 0.6.12:
        # pretrained_cfg = resolve_pretrained_cfg('vit_large_patch16_224',
        #                                         pretrained_cfg=kwargs.pop('pretrained_cfg', None))
        # load_pretrained(
        #     model, pretrained_cfg=pretrained_cfg,
        #     num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


def vit_huge_patch14(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model