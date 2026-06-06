# -*- coding: utf-8 -*-
# Linear probing model for FAS: frozen ViT encoder + one linear classifier.

import os
import sys
from functools import partial

import torch
import torch.nn as nn
import timm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import models_vit
from util.pos_embed import interpolate_pos_embed


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ['model', 'state_dict', 'student', 'teacher']:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _normalize_checkpoint_keys(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ['module.', 'model.', 'backbone.', 'encoder.', 'target_encoder.', 'student.', 'teacher.']:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        new_state[nk] = v
    return new_state


def load_pretrained_vit(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_model = _normalize_checkpoint_keys(_extract_state_dict(checkpoint))
    state_dict = model.state_dict()
    for k in list(checkpoint_model.keys()):
        if k.startswith('head.') or k.startswith('fc_norm.'):
            if k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
                del checkpoint_model[k]
        if k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
            print(f'Removing shape-mismatched key {k}: {checkpoint_model[k].shape} vs {state_dict[k].shape}')
            del checkpoint_model[k]
    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print('Load pre-trained checkpoint from:', ckpt_path)
    print(msg)
    return msg


class FASLinearProbe(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.vit = models_vit.__dict__[args.model](
            num_classes=2,
            global_pool=True,
            drop_path_rate=args.drop_path,
        )
        if args.pt_model is not None:
            load_pretrained_vit(self.vit, args.pt_model)
        else:
            print('[WARN] --pt_model is None; using random/timm initialization for LP baseline.')

        dim_map = {
            'vit_small_patch16': 384,
            'vit_base_patch16': 768,
            'vit_large_patch16': 1024,
            'vit_huge_patch16': 1280,
        }
        dim = dim_map[args.model]
        self.vit.head = nn.Identity()
        self.classifier = nn.Linear(dim, 2)
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.constant_(self.classifier.bias, 0.)

        # Pure linear probe: freeze encoder completely, train only classifier.
        for p in self.vit.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True

    def forward(self, input, norm_flag=True):
        feature = self.vit.forward_features(input)
        logits = self.classifier(feature).unsqueeze(1)  # [B, 1, 2], compatible with FAS evaluate.py
        return logits, feature
