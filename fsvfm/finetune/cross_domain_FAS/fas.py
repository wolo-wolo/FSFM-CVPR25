import os
import sys
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../../')
import timm
from collections import OrderedDict

import models_vit
from utils.pos_embed import interpolate_pos_embed


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class feature_generator(nn.Module):

    def __init__(self, args):
        super(feature_generator, self).__init__()
        global_pool = True

        if args.pt_model is None:
            # self.vit = timm.create_model(args.model+'_224', pretrained=True)
            if not args.scratch:
                print("loading ImageNet pretrained weights....")
            self.vit = models_vit.__dict__[args.model](
                pretrained=False if args.scratch else True,
                num_classes=2,
                global_pool=global_pool,
                drop_path_rate=args.drop_path,
            )

        else:
            model = models_vit.__dict__[args.model](
                num_classes=2,
                global_pool=global_pool,
                drop_path_rate=args.drop_path,
            )

            checkpoint = torch.load(args.pt_model, map_location='cpu')
            print("Load pre-trained checkpoint from: %s" % args.pt_model)
            if 'model' in checkpoint:  # for MAE(CVPR22) and Ours models
                checkpoint_model = checkpoint['model']
            else:  # for DINO(ICCV21) model
                checkpoint_model = checkpoint
            state_dict = model.state_dict()

            convert = any('target_encoder.' in key for key, value in checkpoint_model.items())  # for MCF(ACMMM22) model
            if convert:
                checkpoint_model = {
                    key.replace('target_encoder.', ''): value
                    for key, value in checkpoint_model.items()
                }

            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)
            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg[0])
            if global_pool:
                assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
            else:
                assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
            # manually initialize fc layer
            timm.models.layers.trunc_normal_(model.head.weight, std=2e-5)
            self.vit = model

        self.vit.head = nn.Identity()  # remove the classification head for timm version 0.6.12

    def forward(self, input):
        # feat = self.vit.forward_features(input).detach()
        feat = self.vit.forward_features(input)  # for timm version 0.4.9
        # feat = self.vit.forward_features(input) # for timm version 0.4.9
        # feat = self.vit.forward(input) # for timm version 0.6.12
        # print(f'feat : {feat}')
        return feat


# feature embedders
class feature_embedder(nn.Module):

    def __init__(self, args):
        super(feature_embedder, self).__init__()

        model_dim_map = {
            'vit_small_patch16': 384,  # for ViT-S
            'vit_base_patch16': 768,  # for ViT-B
            'vit_large_patch16': 1024,  # for ViT-L
            'vit_huge_patch16': 1280  # for ViT-H
        }
        input_dim = model_dim_map.get(args.model)
        if input_dim is None:
            raise ValueError(f"Unsupported model: {args.model}")
        self.bottleneck_layer_fc = nn.Linear(input_dim, 512)

        self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
        self.bottleneck_layer_fc.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(self.bottleneck_layer_fc, nn.ReLU(),
                                              nn.Dropout(0.5))

    def forward(self, input, norm_flag=True):
        feature = self.bottleneck_layer(input)
        if (norm_flag):
            feature_norm = torch.norm(
                feature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
            feature = torch.div(feature, feature_norm)
        return feature


# classifier
class classifier(nn.Module):

    def __init__(self):
        super(classifier, self).__init__()
        self.classifier_layer = nn.Linear(512, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, input, norm_flag=True):
        if (norm_flag):
            self.classifier_layer.weight.data = l2_norm(
                self.classifier_layer.weight, axis=0)
            classifier_out = self.classifier_layer(input)
        else:
            classifier_out = self.classifier_layer(input)
        return classifier_out


# base vit
class fas_model_fix(nn.Module):

    def __init__(self, args):
        super(fas_model_fix, self).__init__()
        self.backbone = feature_generator(args)
        self.embedder = feature_embedder(args)
        self.classifier = classifier()

    def forward(self, input, norm_flag=True):
        feature = self.backbone(input)
        feature = self.embedder(feature, norm_flag)
        classifier_out = self.classifier(feature, norm_flag)
        return classifier_out, feature

