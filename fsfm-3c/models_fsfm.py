# -*- coding: utf-8 -*-
# Author: Gaojian Wang@ZJUICSR
# --------------------------------------------------------
# This source code is licensed under the Attribution-NonCommercial 4.0 International License.
# You can find the license in the LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# BYOL: https://github.com/lucidrains/byol-pytorch
# SimSIam: https://github.com/facebookresearch/simsiam
# --------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


def default(val, def_val):
    return def_val if val is None else val


def MaybeSyncBatchnorm(is_distributed=None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    print(is_distributed)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d


class Projector(nn.Module):
    """SimCLR proj"""

    def __init__(self, embed_dim, cl_feat_dim=128):
        super().__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, cl_feat_dim)
        )

    def forward(self, x):
        features_cl = self.projection_head(x)  # [N, cl_feat_dim]
        return features_cl


class BYOLMLP(nn.Module):
    """BYOL proj/pred"""
    def __init__(self, dim, projection_size, hidden_size=4096, sync_batchnorm=None):
        super().__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.LayerNorm(hidden_size),
            # MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.projection_head(x)  # [N, cl_feat_dim]
        return x


class SimSiamMLP(nn.Module):
    """SimSiam proj/pred"""
    def __init__(self, dim, projection_size, hidden_size=4096, sync_batchnorm=None):
        super().__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(dim, hidden_size, bias=False),
            MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size, bias=False),
            MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size, bias=False),
            MaybeSyncBatchnorm(sync_batchnorm)(projection_size),
        )

    def forward(self, x):
        x = self.projection_head(x)  # [N, cl_feat_dim]
        return x


class FSFMViT(nn.Module):
    """ FSFM with VisionTransformer backbone"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 projection_size=256, projection_hidden_size=4096,
                 rep_decoder_embed_dim=768, rep_decoder_depth=2, rep_decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches_axis = img_size // patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # contrastive learning projection&prediction head
        # self.projector = Projector(embed_dim=rep_decoder_embed_dim)
        # self.projector = SimSiamMLP(dim=rep_decoder_embed_dim, projection_size=256, hidden_size=4096)
        self.projector = BYOLMLP(rep_decoder_embed_dim, projection_size, projection_hidden_size)

        self.predictor = BYOLMLP(projection_size, projection_size, projection_hidden_size)

        # --------------------------------------------------------------------------
        # FSFM Rep decoder specifics
        self.mask_token = nn.Parameter(torch.zeros(1, 1, rep_decoder_embed_dim))
        self.rep_decoder_embed = nn.Linear(embed_dim, rep_decoder_embed_dim, bias=True)
        self.rep_decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, rep_decoder_embed_dim),
                                                  requires_grad=False)  # fixed sin-cos embedding
        self.rep_decoder_blocks = nn.ModuleList([
            Block(rep_decoder_embed_dim, rep_decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                  norm_layer=norm_layer)
            for i in range(rep_decoder_depth)])
        self.rep_decoder_norm = norm_layer(rep_decoder_embed_dim)
        self.rep_decoder_pred = nn.Linear(rep_decoder_embed_dim, embed_dim, bias=True)

        # --------------------------------------------------------------------------
        # MAE pixel decoder specifics
        self.decoder_embed = nn.Linear(rep_decoder_embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_encoder(self, x, x_mask, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        N, L, D = x.shape  # batch, length, dim

        # read the binary mask: 0 is kept, 1 is removed
        ids_shuffle = torch.argsort(x_mask, dim=1)  # torch.Size([N, L]), ascend: 0 is keep, 1 is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # torch.Size([N, L])

        # keep the 0 subset
        ids_keep = ids_shuffle[:, :int(L * (1 - mask_ratio))]
        # ids_keep = ids_shuffle[:, :int(L - x_mask.sum(dim=-1).mean())]  # same as above
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, ids_restore

    def forward_rep_decoder(self, x, ids_restore):
        # embed tokens
        x = self.rep_decoder_embed(x)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # add pos embed
        x = x + self.rep_decoder_pos_embed

        # apply Transformer blocks
        for blk in self.rep_decoder_blocks:
            x = blk(x)
        x = self.rep_decoder_norm(x)

        # predictor projection
        x = self.rep_decoder_pred(x)

        # # remove cls token
        # x = x[:, 1:, :]

        return x

    def forward_decoder(self, x, ids_restore):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # embed tokens
        x = self.decoder_embed(x)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, x_mask, specific_facial_region_mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        # mean loss on all removed patches
        loss_rec_all = (loss * x_mask).sum() / x_mask.sum()
        # mean loss on randomly select facial region that patches are fully masked
        loss_rec_sfr = (loss * specific_facial_region_mask).sum() / specific_facial_region_mask.sum()

        return loss_rec_all, loss_rec_sfr

    def forward(self, imgs, imgs_masks, specific_facial_region_mask, mask_ratio=0.75):
        latent, ids_restore = self.forward_encoder(imgs, imgs_masks, mask_ratio)

        feat_all = self.forward_rep_decoder(latent, ids_restore)
        features_proj = self.projector(feat_all.mean(dim=1, keepdim=False))
        features_pred = self.predictor(features_proj)
        features_cl = F.normalize(features_pred, dim=-1)  # [N, cl_feat_dim]

        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss_rec_all, loss_rec_sfr = self.forward_loss(imgs, pred, imgs_masks, specific_facial_region_mask)

        return loss_rec_all, loss_rec_sfr, features_cl, pred, imgs_masks


def fsfm_vit_small_patch16_dec512d8b(**kwargs):
    model = FSFMViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        rep_decoder_embed_dim=384, rep_decoder_depth=2, rep_decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def fsfm_vit_base_patch16_dec512d8b(**kwargs):
    model = FSFMViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        rep_decoder_embed_dim=768, rep_decoder_depth=2, rep_decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def fsfm_vit_large_patch16_dec512d8b(**kwargs):
    model = FSFMViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        rep_decoder_embed_dim=1024, rep_decoder_depth=2, rep_decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def fsfm_vit_huge_patch14_dec512d8b(**kwargs):
    model = FSFMViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        rep_decoder_embed_dim=1280, rep_decoder_depth=2, rep_decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
fsfm_vit_small_patch16 = fsfm_vit_small_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
fsfm_vit_base_patch16 = fsfm_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
fsfm_vit_large_patch16 = fsfm_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
fsfm_vit_huge_patch14 = fsfm_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks


class TargetNetworkViT(nn.Module):
    """
    Target networks in FSFM: Target Encoder; Target Rep Decoder; Target projector;
    share same structure with online counterparts
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 projection_size=256, projection_hidden_size=4096,
                 rep_decoder_embed_dim=768, rep_decoder_depth=2, rep_decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches_axis = img_size // patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # contrastive learning projection&prediction head
        # self.projector = Projector(embed_dim=rep_decoder_embed_dim)
        # self.projector = SimSiamMLP(dim=rep_decoder_embed_dim, projection_size=256, hidden_size=4096)
        self.projector = BYOLMLP(rep_decoder_embed_dim, projection_size, projection_hidden_size)
        # self.predictor = BYOLMLP(projection_size, projection_size, projection_hidden_size)

        # --------------------------------------------------------------------------
        # FSFM Rep decoder specifics
        self.mask_token = nn.Parameter(torch.zeros(1, 1, rep_decoder_embed_dim))
        self.rep_decoder_embed = nn.Linear(embed_dim, rep_decoder_embed_dim, bias=True)
        self.rep_decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, rep_decoder_embed_dim),
                                                  requires_grad=False)  # fixed sin-cos embedding
        self.rep_decoder_blocks = nn.ModuleList([
            Block(rep_decoder_embed_dim, rep_decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None,
                  norm_layer=norm_layer)
            for i in range(rep_decoder_depth)])
        self.rep_decoder_norm = norm_layer(rep_decoder_embed_dim)
        self.rep_decoder_pred = nn.Linear(rep_decoder_embed_dim, embed_dim, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        rep_decoder_pos_embed = get_2d_sincos_pos_embed(self.rep_decoder_pos_embed.shape[-1],
                                                        int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.rep_decoder_pos_embed.data.copy_(torch.from_numpy(rep_decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_encoder(self, x, x_mask, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        N, L, D = x.shape  # batch, length, dim

        # get the binary mask: 0 is kept, 1 is removed
        ids_shuffle = torch.argsort(x_mask, dim=1)  # torch.Size([N, L]), ascend: 0 is keep, 1 is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # torch.Size([N, L])

        # keep the 0 subset
        ids_keep = ids_shuffle[:, :int(L * (1 - mask_ratio))]
        # ids_keep = ids_shuffle[:, :int(L - x_mask.sum(dim=-1).mean())]  # same
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, ids_restore

    def forward_rep_decoder(self, x, ids_restore):
        # embed tokens
        x = self.rep_decoder_embed(x)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # add pos embed
        x = x + self.rep_decoder_pos_embed

        # apply Transformer blocks
        for blk in self.rep_decoder_blocks:
            x = blk(x)
        x = self.rep_decoder_norm(x)

        # predictor projection
        x = self.rep_decoder_pred(x)

        # # remove cls token
        # x = x[:, 1:, :]

        return x

    def forward(self, imgs, imgs_masks, specific_facial_region_mask, mask_ratio=0.75):
        latent, ids_restore = self.forward_encoder(imgs, imgs_masks, mask_ratio)

        feat_all = self.forward_rep_decoder(latent, ids_restore)
        features_proj = self.projector(feat_all.mean(dim=1, keepdim=False))
        features_cl = F.normalize(features_proj, dim=-1)  # [N, cl_feat_dim]

        return features_cl


def vit_target_network(model):
    if model == 'fsfm_vit_small_patch16':
        return TargetNetworkViT(
            patch_size=16, embed_dim=384, depth=12, num_heads=12,
            rep_decoder_embed_dim=384, rep_decoder_depth=2, rep_decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
    
    if model == 'fsfm_vit_base_patch16':
        return TargetNetworkViT(
            patch_size=16, embed_dim=768, depth=12, num_heads=12,
            rep_decoder_embed_dim=768, rep_decoder_depth=2, rep_decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))

    if model == 'fsfm_vit_large_patch16':
        return TargetNetworkViT(
            patch_size=16, embed_dim=1024, depth=24, num_heads=16,
            rep_decoder_embed_dim=1024, rep_decoder_depth=2, rep_decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))

    if model == 'fsfm_vit_huge_patch14':
        return TargetNetworkViT(
            patch_size=14, embed_dim=1280, depth=32, num_heads=16,
            rep_decoder_embed_dim=1280, rep_decoder_depth=2, rep_decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
