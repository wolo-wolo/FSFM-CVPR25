# -*- coding: utf-8 -*-
# Author: Gaojian Wang@ZJUICSR
# --------------------------------------------------------
# This source code is licensed under the Attribution-NonCommercial 4.0 International License.
# You can find the license in the LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# SimCLR: https://github.com/google-research/simclr
# BYOL: https://github.com/lucidrains/byol-pytorch
# SimSiam: https://github.com/facebookresearch/simsiam
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
from __future__ import print_function

import torch
import torch.nn as nn
import math


class SimSiamLoss(nn.Module):
    def __init__(self):
        super(SimSiamLoss, self).__init__()
        self.criterion = nn.CosineSimilarity(dim=1)

    def forward(self, cl_features):

        if len(cl_features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(cl_features.shape) > 3:
            cl_features = cl_features.view(cl_features.shape[0], cl_features.shape[1], -1)  # [BS, 2, feat_cl_dim]

        cl_features_1 = cl_features[:, 0]  # [BS, feat_cl_dim]
        cl_features_2 = cl_features[:, 1]  # [BS, feat_cl_dim]
        loss = -(self.criterion(cl_features_1, cl_features_2).mean()) * 0.5

        # if not math.isfinite(loss):
        #     print(cl_features_1, '\n', cl_features_2)
        #     print(self.criterion(cl_features_1, cl_features_2))

        return loss


class BYOLLoss(nn.Module):
    def __init__(self):
        super(BYOLLoss, self).__init__()

    @staticmethod
    def forward(cl_features):

        if len(cl_features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(cl_features.shape) > 3:
            cl_features = cl_features.view(cl_features.shape[0], cl_features.shape[1], -1)  # [BS, 2, feat_cl_dim]

        cl_features_1 = cl_features[:, 0]  # [BS, feat_cl_dim]
        cl_features_2 = cl_features[:, 1]  # [BS, feat_cl_dim]
        loss = 2 - 2 * (cl_features_1 * cl_features_2).sum(dim=-1)
        # loss = 1 - (cl_features_1 * cl_features_2).sum(dim=-1)
        loss = loss.mean()

        if not math.isfinite(loss):
            print(cl_features_1, '\n', cl_features_2)
            print(2 - 2 * (cl_features_1 * cl_features_2).sum(dim=-1))

        return loss


# different implementation of InfoNCELoss, including MOCOV3Loss; SupConLoss
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1, contrast_sample='all'):
        """
        from CMAE: https://github.com/ZhichengHuang/CMAE/issues/5
        :param temperature: 0.1 0.5 1.0, 1.5 2.0
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        self.contrast_sample = contrast_sample

    def forward(self, cl_features):
        """
        Args:
            :param cl_features: : hidden vector of shape [bsz, n_views, ...]
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if cl_features.is_cuda
                  else torch.device('cpu'))

        if len(cl_features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(cl_features.shape) > 3:
            cl_features = cl_features.view(cl_features.shape[0], cl_features.shape[1], -1)  # [BS, 2, feat_cl_dim]

        cl_features_1 = cl_features[:, 0]  # [BS, feat_cl_dim]
        cl_features_2 = cl_features[:, 1]  # [BS, feat_cl_dim]
        score_all = torch.matmul(cl_features_1, cl_features_2.transpose(1, 0))  # [BS, BS]
        score_all = score_all / self.temperature
        bs = score_all.size(0)

        if self.contrast_sample == 'all':
            score = score_all
        elif self.contrast_sample == 'positive':
            mask = torch.eye(bs, dtype=torch.float).to(device)  # torch.Size([BS, BS])
            score = score_all * mask
        else:
            raise ValueError('Contrastive sample: all{pos&neg} or positive(positive)')

        # label = (torch.arange(bs, dtype=torch.long) +
        #          bs * torch.distributed.get_rank()).to(device)
        label = torch.arange(bs, dtype=torch.long).to(device)

        loss = 2 * self.temperature * self.criterion(score, label)

        if not math.isfinite(loss):
            print(cl_features_1, '\n', cl_features_2)
            print(score_all, '\n', score, '\n', mask)

        return loss


class MOCOV3Loss(nn.Module):
    def __init__(self, temperature=0.1):
        super(MOCOV3Loss, self).__init__()
        self.temperature = temperature

    def forward(self, cl_features):

        if len(cl_features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(cl_features.shape) > 3:
            cl_features = cl_features.view(cl_features.shape[0], cl_features.shape[1], -1)  # [BS, 2, feat_cl_dim]

        cl_features_1 = cl_features[:, 0]  # [BS, feat_cl_dim]
        cl_features_2 = cl_features[:, 1]  # [BS, feat_cl_dim]

        # normalize
        cl_features_1 = nn.functional.normalize(cl_features_1, dim=1)
        cl_features_2 = nn.functional.normalize(cl_features_2, dim=1)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [cl_features_1, cl_features_2]) / self.temperature
        N = logits.shape[0]
        labels = (torch.arange(N, dtype=torch.long)).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.temperature)


class SupConLoss(nn.Module):
    """
    from: https://github.com/HobbitLong/SupContrast
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.1, contrast_mode='all', contrast_sample='all',
                 base_temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.contrast_sample = contrast_sample
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)  # [BS, 2, feat_cl_dim]

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)  # torch.Size([BS, BS])
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # contrast_count(2)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [BS*contrast_count, D]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]  # [BS, D]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature  # [BS*contrast_count, D]
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)  # [BS*contrast_count, BS*contrast_count]
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # [BS*contrast_count, 1]
        logits = anchor_dot_contrast - logits_max.detach()  # [BS*contrast_count, BS*contrast_count]

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)  # [BS*anchor_count, BS*contrast_count]
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )  # [BS*anchor_count, BS*contrast_count]
        mask = mask * logits_mask  # [BS*anchor_count, BS*contrast_count]

        """
        logits_mask is used to get the denominator(positives and negatives).
        mask is used to get the numerator(positives). mask is applied to log_prob.
        """

        # compute log_prob，logits_mask is contrast anchor with both positives and negatives
        exp_logits = torch.exp(logits) * logits_mask  # [BS*anchor_count, BS*contrast_count]
        # compute log_prob，logits_mask is contrast anchor with negatives, i.e., denominator only negatives contrast:
        # exp_logits = torch.exp(logits) * (logits_mask-mask)

        if self.contrast_sample == 'all':
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # [BS*anchor_count, BS*anchor_count]
            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # [BS*anchor_count]
        elif self.contrast_sample == 'positive':
            mean_log_prob_pos = (mask * logits).sum(1) / mask.sum(1)
        else:
            raise ValueError('Contrastive sample: all{pos&neg} or positive(positive)')

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class InfoNCELossPatchLevel(nn.Module):
    """
    test: ref ConMIM: https://github.com/TencentARC/ConMIM.
    """
    def __init__(self, temperature=0.1, contrast_sample='all'):
        """
        :param temperature: 0.1 0.5 1.0, 1.5 2.0
        """
        super(InfoNCELossPatchLevel, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        self.contrast_sample = contrast_sample

        self.facial_region_group = [
            [2, 3],  # eyebrows
            [4, 5],  # eyes
            [6],  # nose
            [7, 8, 9],  # mouth
            [10, 1, 0],  # face boundaries
            [10],  # hair
            [1],  # facial skin
            [0]  # background
        ]

    def forward(self, cl_features, parsing_map=None):
        """
        Args:
            :param parsing_map:
            :param cl_features: : hidden vector of shape [bsz, n_views, ...]
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if cl_features.is_cuda
                  else torch.device('cpu'))

        if len(cl_features.shape) < 4:
            raise ValueError('`features` needs to be [bsz, n_views, n_cl_patches, ...],'
                             'at least 4 dimensions are required')
        if len(cl_features.shape) > 4:
            cl_features = cl_features.view(cl_features.shape[0], cl_features.shape[1], cl_features.shape[2], -1)
            # [BS, 2, num_cl_patches, feat_cl_dim]

        cl_features_1 = cl_features[:, 0]
        cl_features_2 = cl_features[:, 1]
        score = torch.matmul(cl_features_1, cl_features_2.permute(0, 2, 1))  # [BS, num_cl_patches, num_cl_patches]
        score = score / self.temperature
        bs = score.size(0)
        num_cl_patches = score.size(1)

        if self.contrast_sample == 'all':
            score = score
        elif self.contrast_sample == 'positive':
            mask = torch.eye(num_cl_patches, dtype=torch.float32)  # torch.Size([num_cl_patches, num_cl_patches])
            mask_batch = mask.unsqueeze(0).expand(bs, -1).to(device)  # [bs, num_cl_patches, num_cl_patches]
            score = score*mask_batch
        elif self.contrast_sample == 'region':
            cl_features_1_fr = []
            cl_features_2_fr = []
            for facial_region_index in self.facial_region_group:
                fr_mask = (parsing_map == facial_region_index).unsqueeze(2).expand(-1, -1, cl_features_1.size(-1))
                cl_features_1_fr.append((cl_features_1 * fr_mask).mean(dim=1, keepdim=False))
                cl_features_2_fr.append((cl_features_1 * fr_mask).mean(dim=1, keepdim=False))
            cl_features_1_fr = torch.stack(cl_features_1_fr, dim=1)
            cl_features_2_fr = torch.stack(cl_features_2_fr, dim=1)
            score = torch.matmul(cl_features_1_fr, cl_features_2_fr.permute(0, 2, 1))  # [BS, 8, 8]
            score = score / self.temperature
            # mask = torch.eye(cl_features_1_fr.size(1), dtype=torch.bool)
            # torch.Size([cl_features_1_fr.size(1), cl_features_1_fr.size(1)])
            # mask_batch = mask.unsqueeze(0).expand(bs, -1).to(device)
            # [bs, cl_features_1_fr.size(1), cl_features_1_fr.size(1)]
            # score = score*mask_batch
            label = torch.arange(cl_features_1_fr.size(1), dtype=torch.long).to(device)
            labels_batch = label.unsqueeze(0).expand(bs, -1)
            loss = 2 * self.temperature * self.criterion(score, labels_batch)
            return loss
        else:
            raise ValueError('Contrastive sample: all{pos&neg} or positive(positive)')

        # label = (torch.arange(bs, dtype=torch.long) +
        #          bs * torch.distributed.get_rank()).to(device)
        label = torch.arange(num_cl_patches, dtype=torch.long).to(device)
        labels_batch = label.unsqueeze(0).expand(bs, -1)

        loss = 2 * self.temperature * self.criterion(score, labels_batch)

        return loss


class MSELoss(nn.Module):
    """
    test: unused
    """
    def __init__(self):
        super(MSELoss, self).__init__()

    @staticmethod
    def forward(cl_features):

        if len(cl_features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, n_patches, ...],'
                             'at least 3 dimensions are required')
        if len(cl_features.shape) > 3:
            cl_features = cl_features.view(cl_features.shape[0], cl_features.shape[1], -1)  # [BS, 2, feat_cl_dim]

        cl_features_1 = cl_features[:, 0].float()  # [BS, feat_cl_dim]
        cl_features_2 = cl_features[:, 1].float()  # [BS, feat_cl_dim]

        return torch.nn.functional.mse_loss(cl_features_1, cl_features_2, reduction='mean')
