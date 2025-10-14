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
import torch.nn.functional as F


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
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
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
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)  # avoid log(0) that causes NaN loss

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class RealAnchorConLoss(SupConLoss):
    """Face-Security-Specific Supervised Contrastive Loss.
    Only pulls pos-pos (real-real) together and pushes pos-neg (real-fake) pairs apart.
    Does not affect neg-neg (fake-fake) pairs.
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07,
                 pos_class_index=0):
        super().__init__(temperature, contrast_mode, base_temperature)
        self.pos_class_index = pos_class_index  # real class index (0 or 1)

    def forward(self, features, labels=None, mask=None):
        device = features.device

        # Check if batch contains any real samples
        labels = labels.contiguous().view(-1, 1)
        pos_samples = (labels == self.pos_class_index).float()
        num_pos = pos_samples.sum()

        # If no real samples in batch, return zero loss
        if num_pos == 0:
            return torch.tensor(1e-6, device=device, requires_grad=True)

        # Create binary mask: 1 for pos-pos pairs, 0 for pos-neg pairs
        mask = torch.eq(labels, labels.T).float().to(device)

        # Zero out neg-neg pairs in mask
        has_pos = torch.matmul(pos_samples, pos_samples.T)
        mask = mask * has_pos

        # Call parent class forward with computed mask
        return super().forward(features, mask=mask)


class PosOnlySupConLoss(SupConLoss):
    """Ablating Face-Security-Specific Supervised Contrastive Loss.
    Only pulls pos-pos (real-real) together.
    Does not push pos-neg (real-fake) pairs apart or affect neg-neg (fake-fake) pairs.
    """

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, pos_class_index=0):
        super().__init__(temperature, contrast_mode, base_temperature)
        self.pos_class_index = pos_class_index  # index of the “real” class

    def forward(self, features, labels=None, mask=None):
        device = features.device

        if labels is None:
            raise ValueError("Labels must be provided for this loss")
        labels = labels.to(device)

        # Standard shape checks from SupConLoss
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dims required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        else:  # 'all'
            anchor_feature = contrast_feature
            anchor_count = contrast_count

        # Build a mask that is 1 ONLY for real–real pairs
        labels = labels.contiguous().view(-1, 1)
        pos_samples = (labels == self.pos_class_index).float()
        num_pos = pos_samples.sum()
        # No real samples → zero loss
        if num_pos < 1:
            return torch.tensor(0.0, device=device, dtype=features.dtype, requires_grad=True)

        # mask_pos[i,j] == 1 iff both i and j are real AND have same label
        mask_pos = torch.eq(labels, labels.T).float().to(device)
        has_pos = pos_samples @ pos_samples.T  # outer product
        mask_pos = mask_pos * has_pos
        mask_pos.fill_diagonal_(0)

        # Compute logits
        anchor_dot_contrast = torch.div(
            anchor_feature @ contrast_feature.T,
            self.temperature
        )
        # Numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Create a denominator-mask that only includes positive pairs (to prevent any pushing)
        logits_mask = mask_pos.repeat(anchor_count, contrast_count)
        # Additionally zero out self-contrast entries
        diag_idx = torch.arange(batch_size * anchor_count, device=device)
        logits_mask[diag_idx, diag_idx] = 0

        # Compute log-probabilities over ONLY the real–real pairs
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # Mean log-probability of positive (real–real) pairs
        mask_sum = mask_pos.sum(1).repeat(anchor_count)
        mask_sum = torch.where(mask_sum < 1e-6, 1, mask_sum)
        mean_log_prob_pos = (mask_pos.repeat(anchor_count, contrast_count) * log_prob).sum(1) / mask_sum

        # Final loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


class PosNegOnlySupConLoss(SupConLoss):
    """Ablating Face-Security-Specific Supervised Contrastive Loss.
    Only pushes pos-neg (real-fake) pairs apart.
    Does not pull pos-pos (real-real) together or affect neg-neg (fake-fake) pairs.
    """

    def __init__(self, temperature=0.07, pos_class_index=0):
        # contrast_mode and base_temperature unused here, but kept for API consistency
        super().__init__(temperature, contrast_mode='all', base_temperature=0.07)
        self.pos_class_index = pos_class_index

    def forward(self, features, labels=None, mask=None):
        device = features.device

        if labels is None:
            raise ValueError("Labels must be provided for this loss")
        labels = labels.to(device)

        # Shape checks
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dims required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        # Collapse views (we treat each sample via its first view)
        # If you have multiple views, you could average them instead
        feat = features[:, 0, :]  # [B, D]

        # Labels
        labels = labels.contiguous().view(-1)
        is_pos = (labels == self.pos_class_index)  # bool mask for real
        real_idx = torch.nonzero(is_pos, as_tuple=False).view(-1)
        fake_idx = torch.nonzero(~is_pos, as_tuple=False).view(-1)

        # Need at least one real and one fake to compute
        if real_idx.numel() < 1 or fake_idx.numel() < 1:
            return torch.tensor(0.0, device=device, dtype=features.dtype, requires_grad=True)

        # Compute pairwise cosine similarities [B, B]
        sim_matrix = feat @ feat.T  # already cosine since normalized
        # Extract only real–fake pairs
        sims = sim_matrix[real_idx.unsqueeze(1), fake_idx.unsqueeze(0)]  # [#real, #fake]

        # Push them apart using a smooth “softplus” penalty
        # loss = mean( log(1 + exp(sim / temperature)) )
        raw = sims / self.temperature
        loss = F.softplus(raw)  # softplus(x) = log(1 + exp(x))
        return loss.mean()


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
