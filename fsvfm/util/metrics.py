# -*- coding: utf-8 -*-
# Author: Gaojian Wang@ZJUICSR
# --------------------------------------------------------
# This source code is licensed under the Attribution-NonCommercial 4.0 International License.
# You can find the license in the LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc, accuracy_score, balanced_accuracy_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# can reuse frame_level metrics for image-level
def frame_level_acc(labels, y_preds):
    return accuracy_score(labels, y_preds) * 100.


def frame_level_balanced_acc(labels, y_preds):
    return balanced_accuracy_score(labels, y_preds) * 100.


def frame_level_auc(labels, preds):
    return roc_auc_score(labels, preds) * 100.


def frame_level_eer(labels, preds):
    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    # eer_thresh = interp1d(fpr, thresholds)(eer)
    return eer


# def frame_level_eer(labels, preds):
#     fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=1)
#     eer_threshold = thresholds[(fpr + (1 - tpr)).argmin()]
#     fpr_eer = fpr[thresholds == eer_threshold][0]
#     fnr_eer = 1 - tpr[thresholds == eer_threshold][0]
#     eer = (fpr_eer + fnr_eer) / 2
#     metric_logger.meters['eer'].update(eer)
#     return eer, eer_thresh


def get_video_level_label_pred(f_label_list, v_name_list, f_pred_list):
    """
    References:
    CADDM: https://github.com/megvii-research/CADDM
    """
    video_res_dict = dict()
    video_pred_list = list()
    video_y_pred_list = list()
    video_label_list = list()
    # summarize all the results for each video
    for label, video, score in zip(f_label_list, v_name_list, f_pred_list):
        if video not in video_res_dict.keys():
            video_res_dict[video] = {"scores": [score], "label": label}
        else:
            video_res_dict[video]["scores"].append(score)
    # get the score and label for each video
    for video, res in video_res_dict.items():
        score = sum(res['scores']) / len(res['scores'])
        label = res['label']
        video_pred_list.append(score)
        video_label_list.append(label)
        video_y_pred_list.append(score >= 0.5)

    return video_label_list, video_pred_list, video_y_pred_list


def video_level_acc(video_label_list, video_y_pred_list):
    return accuracy_score(video_label_list, video_y_pred_list) * 100.


def video_level_balanced_acc(video_label_list, video_y_pred_list):
    return balanced_accuracy_score(video_label_list, video_y_pred_list) * 100.


def video_level_auc(video_label_list, video_pred_list):
    return roc_auc_score(video_label_list, video_pred_list) * 100.


def video_level_eer(video_label_list, video_pred_list):
    fpr, tpr, thresholds = roc_curve(video_label_list, video_pred_list, pos_label=1)
    v_eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    # eer_thresh = interp1d(fpr, thresholds)(eer)
    return v_eer
