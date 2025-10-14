from utils.utils import AverageMeter, accuracy
from utils.statistic import get_EER_states, get_HTER_at_thr, calculate, calculate_threshold
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import roc_curve


def eval(valid_dataloader, model, norm_flag, return_prob=False):
    """
    Eval the FAS model on the target data.
    The function evaluates the model on the target data and return TPR@FPR, HTER
    and AUC.
    Used in train.py to eval the model after each epoch; used in test.py to eval
    the final model.
    """

    criterion = nn.CrossEntropyLoss()
    valid_losses = AverageMeter()
    valid_top1 = AverageMeter()
    prob_dict = {}
    label_dict = {}
    model.eval()
    output_dict_tmp = {}
    target_dict_tmp = {}
    with torch.no_grad():
        for iter, (input, target, videoID, name) in enumerate(valid_dataloader):
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()

            cls_out, feature = model(input, norm_flag)

            prob = F.softmax(cls_out, dim=-1).cpu().data.numpy()[:, 0][:, 1]
            label = target.cpu().data.numpy()
            videoID = videoID.cpu().data.numpy()

            for i in range(len(prob)):
                if (videoID[i] in prob_dict.keys()):
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]].append(cls_out[i][0].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                else:
                    prob_dict[videoID[i]] = []
                    label_dict[videoID[i]] = []
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]] = []
                    target_dict_tmp[videoID[i]] = []
                    output_dict_tmp[videoID[i]].append(cls_out[i][0].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
    prob_list = []
    label_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
        # avg_single_video_prob = np.argmax(avg_single_video_prob, axis=0)
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)
        # compute loss and acc for every video
        avg_single_video_output = sum(output_dict_tmp[key]) / len(
            output_dict_tmp[key])
        avg_single_video_target = sum(target_dict_tmp[key]) / len(
            target_dict_tmp[key])
        loss = criterion(avg_single_video_output, avg_single_video_target.long())
        acc_valid = accuracy(
            avg_single_video_output, avg_single_video_target, topk=(1,))
        valid_losses.update(loss.item())
        valid_top1.update(acc_valid[0])

    auc_score = roc_auc_score(label_list, prob_list)
    cur_EER_valid, threshold, _, _ = get_EER_states(prob_list, label_list)
    ACC_threshold = calculate_threshold(prob_list, label_list, threshold)
    cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)

    fpr, tpr, thr = roc_curve(label_list, prob_list)
    tpr_filtered = tpr[fpr <= 1 / 100]
    if len(tpr_filtered) == 0:
        rate = 0
    else:
        rate = tpr_filtered[-1]
    print("TPR@FPR = ", rate)

    if not return_prob:
        return [
            valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid,
            auc_score, threshold, ACC_threshold * 100, rate
        ]
    else:
        return [
            valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid,
            auc_score, threshold, ACC_threshold * 100, rate], [prob_list, label_list]
