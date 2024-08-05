import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import SimpleITK as sitk
from sklearn.metrics import confusion_matrix
import torch
import numpy as np
import torch.nn as nn
import json


def format_conversion(tl, pl):
    tru, pre = [], []
    for i in range(len(tl)):
        for j in range(len(tl[i])):
            pre.append(int(pl[i][j]))
            tru.append(int(tl[i][j]))

    return tru, pre


def confusion(con_str, num_class, tl, pl, save=False):
    tru, pre = format_conversion(tl, pl)
    conf_mat = confusion_matrix(tru, pre)  # (bs, num_cla)
    te_acc, tev_npv, te_ppv, te_sen, te_spe, te_fos = calculate_metrics_cls(conf_mat, con_str)
    if save:
        labels = [f"Class {i+1}" for i in range(num_class)]
        df_cm = pd.DataFrame(conf_mat, index=labels, columns=labels)
        sns.set(font_scale=1.4)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu", square=False)
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=90, ha='right')
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='center')
        matplotlib.rcParams['font.sans-serif'] = 'Arial'
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.savefig('./recording/classification/' + str(con_str) + 'matrix.pdf', bbox_inches='tight', pad_inches=0.0, dpi=300)
        plt.show()

    return te_acc, tev_npv, te_ppv, te_sen, te_spe, te_fos


def calculate_metrics_cls(conf_mat, con_str, e=1e-6):
    num_class = conf_mat.shape[0]
    TP = np.diag(conf_mat)
    FN = conf_mat.sum(axis=1) - TP
    FP = conf_mat.sum(axis=0) - TP
    TN = conf_mat.sum() - (TP + FN + FP)

    accuracy = np.sum(TP) / np.sum(conf_mat)
    npv = TN / (TN + FN + e)
    ppv = TP / (TP + FP + e)
    sensitivity = TP / (TP + FN + e)
    specificity = TN / (TN + FP + e)
    f1_score = 2 * ppv * sensitivity / (ppv + sensitivity + e)

    macro_npv = np.mean(npv)
    macro_ppv = np.mean(ppv)
    macro_sensitivity = np.mean(sensitivity)
    macro_specificity = np.mean(specificity)
    macro_f1_score = np.mean(f1_score)

    metrics = {
        "accuracy": accuracy,
        "npv": {f"Class {i + 1}": npv[i] for i in range(num_class)},
        "ppv": {f"Class {i + 1}": ppv[i] for i in range(num_class)},
        "sensitivity": {f"Class {i + 1}": sensitivity[i] for i in range(num_class)},
        "specificity": {f"Class {i + 1}": specificity[i] for i in range(num_class)},
        "f1_score": {f"Class {i + 1}": f1_score[i] for i in range(num_class)},
        "macro_npv": macro_npv,
        "macro_ppv": macro_ppv,
        "macro_sensitivity": macro_sensitivity,
        "macro_specificity": macro_specificity,
        "macro_f1_score": macro_f1_score,
    }

    with open('./recording/classification/' + str(con_str) + 'metrics.json', "w") as f:
        json.dump(metrics, f, indent=4)

    return accuracy, macro_npv, macro_ppv, macro_sensitivity, macro_specificity, macro_f1_score


def soft_iou(out, target, e=1e-6):
    target, out = target.squeeze(dim=1), out.squeeze(dim=1)
    target, out = target.reshape(target.shape[0], -1), out.reshape(target.shape[0], -1)
    out = torch.sigmoid(out)
    num = (out*target).sum(1, True)
    den = (out+target-out*target).sum(1, True) + e
    iou = num / den
    cost = 1 - iou

    return cost.squeeze()


class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss,self).__init__()
    def forward(self, y_pred, y_true):
        costs = soft_iou(y_pred, y_true).view(-1,1)
        costs = torch.mean(costs)
        return costs


def calculate_metrics_seg(target, out, e=1e-6):
    out = torch.sigmoid(out)
    out_binary = (out > 0.5).int()
    target_binary = (target > 0.5).int()
    intersection = torch.logical_and(out_binary, target_binary).sum(dim=(2, 3))
    union_dice = out_binary.sum(dim=(2, 3)) + target_binary.sum(dim=(2, 3))
    union_iou = union_dice - intersection
    dice_score = 2 * intersection / (union_dice + e)
    iou_score = intersection / (union_iou + e)
    correct_pixels = torch.eq(out_binary, target_binary).sum()
    total_pixels = target_binary.numel()
    pixel_accuracy = correct_pixels / total_pixels

    return dice_score.mean(), iou_score.mean(), pixel_accuracy
