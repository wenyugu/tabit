import torch
import torch.nn.functional as F

def tab_precision(output, target):
    # get rid of "closed" class, as we only want to count positives
    tab_pred = F.one_hot(torch.max(F.softmax(output, 2), 2)[1], num_classes=21)[:, :, 1:]
    tab_gt = target[:, :, 1:]
    numerator = torch.sum(tab_pred * tab_gt)
    denominator = torch.sum(tab_pred)
    return (1.0 * numerator) / denominator

def tab_recall(output, target):
    # get rid of "closed" class, as we only want to count positives
    tab_pred = F.one_hot(torch.max(F.softmax(output, 2), 2)[1], num_classes=21)[:, :, 1:]
    tab_gt = target[:, :, 1:]
    numerator = torch.sum(tab_pred * tab_gt)
    denominator = torch.sum(tab_gt)
    return (1.0 * numerator) / denominator

def f_score(p, r):
    return (2 * p * r) / (p + r)