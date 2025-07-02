import os
import numpy as np
import random
import cv2

import torch
from torch import nn
import torch.nn.functional as F

def smooth(gt_label, video_length, epsilon=0.1):
    frame_label = torch.zeros([video_length, 2])
    for i in range(video_length):
        if i in gt_label:
            frame_label[i, :] = torch.tensor([epsilon, 1 - epsilon])
        else:
            frame_label[i, :] = torch.tensor([1 - epsilon, epsilon])
    return frame_label

def extent(label):
    e_label = torch.zeros(len(label), 2)
    e_label[:, 0] = 1 - label
    e_label[:, 1] = label
    return e_label.to(label.device)

# gaussian blur, sigma = 2
# [0.32465247, 0.60653066, 0.8824969, 1., 0.8824969, 0.60653066, 0.32465247]    sigma = 2.0
# [0.011109, 0.13533528, 0.60653066, 1., 0.60653066, 0.13533528, 0.011109]      sigma = 1.0
# [0.13533528, 0.41111229, 0.8007374, 1., 0.8007374, 0.41111229, 0.13533528]    sigma = 1.5
def gauss_blur(label):
    # gauss = [0, 0, 0.8824969, 1., 0.8824969, 0, 0]
    gauss = [0.32465247, 0.60653066, 0.8824969, 1., 0.8824969, 0.60653066, 0.32465247]
    max_len = len(label)
    glabel = torch.zeros_like(label)
    gt = label.nonzero().detach()
    for i in gt:
        for j in range(0, 7):
            idx = i + j - 3
            if idx >= 0 and idx < max_len:
                glabel[idx] += gauss[j]
    glabel = glabel.clamp(0, 1)
    return glabel

def blur2D(label): # [8 64]
    blur_label = []
    for i in range(len(label)):
        blur_label.append(gauss_blur(label[i]))
    return torch.stack(blur_label, dim = 0)

#############  debug util  #############
def check_grad(net):
    flag = True
    for name, param in net.named_parameters():
        # if param.grad is None:
        if not param.requires_grad or param.grad is None:
            continue
        if torch.isnan(param.grad).any():
            print(name, torch.mean(param.grad))
            flag = False
    return flag

def check_param(net):
    for name, param in net.named_parameters():
        # if not param.requires_grad:
        #     continue
        if torch.isnan(param).sum() != 0:
            print(name)
            return False
    return True

############   testing&metrics     ##########
def num_pos_error(out, label):
    num = 0
    pos = 0
    gt = label.nonzero()
    pred = out.nonzero()
    if len(pred) < 0:
        return 0, 0
    for i in pred:
        dist = np.abs(gt - i)
        num += 1 if min(dist) <= 3 else 0
        pos += min(dist) if min(dist) <= 3 else 0
    num = max(num, 0.5)
    return num, pos / num

def metrics(pred, label):
    # pred = pred.nonzero()
    # label = label.nonzero()
    if len(pred) <= 0 or len(label) <= 0:  # nothing to predict
        return 0, 0, 0, 0, 0
    pos = 0
    num_t1 = 0
    num_t2 = 0
    for i in label:
        dist = np.abs(pred - i)
        num_t1 += 1 if min(dist) <= 3 else 0 # label find
        pos += min(dist) if min(dist) <= 3 else 0
    for i in pred:
        dist = np.abs(label - i)
        num_t2 += 1 if min(dist) <= 3 else 0 # pred right
    if num_t1 <= 0:     # nothing to find
        return 0, 0, 0, 0, 0
    tp = num_t1
    fn = len(label) - num_t1
    fp = len(pred) - num_t2

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    fscore = 2 * (recall * precision) / (recall + precision)
    num_error = np.abs(num_t1 - len(label))
    poz_error = pos / num_t1
    return recall, precision, num_error, poz_error, fscore

def prune_label(label):
    count = 0
    actions = np.zeros(len(label))
    for i in range(len(label)):
        if (i > 0) and (label[i] == label[i - 1] + 1):
            actions[i] = actions[i - 1]
        else:
            actions[i] = count + 1
            count += 1
    # cluster = {}
    # for i in range(len(label)):
    #     if actions[i] not in cluster:
    #         cluster[actions[i]] = []
    #     cluster[actions[i]].append(label[i])
    return actions

def eval(pred, label):
    if len(pred) <= 0 or len(label) <= 0: 
        return 0, 0, 0, 0, 0
    
    actions = prune_label(label)
    cluster = np.zeros(len(pred))
    for idx, point in enumerate(pred):
        dist = np.abs(label - point)
        if min(dist) <= 3:
            cluster[idx] = actions[np.argmin(dist)]
            pos += min(dist)
        else:
            cluster[idx] = -1
    tp = 0

    for i in range(len(label)):
        if (i > 0) and (label[i] == label[i - 1] + 1):
            actions[i] = actions[i - 1]
        else:
            actions[i] = count + 1
            count += 1


    for center in np.unique(cluster):
        if center in np.unique(actions):
            tp += 1
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    fscore = 2 * (recall * precision) / (recall + precision)
    num_error = np.abs(num_t1 - len(label))
    poz_error = pos / num_t1
    return recall, precision, num_error, poz_error, fscore

#############
'''
conns = []
for i in range(diff.shape[1] - 1):
    conn = compute_connectivity_error(diff[0, i].cpu().numpy(), diff[0, i+1].cpu().numpy())
    conns.append(conn)
conns = np.array(conns)
conn_e = np.argsort(np.fabs(conns[1:] - conns[:-1]))
print(np.argsort(conns))
print(conn_e.squeeze())
print(label[0].nonzero().squeeze())
'''

###########      distributed       ##############
def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return torch.utils.data.RandomSampler(dataset)
    else:
        return torch.utils.data.SequentialSampler(dataset)
    
