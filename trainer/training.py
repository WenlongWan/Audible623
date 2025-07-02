"""train or valid looping """
import os
import numpy as np
import random
import math
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm import tqdm
from tensorboardX import SummaryWriter
import datetime


from trainer.losses import *
from trainer.train_util import blur2D, extent

# torch.manual_seed(1)  # random seed. We not yet optimization it.

def train_loop(n_epochs, model, train_set, valid_set, batch_size=1, lr=1e-6,
               ckpt_name='ckpt', lastckpt=None, log_dir='log/eval', device_ids=[0]):
    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
    currEpoch = 0
    trainloader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=4)
    validloader = DataLoader(valid_set, batch_size=1, pin_memory=False, shuffle=True, num_workers=2)
    param_groups = model.get_parameter_groups()
    model = nn.DataParallel(model.to(device), device_ids=device_ids)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    optimizer = torch.optim.Adam([
        {'params': param_groups[0], 'lr': lr},
        {'params': param_groups[1], 'lr': 2 * lr},
        {'params': param_groups[2], 'lr': 10 * lr},
        {'params': param_groups[3], 'lr': 20 * lr},
    ], lr=lr)
    scaler = GradScaler()

    if lastckpt is not None:
        print("loading checkpoint")
        checkpoint = torch.load(lastckpt)
        currEpoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        del checkpoint

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    batch_idx = 0
    for epoch in tqdm(range(currEpoch, n_epochs + currEpoch)):
        pbar = tqdm(trainloader, total=len(trainloader))
        trainLosses = []

        for input, diff, d2f, target in pbar:
            with autocast():
                model.train()
                optimizer.zero_grad()
                input = input.type(torch.FloatTensor).to(device)
                diff = diff.type(torch.FloatTensor).to(device)
                d2f = d2f.type(torch.FloatTensor).to(device)
  
                target = target.to(device).detach()
                target.requires_grad_ = False
    
                pred, pred_l, m1, m2, c = model(input, diff, d2f)

                density = blur2D(target)
                density = density.type(torch.FloatTensor).to(device)

                predict_density = pred
                loss = cal_loss(predict_density, target, pred_l, m1, m2, c)

                assert torch.isnan(loss).sum() == 0, print(loss)
                pbar.set_postfix({'Epoch': epoch, 'batch_idx': batch_idx, 'loss': loss.item()})
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                batch_idx += 1

        # scheduler.step()
        if not os.path.exists('checkpoint/{0}/'.format(ckpt_name)):
            os.mkdir('checkpoint/{0}/'.format(ckpt_name))
        if (epoch > 0 and epoch % 20 == 0):
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
            }
            torch.save(checkpoint,
                       'checkpoint/{0}/'.format(ckpt_name) + str(epoch) + '.pt')

def cal_loss(predict_, target, pred_l, m1, m2, c):
    density = blur2D(target)
    density = density.type(torch.FloatTensor).to(predict_.device)

    focal_loss = FocalLoss()
    loss1 = focal_loss(predict_.view(-1, 2), target.view(-1).long())
    loss2 = soft_cross_entropy(predict_.view(-1, 2), extent(density.view(-1)))
    loss_ce = 0.1 * loss2 + loss1

    # loss_ce_l = focal_loss(pred_l.view(-1, 2), target.view(-1).long())
    loss_loc = fb_loss(m1, m2, c)
    loss_tmp = tmp_loss(c)
    loss = loss_ce + 0.01 * loss_loc + 0.002 * loss_tmp
    return loss

def fb_loss(m1, m2, c):
    bsz = len(m1)
    c.requires_grad_ = False
    c = c.detach()
    criterion = [SimMaxLoss(metric='cos', alpha=0.25).cuda(), SimMinLoss(metric='cos').cuda(),
                 SimMaxLoss(metric='cos', alpha=0.25).cuda()]
    ################# Inter-Frame Sampling #################
    fg_feats = torch.stack([m1[i, c[i]] for i in range(bsz)])
    bg_feats = torch.stack([m2[i, c[i]] for i in range(bsz)])
    # loss  = multi_loss(key_side, key_mask)
    loss1 = criterion[0](fg_feats)
    loss2 = criterion[1](bg_feats, fg_feats)
    loss3 = criterion[2](bg_feats)
    loss = loss1 + loss2 + loss3
    return loss