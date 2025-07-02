import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_cross_entropy(pred, soft_targets):
    logsoftmax = nn.LogSoftmax(dim = 1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

class FocalLoss(nn.Module):
    def __init__(self, weight=None, reduction='none', gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
    
class Balanced_CE_loss(torch.nn.Module):
    def __init__(self):
        super(Balanced_CE_loss, self).__init__()

    def forward(self, input, target):
        input = input.view(input.shape[0], -1)
        target = target.view(target.shape[0], -1)
        loss = 0.0
        # version2
        for i in range(input.shape[0]):
            beta = 1-torch.sum(target[i])/target.shape[1]
            x = torch.max(torch.log(input[i]), torch.tensor([-100.0]))
            y = torch.max(torch.log(1-input[i]), torch.tensor([-100.0]))
            l = -(beta*target[i] * x + (1-beta)*(1 - target[i]) * y)
            loss += torch.sum(l)
        return loss
    
class BlanceLoss(torch.nn.Module):
    def __init__(self):
        super(BlanceLoss, self).__init__()

    def forward(self, input, target):
        loss1 = F.binary_cross_entropy_with_logits(input, target)

        loss_positive = torch.mean(loss1 * target) * 10
        loss_negative = torch.mean(loss1 * (1 - target))

        loss = loss_positive + loss_negative
        return loss
    

###########  unsupervised learning & contrasive learning  #############
# Minimize Similarity, e.g., push representation of foreground and background apart.

def multi_loss(loc, mask):
    # loc = loc.view(B, 1, H * W).permute(0, 2, 1).contiguous()    # [B, H*W, 1]
    # ccam = ccam.view(B, 1, B * W)                                # [B, 1, H*W]
    # fg_feats = torch.matmul(ccam, loc) / (H * W)                # [B, 1, C]
    # bg_feats = torch.matmul(1 - ccam, loc) / (H * W)            # [B, 1, C]
    B, _, H, W = loc.shape
    mask = F.upsample(mask, size=(H,W), mode='bilinear')
    # avg_pool = nn.AdaptiveAvgPool2d((H, W))
    # loc = avg_pool(loc)

    epsilon = 0.6
    epsilon2 = 0.4
    tau = 0.03
    logit_temperature = 0.07
    loc = loc.view(B, H * W)
    mask = mask.view(B, H * W)
    pos = F.sigmoid((mask - epsilon)/tau)       # [B, H*W]
    pos2 = F.sigmoid((mask - epsilon2)/tau)     # [B, H*W]
    neg = 1 - pos2                              # [B, H*W]

    sim1 = ((loc * pos).sum(-1) / (pos.sum(-1))).unsqueeze(1)   # [B]
    sim2 = ((loc * neg).sum(-1) / (neg.sum(-1))).unsqueeze(1)   # [B]
    dis = torch.matmul(loc, pos.T) / (pos.sum(-1))              # [B, B]
    # sim = list()
    # for i in range(B):
    #     d3 = (dis[i, :i].sum(-1) + dis[i, i:].sum(-1)) / (B - 1)
    #     sim.append(d3)
    # sim = torch.stack(sim, dim = 0)
    # loss = (-1.0 / B) * torch.log( torch.exp(sim1) / (torch.exp(sim1) + torch.exp(sim2 + sim3) + 1e-7) )

    sim = dis * ( 1 -100 * torch.eye(B,B)).to(loc.device)
    logits = torch.cat((sim1,sim,sim2),1)/logit_temperature
    target = torch.zeros((B), dtype=torch.long).to(loc.device)
    loss = F.cross_entropy(logits, target)

    return torch.mean(loss), sim1, sim2, sim

    # embedded_fg = F.normalize(fg_feats.view(B, -1), dim=1)
    # embedded_bg = F.normalize(bg_feats.view(B, -1), dim=1)
    # sim3 = torch.matmul(embedded_fg, embedded_bg.T)

def perframe_loss(loc, mask):
    B, _, H, W = mask.shape
    # avg_pool = nn.AdaptiveAvgPool2d((H, W))
    # loc = avg_pool(loc)

    epsilon = 0.6
    epsilon2 = 0.4
    tau = 0.1
    logit_temperature = 0.07
    loc = loc.view(B, H * W)
    mask = mask.view(B, H * W)
    pos = F.sigmoid((mask - epsilon)/tau)       # [B, H*W]
    pos2 = F.sigmoid((mask - epsilon2)/tau)     # [B, H*W]
    neg = 1 - pos2                              # [B, H*W]

    # positive
    sim1 = ((loc * pos).sum(-1) / (pos.sum(-1))).unsqueeze(1)  # [B]
    # negative
    sim2 = ((loc * neg).sum(-1) / (neg.sum(-1))).unsqueeze(1)  # [B]
    # loss = -torch.log( torch.exp(sim1) / (torch.exp(sim1) + torch.exp(sim2) + 1e-7) )
    logits = torch.cat((sim1,sim2),1)/logit_temperature
    target = torch.zeros((B), dtype=torch.long).to(loc.device)
    loss = F.cross_entropy(logits, target, reduction='none')
    return loss

def intra_loss(loc, mask, center, width, epoch):
    batch_size, frame_len, _, _, _ = loc.shape    # B N C H W
    beta1 = 0.1 * 1e-3
    beta2 = 0.15 * 1e-3
    center = center / frame_len
    pos_weight = generate_gauss_weight(frame_len, center, width)
    neg_weight1, neg_weight2 = negative_proposal_mining(frame_len, center, width, epoch)
    pos_weight = F.normalize(pos_weight, p=1, dim=-1)
    neg_weight1 = F.normalize(neg_weight1, p=1, dim=-1)
    neg_weight2 = F.normalize(neg_weight2, p=1, dim=-1)
    loss = []
    ref_losses = []
    neg_losses_1 = []
    neg_losses_2 = []
    
    for i in range(batch_size):
        frame_loss = perframe_loss(loc[i], mask[i])
        loss_pos = (frame_loss * pos_weight[i]).mean()
        loss_neg1 = (frame_loss * neg_weight1[i]).mean()
        loss_neg2 = (frame_loss * neg_weight2[i]).mean()
        loss_ref = (frame_loss * (1 / frame_len)).mean()

        zero_0 = torch.zeros_like(loss_pos).to(loss_pos.device)
        zero_0.requires_grad = False
        zero_1 = torch.zeros_like(loss_pos).to(loss_pos.device)
        zero_1.requires_grad = False
        zero_2 = torch.zeros_like(loss_pos).to(loss_pos.device)
        zero_2.requires_grad = False
        ref_loss = torch.max(loss_pos - loss_ref + beta1, zero_0)
        neg_loss_1 = torch.max(loss_pos - loss_neg1 + beta2, zero_1)
        neg_loss_2 = torch.max(loss_pos - loss_neg2 + beta2, zero_2)
        rank_loss = ref_loss.mean() + neg_loss_1.mean() + neg_loss_2.mean() 
        # rank_loss = frame_loss.mean()
        loss.append(rank_loss)
        ref_losses.append(ref_loss.mean())
        neg_losses_1.append(neg_loss_1.mean())
        neg_losses_2.append(neg_loss_2.mean())
    loss = torch.stack(loss)
    # if loss.isnan().any():
    #     print(loss)
    #     perframe_loss(loc[0], mask[0])
    return torch.mean(loss), torch.stack(ref_losses), torch.stack(neg_losses_1), torch.stack(neg_losses_2)
    # return torch.mean(loss), torch.mean(loss), torch.mean(loss), torch.mean(loss)

def iou_loss(area1, area2):
    # B C H W
    ep = 1e-7
    intersection = 2 * torch.sum(area1 * area2, dim=(1,2,3)) + ep
    union = torch.sum(area1, dim=(1,2,3)) + torch.sum(area2, dim=(1,2,3)) + ep
    loss = 1 - (intersection / union) 
    # dice loss, less to be better
    return loss

def reconstruct_loss(area1, area2, mask):
    mse_loss = torch.nn.MSELoss()
    sum_area = (area1 + area2).clamp(0, 1)
    return mse_loss(sum_area, mask)

def intra_iou_loss(mask, area1, area2, target):
    batch_size, frame_len, _, _, _ = mask.shape    # B N C H W
    loss = []
    aloss = []
    rloss = []
    for i in range(batch_size):
        frame_loss = iou_loss(area1[i], area2[i])
        rec_loss = reconstruct_loss(area1[i], area2[i], mask[i])
        area_loss = (-frame_loss * (target[i] * 10 - 1)).mean()
        vloss = rec_loss + area_loss
        loss.append(vloss)
        aloss.append(area_loss)
        rloss.append(rec_loss)
    loss = torch.stack(loss)
    return torch.mean(loss), torch.stack(rloss), torch.stack(aloss)

########### gaussian function #############
def generate_gauss_weight(props_len, center, width):
    sigma = 6
    weight = torch.linspace(0, 1, props_len)
    weight = weight.view(1, -1).expand(center.size(0), -1).to(width.device)
    center = center.unsqueeze(-1)
    width = width.unsqueeze(-1).clamp(1e-2) / sigma
    w = 0.3989422804014327      # 1 / sqrt(2 * pi)
    weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))
    return weight/weight.max(dim=-1, keepdim=True)[0]

def negative_proposal_mining(props_len, center, width, epoch):
    sigma = 6
    gamma = 0.5
    max_epoch = 150
    def Gauss(pos, w1, c):
        w1 = w1.unsqueeze(-1).clamp(1e-2) / (sigma/2)
        c = c.unsqueeze(-1)
        w = 0.3989422804014327
        y1 = w/w1*torch.exp(-(pos-c)**2/(2*w1**2))
        return y1/y1.max(dim=-1, keepdim=True)[0]
    weight = torch.linspace(0, 1, props_len)
    weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
    left_width = torch.clamp(center-width/2, min=0)
    left_center = left_width * min(epoch/max_epoch, 1)**gamma * 0.5
    right_width = torch.clamp(1-center-width/2, min=0)
    right_center = 1 - right_width * min(epoch/max_epoch, 1)**gamma * 0.5
    left_neg_weight = Gauss(weight, left_center, left_center)
    right_neg_weight = Gauss(weight, 1-right_center, right_center)
    return left_neg_weight, right_neg_weight

def tmp_loss(c):
    return torch.sum(torch.abs(c[2:] + c[:-2] - 2 * c[1:-1])) * 0.1


##########################################################################
def cos_simi(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return torch.clamp(sim, min=0.0005, max=0.9995)


def cos_distance(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return 1 - sim


def l2_distance(embedded_fg, embedded_bg):
    N, C = embedded_fg.size()

    # embedded_fg = F.normalize(embedded_fg, dim=1)
    # embedded_bg = F.normalize(embedded_bg, dim=1)

    embedded_fg = embedded_fg.unsqueeze(1).expand(N, N, C)
    embedded_bg = embedded_bg.unsqueeze(0).expand(N, N, C)

    return torch.pow(embedded_fg - embedded_bg, 2).sum(2) / C

class SimMinLoss(nn.Module):
    def __init__(self, metric='cos', reduction='mean'):
        super(SimMinLoss, self).__init__()
        self.metric = metric
        self.reduction = reduction

    def forward(self, embedded_bg, embedded_fg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError
        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_fg)
            loss = -torch.log(1 - sim)
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)

class SimMaxLoss(nn.Module):
    def __init__(self, metric='cos', alpha=0.25, reduction='mean'):
        super(SimMaxLoss, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, embedded_bg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError

        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_bg)
            loss = -torch.log(sim)
            loss[loss < 0] = 0
            _, indices = sim.sort(descending=True, dim=1)
            _, rank = indices.sort(dim=1)
            rank = rank - 1
            rank_weights = torch.exp(-rank.float() * self.alpha)
            loss = loss * rank_weights
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


if __name__ == '__main__':
    # loc = torch.ones(4, 1, 112, 112)
    # ccam = torch.Tensor(4, 1, 112, 112)
    # multi_loss(loc, ccam)

    # center = torch.ones(4) * 0.7   # bsz*self.num_props
    # width = torch.ones(4) * 0.2   # bsz*self.num_props
    # # center = torch.tensor(0.7)
    # # width = torch.tensor(0.1)
    # weight = generate_gauss_weight(64, center, width)
    # neg_weight1, neg_weight2 = negative_proposal_mining(64, center, width, 300)
    # pos_weight = F.normalize(weight, p=1, dim=-1)
    # print(weight)

    a1 = torch.Tensor(16, 1, 32, 32)
    a2 = torch.Tensor(16, 1, 32, 32)
    iou_loss(a1, a2)