"""
This part is the available loss function
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class LossMulti:
    def __init__(self, jaccard_weight=0, class_weights=None, num_classes=1):
        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights.astype(np.float32)).cuda()
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss(weight=nll_weight)  # Not include softmax
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
        return loss

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(torch.log(inputs), targets)  

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets): # 包含了log_softmax函数，调用时网络输出层不需要加log_softmax
        return self.nll_loss((1 - F.softmax(inputs,1)) ** self.gamma * F.log_softmax(inputs,1), targets)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss
class DiceLoss_1(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super(DiceLoss_1, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, probs, target):
        """
        probs: (B, 2, H, W) —— softmax output (background, vessel)
        target: (B, H, W) —— ground truth with values in {0, 1}, possibly 255 for ignore
        """
        # 只取血管类（class=1）的概率
        pred = probs[:, 1, :, :]  # (B, H, W)

        # 构建二值化目标：1 表示血管，0 表示非血管
        target_fg = (target == 1).float()  # (B, H, W)

        # 处理 ignore_index（如 255）
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            pred = pred * valid_mask
            target_fg = target_fg * valid_mask

        # 展平所有像素
        pred = pred.contiguous().view(-1)
        target_fg = target_fg.contiguous().view(-1)

        intersection = (pred * target_fg).sum()
        dice_coef = (2. * intersection + self.smooth) / (
            pred.sum() + target_fg.sum() + self.smooth
        )
        return 1.0 - dice_coef
class CE_Dice_Loss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(CE_Dice_Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = CrossEntropyLoss2d()
        self.dice_loss = DiceLoss_1()

    def forward(self, softmax_probs, targets):

        ce = self.ce_loss(softmax_probs, targets)
        dice = self.dice_loss(softmax_probs, targets)
        return self.alpha * ce + self.beta * dice