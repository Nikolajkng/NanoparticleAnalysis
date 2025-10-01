import torch
from torch.nn import Module
import numpy as np
import torch.nn.functional as F
class DiceLoss(Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = F.softmax(prediction, dim=1)

        target_one_hot = F.one_hot(target, num_classes=prediction.shape[1])  # (B, H, W, C)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        prediction = prediction.contiguous().view(prediction.shape[0], prediction.shape[1], -1)
        target_one_hot = target_one_hot.contiguous().view(target_one_hot.shape[0], target_one_hot.shape[1], -1)


        intersection = (prediction * target_one_hot).sum(dim=2)
        total = prediction.sum(dim=2) + target_one_hot.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        return 1 - dice.mean()
    
class BinarySymmetricDiceLoss(Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = torch.softmax(prediction, dim=1)[:, 1, :, :]  
        target = target.float()

        # Flatten
        prediction = prediction.contiguous().view(-1)
        target = target.contiguous().view(-1)

        # Foreground Dice
        fg_num = (prediction * target).sum()
        fg_denom = prediction.sum() + target.sum()

        # Background Dice
        bg_pred = 1 - prediction
        bg_target = 1 - target
        bg_num = (bg_pred * bg_target).sum()
        bg_denom = bg_pred.sum() + bg_target.sum()

        # Total loss
        dice_fg = (fg_num + self.smooth) / (fg_denom + self.smooth)
        dice_bg = (bg_num + self.smooth) / (bg_denom + self.smooth)

        return 1 - (dice_fg + dice_bg)
        
class WeightedDiceLoss(Module):
    def __init__(self, class_weights, smooth=1e-6):
        super(WeightedDiceLoss, self).__init__()
        self.class_weights = torch.tensor(class_weights).float()
        self.smooth = smooth

    def forward(self, prediction, target):
        # prediction: (B, C, H, W), logits
        # target: (B, H, W), class indices

        prediction = F.softmax(prediction, dim=1)
        target_one_hot = F.one_hot(target, num_classes=prediction.shape[1])
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        # Flatten
        prediction = prediction.contiguous().view(prediction.shape[0], prediction.shape[1], -1)
        target_one_hot = target_one_hot.contiguous().view(target_one_hot.shape[0], target_one_hot.shape[1], -1)

        # Per-class intersection and total
        intersection = (prediction * target_one_hot).sum(dim=2)
        total = prediction.sum(dim=2) + target_one_hot.sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)  # (B, C)
        
        # Average over batch and apply weights per class
        dice = dice.mean(dim=0)  # mean over batch → shape (C,)
        self.class_weights = self.class_weights.to(dice.device) 
        weighted_dice = (self.class_weights * dice).sum() / self.class_weights.sum()
        return 1 - weighted_dice
        