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

# ==================== NEW LOSS FUNCTIONS ====================

class FocalLoss(Module):
    """
    Focal Loss for addressing class imbalance.
    Great for nanoparticle segmentation where particles are small vs background.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, prediction, target):
        # prediction: (B, C, H, W), target: (B, H, W)
        ce_loss = F.cross_entropy(prediction, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(Module):
    """
    Combination of Cross Entropy and Dice Loss.
    CE helps with classification, Dice helps with overlap.
    """
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, prediction, target):
        ce = self.ce_loss(prediction, target)
        dice = self.dice_loss(prediction, target)
        return self.ce_weight * ce + self.dice_weight * dice

class TverskyLoss(Module):
    """
    Tversky Loss - great for unbalanced datasets.
    Controls false positives vs false negatives.
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # Controls false positives
        self.beta = beta    # Controls false negatives
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = F.softmax(prediction, dim=1)
        target_one_hot = F.one_hot(target, num_classes=prediction.shape[1])
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        # Flatten
        prediction = prediction.contiguous().view(prediction.shape[0], prediction.shape[1], -1)
        target_one_hot = target_one_hot.contiguous().view(target_one_hot.shape[0], target_one_hot.shape[1], -1)

        # True Positives, False Positives & False Negatives
        TP = (prediction * target_one_hot).sum(dim=2)
        FP = (prediction * (1 - target_one_hot)).sum(dim=2)
        FN = ((1 - prediction) * target_one_hot).sum(dim=2)

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky.mean()

class BoundaryLoss(Module):
    """
    Emphasizes particle boundaries - important for nanoparticle separation.
    """
    def __init__(self, boundary_weight=5.0):
        super(BoundaryLoss, self).__init__()
        self.boundary_weight = boundary_weight
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, prediction, target):
        # Standard CE loss
        ce = self.ce_loss(prediction, target)
        
        # Extract boundaries using gradients
        boundary_target = self._extract_boundaries(target)
        boundary_pred = self._extract_boundaries(torch.argmax(prediction, dim=1))
        
        # Boundary loss
        boundary_loss = F.binary_cross_entropy_with_logits(
            boundary_pred.float(), boundary_target.float()
        )
        
        return ce + self.boundary_weight * boundary_loss
    
    def _extract_boundaries(self, mask):
        """Extract boundaries using Sobel-like filters"""
        # Convert to float for gradient computation
        mask_float = mask.float()
        
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=mask.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=mask.device)
        
        # Add batch and channel dimensions
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        if len(mask_float.shape) == 3:
            mask_float = mask_float.unsqueeze(1)  # Add channel dimension
        
        # Apply filters
        grad_x = F.conv2d(mask_float, sobel_x, padding=1)
        grad_y = F.conv2d(mask_float, sobel_y, padding=1)
        
        # Gradient magnitude
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold to get boundaries
        boundaries = (gradient_magnitude > 0.1).float()
        
        return boundaries.squeeze(1)  # Remove channel dimension

class SizePenaltyLoss(Module):
    """
    Penalizes unrealistic particle sizes for nanoparticle segmentation.
    """
    def __init__(self, expected_size_range=(50, 500), penalty_weight=0.1):
        super(SizePenaltyLoss, self).__init__()
        self.min_size, self.max_size = expected_size_range
        self.penalty_weight = penalty_weight
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, prediction, target):
        # Standard CE loss
        ce = self.ce_loss(prediction, target)
        
        # Size penalty
        pred_binary = torch.argmax(prediction, dim=1)
        size_penalty = self._calculate_size_penalty(pred_binary)
        
        return ce + self.penalty_weight * size_penalty
    
    def _calculate_size_penalty(self, pred_binary):
        """Calculate penalty for unrealistic particle sizes"""
        penalty = 0.0
        batch_size = pred_binary.shape[0]
        
        for b in range(batch_size):
            mask = pred_binary[b].cpu().numpy()
            # Count connected components (particles)
            from scipy.ndimage import label
            labeled, num_features = label(mask)
            
            for i in range(1, num_features + 1):
                component_size = (labeled == i).sum()
                
                if component_size < self.min_size:
                    penalty += (self.min_size - component_size) / self.min_size
                elif component_size > self.max_size:
                    penalty += (component_size - self.max_size) / self.max_size
        
        return torch.tensor(penalty / batch_size, device=pred_binary.device)