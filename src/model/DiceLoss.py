from torch import Module
class DiceLoss(Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, ground_truth):
        # Flatten the tensors
        prediction = prediction.view(-1)
        ground_truth = ground_truth.view(-1)

        # Calculate intersection and union
        intersection = (prediction * ground_truth).sum()
        total = prediction.sum() + ground_truth.sum()

        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)

        # Calculate Dice loss
        return 1 - dice
        