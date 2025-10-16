class EvaluationResult:
    def __init__(self, iou_scores, dice_scores):
        import numpy as np
        self.iou_scores = iou_scores
        self.dice_scores = dice_scores

        self.mean_iou = np.mean(iou_scores)
        self.mean_dice = np.mean(dice_scores)
        self.min_iou = np.min(iou_scores)
        self.min_dice = np.min(dice_scores)
        self.max_iou = np.max(iou_scores)
        self.max_dice = np.max(dice_scores)

    def __len__(self):
        return len(self.iou_scores)

    