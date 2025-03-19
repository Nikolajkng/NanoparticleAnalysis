from model.UNet import UNet
from torch.utils.data import DataLoader
import numpy as np
import torch

class ModelEvaluator():
    @staticmethod
    def __get_single_image_iou(prediction: np.ndarray, ground_truth: np.ndarray):
        assert np.isin(prediction, [0, 1]).all(), "prediction must be binary image"
        assert np.isin(ground_truth, [0, 1]).all(), "ground truth must be binary image"

        intersection = np.logical_and(prediction, ground_truth).sum()
        union = np.logical_or(prediction, ground_truth).sum()
        
        iou = intersection / union if union > 0 else 0.0
        return iou
    @staticmethod
    def __get_single_image_pixel_accuracy(prediction: np.ndarray, ground_truth: np.ndarray):
        correct_pixels = (prediction == ground_truth).sum()
        total_pixels = ground_truth.numel()
        return correct_pixels.float() / total_pixels
    
    @staticmethod
    def calculate_average_iou(predictions, ground_truths):
        total_iou = 0.0
        for prediction, ground_truth in zip(predictions, ground_truths):
            total_iou += ModelEvaluator.__get_single_image_iou(prediction, ground_truth)
        return total_iou / len(predictions)
    @staticmethod
    def calculate_average_pixel_accuracy(predictions, ground_truths):
        total_pixel_accuracy = 0.0
        for prediction, ground_truth in zip(predictions, ground_truths):
            total_pixel_accuracy += ModelEvaluator.__get_single_image_pixel_accuracy(prediction, ground_truth)
        return total_pixel_accuracy / len(predictions)
    
    @staticmethod
    def get_predictions(unet: UNet, dataloader: DataLoader):
        predictions = []
        labels = []
        unet.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                input, label = data
                label = label.long().squeeze(1)
                prediction = unet.segment(input)
                predictions.append(prediction)
                labels.append(label)
        return predictions, labels

    @staticmethod
    def evaluate_model(unet: UNet, test_dataloader: DataLoader) -> tuple[float, float]:
        predictions, labels = ModelEvaluator.get_predictions(unet, test_dataloader)
        iou = ModelEvaluator.calculate_average_iou(predictions, labels)
        pixel_accuracy = ModelEvaluator.calculate_average_pixel_accuracy(predictions, labels)
        return iou.float(), pixel_accuracy.float()

        
        