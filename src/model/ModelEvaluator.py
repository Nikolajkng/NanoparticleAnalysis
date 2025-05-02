from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
import random

from src.model.UNet import UNet
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
    def calculate_ious(predictions, ground_truths):
        ious = []
        for prediction, ground_truth in zip(predictions, ground_truths):
            ious.append(ModelEvaluator.__get_single_image_iou(prediction, ground_truth))
        return ious
    @staticmethod
    def calculate_pixel_accuracies(predictions, ground_truths):
        accuracies = []
        for prediction, ground_truth in zip(predictions, ground_truths):
            accuracies.append(ModelEvaluator.__get_single_image_pixel_accuracy(prediction, ground_truth))
        return accuracies
    
    @staticmethod
    def get_predictions(unet: UNet, dataloader: DataLoader):
        predictions = []
        labels = []
        unet.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                input, label = data
                input, label = input.to(unet.device), label.to(unet.device)
                label = label.long().squeeze(1)
                prediction = unet.segment(input)
                predictions.append(prediction)
                labels.append(label)
        return predictions, labels

    @staticmethod
    def evaluate_model(unet: UNet, test_dataloader: DataLoader, test_callback = None) -> tuple[float, float]:
        predictions, labels = ModelEvaluator.get_predictions(unet, test_dataloader)

        ious = ModelEvaluator.calculate_ious(predictions, labels)
        pixel_accuracies = ModelEvaluator.calculate_pixel_accuracies(predictions, labels)


        number_of_predictions_to_show = np.min([4, len(predictions)]) 
        indicies = random.sample(range(len(predictions)), number_of_predictions_to_show)
        if not test_callback:
            return np.mean(ious), np.mean(pixel_accuracies)
        
        try:
            for i in indicies:
                test_callback(predictions[i], labels[i], ious[i], pixel_accuracies[i])
        except Exception:
            return np.mean(ious), np.mean(pixel_accuracies)

        return np.mean(ious), np.mean(pixel_accuracies)

        
        