from model.UNet import UNet
from torch.utils.data import DataLoader
import numpy as np
import torch
from model.DataTools import showTensor
import cv2
import matplotlib.pyplot as plt
import random
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
                label = label.long().squeeze(1)
                prediction = unet.segment(input)
                predictions.append(prediction)
                labels.append(label)
        return predictions, labels

    @staticmethod
    def plot_difference(prediction, label, iou, pixel_accuracy):
        prediction_uint8 = (np.array(prediction) * 255).astype(np.uint8).squeeze(0)
        label_uint8 = (np.array(label) * 255).astype(np.uint8).squeeze(0)

        false_positives = ((prediction_uint8 == 255) & (label_uint8 == 0))  # FP: Red
        false_negatives = ((prediction_uint8 == 0) & (label_uint8 == 255))  # FN: Blue

        overlay = np.zeros((*false_positives.shape, 3), dtype=np.uint8)

        overlay[..., 0] = false_positives * 255
        overlay[..., 2] = false_negatives * 255  # Blue channel for FN
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharex=True, sharey=True)

        axes[0].imshow(prediction_uint8, cmap='gray')
        axes[0].set_title("Prediction")

        axes[1].imshow(label_uint8, cmap='gray')
        axes[1].set_title("Label")

        axes[2].imshow(overlay)
        axes[2].set_title("Difference (FP: Red, FN: Blue)")

        fig.text(0.5, 0.95, f"IoU: {iou:.2f}   Pixel Accuracy: {pixel_accuracy:.2f}",
         ha='center', va='top', fontsize=14, bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def evaluate_model(unet: UNet, test_dataloader: DataLoader) -> tuple[float, float]:
        predictions, labels = ModelEvaluator.get_predictions(unet, test_dataloader)

        ious = ModelEvaluator.calculate_ious(predictions, labels)
        pixel_accuracies = ModelEvaluator.calculate_pixel_accuracies(predictions, labels)


        number_of_predictions_to_show = np.min([4, len(predictions)]) 
        indicies = random.sample(range(len(predictions)), number_of_predictions_to_show)
        for i in indicies:
            ModelEvaluator.plot_difference(predictions[i], labels[i], ious[i], pixel_accuracies[i])


        return np.mean(ious), np.mean(pixel_accuracies)

        
        