from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
import random

from src.model.UNet import UNet
from src.model.DataTools import center_crop, construct_image_from_patches, mirror_fill, extract_slices
class ModelEvaluator():
    @staticmethod
    def __get_single_image_iou(prediction: np.ndarray, ground_truth: np.ndarray):
        assert np.isin(prediction, [0, 1]).all(), "prediction must be binary image"
        assert np.isin(ground_truth, [0, 1]).all(), "ground truth must be binary image"

        if ground_truth.sum() == 0: # No ground truth / only background -> use pixel accuracy instead
            return ModelEvaluator.__get_single_image_pixel_accuracy(prediction, ground_truth)
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
                
                stride_length = unet.preffered_input_size[0]*4//5
                tensor_mirror_filled = mirror_fill(input, unet.preffered_input_size, (stride_length,stride_length))
                patches = extract_slices(tensor_mirror_filled, unet.preffered_input_size, (stride_length,stride_length))
                segmentations = np.empty((patches.shape[0], 2, patches.shape[2], patches.shape[3]), dtype=patches.dtype)
                unet.to(input.device)
                patches_tensor = torch.tensor(patches, dtype=input.dtype, device=input.device)
                segmentations = unet(patches_tensor).cpu().detach().numpy()
                segmented_image = construct_image_from_patches(segmentations, tensor_mirror_filled.shape[2:], (stride_length,stride_length))
                segmented_image = center_crop(segmented_image, (input.shape[2], input.shape[3])).argmax(axis=1)
                #prediction = unet.segment(input)
                predictions.append(torch.tensor(segmented_image, dtype=input.dtype, device=input.device))
                labels.append(label)
        return predictions, labels

    @staticmethod
    def evaluate_model(unet: UNet, test_dataloader: DataLoader, test_callback = None) -> tuple[float, float]:
        predictions, labels = ModelEvaluator.get_predictions(unet, test_dataloader)
        predictions = [pred.cpu() for pred in predictions]
        labels = [label.cpu() for label in labels]
        ious = ModelEvaluator.calculate_ious(predictions, labels)
        pixel_accuracies = ModelEvaluator.calculate_pixel_accuracies(predictions, labels)


        number_of_predictions_to_show = np.min([5, len(predictions)]) 
        indicies = random.sample(range(len(predictions)), number_of_predictions_to_show)
        if not test_callback:
            return predictions, labels, np.mean(ious), np.mean(pixel_accuracies)
        
        try:
            for i in indicies:
                test_callback(predictions[i], labels[i], ious[i], pixel_accuracies[i])
        except Exception:
            return predictions, labels, np.mean(ious), np.mean(pixel_accuracies)

        return predictions, labels, np.mean(ious), np.mean(pixel_accuracies)

        
        