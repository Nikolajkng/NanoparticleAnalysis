from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
import random

class ModelEvaluator():
    @staticmethod
    def __get_single_image_iou(prediction: np.ndarray, ground_truth: np.ndarray):
        assert np.isin(prediction, [0, 1]).all(), "prediction must be binary image"
        assert np.isin(ground_truth, [0, 1]).all(), "ground truth must be binary image"
        epsilon = 1e-6
        intersection = np.logical_and(prediction, ground_truth).sum()
        union = np.logical_or(prediction, ground_truth).sum()
        
        iou = (intersection + epsilon) / (union + epsilon)
        return iou
    
    @staticmethod
    def __get_single_image_dice_score(prediction: np.ndarray, ground_truth: np.ndarray):
        assert np.isin(prediction, [0, 1]).all(), "prediction must be binary image"
        assert np.isin(ground_truth, [0, 1]).all(), "ground truth must be binary image"
        epsilon = 1e-6
        intersection = np.logical_and(prediction, ground_truth).sum()
        dice_score = (2 * intersection + epsilon) / (prediction.sum() + ground_truth.sum() + epsilon)
        return dice_score

    @staticmethod
    def calculate_ious(predictions, ground_truths):
        ious = []
        for prediction, ground_truth in zip(predictions, ground_truths):
            ious.append(ModelEvaluator.__get_single_image_iou(prediction, ground_truth))
        return ious

    @staticmethod
    def calculate_dice_scores(predictions, ground_truths):
        dice_scores = []
        for prediction, ground_truth in zip(predictions, ground_truths):
            dice_scores.append(ModelEvaluator.__get_single_image_dice_score(prediction, ground_truth))
        return dice_scores
    
    def get_predictions(unet, dataloader: DataLoader):
        from src.model.DataTools import binarize_segmentation_output, center_crop, construct_image_from_patches, mirror_fill, extract_slices
        inputs = []
        predictions = []
        labels = []
        unet.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                input, label = data           
                input, label = input.to(unet.device), label.to(unet.device)
                label = (label > 0.5).long().squeeze(1)
                stride_length = unet.preferred_input_size[0]*4//5
                tensor_mirror_filled = mirror_fill(input, unet.preferred_input_size, (stride_length,stride_length))
                patches = extract_slices(tensor_mirror_filled, unet.preferred_input_size, (stride_length,stride_length))
                segmentations = np.empty((patches.shape[0], 2, patches.shape[2], patches.shape[3]), dtype=patches.dtype)
                unet.to(input.device)
                patches_tensor = torch.tensor(patches, dtype=input.dtype, device=input.device)

                if unet.device.type == 'cuda':
                    with torch.autocast("cuda"):
                        segmentations = unet(patches_tensor).cpu().detach().numpy()
                else:
                    segmentations = unet(patches_tensor).cpu().detach().numpy()
                segmented_image = construct_image_from_patches(segmentations, tensor_mirror_filled.shape[2:], (stride_length,stride_length))
                segmented_image = center_crop(segmented_image, (input.shape[2], input.shape[3]))
                segmented_image = binarize_segmentation_output(segmented_image)
                predictions.append(torch.tensor(segmented_image, dtype=input.dtype, device=input.device))
                labels.append(label)
                inputs.append(input.cpu())
        return inputs, predictions, labels

    @staticmethod
    def evaluate_model(unet, test_dataloader: DataLoader, test_callback = None) -> tuple[float, float]:
        inputs, predictions, labels = ModelEvaluator.get_predictions(unet, test_dataloader)
        predictions = [pred.cpu() for pred in predictions]
        labels = [label.cpu() for label in labels]
        ious = ModelEvaluator.calculate_ious(predictions, labels)
        dice_scores = ModelEvaluator.calculate_dice_scores(predictions, labels)

        print(f"IOUS: {ious}")
        print(f"Dice scores: {dice_scores}")

        number_of_predictions_to_show = np.min([5, len(predictions)]) 
        indicies = random.sample(range(len(predictions)), number_of_predictions_to_show)
        if not test_callback:
            return np.mean(ious), np.mean(dice_scores)
        try:
            for i in indicies:
                test_callback(inputs[i], predictions[i], labels[i], ious[i], dice_scores[i])
        except Exception as e:
            return np.mean(ious), np.mean(dice_scores)

        return np.mean(ious), np.mean(dice_scores)

        
        