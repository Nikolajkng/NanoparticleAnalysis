import datetime
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from src.model.SegmentationDataset import SegmentationDataset
from src.model.PlottingTools import *
from src.model.DataTools import get_dataloaders, get_dataloaders_without_testset, get_normalizer
from src.model.DataAugmenter import DataAugmenter
from src.model.UNet import UNet
from src.model.ModelEvaluator import ModelEvaluator
from src.shared.ModelConfig import ModelConfig

def cv_holdout(unet: UNet, model_config: ModelConfig, input_size, stop_training_event = None, loss_callback = None, test_callback = None):
    
    # Set parameters:
    train_subset_size = 0.6
    validation_subset_size = 0.2
    print(f"Training model using holdout [train_split_size={train_subset_size}, epochs={model_config.epochs}, learnRate={model_config.learning_rate}]...")
    print("---------------------------------------------------------------------------------------")
    dataset = SegmentationDataset(model_config.images_path, model_config.masks_path)
    train_dataloader, validation_dataloader, test_dataloader = None, None, None

    if model_config.test_images_path and model_config.test_masks_path:
        train_dataloader, validation_dataloader = get_dataloaders_without_testset(dataset, train_subset_size, unet.preffered_input_size)
        test_dataset = SegmentationDataset(model_config.test_images_path, model_config.test_masks_path)
        test_dataloader = DataLoader(test_dataset, batch_size=1)
    else:
        train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(dataset, train_subset_size, validation_subset_size, unet.preffered_input_size)
    unet.normalizer = get_normalizer(torch.stack(train_dataloader.dataset.images))
    unet.train_model(
        training_dataloader=train_dataloader, 
        validation_dataloader=validation_dataloader, 
        epochs=model_config.epochs, 
        learningRate=model_config.learning_rate, 
        model_name="UNet_" + datetime.datetime.now().strftime('%d.%m.%Y_%H-%M-%S')+".pt",
        cross_validation="holdout",
        with_early_stopping=model_config.with_early_stopping,
        stop_training_event=stop_training_event,
        loss_callback=loss_callback
        )
    
    iou, pixel_accuracy = ModelEvaluator.evaluate_model(unet, test_dataloader, test_callback)
    return iou, pixel_accuracy

