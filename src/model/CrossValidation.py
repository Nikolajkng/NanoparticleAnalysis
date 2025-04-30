import datetime
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from model.SegmentationDataset import SegmentationDataset
from model.PlottingTools import *
from model.DataTools import get_dataloaders, get_dataloaders_without_testset
from model.DataAugmenter import DataAugmenter
from model.UNet import UNet
from model.ModelEvaluator import ModelEvaluator
from shared.ModelConfig import ModelConfig

def cv_holdout(unet: UNet, model_config: ModelConfig, input_size, stop_training_event = None, loss_callback = None):
    
    # Set parameters:
    train_subset_size = 0.7
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
    
    iou, pixel_accuracy = ModelEvaluator.evaluate_model(unet, test_dataloader)
    return iou, pixel_accuracy


def cv_kfold(unet, images_path, masks_path):
    fold_results = []   
    
    # Set parameters:
    K1 = 2  
    K2 = 2 
    learning_rates = [0.001, 0.0001, 0.00001]  
    S = len(learning_rates)
    epochs = 2  
    print(f"\nTraining model using two-level cross-validation with K1={K1} and K2={K2}")

    # Load data
    dataset = SegmentationDataset(images_path, masks_path)
    data_augmenter = DataAugmenter()
    dataset = data_augmenter.augment_dataset(dataset)
    dataset_size = len(dataset)  # FIXED

    cv = KFold(n_splits=K1, shuffle=True, random_state=42)
    
    fold_results = []

    for outerfold, (par_idx, test_idx) in enumerate(cv.split(np.arange(dataset_size))): 
        print(f"\n ---------------- Outer Fold {outerfold+1}/{K1} ----------------") 

        # Outer split: partition & test set
        par_split = Subset(dataset, par_idx.tolist())
        test_split = Subset(dataset, test_idx.tolist())

        outer_train_dataloader, outer_val_dataloader = get_dataloaders_without_testset(par_split, 0.7)
        outer_test_dataloader = DataLoader(test_split, batch_size=1, shuffle=False)

        # Inner CV: Find the best learning rate
        best_val_loss = np.inf
        best_s = None
        val_results = {s: [] for s in range(1, S+1)}

        for innerfold, (train_idx, val_idx) in enumerate(cv.split(par_idx)): 
            print(f"\n ------------ Inner Fold {innerfold+1}/{K2} -------------") 

            train_split = Subset(par_split, train_idx.tolist())
            inner_test_data = Subset(par_split, val_idx.tolist())

            inner_train_dataloader, inner_validation_dataloader = get_dataloaders_without_testset(train_split, 0.7)
            inner_test_dataloader = DataLoader(inner_test_data, batch_size=1, shuffle=False)

            for s in range(1, S+1):
                unet = UNet()
                model_name = f"UNet{s}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                learning_rate = learning_rates[s-1]  

                print(f"\nTraining model {s} with name: {model_name} and learning rate: {learning_rate}")
                unet.train_model(
                    training_dataloader=inner_train_dataloader,
                    validation_dataloader=inner_validation_dataloader,
                    epochs=epochs,
                    learningRate=learning_rate,
                    model_name=model_name,
                    cross_validation="kfold"
                )

                validation_loss = unet.get_validation_loss(inner_test_dataloader)
                val_results[s].append(validation_loss)

        # Compute weighted validation loss
        weighted_val_loss = {s: sum(val_results[s]) / len(val_results[s]) for s in range(1, S+1)}
        best_s = min(weighted_val_loss, key=weighted_val_loss.get)
        best_learning_rate = learning_rates[best_s-1]
        print(f"\nSelected best model: UNet{best_s} with weighted validation loss: {weighted_val_loss[best_s]:.5f} and learning rate {best_learning_rate}")

        # Train best model on the full partitioned dataset
        best_model = UNet()
        best_model.train_model(
            training_dataloader=outer_train_dataloader,
            validation_dataloader=outer_val_dataloader, 
            epochs=epochs,
            learningRate=best_learning_rate,
            model_name=f"Best_UNet_Outer{outerfold+1}",
            cross_validation="kfold"
        )

        # **NEW STEP: Evaluate best model on the test set**
        test_loss = best_model.get_validation_loss(outer_test_dataloader)
        fold_results.append((len(test_split), test_loss))  # Store test errors

    # **Final Generalization Error Computation**
    gen_error_estimate = sum(test_size * test_loss for test_size, test_loss in fold_results) / dataset_size

    print(f"\n############## K-Fold Cross Validation Summary ##############")
    for i, (size, loss) in enumerate(fold_results):
        print(f"Outer Fold {i+1}: Test Loss = {loss:.5f}")

    print(f"\n############## Estimated Generalization Error: {gen_error_estimate:.5f} ##############")

