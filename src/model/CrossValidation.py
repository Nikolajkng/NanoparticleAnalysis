import copy
import datetime
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from src.model.SegmentationDataset import SegmentationDataset
from src.model.PlottingTools import *
from src.model.DataTools import get_dataloaders, get_dataloaders_kfold, get_dataloaders_without_testset, process_and_slice, slice_dataset_in_four
from src.model.DataAugmenter import DataAugmenter
from src.model.UNet import UNet
from src.model.ModelEvaluator import ModelEvaluator
from src.shared.ModelConfig import ModelConfig

def cv_holdout(unet: UNet, model_config: ModelConfig, input_size, stop_training_event = None, loss_callback = None, testing_callback = None):
    # Set parameters:
    train_subset_size = 0.7
    validation_subset_size = 0.1
    print(f"Training model using holdout [train_split_size={train_subset_size}, epochs={model_config.epochs}, learnRate={model_config.learning_rate}]...")
    print("---------------------------------------------------------------------------------------")
    dataset = SegmentationDataset(model_config.images_path, model_config.masks_path)
    train_dataloader, validation_dataloader, test_dataloader = None, None, None

    if model_config.test_images_path and model_config.test_masks_path:
        train_dataloader, validation_dataloader = get_dataloaders_without_testset(dataset, train_subset_size, unet.preferred_input_size)
        test_dataset = SegmentationDataset(model_config.test_images_path, model_config.test_masks_path)
        test_dataloader = DataLoader(test_dataset, batch_size=1)
    else:
        train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(dataset, train_subset_size, validation_subset_size, unet.preferred_input_size)

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
    
    iou, pixel_accuracy = ModelEvaluator.evaluate_model(unet, test_dataloader, testing_callback)
    return iou, pixel_accuracy


def cv_kfold(images_path, masks_path):
    fold_results = []   
    
    # Set parameters:
    K1 = 3  
    K2 = 5
    learning_rates = [0.001, 0.0001, 0.00001]  
    S = len(learning_rates)
    models = [UNet() for _ in range(S)]
    epochs = 15  
    print(f"\nTraining model using two-level cross-validation with K1={K1} and K2={K2}")

    # Load data
    dataset = SegmentationDataset(images_path, masks_path)
    dataset = slice_dataset_in_four(dataset)
    dataset_size = len(dataset)
    sliced_dataset_size = len(process_and_slice(dataset))

    cv_outer = KFold(n_splits=K1, shuffle=True, random_state=42)

    fold_results = {"test_sizes": [], "test_losses": [], "test_ious": []}
    model_losses = {s+1: [] for s in range(S)}
    best_models = []
    for outerfold, (par_idx, test_idx) in enumerate(cv_outer.split(np.arange(dataset_size))): 
        fold_model_losses, best_model = outer_fold(outerfold, dataset, par_idx, test_idx, K1, K2, learning_rates, epochs, S, models, fold_results)
        best_models.append(best_model)
        for i, loss in enumerate(fold_model_losses):
            model_losses[i+1].append(loss) 

    # **Final Generalization Error Computation**

    #print(inner_test_results)
    
    test_sizes = fold_results["test_sizes"]
    test_losses = fold_results["test_losses"]
    test_ious = fold_results["test_ious"]
    gen_error_estimate = sum(test_size * test_loss for test_size, test_loss in zip(test_sizes, test_losses)) / sliced_dataset_size 

    print(f"\n############## K-Fold Cross Validation Summary ##############")
    for i, loss in enumerate(fold_results["test_losses"]):
        print(f"Outer Fold {i+1}: Test Loss = {loss:.5f}, Best Learning Rate = {learning_rates[best_models[i]]}")

    print("############## Model Losses ##############")
    for idx, (key, value) in enumerate(model_losses.items()):
        print(f"Model {key} (learning rate={learning_rates[key-1]}): {value}")

    print(f"\n############## Estimated Generalization Error: {gen_error_estimate:.5f} ##############")

def outer_fold(idx, dataset, par_idx, test_idx, K1, K2, learning_rates, epochs, S, models, fold_results):
    print(f"\n ---------------- Outer Fold {idx+1}/{K1} ----------------") 
    cv_inner = KFold(n_splits=K2, shuffle=True, random_state=42)
    # Outer split: partition & test set
    par_split = Subset(dataset, par_idx.tolist())
    sliced_par_split_size = len(process_and_slice(par_split))
    test_split = Subset(dataset, test_idx.tolist())

    # Inner CV: Find the best learning rate
    best_val_loss = np.inf
    best_s = None
    inner_test_results = {s: {"test_sizes": [], "test_losses": [], "test_ious": []} for s in range(1, S+1)}

    for innerfold, (train_idx, inner_test_idx) in enumerate(cv_inner.split(par_idx)): 
        inner_fold(innerfold, K2, par_split, learning_rates, epochs, train_idx, inner_test_idx, inner_test_results, models)

    # Compute weighted validation loss
    E_gen_s = []
    for s in range(1, S+1):
        test_sizes = inner_test_results[s]["test_sizes"]
        test_losses = inner_test_results[s]["test_losses"]
        test_ious = inner_test_results[s]["test_ious"]
        gen_error_estimate = sum(test_size * test_loss for test_size, test_loss in zip(test_sizes, test_losses)) / sliced_par_split_size
        E_gen_s.append(gen_error_estimate)
    best_s = E_gen_s.index(min(E_gen_s))
    best_learning_rate = learning_rates[best_s]
    print(f"\nSelected best model: UNet{best_s+1} with weighted validation loss: {E_gen_s[best_s]:.5f} and learning rate {best_learning_rate}")


    outer_train_dataloader, outer_val_dataloader = get_dataloaders_kfold(par_split, 0.8, (256, 256))
    test_split = process_and_slice(test_split)
    outer_test_dataloader = DataLoader(test_split, batch_size=1, shuffle=False)

    # Train best model on the full partitioned dataset
    best_model = UNet()
    best_model.train_model(
        training_dataloader=outer_train_dataloader,
        validation_dataloader=outer_val_dataloader, 
        epochs=epochs,
        learningRate=best_learning_rate,
        model_name=f"Best_UNet_Outer{idx+1}",
        cross_validation="kfold",
        with_early_stopping=False
    )

    # **NEW STEP: Evaluate best model on the test set**
    test_loss = best_model.get_validation_loss(outer_test_dataloader)
    test_iou = ModelEvaluator.evaluate_model(best_model, outer_test_dataloader)[0]
    fold_results["test_sizes"].append(len(test_split))
    fold_results["test_losses"].append(test_loss)
    fold_results["test_ious"].append(test_iou)

    return E_gen_s, best_s

def inner_fold(idx, K2, par_split, learning_rates, epochs, train_idx, test_idx, test_results, models):
    print(f"\n ------------ Inner Fold {idx+1}/{K2} -------------") 
    train_split = Subset(par_split, train_idx.tolist())
    inner_test_data = Subset(par_split, test_idx.tolist())

    inner_train_dataloader, inner_validation_dataloader = get_dataloaders_kfold(train_split, 0.8, (256, 256))
    
    inner_test_data = process_and_slice(inner_test_data)
    inner_test_dataloader = DataLoader(inner_test_data, batch_size=1, shuffle=False)

    for s in range(1, len(learning_rates)+1):
        unet: UNet = copy.deepcopy(models[s-1])
        model_name = f"UNet{s}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        learning_rate = learning_rates[s-1]  

        print(f"\nTraining model {s} with name: {model_name} and learning rate: {learning_rate}")
        unet.train_model(
            training_dataloader=inner_train_dataloader,
            validation_dataloader=inner_validation_dataloader,
            epochs=epochs,
            learningRate=learning_rate,
            model_name=model_name,
            cross_validation="kfold",
            with_early_stopping=False,
        )

        test_loss = unet.get_validation_loss(inner_test_dataloader)
        test_iou = ModelEvaluator.evaluate_model(unet, inner_test_dataloader)[0]
        test_results[s]["test_sizes"].append(len(inner_test_data))
        test_results[s]["test_losses"].append(test_loss)
        test_results[s]["test_ious"].append(test_iou)