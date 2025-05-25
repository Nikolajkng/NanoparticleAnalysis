import copy
import os
import datetime
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, random_split
from src.model.SegmentationDataset import SegmentationDataset
from src.model.PlottingTools import *
from src.model.DataTools import get_dataloaders, get_dataloaders_kfold, get_dataloaders_kfold_already_split, get_dataloaders_without_testset, process_and_slice, slice_dataset_in_four, get_normalizer
from src.model.DataAugmenter import DataAugmenter
from src.model.ModelEvaluator import ModelEvaluator
from src.shared.ModelConfig import ModelConfig

def cv_holdout(unet, model_config: ModelConfig, input_size, stop_training_event = None, loss_callback = None, testing_callback = None):
    # Set parameters:
    train_subset_size = 0.6
    validation_subset_size = 0.2
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
    normalizer = get_normalizer(train_dataloader.dataset.dataset)
    unet.normalizer = normalizer
    unet.train_model(
        training_dataloader=train_dataloader, 
        validation_dataloader=validation_dataloader, 
        epochs=model_config.epochs, 
        learningRate=model_config.learning_rate, 
        model_name="UNet_" + datetime.datetime.now().strftime('%d.%m.%Y_%H-%M-%S')+".pt",
        cross_validation="holdout",
        with_early_stopping=model_config.with_early_stopping,
        loss_function="cross_entropy",
        stop_training_event=stop_training_event,
        loss_callback=loss_callback
        )
    
    iou, dice_score = ModelEvaluator.evaluate_model(unet, test_dataloader, testing_callback)
    return iou, dice_score


def cv2_kfold(images_path, masks_path):
    fold_results = []   
    
    # Set parameters:
    K1 = 3
    K2 = 5
    learning_rates = [0.001, 0.0001, 0.00001]  
    #loss_functions = ["cross_entropy", "dice"]
    S = len(learning_rates)#len(learning_rates)
    #models = [UNet() for _ in range(S)]
    epochs = 100
    print(f"\nTraining model using two-level cross-validation with K1={K1} and K2={K2}")

    # Load data
    dataset = SegmentationDataset(images_path, masks_path)
    dataset = slice_dataset_in_four(dataset)
    dataset_size = len(dataset)
    sliced_dataset_size = len(process_and_slice(dataset))

    cv_outer = KFold(n_splits=K1, shuffle=True, random_state=42)

    fold_results = {"test_sizes": [], "test_losses": [], "test_ious": [], "test_dice_scores": []}
    model_ious = {s+1: [] for s in range(S)}
    best_models = []
    for outerfold, (par_idx, test_idx) in enumerate(cv_outer.split(np.arange(dataset_size))): 
        fold_model_ious, best_model = outer_fold(outerfold, dataset, par_idx, test_idx, K1, K2, learning_rates, epochs, S, fold_results)
        best_models.append(best_model)
        for i, iou in enumerate(fold_model_ious):
            model_ious[i+1].append(iou) 

    # **Final Generalization Error Computation**
        
    test_sizes = fold_results["test_sizes"]
    test_losses = fold_results["test_losses"]
    test_ious = fold_results["test_ious"]
    test_dice_scores = fold_results["test_dice_scores"]
    gen_error_estimate = sum(test_size * test_iou for test_size, test_iou in zip(test_sizes, test_ious)) / sliced_dataset_size 

    print(f"\n############## K-Fold Cross Validation Summary ##############")
    for i, iou in enumerate(fold_results["test_ious"]):
        print(f"Outer Fold {i+1}: Test IOU = {iou:.5f}, Best Learning Rate = {learning_rates[best_models[i]]}")

    print("############## Model Losses ##############")
    for idx, (key, value) in enumerate(model_ious.items()):
        print(f"Model {key} (Learning rate ={learning_rates[key-1]}): {[float(x) for x in value]}")

    print(f"\n############## Estimated Generalization Error: {gen_error_estimate:.5f} ##############")

    log_outer_folds_results(fold_results, learning_rates, best_models, model_ious, gen_error_estimate)

def cv_kfold(images_path, masks_path):
    fold_results = []   
    
    # Set parameters:
    K = 5
    learning_rates = [0.001, 0.0001, 0.00001]  
    #loss_functions = ["cross_entropy", "dice2"]#, "dice", "weighted_cross_entropy", "weighted_dice"] 
    augmentations = [(True, True, False, False, False, False)]
    random_cropping = [False, True]
    S = len(learning_rates)#len(learning_rates)
    #models = [UNet() for _ in range(S)]
    epochs = 100
    print(f"\nTraining model using one-level cross-validation with K={K}")

    # Load data
    dataset = SegmentationDataset(images_path, masks_path)
    dataset = slice_dataset_in_four(dataset)
    dataset_size = len(dataset)
    sliced_dataset_size = len(process_and_slice(dataset))

    cv = KFold(n_splits=K, shuffle=True)

    fold_results = {s: {"test_sizes": [], "test_losses": [], "test_ious": [], "test_dice_scores": []} for s in range(1, S+1)}
    for fold, (par_idx, test_idx) in enumerate(cv.split(np.arange(dataset_size))): 
        inner_fold(fold, K, dataset, learning_rates, epochs, par_idx, test_idx, fold_results)

    
    E_gen_loss_s = []
    E_gen_iou_s = []
    E_gen_dice_s = []
    for s in range(1, S+1):
        test_sizes = fold_results[s]["test_sizes"]
        test_losses = fold_results[s]["test_losses"]
        test_ious = fold_results[s]["test_ious"]
        test_dice_scores = fold_results[s]["test_dice_scores"]
        gen_error_estimate_loss = sum(test_size * test_loss for test_size, test_loss in zip(test_sizes, test_losses)) / sliced_dataset_size
        gen_error_estimate_iou = np.mean(test_ious)#sum(test_size * test_iou for test_size, test_iou in zip(test_sizes, test_ious)) / sliced_dataset_size
        gen_error_estimate_dice = np.mean(test_dice_scores)#sum(test_size * test_dice for test_size, test_dice in zip(test_sizes, test_dice_scores)) / sliced_dataset_size
        E_gen_loss_s.append(gen_error_estimate_loss)
        E_gen_iou_s.append(gen_error_estimate_iou)
        E_gen_dice_s.append(gen_error_estimate_dice)
    best_s = E_gen_iou_s.index(max(E_gen_iou_s))
    best_parameter = learning_rates[best_s]

    print(f"\nSelected best model: UNet{best_s+1} with Mean IOU: {E_gen_iou_s[best_s]:.5f} and loss function: {best_parameter}")

    log_one_layer_cv_results(learning_rates, fold_results, best_parameter)

def outer_fold(idx, dataset, par_idx, test_idx, K1, K2, parameters, epochs, S, fold_results):
    print(f"\n ---------------- Outer Fold {idx+1}/{K1} ----------------") 
    cv_inner = KFold(n_splits=K2, shuffle=True, random_state=42)
    # Outer split: partition & test set
    par_split = Subset(dataset, par_idx.tolist())
    sliced_par_split_size = len(process_and_slice(par_split))
    test_split = Subset(dataset, test_idx.tolist())

    # Inner CV: Find the best learning rate
    best_val_loss = np.inf
    best_s = None
    inner_test_results = {s: {"test_sizes": [], "test_losses": [], "test_ious": [], "test_dice_scores": []} for s in range(1, S+1)}

    for innerfold, (train_idx, inner_test_idx) in enumerate(cv_inner.split(par_idx)): 
        inner_fold(innerfold, K2, par_split, parameters, epochs, train_idx, inner_test_idx, inner_test_results)

    # Compute weighted validation loss
    E_gen_iou_s = []
    for s in range(1, S+1):
        test_sizes = inner_test_results[s]["test_sizes"]
        test_losses = inner_test_results[s]["test_losses"]
        test_ious = inner_test_results[s]["test_ious"]
        test_dice_scores = inner_test_results[s]["test_dice_scores"]
        gen_error_estimate = sum(test_size * test_iou for test_size, test_iou in zip(test_sizes, test_ious)) / sliced_par_split_size
        E_gen_iou_s.append(gen_error_estimate)
    best_s = E_gen_iou_s.index(max(E_gen_iou_s))
    best_parameter = parameters[best_s]
    learning_rate = 1e-3
    loss_function = "dice"
    print(f"\nSelected best model: UNet{best_s+1} with Mean IOU: {E_gen_iou_s[best_s]:.5f} and loss function: {best_parameter}")


    outer_train_dataloader, outer_val_dataloader = get_dataloaders_kfold(par_split, 0.8, 32, (256, 256))
    #test_split = process_and_slice(test_split)
    outer_test_dataloader = DataLoader(test_split, batch_size=1, shuffle=False)

    # Train best model on the full partitioned dataset
    from src.model.UNet import UNet

    best_model = UNet()
    best_model.normalizer = get_normalizer(outer_train_dataloader.dataset.dataset)
    best_model.train_model(
        training_dataloader=outer_train_dataloader,
        validation_dataloader=outer_val_dataloader, 
        epochs=epochs,
        learningRate=best_parameter,#learning_rate,
        model_name=f"Best_UNet_Outer{idx+1}",
        cross_validation="kfold",
        with_early_stopping=True,
        loss_function=loss_function#best_parameter
    )

    # **NEW STEP: Evaluate best model on the test set**
    test_loss = best_model.get_validation_loss(outer_test_dataloader)
    test_iou, test_dice = ModelEvaluator.evaluate_model(best_model, outer_test_dataloader)
    fold_results["test_sizes"].append(len(test_split))
    fold_results["test_losses"].append(test_loss)
    fold_results["test_ious"].append(test_iou)
    fold_results["test_dice_scores"].append(test_dice)

    log_inner_fold_results(idx, parameters, inner_test_results, S)
    return E_gen_iou_s, best_s
    

def inner_fold(idx, K2, par_split, parameters, epochs, train_idx, test_idx, test_results):
    print(f"\n ------------ Inner Fold {idx+1}/{K2} -------------") 
    train_split = Subset(par_split, train_idx.tolist())
    inner_test_data = Subset(par_split, test_idx.tolist())
    train_data, val_data = random_split(train_split, [0.8, 0.2])
    inner_test_data = process_and_slice(inner_test_data)
    inner_test_dataloader = DataLoader(inner_test_data, batch_size=1, shuffle=False)
    data_augmenter = DataAugmenter()
    from src.model.UNet import UNet

    for s in range(1, len(parameters)+1):
        unet = UNet()
        inner_train_dataloader, inner_validation_dataloader = get_dataloaders_kfold_already_split(train_data, val_data, 32, (256, 256))
        print(len(inner_validation_dataloader.dataset))
        unet.normalizer = get_normalizer(inner_train_dataloader.dataset.dataset)
        #inner_train_dataloader.dataset.dataset.transform = data_augmenter.get_transformer(True, *parameters[s-1])
        
        model_name = f"UNet{s}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        learning_rate = parameters[s-1]  
        loss_function = "cross_entropy"#parameters[s-1]
        print(parameters[s-1])
        print(f"\nTraining model {s} with \nName: {model_name}\n Loss function: {loss_function}\n Learning rate: {learning_rate}")
        unet.train_model(
            training_dataloader=inner_train_dataloader,
            validation_dataloader=inner_validation_dataloader,
            epochs=epochs,
            learningRate=learning_rate,
            model_name=model_name,
            cross_validation="kfold",
            with_early_stopping=True,
            loss_function=loss_function
        )

        test_loss = unet.get_validation_loss(inner_test_dataloader)
        test_iou, test_dice = ModelEvaluator.evaluate_model(unet, inner_test_dataloader)

        test_results[s]["test_sizes"].append(len(inner_test_data))
        test_results[s]["test_losses"].append(test_loss)
        test_results[s]["test_ious"].append(test_iou)
        test_results[s]["test_dice_scores"].append(test_dice)
        print(f"Test IOU: {test_iou}")
        with open(f"cv_loss_functions_inner{idx}_model{s}.txt", "w") as f:
            f.write(f"Model {s} in fold {idx}\n")
            f.write(f"Mean IOU: {test_iou}\n")
            f.write(f"Mean Dice: {test_dice}")

def log_outer_folds_results(fold_results, parameters, best_models, model_ious, gen_error_estimate):
    with open("cross_validation_clahe_results.txt", "w") as f:
        f.write(f"############## K-Fold Cross Validation Summary ##############\n")
        for i, (loss, iou, dice) in enumerate(zip(fold_results["test_losses"], fold_results["test_ious"], fold_results["test_dice_scores"])):
            f.write(f"Outer Fold {i+1}:\n")
            f.write(f"  Test Size: {fold_results["test_sizes"][i]}\n")
            f.write(f"  Test Loss: {loss:.5f}\n")
            f.write(f"  Test IOU: {iou:.5f}\n")
            f.write(f"  Test Dice Score: {dice:.5f}\n")
            f.write(f"  Best Learning rate: {parameters[best_models[i]]}\n\n")

        f.write("############## Estimated Model IOUs from Inner folds per Outer Fold ##############\n")
        for idx, (key, value) in enumerate(model_ious.items()):
            f.write(f"Model {key} (Learning rate={parameters[key-1]}): {[float(x) for x in value]}\n")

        f.write(f"\n############## Estimated Generalization Error: {gen_error_estimate:.5f} ##############\n")

def log_inner_fold_results(idx, parameters, inner_test_results, S):
    results_dir = "cv_loss_functions_logs"
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, f"inner_fold_results_outer{idx+1}.txt")

    with open(log_file, "w") as f:
        f.write(f"Inner Fold Results for Outer Fold {idx+1}\n")
        f.write("=" * 50 + "\n")
        for s in range(1, S+1):
            f.write(f"\nModel {s} (Learning rate = {parameters[s-1]}):\n")
            for i in range(len(inner_test_results[s]["test_ious"])):
                f.write(f"  Inner Fold {i+1}:\n")
                f.write(f"    Test Size: {inner_test_results[s]['test_sizes'][i]}\n")
                f.write(f"    Loss: {inner_test_results[s]['test_losses'][i]:.5f}\n")
                f.write(f"    IOU: {inner_test_results[s]['test_ious'][i]:.5f}\n")
                f.write(f"    Dice: {inner_test_results[s]['test_dice_scores'][i]:.5f}\n")

def log_one_layer_cv_results(parameters, fold_results, best_parameter):
    with open("cross_validation_learning_rates_results.txt", "w") as f:
        f.write(f"############## K-Fold Cross Validation Summary ##############\n")
        for s in range(1, len(parameters)+1):
            f.write(f"Model with {parameters[s-1]}:\n")
            f.write(f"  Test Sizes: {fold_results[s]['test_sizes']}\n")
            f.write(f"  Test Losses: {[float(x) for x in fold_results[s]['test_losses']]} -> Mean = {np.mean(fold_results[s]['test_losses'])}\n")
            f.write(f"  Test IOUs: {[float(x) for x in fold_results[s]['test_ious']]} -> Mean = {np.mean(fold_results[s]['test_ious'])}\n")
            f.write(f"  Test Dices Scores: {[float(x) for x in fold_results[s]['test_dice_scores']]} -> Mean = {np.mean(fold_results[s]['test_dice_scores'])}\n\n")
        f.write(f"Best loss function: {best_parameter}")