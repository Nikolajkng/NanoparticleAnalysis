import datetime
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from model.SegmentationDataset import SegmentationDataset
from model.TensorTools import *
from model.PlottingTools import *
from model.DataTools import get_dataloaders
from model.DataAugmenter import DataAugmenter
from model.UNet import UNet


def cv_holdout(unet, images_path, masks_path):
    
    # Set parameters:
    train_subset_size = 0.75  # 3/4
    epochs = 10
    learning_rate = 0.01
    print("-----------------------------------------------------------------------")
    print(f"Training model using holdout [train_split_size={train_subset_size}, epochs={epochs}, learnRate={learning_rate}]...")

    dataset = SegmentationDataset(images_path, masks_path)
    train_dataloader, validation_dataloader = get_dataloaders(dataset, train_subset_size)
    
    unet.train_model(
        training_dataloader=train_dataloader, 
        validation_dataloader=validation_dataloader, 
        epochs=epochs, 
        learningRate=learning_rate, 
        model_name="UNet_"+datetime.datetime.now().strftime('%Y%m%d%H%M%S'),
        cross_validation="holdout"
        )


def cv_kfold(unet, images_path, masks_path):
    fold_results = []   
    train_split = None
    val_split = None
    best_val_loss = np.inf  
    best_split_subset = (None, None) 
    best_split_indices = ([], [])
    best_results = []
    
    # Set parameters:
    outer_k = 2
    inner_k = 2
    epochs = 2
    learning_rate = 0.01
    
    print(f"Training model using two layered k-fold [k1={outer_k}, k2={inner_k}, epochs={epochs}, learnRate={learning_rate}]...")
    print("---------------------------------------------------------------------------------------")
    
    dataset = SegmentationDataset(images_path, masks_path)
    data_augmenter = DataAugmenter()
    dataset = data_augmenter.augment_dataset(dataset)
    dataset_size = np.arange(len(dataset))
    
    # Outer K-Fold Cross Validation
    outer_kfold = KFold(n_splits=outer_k, shuffle=True, random_state=42)
    
    for outerfold, (train_idx, val_idx) in enumerate(outer_kfold.split(dataset_size)): 
        print(f"\n############## Outer Fold {outerfold+1}/{outer_k} ##############") 
        train_split = Subset(dataset, train_idx.tolist())
        val_split = Subset(dataset, val_idx.tolist())
        print(f"Training-split ({len(train_split.indices)}/{len(dataset)}): {train_split.indices}")
        print(f"Validation-split ({len(val_split.indices)}/{len(dataset)}): {val_split.indices}")

        # Inner K-Fold Cross Validation on the training data of the outer fold
        inner_fold_results = []
        inner_kfold = KFold(n_splits=inner_k, shuffle=True, random_state=42)
        
        for innerfold, (inner_train_idx, inner_val_idx) in enumerate(inner_kfold.split(train_split.indices)): 
            print(f"\n############## Inner Fold {innerfold+1}/{inner_k} for Outer Fold {outerfold+1} ##############") 
            inner_train_split = Subset(train_split, inner_train_idx.tolist())
            inner_val_split = Subset(train_split, inner_val_idx.tolist())
            print(f"Inner Training-split ({len(inner_train_split.indices)}/{len(train_split.indices)}): {inner_train_split.indices}")
            print(f"Inner Validation-split ({len(inner_val_split.indices)}/{len(train_split.indices)}): {inner_val_split.indices}")
            
            train_dataloader = DataLoader(inner_train_split, batch_size=4, shuffle=True)
            val_dataloader = DataLoader(inner_val_split, batch_size=1, shuffle=False)
            
            # Initialize a fresh model for each fold
            model_name = f"UNet_Fold{outerfold+1}_InnerFold{innerfold+1}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            unet = UNet()
            
            # Train the model on the inner fold
            unet.train_model(
                training_dataloader=train_dataloader,
                validation_dataloader=val_dataloader,
                epochs=epochs,
                learningRate=learning_rate,
                model_name=model_name,
                cross_validation="k-fold"
            )
            
            # Evaluate inner fold performance
            validation_loss = unet.get_validation_loss(val_dataloader)
            print(f"Inner Fold {innerfold+1} Validation Loss: {validation_loss:.5f}")
            inner_fold_results.append(validation_loss)
        
        # After inner loop, evaluate the outer fold performance (average of inner folds)
        avg_inner_loss = np.mean(inner_fold_results)
        print(f"\nAverage Inner Fold Validation Loss for Outer Fold {outerfold+1}: {avg_inner_loss:.5f}")
        
        # Now train on the entire training split of the outer fold
        train_dataloader = DataLoader(train_split, batch_size=4, shuffle=True)
        val_dataloader = DataLoader(val_split, batch_size=1, shuffle=False)
        
        # Initialize a fresh model for the outer fold
        model_name = f"UNet_OuterFold{outerfold+1}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        unet = UNet()
        
        # Train the model on the outer fold
        unet.train_model(
            training_dataloader=train_dataloader,
            validation_dataloader=val_dataloader,
            epochs=epochs,
            learningRate=learning_rate,
            model_name=model_name,
            cross_validation="k-fold"
        )
        
        # Evaluate outer fold performance
        validation_loss = unet.get_validation_loss(val_dataloader)
        print(f"Outer Fold {outerfold+1} Validation Loss: {validation_loss:.5f}")
        fold_results.append(validation_loss)
        
        # Track the best validation result across all folds
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_split_subset = (train_split, val_split)                    # Store best data split
            best_split_indices = (train_split.indices, val_split.indices)   # For tracking the split indices
        
    # Summary of results:
    print(f"\n############## K-Fold Cross Validation Summary ##############")
    for i, loss in enumerate(fold_results):
        print(f"Outer Fold {i+1}: Validation Loss = {loss:.5f}")
    
    avg_loss = np.mean(fold_results)
    print(f"\nAverage Validation Loss across all Outer Folds: {avg_loss:.5f}")
    
    # Summary of the best fold
    print(f"\n############## Best split found: ##############")
    best_results.append((best_val_loss, best_split_indices, best_split_subset))   
    print(f"Lowest Validation Loss = {best_val_loss} with split = {best_split_indices}")
