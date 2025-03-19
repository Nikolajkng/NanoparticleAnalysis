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
    val_results = []  # Validation results for each inner fold
    
    # Set parameters:
    K1 = 2  # Number of outer folds
    K2 = 2  # Number of inner folds
    S = 2   # Number of different learning rates
    epochs = 2  # Number of epochs
    print(f"\nTraining model using two-level cross-validation with K1={K1} and K2={K2}")
    
    # Load data
    dataset = SegmentationDataset(images_path, masks_path)
    data_augmenter = DataAugmenter()
    dataset = data_augmenter.augment_dataset(dataset)
    dataset_size = np.arange(len(dataset))
    
    cv = KFold(n_splits=K1, shuffle=True, random_state=42)
    
    ####################### Outer fold #######################
    for outerfold, (par_idx, test_idx) in enumerate(cv.split(dataset_size)): 
        print(f"\n ---------------- Outer Fold {outerfold+1}/{K1} ----------------") 
        
        # split into partition and test set
        par_split = Subset(dataset, par_idx.tolist())
        test_split = Subset(dataset, test_idx.tolist())

        par_dataloader = DataLoader(par_split, batch_size=4, shuffle=True)
        test_dataloader = DataLoader(test_split, batch_size=1, shuffle=False)        
        
        ####################### Inner fold #######################
        best_val_loss = np.inf
        best_s = None
        val_results = {s: [] for s in range(1, S+1)}  # Store validation losses for each learning rate
        
        for innerfold, (train_idx, val_idx) in enumerate(cv.split(par_idx)): 
            print(f"\n ------------ Inner Fold {innerfold+1}/{K2} -------------") 
            
            train_split = Subset(par_split, train_idx.tolist())
            val_split = Subset(par_split, val_idx.tolist())

            train_dataloader = DataLoader(train_split, batch_size=4, shuffle=True)
            val_dataloader = DataLoader(val_split, batch_size=1, shuffle=False)
            
            # Train S different models (different learning rates)
            for s in range(1, S+1):
                unet = UNet()
                model_name = f"UNet{s}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                learning_rate = 10**-s
                
                print(f"\nTraining model {s} with name: {model_name} and learning rate: {learning_rate}")
                unet.train_model(
                    training_dataloader=train_dataloader,
                    validation_dataloader=val_dataloader,
                    epochs=epochs,
                    learningRate=learning_rate,
                    model_name=model_name,
                    cross_validation="k-fold"
                )
                
                # Get the validation loss on validation set
                validation_loss = unet.get_validation_loss(val_dataloader)
                val_results[s].append(validation_loss) 
        
        # Compute the weighted validation loss for each learning rate
        weighted_val_loss = {s: sum(val_results[s]) / len(val_results[s]) for s in range(1, S+1)} 
        
        # Select the best model based on the lowest validation loss
        best_s = min(weighted_val_loss, key=weighted_val_loss.get)
        best_learning_rate = 10**-best_s
        print(f"\nSelected best model: UNet{best_s} with weighted validation loss: {weighted_val_loss[best_s]:.5f} and learning rate {best_learning_rate}")
        
        ####################### Retrain best model #######################
        best_model = UNet()
        best_model.train_model(
            training_dataloader=par_dataloader,
            validation_dataloader=val_dataloader, 
            epochs=epochs,
            learningRate=best_learning_rate,
            model_name=f"Best_UNet_Outer{outerfold+1}",
            cross_validation="k-fold"
        )

        ####################### Evaluate on Test Set #######################
        test_loss = best_model.get_validation_loss(test_dataloader)
        fold_results.append((len(test_split), test_loss)) 
    
    ####################### Compute Final Generalization Error #######################
    gen_error_estimate = sum(test_size * test_loss for test_size, test_loss in fold_results) / dataset_size

    
    print(f"\n############## K-Fold Cross Validation Summary ##############")
    for i, (size, loss) in enumerate(fold_results):
        print(f"Outer Fold {i+1}: Test Loss = {loss:.5f}")

    print(f"\n############## Estimated Generalization Error: {gen_error_estimate:.5f} ##############")