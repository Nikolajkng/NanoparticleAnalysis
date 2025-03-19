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
from model.ModelEvaluator import ModelEvaluator

def cv_holdout(unet: UNet, images_path, masks_path):
    
    # Set parameters:
    train_subset_size = 0.7
    validation_subset_size = 0.2
    epochs = 1
    learning_rate = 0.0005
    print(f"Training model using holdout [train_split_size={train_subset_size}, epochs={epochs}, learnRate={learning_rate}]...")
    print("---------------------------------------------------------------------------------------")

    dataset = SegmentationDataset(images_path, masks_path)
    data_augmenter = DataAugmenter()
    dataset = data_augmenter.augment_dataset(dataset)
    train_dataloader, validation_dataloader, test_dataloder = get_dataloaders(dataset, train_subset_size, validation_subset_size)
    
    unet.train_model(
        training_dataloader=train_dataloader, 
        validation_dataloader=validation_dataloader, 
        epochs=epochs, 
        learningRate=learning_rate, 
        model_name="UNet_"+datetime.datetime.now().strftime('%Y%m%d%H%M%S'),
        cross_validation="holdout"
        )
    
    iou, pixel_accuracy = ModelEvaluator.evaluate_model(unet, test_dataloder)
    return iou, pixel_accuracy



def cv_kfold(self, images_path, masks_path):
        from model.UNet import UNet  # Her -> Undgå circle import
                
        # Set parameters:
        k_folds = 5
        epochs = 3
        learning_rate = 0.01
     
        print(f"Training model using {k_folds}-fold [k={k_folds}, epochs={epochs}, learnRate={learning_rate}]...")
        print("---------------------------------------------------------------------------------------")
     
        dataset = SegmentationDataset(images_path, masks_path)
        data_augmenter = DataAugmenter()
        dataset = data_augmenter.augment_dataset(dataset)
        datasetSize = np.arange(len(dataset))
        
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        fold_train_loss = []
        fold_val_loss = []
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(datasetSize)):
        
            print(f"############## Fold {fold+1}/{k_folds} ##############") 
            train_subset = Subset(dataset, train_idx.tolist())
            val_subset = Subset(dataset, val_idx.tolist())
            print(f"Training-split ({len(train_subset.indices)}/{len(dataset)}): {train_subset.indices}")
            print(f"Validation-split ({len(val_subset.indices)}/{len(dataset)}): {val_subset.indices}")

            train_dataloader = DataLoader(train_subset, batch_size=4, shuffle=True)
            val_dataloader = DataLoader(val_subset, batch_size=1, shuffle=False)

            unet = UNet()
            model_name = f"UNet_Fold{fold+1}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

            # Train model
            train_loss, val_loss = unet.train_model(
                training_dataloader=train_dataloader,
                validation_dataloader=val_dataloader,
                epochs=epochs,
                learningRate=learning_rate,
                model_name=model_name,
                cross_validation=""
            )

            # Evaluate fold performance
            validation_loss = unet.get_validation_loss(val_dataloader)
            fold_results.append(validation_loss)
            
            # For plotting later
            fold_train_loss.append(train_loss)
            fold_val_loss.append(val_loss)
            
            print(f"Fold {fold+1} Validation Loss: {validation_loss:.5f}")


        # Final results:
        print(f"\nK-Fold Cross Validation Results:\n--------------------------------")
        for i, loss in enumerate(fold_results):
            print(f"Fold {i+1}: Validation Loss = {loss:.5f}")
            
        avg_loss = np.mean(fold_results)
        print(f"\nAverage Validation Loss: {avg_loss:.5f}")