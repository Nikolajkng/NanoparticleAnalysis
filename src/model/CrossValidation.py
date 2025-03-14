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
     
        print(f"Training model using {outer_k}-fold [k={outer_k}, epochs={epochs}, learnRate={learning_rate}]...")
        print("---------------------------------------------------------------------------------------")
     
        dataset = SegmentationDataset(images_path, masks_path)
        data_augmenter = DataAugmenter()
        dataset = data_augmenter.augment_dataset(dataset)
        datasetSize = np.arange(len(dataset))
        outer_kfold = KFold(n_splits=outer_k, shuffle=True, random_state=42)
        inner_kfold = KFold(n_splits=inner_k, shuffle=True, random_state=42)

        for fold1, (train_idx, val_idx) in enumerate(outer_kfold.split(datasetSize)): 
            print(f"\n############## Outer Fold {fold1+1}/{outer_k} ##############") 
            train_split = Subset(dataset, train_idx.tolist())
            val_split = Subset(dataset, val_idx.tolist())
            print(f"Training-split ({len(train_split.indices)}/{len(dataset)}): {train_split.indices}")
            print(f"Validation-split ({len(val_split.indices)}/{len(dataset)}): {val_split.indices}")


            for fold2, (train_idx, val_idx) in enumerate(inner_kfold.split(np.arange(len(train_split)))): 
                print(f"\n############## Inner Fold {fold2+1}/{inner_k} ##############") 

                train_dataloader = DataLoader(train_split, batch_size=4, shuffle=True)
                val_dataloader = DataLoader(val_split, batch_size=1, shuffle=False)

                unet = UNet()
                model_name = f"UNet_Fold{fold2+1}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

                # Train model
                training_loss, validation_loss = unet.train_model(
                    training_dataloader=train_dataloader,
                    validation_dataloader=val_dataloader,
                    epochs=epochs,
                    learningRate=learning_rate,
                    model_name=model_name,
                    cross_validation="k-fold"
                )

                # Evaluate fold performance and store the best
                validation_loss = unet.get_validation_loss(val_dataloader)
                if(validation_loss < best_val_loss or best_val_loss == None):
                    best_val_loss = validation_loss
                    best_split_subset = (train_split, val_split)                    
                    best_split_indices = (train_split.indices, val_split.indices)   
                    

                print(f"Fold {fold2+1} Validation Loss: {validation_loss:.5f}")
                
                fold_results.append(validation_loss)
          
        
        # Summary of results:
        print(f"\n############## K-Fold Cross Validation Summary ##############")
        for i, loss in enumerate(fold_results):
            print(f"Fold {i+1}: Validation Loss = {loss:.5f}")
            
        avg_loss = np.mean(fold_results)
        print(f"\nAverage Validation Loss: {avg_loss:.5f}")
        
        
        # Summary of fold with best result based on validation loss:
        print(f"\n############## Best split found: ##############")
        best_results.append((best_val_loss, best_split_indices, best_split_subset))   
        print(f"Lowest Validation Loss = {best_val_loss} with split = {best_split_indices}")
        
