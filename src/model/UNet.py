import datetime
from functools import partial
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
import threading

# Model-related imports
from model.SegmentationDataset import SegmentationDataset
from model.TensorTools import *
from model.PlottingTools import *
from model.DataTools import get_dataloaders



class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        x = F.relu(self.bn(self.conv1(input)))
        x = F.relu(self.bn(self.conv2(x)))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, input, concat_map):
        x: Tensor = self.upconv(input)

        x = torch.cat((concat_map, x), dim=1)
    
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.bn(self.conv2(x)))
        return x
        

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder1 = EncoderBlock(1, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        self.bottleneck = EncoderBlock(512, 1024)

        self.decoder1 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)

        self.mappingConvolution = nn.Conv2d(64, 2, 1)

        self.optimizer = None
        # self.criterion = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device}")

    def forward(self, input):
        e1 = self.encoder1(input)
        pooled = nn.MaxPool2d(2,2)(e1)
        e2 = self.encoder2(pooled)
        pooled = nn.MaxPool2d(2,2)(e2)
        e3 = self.encoder3(pooled)
        pooled = nn.MaxPool2d(2,2)(e3)
        e4 = self.encoder4(pooled)
        pooled = nn.MaxPool2d(2,2)(e4)
        b = self.bottleneck(pooled)
        d1 = self.decoder1(b, e4)
        d2 = self.decoder2(d1, e3)
        d3 = self.decoder3(d2, e2)
        d4 = self.decoder4(d3, e1)
        m = self.mappingConvolution(d4)
        return m

    def train_model(self, training_dataloader: DataLoader, validation_dataloader: DataLoader, epochs: int, learningRate: float, model_name: str):
        self.to(self.device)
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learningRate, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

        training_loss_values = []
        validation_loss_values = []
        best_loss = np.inf
        no_improvement_epochs = 0

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0

            for i, data in enumerate(training_dataloader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.long().squeeze(1)
                outputs = self(inputs)
                
                self.optimizer.zero_grad()
                loss = self.criterion(outputs, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                running_loss += loss.item()

            epoch_training_loss = running_loss / len(training_dataloader)
            training_loss_values.append(epoch_training_loss)

            epoch_validation_loss = self.get_validation_loss(validation_dataloader)
            validation_loss_values.append(epoch_validation_loss)

            print(f'Epoch {epoch + 1}: Training loss: {epoch_training_loss:.5f}, Validation loss: {epoch_validation_loss:.5f}')

            if epoch_validation_loss < best_loss:
                self.save_model("data/models/" + model_name)
                best_loss = epoch_validation_loss
                no_improvement_epochs = 0
                
                
            else:
                no_improvement_epochs += 1
                if no_improvement_epochs >= 20:
                    break
        
        print('Finished Training')
        self.load_model("data/models/" + model_name)
        return training_loss_values, validation_loss_values


    def get_validation_loss(self, validation_dataloader: DataLoader) -> float:
        self.eval()
        running_loss = 0
        with torch.no_grad():
            for i, data in enumerate(validation_dataloader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.long().squeeze(1)

                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

        return running_loss / len(validation_dataloader)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)
        
    # This should be in another communicator class
    def process_request_train(self, images_path, masks_path):
        try:
            k_folds = 5
            
            dataset = SegmentationDataset(images_path, masks_path)
            datasetSize = np.arange(len(dataset))
            
            kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            
            fold_train_loss = []
            fold_val_loss = []
            fold_results = []

            for fold, (train_idx, val_idx) in enumerate(kfold.split(datasetSize)):
            
                print(f"############## Fold {fold+1}/{k_folds} ##############") 
                train_subset = Subset(dataset, train_idx.tolist())
                val_subset = Subset(dataset, val_idx.tolist())
                print(f"Training-split: {train_subset.indices}")
                print(f"Validation-split: {val_subset.indices}")

                train_dataloader = DataLoader(train_subset, batch_size=4, shuffle=True)
                val_dataloader = DataLoader(val_subset, batch_size=1, shuffle=False)

                unet = UNet()
                model_name = f"UNet_Fold{fold+1}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

                # Train model
                train_loss, val_loss = unet.train_model(
                    training_dataloader=train_dataloader,
                    validation_dataloader=val_dataloader,
                    epochs=3,
                    learningRate=0.01,
                    model_name=model_name
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

            return (None, 0)
        except Exception as e:
            return (e, 1)

    

    def process_request_segment(self, image_path):
        
        image = Image.open(image_path).convert("L")
        image = TF.to_tensor(image).unsqueeze(0)
       
        output = self(image)
        segmentation = segmentation_to_image(output)
        return (segmentation, 0)
    
    def process_request_load_model(self, model_path):
        try:
            self.load_model(model_path)
            return (None, 0)
        except Exception as e:
            return (e, 1)

def main():
   

    if __name__ == '__main__':
        main()