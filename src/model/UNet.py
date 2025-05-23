import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
import numpy as np
from threading import Event
import os

# Model-related imports
from src.model.PlottingTools import *
from src.model.CrossValidation import *
from src.shared.ModelTrainingStats import ModelTrainingStats
from src.model.DataTools import resource_path

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        x = F.relu(self.bn(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, input, concat_map):
        x: Tensor = self.upconv(input)

        x = torch.cat((concat_map, x), dim=1)
    
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x
        

class UNet(nn.Module):
    def __init__(self, pre_loaded_model_path = None):
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
        
        self.criterion = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device}")

        self.preffered_input_size = (256, 256)
    
        if pre_loaded_model_path:
            model_path = resource_path(pre_loaded_model_path)
            self.load_model(model_path)

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

    def train_model(self, training_dataloader: DataLoader, validation_dataloader: DataLoader, epochs: int, learningRate: float, model_name: str, cross_validation: str, with_early_stopping: bool, stop_training_event: Event = None, loss_callback = None):
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learningRate)
        self.criterion = nn.CrossEntropyLoss()

        training_loss_values = []
        validation_loss_values = []
        best_loss = np.inf
        no_improvement_epochs = 0
        batches_in_epoch = len(training_dataloader.dataset)//training_dataloader.batch_size
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            
            for i, data in enumerate(training_dataloader):
                if (stop_training_event is not None) and stop_training_event.is_set():
                    print("Training stopped by user.")
                    return training_loss_values, validation_loss_values
                
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
                print(f"Epoch {epoch + 1}: Finished batch {i + 1} of {batches_in_epoch}")
            epoch_training_loss = running_loss / len(training_dataloader)
            training_loss_values.append(epoch_training_loss)

            epoch_validation_loss = self.get_validation_loss(validation_dataloader)
            validation_loss_values.append(epoch_validation_loss)
            
            
            
            print(f'---Epoch {epoch + 1}: Training loss: {epoch_training_loss:.5f}, Validation loss: {epoch_validation_loss:.5f}---')

            if epoch_validation_loss < best_loss:
                self.save_model("data/models/", model_name)
                best_loss = epoch_validation_loss
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
                if with_early_stopping and no_improvement_epochs >= 50:
                    break
            
            if loss_callback:
                stats = ModelTrainingStats(training_loss=epoch_training_loss,
                                           val_loss=epoch_validation_loss,
                                           best_loss=best_loss,
                                           epoch=epoch+1,
                                           best_epoch=epoch+1 - no_improvement_epochs)
                loss_callback(stats)

            
        
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

    def save_model(self, folder_path: str, model_name: str):
        path = folder_path + model_name
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save(self.state_dict(), path)
        
    def load_model(self, path):
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.load_state_dict(state_dict)

    def segment(self, tensor: Tensor):
        output = self(tensor)
        arg = output.argmax(dim=1)
        return arg