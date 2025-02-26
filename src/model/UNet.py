from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.transforms.functional as TF
import os
from SegmentationDataset import SegmentationDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import datetime
from TensorTools import *
from PlottingTools import *


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, input, concat_map):
        x: Tensor = self.upconv(input)

        x = torch.cat((concat_map, x), dim=1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
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
        self.criterion = None

        self.device = None
        if torch.cuda.is_available():
            print("Using CUDA")
            self.device = torch.device("cuda")
        else:
            print("Using CPU")
            self.device = torch.device("cpu")
    
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

    def train_model(self, dataloader: DataLoader, epochs: int, learningRate: float):
        self.to(self.device)
        self.train()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learningRate)
        self.criterion = nn.CrossEntropyLoss()

        lossValues = []
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                labels = labels.long().squeeze(1)
                self.optimizer.zero_grad()
                outputs = self(inputs)
                
                loss = self.criterion(outputs, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                self.optimizer.step()

                if epoch % 20 == 19:
                    showTensor(outputs)
                running_loss += loss.item()
                

            epoch_loss = running_loss / len(dataloader)
            print(f'Epoch {epoch + 1} loss: {epoch_loss:.3f}')
            lossValues.append(epoch_loss)

            plotLoss(lossValues)

        print('Finished Training')
        plt.show()


    def save_model(self, fileName):
        torch.save(self.state_dict(), "data/models/" + fileName)

    def load_model(self, fileName):
        state_dict = torch.load("data/models/" + fileName)
        self.load_state_dict(state_dict)


def main():
    unet = UNet()
    dataset = SegmentationDataset("data/images/", "data/masks/")
    dataloader = DataLoader(dataset, batch_size=1)

    unet.train_model(dataloader=dataloader, epochs=1000, learningRate=0.0005)
    unet.save_model("UNet_"+datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    unet.eval()


    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        outputs = unet(inputs)

        showTensor(outputs)


if __name__ == '__main__':
    main()