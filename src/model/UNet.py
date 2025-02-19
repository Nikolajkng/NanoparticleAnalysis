from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.transforms.functional as TF
import os
from SegmentationDataset import SegmentationDataset
from torch.utils.data import DataLoader

def crop(tensor: Tensor, target_size: tuple[int, int]) -> Tensor:
    _, _, h, w = tensor.shape
    th, tw = target_size

    start_h = (h-th) // 2
    start_w = (w-tw) // 2
    return tensor[:, :, start_h:start_h + th, start_w:start_w + tw]

def normalize_tensor_to_pixels(tensor: Tensor) -> Tensor:
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    tensor = tensor * 255
    return tensor


class encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        return x

class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3)
    
    def forward(self, input, concat_map):
        x: Tensor = self.upconv(input)
        concat_map = crop(concat_map, (x.size(dim=2), x.size(dim=3)))

        x = torch.cat((concat_map, x), dim=1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x
        

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder1 = encoder_block(1, 64)
        self.encoder2 = encoder_block(64, 128)
        self.encoder3 = encoder_block(128, 256)
        self.encoder4 = encoder_block(256, 512)

        self.bottleneck = encoder_block(512, 1024)

        self.decoder1 = decoder_block(1024, 512)
        self.decoder2 = decoder_block(512, 256)
        self.decoder3 = decoder_block(256, 128)
        self.decoder4 = decoder_block(128, 64)

        self.mapping_conv = nn.Conv2d(64, 2, 1)
    
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
        m = self.mapping_conv(d4)
        return m


def train(model: UNet, dataloader: DataLoader, criterion: nn.CrossEntropyLoss, optimizer: torch.optim.SGD, epochs: int) -> UNet:
    model.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            labels = labels.long()
            labels = labels.squeeze(1)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 9:
                print(f'Output min: {outputs.min()}, max: {outputs.max()}')
                probabilities = F.softmax(outputs, dim=1)  
                #predicted_classes = torch.argmax(probabilities, dim=1)
                probabilities = probabilities.squeeze(0)
                
                pixels = normalize_tensor_to_pixels(probabilities[1, :, :])
                
                img = TF.to_pil_image(pixels.byte())
                img.show()

            # print statistics
            running_loss += loss.item()
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}')
            running_loss = 0.0

    print('Finished Training')
    return model

def main():
    unet = UNet()
    dataset = SegmentationDataset("data/images/", "data/masks/")
    dataloader = DataLoader(dataset)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unet.parameters(), lr=0.005, momentum=0.99)
    
    
    unet = train(unet, dataloader, criterion, optimizer, 100)
    torch.save(unet.state_dict(), "data/model/")
    unet.eval()





if __name__ == '__main__':
    main()