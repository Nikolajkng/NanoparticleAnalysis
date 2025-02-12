import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def crop(tensor, target_size):
    _, h, w = tensor.shape
    th, tw = target_size

    start_h = (h-th) // 2
    start_w = (w-tw) // 2
    return tensor[:, start_h:start_h + th, start_w:start_w + tw]


class encoder_block():
    def __init__(self, in_channels, out_channels):
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        return x

class decoder_block():
    def __init__(self, in_channels, out_channels):
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3)
    
    def forward(self, input, concat_map):
        x: Tensor = self.upconv(input)

      
        concat_map = crop(concat_map, (x.size(dim=1), x.size(dim=2)))
        
        x = torch.cat((concat_map, x))

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

        self.mapping_conv = nn.Conv2d(64, 1, 1)
    
    def forward(self, input):
        e1 = self.encoder1.forward(input)
        pooled = nn.MaxPool2d(2,2)(e1)
        e2 = self.encoder2.forward(pooled)
        pooled = nn.MaxPool2d(2,2)(e2)
        e3 = self.encoder3.forward(pooled)
        pooled = nn.MaxPool2d(2,2)(e3)
        e4 = self.encoder4.forward(pooled)
        pooled = nn.MaxPool2d(2,2)(e4)
        b = self.bottleneck.forward(pooled)
        print(b.size())
        d1 = self.decoder1.forward(b, e4)
        d2 = self.decoder2.forward(d1, e3)
        d3 = self.decoder3.forward(d2, e2)
        d4 = self.decoder4.forward(d3, e1)
        m = self.mapping_conv(d4)
        return m

def main():
    unet = UNet()
    x = torch.randn(1, 572, 572)
    y = unet(x)
    print(y)




if __name__ == '__main__':
    main()