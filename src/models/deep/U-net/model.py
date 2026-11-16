"""
model.py
U-net for binary Carvana vehicle segmentation

Input:
    Float tensor [B, 3, H, W], normally RGB values normalized to [0,1].

Output:
    Raw logits [B, 1, H, W]. Apply sigmoid outside the model to obtain 
    per pixel probabilities.
"""


import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by Batch Norm and ReLU"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        def forward(self, x:torch.Tensor) -> torch.Tensor:
            return self.block(x)

class UNet(nn.Module):
    """
    Compare U-Net   
    
    The input height and width should be divisible by 8 because there are 
    three pooling operations. 256x256 is a suitable initial image size.
    """

    def __init__(self, in_channels: int =3, out_channels: int = 1,
                 features=(32,64,128)):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = DoubleConv(in_channels, features[0])
        self.encoder2 = DoubleConv(features[0], features[1])
        self.encoder3 = DoubleConv(features[1], features[2])

        self.bottleneck = DoubleConv(features[2], features[2] * 2)

        self.up3 = nn.ConvTranspose2d(features[2] * 2, features[2],
                                      kernel_size=2, stride=2)

        self.decoder3 = DoubleConv(features[2] * 2, features)

        self.up2 = nn.ConvTranspose2d(features[2], features[1],
                                      kernel_size=2, stride=2)

        self.decoder2 = DoubleConv(features[1] * 2, features[1])

        self.up1 = nn.ConvTranspose2d(features[1], features[0],
                                      kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(features[0] * 2, features[0])

        self.output = nn.Conv2d(features[0], out_channels, kernel_size=1)
        