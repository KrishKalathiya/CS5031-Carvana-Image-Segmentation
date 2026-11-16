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

