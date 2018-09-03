# -*- coding: utf-8 -*-
"""
Created on Mon Sep 03 14:22:37 2018

@author: user
"""

from torch import nn

class Conv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv, self).__init__()
        
        self.function = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.RELU()
                )
        
    def forward(self, x):
        
        return self.function(x)
    
class ResidualBlock(nn.Module):
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.function = nn.Sequential(
                Conv(channels, channels, kernel_size=3, stride=1, padding=1),
                Conv(channels, channels, kernel_size=3, stride=1, padding=1)
                )
        
    def forward(self, x):
        
        return x + self.function(x)

class deConv(nn.module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        self.function = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, output_padding=output_padding),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.RELU()
                )
        
    def forward(self, x):
        
        return x + self.function(x)
        
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.initial = Conv(3, 32, 7, 1, 3)
        self.dlayers = nn.Sequential(
                Conv(32, 64, 3, 2, 2),
                Conv(64, 128, 3, 2, 2)
                )
        self.residualblocks = nn.Sequential(
                ResidualBlock(128)
                ResidualBlock(128)
                ResidualBlock(128)
                ResidualBlock(128)
                ResidualBlock(128)
                ResidualBlock(128)
                ResidualBlock(128)
                ResidualBlock(128)
                ResidualBlock(128)
                )
        self.ulayers = nn.Sequential(
                deConv(128, 64),
                deConv(64, 32)
                )
        self.generate = Conv(32, 3, 7, 1, 3)
        
    def forward(self, x):
        
        x = self.initial(x)
        x = self.dlayers(x) 
        x = self.residualblocks(x)
        x = self.ulayers(x)
        x = self.generate(x)       
        
        return x
        
        