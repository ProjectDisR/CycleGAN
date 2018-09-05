# -*- coding: utf-8 -*-
"""
Created on Mon Sep 03 14:22:37 2018

@author: user
"""
import torch as t
from torch import nn

class G_Conv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(G_Conv, self).__init__()
        
        self.function = nn.Sequential(
                nn.ReflectionPad2d(padding),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.ReLU()
                )
        
    def forward(self, x):
        
        return self.function(x)
    
class ResidualBlock(nn.Module):
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.function = nn.Sequential(
                G_Conv(channels, channels, kernel_size=3, stride=1, padding=1),
                nn.ReflectionPad2d(1),
                nn.Conv2d(channels, channels, kernel_size=3, stride=1),
                nn.InstanceNorm2d(channels, affine=True),
                )
        
    def forward(self, x):
        
        return x + self.function(x)

class upConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super(upConv, self).__init__()
        
        self.function = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.ReLU()
                )
        
    def forward(self, x):
        
        return self.function(x)
        
class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        
        self.init = G_Conv(3, 32, 7, 1, 3)
        
        self.dlayers = nn.Sequential(
                G_Conv(32, 64, 3, 2, 1),
                G_Conv(64, 128, 3, 2, 1)
                )
        
        self.residualblocks = nn.Sequential(
                ResidualBlock(128),
                ResidualBlock(128),
                ResidualBlock(128),
                ResidualBlock(128),
                ResidualBlock(128),
                ResidualBlock(128),
                ResidualBlock(128),
                ResidualBlock(128),
                ResidualBlock(128)
                )
        
        self.ulayers = nn.Sequential(
                upConv(128, 64),
                upConv(64, 32)
                )
        
        self.generate = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(32, 3, kernel_size=7, stride=1),
                nn.Tanh()
                )
        
    def forward(self, x):
        
        x = self.init(x)
        x = self.dlayers(x) 
        x = self.residualblocks(x)
        x = self.ulayers(x)
        x = self.generate(x)       
        
        return x
    
class D_Conv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(D_Conv, self).__init__()
        
        self.function = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.LeakyReLU(0.2)
                )
        
    def forward(self, x):
        
        return self.function(x)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.init = nn.Sequential(
                nn.Conv2d(3, 64, 4, 1, 1),
                nn.LeakyReLU(0.2)
                )
        
        self.layers = nn.Sequential(
                D_Conv(64, 128),
                D_Conv(128, 256),
                D_Conv(256, 512)
                )
        
        self.discriminate = nn.Sequential(
                nn.Conv2d(512, 1, 4, 1, 1),
                nn.AvgPool2d(30, 1)
                )
        
    def forward(self, x):
        
        x = self.init(x)
        x = self.layers(x)
        x = self.discriminate(x)
        x = x.view(-1, 1)
        
        return x

#if __name__ == '__main__': 
#    a = t.randn(2, 3, 256, 256)
#    G = Generator()
#    D = Discriminator()
#    a = G(a)
#    print(a.size())
#    a = D(a)
#    a = a.view(-1, 1)
#    print(a.size())

