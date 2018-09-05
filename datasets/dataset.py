# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:04:00 2018

@author: user
"""

import os
from skimage.io import imread

from torch.utils import data
import torchvision as tv


class StyleAB(data.Dataset):
    
    def __init__(self, root, train=True):
        folder = 'test'
        if train:
            folder = 'train'
            
        A_img_ls = os.listdir(os.path.join(root, folder+'A'))
        B_img_ls = os.listdir(os.path.join(root, folder+'B'))
        self.A_img_ls = [os.path.join(root, folder+'A', img_name) for img_name in A_img_ls]
        self.B_img_ls = [os.path.join(root, folder+'B', img_name) for img_name in B_img_ls]
        self.A_img_ls.sort()
        self.B_img_ls.sort()
        
        self.transforms = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
        
    def __getitem__(self, index):
        I_A = imread(self.A_img_ls[index % len(self.A_img_ls)])
        I_B = imread(self.B_img_ls[index % len(self.B_img_ls)])
        I_A = self.transforms(I_A)
        I_B = self.transforms(I_B)
        
        return I_A, I_B
    
    def __len__(self):
        
        return max(len(self.A_img_ls), len(self.B_img_ls))
    
def remove_gray(root, folder):
    
    print(os.path.join(root, folder))
    
    img_ls = os.listdir(os.path.join(root, folder))
    img_ls = [os.path.join(root, folder, img_name) for img_name in img_ls]
    img_ls.sort()
    
    for img_dir in img_ls:
        
        I = imread(img_dir)
        
        if I.shape[-1] != 3:
            os.remove(img_dir) 
            print(img_dir)
    return

#remove_gray('horse2zebra', 'trainA')
#remove_gray('horse2zebra', 'trainB')
#remove_gray('horse2zebra', 'testA')
#remove_gray('horse2zebra', 'testB')