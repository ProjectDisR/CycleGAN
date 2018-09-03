# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:04:00 2018

@author: user
"""

import os
from torch.utils import data
import torchvision as tv
from skimage.io import imread

class StyleAB(data.Dataset):
    
    def __init__(self, root, train=True):
        folder = 'test'
        if train == True:
            folder = 'train'
            
        A_img_ls = os.listdir(os.path.join(root, folder+'A'))
        B_img_ls = os.listdir(os.path.join(root, folder+'B'))
        self.A_img_ls = [os.path.join(root, folder+'A', img) for img in A_img_ls]
        self.B_img_ls = [os.path.join(root, folder+'B', img) for img in B_img_ls]
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
    

#dataset = StyleAB('horse2zebra', train=True)
#for i in range(3000):
#    A, B = dataset[i]
#    print(A.size(), B.size())
#    print(i)
#root = 'horse2zebra'
#folder = 'train'
#A_img_ls = os.listdir(os.path.join(root, folder+'A'))
#A_img_ls = [os.path.join(root, folder+'A', img) for img in A_img_ls]
#A_img_ls.sort()
#for i in range(len(A_img_ls)):
#    I = imread(A_img_ls[i])
#    if I.shape[-1] != 3:
#        os.remove(A_img_ls[i]) 
#        print(i)
