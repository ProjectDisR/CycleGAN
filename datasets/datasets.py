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
        
        self.train = train
        
        folder = 'test'
        if self.train:
            folder = 'train'
            
        self.img_name_ls_A = os.listdir(os.path.join(root, folder+'A'))
        self.img_name_ls_B = os.listdir(os.path.join(root, folder+'B'))
        self.img_name_ls_A.sort()
        self.img_name_ls_B.sort()
        self.img_ls_A = [os.path.join(root, folder+'A', img_name) for img_name in self.img_name_ls_A]
        self.img_ls_B = [os.path.join(root, folder+'B', img_name) for img_name in self.img_name_ls_B]
        
        self.transforms = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
        
    def __getitem__(self, index):
        I_A = imread(self.img_ls_A[index % len(self.img_ls_A)])
        I_B = imread(self.img_ls_B[index % len(self.img_ls_B)])
        I_A = self.transforms(I_A)
        I_B = self.transforms(I_B)
        
        return I_A, I_B, self.img_name_ls_A[index % len(self.img_name_ls_A)], self.img_name_ls_B[index % len(self.img_name_ls_B)]
    
    def __len__(self):
        
        return max(len(self.img_ls_A), len(self.img_ls_B))
    
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

if __name__ == '__main__': 
    remove_gray('horse2zebra', 'trainA')
    remove_gray('horse2zebra', 'trainA')
    remove_gray('horse2zebra', 'trainB')
    remove_gray('horse2zebra', 'trainB')
    remove_gray('horse2zebra', 'testA')
    remove_gray('horse2zebra', 'testA')
    remove_gray('horse2zebra', 'testB')
    remove_gray('horse2zebra', 'testB')