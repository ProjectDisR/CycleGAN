# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:49:28 2018

@author: user
"""

class DefaultConfig():
    env = 'CycleGAN'
    
    data_root = 'datasets/horse2zebra'
    batch_size = 2
    n_epoch = 200
    lr = 0.0002
    lambda_cyle = 10
    
    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                print('Unknown attr ', k)
            else:
                setattr(self, k, v)
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, v)
