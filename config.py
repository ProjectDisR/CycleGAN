# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:49:28 2018

@author: user
"""

class DefaultConfig():
    
    def __init__(self):
        
        self.env = 'CycleGAN'
        self.data_root = 'datasets/horse2zebra'
        self.ckpts_root = 'checkpoints'
        
        self.n_epoch = 200
        self.batch_size = 2
        self.lr = 0.0002
        self.lambda_cycle = 10
        
    def print_config(self):
        
        print('\n')
        
        import inspect
        
        for k in dir(self):   
            if not k.startswith('__') and not inspect.ismethod(getattr(self, k)):
                print('   ', k, ':', getattr(self, k))
                
        return
    
    def parse(self, kwargs):
        
        for k, v in kwargs.items():
            
            if not hasattr(self, k):
                print('Unknown attr ', k, '!')
            else:
                setattr(self, k, v)
                
        self.print_config()
        
        return