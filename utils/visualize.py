# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:09:16 2018

@author: user
"""

import numpy as np

import visdom

class Visualizer():
    
    def __init__(self, env='CycleGAN'):
        
        self.vis = visdom.Visdom(env=env)
        self.names = set()
        self.log_text = ''
    
    def add_names(self, *args):
        
        for name in args:
            self.names.add(name)
            
        return
    
    def plot(self, name, epoch, value):
        
        if not name in self.names:
            print('Unknown name for plotting!')
            print('Use add_names to add a new name.')
            
        else:
            opts = {'xlabel':'epoch', 'ylabel':name} 
            self.vis.line(Y=np.array([value]), X=np.array([epoch]), win=name,
                          opts=opts, update=None if epoch == 0 else 'append')
            
        return
    
    def imgs(self, name, I):
        
        self.vis.images(I, nrow=4, win=name)
        
        return
        
    def log(self, info, win='log'):
        
        self.log_text += '{} <br>'.format(info)
        self.vis.text(self.log_text, win)
        
        return