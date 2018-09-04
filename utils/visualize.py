# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:09:16 2018

@author: user
"""
import visdom
import numpy as np
class Visualizer():
    
    def __init__(self, env='CycleGAN'):
        self.vis = visdom.Visdom(env=env)
        self.names = set()
        self.log_text = ''
    
    def add_names(self, name_ls):
        for name in name_ls:
            self.names.add(name)
            
    def plot(self, name, epoch, value):
        
        if not name in self.names:
            
            print('Unknown name!')
            print('Use add_names to add a new name.')
            
        else:
            
            opts = {'xlabel':'epoch', 'ylabel':name} 
            self.vis.line(Y=np.array[value], X=np.array(epoch), win=name,
                          opts=opts, update=None if epoch == 0 else 'append')
            
    def imgs(self, name, I):
        self.vis.images(I, win=name)
        
    def log(self, info, win='log'):
        self.log_text += '{info} <br>'.format(info)
        self.vis.text(self.log_text, win)