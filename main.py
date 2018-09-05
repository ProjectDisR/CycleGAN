# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 22:25:09 2018

@author: user
"""
import os
import itertools
import numpy as np

from config import DefaultConfig
from datasets.datasets import StyleAB
import torch as t
from torch.utils.data import DataLoader
from torch import nn
from models.cyclegan import Generator, Discriminator
from utils.visualize import Visualizer

from skimage.io import imsave

import fire

def train(**kwargs):
    
    opt = DefaultConfig()
    opt.parse(kwargs)
    
    dataset = StyleAB(opt.data_root, train=True)
    train_dataloader = DataLoader(dataset, opt.batch_size, shuffle=True)
    dataset = StyleAB(opt.data_root, train=False)
    test_dataloader = DataLoader(dataset, opt.batch_size, shuffle=False)
       
    G_A2B = Generator()
    G_B2A = Generator()
    D_A = Discriminator()
    D_B = Discriminator()
    
    G_A2B = G_A2B.cuda()
    G_B2A = G_B2A.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    
    criterion_real = nn.MSELoss()
    criterion_recons = nn.L1Loss()
    
    optimizer_G = t.optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), 
                               lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = t.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = t.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    vis = Visualizer(opt.env)
    vis.add_names('loss_G', 'loss_D_A', 'loss_D_B')
    
    for epoch in range(opt.n_epoch):
        
        avgloss_G = 0
        avgloss_D_A = 0
        avgloss_D_B = 0
        
        
        for i, (real_A, real_B, img_names_A, img_names_B) in enumerate(train_dataloader):
            print(i)
            real_A = real_A.cuda()
            real_B = real_B.cuda()
            
            yes = t.ones(real_A.size()[0], 1).cuda()
            no = t.zeros(real_A.size()[0], 1).cuda()
            
            gener_B = G_A2B(real_A)
            recons_A = G_B2A(gener_B)
    
            gener_A = G_B2A(real_B)
            recons_B = G_A2B(gener_A)
            
            loss_G1 = criterion_real(D_B(gener_B), yes) + criterion_recons(recons_A, real_A)*opt.lambda_cyle
            loss_G2 = criterion_real(D_A(gener_A), yes) + criterion_recons(recons_B, real_B)*opt.lambda_cyle
            loss_G = loss_G1 + loss_G2
            
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            
            gener_A = gener_A.detach()
            gener_B = gener_B.detach()
            
            loss_D_A = criterion_real(D_A(gener_A), no) + criterion_real(D_A(real_A), yes)
            loss_D_A = loss_D_A*0.5
            
            optimizer_D_A.zero_grad()
            loss_D_A.backward()
            optimizer_D_A.step()
            
            loss_D_B = criterion_real(D_B(gener_B), no) + criterion_real(D_B(real_B), yes)
            loss_D_B = loss_D_B*0.5
            
            optimizer_D_B.zero_grad()
            loss_D_B.backward()
            optimizer_D_B.step()
            
            avgloss_G += loss_G.cpu().item()
            avgloss_D_A += loss_D_A.cpu().item()
            avgloss_D_B += loss_D_B.cpu().item()
        
        vis.plot('loss_G', epoch, avgloss_G/i)
        vis.plot('loss_D_A', epoch, avgloss_D_A/i)
        vis.plot('loss_D_B', epoch, avgloss_D_B/i)
        
        test_iter = iter(test_dataloader)
        real_A, real_B, img_names_A, img_names_B = next(test_iter)
        
        real_A = real_A.cuda()
        real_B = real_B.cuda()

        gener_B = G_A2B(real_A)
        gener_A = G_B2A(real_B)
        
        real_A = real_A*0.5 + 0.5 
        real_B = real_B*0.5 + 0.5
        gener_A = gener_A*0.5 + 0.5
        gener_B = gener_B*0.5 + 0.5
        
        real_A = real_A.cpu()
        real_B = real_B.cpu()
        gener_A = gener_A.cpu()
        gener_B = gener_B.cpu()
        
        vis.imgs('real_A', real_A)
        vis.imgs('real_B', real_B)
        vis.imgs('gener_A', gener_A)
        vis.imgs('gener_B', gener_B)
        
        vis.log('epoch:{}, loss_G:{}, loss_D_A:{}, loss_D_B:{}'.format(epoch, avgloss_G/i, avgloss_D_A/i, avgloss_D_B/i))
        
        t.save(G_A2B.state_dict(), 'checkpoints/G_A2B_e{}.ckpt'.format(epoch+1))
        t.save(G_B2A.state_dict(), 'checkpoints/G_B2A_e{}.ckpt'.format(epoch+1))
        
    if epoch > 100:
        opt.lr -= 0.0002*0.01
        
    optimizer_G = t.optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), 
                               lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = t.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = t.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    return

def test(**kwargs):
    
    opt = DefaultConfig()
    opt.parse(kwargs)
    
    dataset = StyleAB(opt.data_root, train=False)
    test_dataloader = DataLoader(dataset, opt.batch_size, shuffle=False)
       
    G_A2B = Generator()
    G_B2A = Generator()
    G_A2B.load_state_dict(os.path.join(opt.ckpts_root, 'G_A2B_e'+str(opt.n_epoch)+'.ckpt'))
    G_B2A.load_state_dict(os.path.join(opt.ckpts_root, 'G_B2A_e'+str(opt.n_epoch)+'.ckpt'))
    G_A2B = G_A2B.cuda().eval()
    G_B2A = G_B2A.cuda().eval()
    
    for i, (real_A, real_B, img_names_A, img_names_B) in enumerate(test_dataloader):
        
        real_A = real_A.cuda()
        real_B = real_B.cuda()
        gener_B = G_A2B(real_A)
        gener_A = G_B2A(real_B)
        
        gener_A = (gener_A*0.5 + 0.5)*255
        gener_B = (gener_B*0.5 + 0.5)*255
        gener_A = t.clamp(gener_A, 0, 255)
        gener_B = t.clamp(gener_B, 0, 255)
        gener_A = gener_A.cpu().numpy()
        gener_B = gener_B.cpu().numpy()

        if not os.path.isdir(os.path.join(opt.data_root, 'generA')):
            os.mkdir(os.path.join(opt.data_root, 'generA'))
        
        if not os.path.isdir(os.path.join(opt.data_root, 'generB')):
            os.mkdir(os.path.join(opt.data_root, 'generB'))
        
        for I_A, I_B, img_name_A, img_name_B in zip(gener_A, gener_B, img_names_A, img_names_B):
            I_A = np.transpose(I_A, (1, 2, 0))
            I_B = np.transpose(I_B, (1, 2, 0))
            imsave(os.path.join(opt.data_root, 'generB', img_name_A), I_B)
            imsave(os.path.join(opt.data_root, 'generA', img_name_B), I_A)
            
    return

def help():
    print("""
    usage : python {0} <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test
            python {0} help
    avaiable args:""".format(__file__))
    opt = DefaultConfig()
    opt.parse()

if __name__ == '__main__':
    fire.Fire()