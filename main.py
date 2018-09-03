# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 22:25:09 2018

@author: user
"""
import itertools

from config import DefaultConfig
from datasets.dataset import StyleAB
import torch as t
from torch.utils.data import DataLoader
from torch import nn
from model import Generator, Discriminator
#from utils.visualize import Visaulizer


def train(**kwargs):
    opt = DefaultConfig()
    opt.parse(kwargs)
    
    dataset = StyleAB(opt.data_root, train=True)
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=True)
    
#    vis = Visualizer(opt.env)
    
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
    for epoch in range(opt.n_epoch):
        for i, (real_A, real_B) in enumerate(dataloader):
            
            real_A = real_A.cuda()
            real_B = real_B.cuda()
            
            yes = t.ones(opt.batch_size, 1).cuda()
            no = t.zeros(opt.batch_size, 1).cuda()
            
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
            
            optimizer_D_A.zero_grad()
            loss_D_A.backward()
            optimizer_D_A.step()
            
            loss_D_B = criterion_real(D_B(gener_B), no) + criterion_real(D_B(real_B), yes)
            
            optimizer_D_B.zero_grad()
            loss_D_B.backward()
            optimizer_D_B.step()
            
            print(loss_G.cpu().item(), loss_D_A.cpu().item(), loss_D_B.cpu().item())
train()