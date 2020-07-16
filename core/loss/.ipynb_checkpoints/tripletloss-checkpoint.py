#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import torch
from IPython import embed


class TripletLoss_self(nn.Module):
    
    def __init__(self, margin=0.5):
        super(TripletLoss_self, self).__init__()
        self.margin = margin

    def forward(self, pred, target):
        
        





















