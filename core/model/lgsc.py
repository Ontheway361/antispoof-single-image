#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import thop
import torch
import numpy as np
import torch.nn as nn
# from decoder import DeCoder
# from encoder import EnCoder
# from classifier import Classifier
from core.model.decoder import DeCoder
from core.model.encoder import EnCoder
from core.model.classifier import Classifier
from IPython import embed

class LGSC(nn.Module):

    def __init__(self, drop_ratio = 0.5, pretrained = True):
        super(LGSC, self).__init__()
        self.encoder = EnCoder(pretrained=pretrained)
        self.decoder = DeCoder()
        self.auxcfer = Classifier(drop_ratio=drop_ratio)
        
    def forward(self, x):
        outs = self.encoder(x)
        outs = self.decoder(outs)
        overlayed = x + outs[-1]
        clfo = self.auxcfer(overlayed)
        return outs, clfo
    
    
if __name__ == "__main__":
    
    
    input  = torch.Tensor(1, 3, 224, 224)
    antisp = LGSC()
    # Total: 27.51932M  Trainable: 27.50012M | org
    # FLOPS : 14.599G Params : 28.008M | lgsc-self
    flops, params = thop.profile(antisp, inputs=(input, ))
    flops, params = thop.clever_format([flops, params], "%.3f")
    print(flops, params)
#     antisp.eval()
#     outs, clfo = antisp(input)
#     print(clfo)
#     for feat in outs:
#         print(feat.shape)
