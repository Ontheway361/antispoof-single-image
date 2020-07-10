#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import torch
import torch.nn as nn
from core.model.decoder import DeCoder
from core.model.encoder import EnCoder, Classifier
from IPython import embed

class LGSC(nn.Module):

    def __init__(self, drop_ratio = 0.5, pretrained = True):
        super(LGSC, self).__init__()
        self.encoder = EnCoder(pretrained=pretrained)
        self.decoder = DeCoder()
        self.auxcfer = Classifier(drop_ratio=drop_ratio, pretrained=pretrained)

    def forward(self, x):
        outs = self.encoder(x)
        outs = self.decoder(outs)
        s    = x + outs[-1]
        clfo = self.auxcfer(s)
        return outs, clfo


if __name__ == "__main__":

    input  = torch.Tensor(10, 3, 224, 224)
    antisp = LGSC()
    outs, clfo = antisp(input)
    for feat in outs:
        print(feat.shape)
