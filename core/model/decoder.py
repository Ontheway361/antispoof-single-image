#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import torch
import torch.nn as nn
from .encoder import BasicBlock
from IPython import embed

class Decoder(nn.Module):

    def __init__(self):

        super(DeCoder, self).__init__()
        self.in_channels  = (64, 64, 128, 256, 512)
        self.out_channels = (512, 256, 128, 64, 64, 3)
        self.layer1 = self._make_layer(BasicBlock, 512, 256, 2)
        self.layer2 = self._make_layer(BasicBlock, 256, 128, 2)
        self.layer3 = self._make_layer(BasicBlock, 128, 64,  2)
        self.layer4 = self._make_layer(BasicBlock, 128, 64,  2)
        self.layer5 = self._make_layer(BasicBlock, 64,  3,)
        


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def _gen_layers(self):
