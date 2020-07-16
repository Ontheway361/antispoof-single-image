#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import torch
import torch.nn as nn
from torchvision import models

from IPython import embed

# source-code : https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

class EnCoder(nn.Module):

    def __init__(self, pretrained = True):

        super(EnCoder, self).__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        # self._freeze_model()

    def _freeze_model(self):
        for p in self.backbone.parameters():
            p.required_grad = False

    def forward(self, x):
        outs = []
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        outs.append(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        outs.append(x)
        x = self.backbone.layer2(x)
        outs.append(x)
        x = self.backbone.layer3(x)
        outs.append(x)
        x = self.backbone.layer4(x)
        outs.append(x)
        return tuple(outs)
    

if __name__ == "__main__":

    input  = torch.Tensor(1, 3, 224, 224)
    model  = EnCoder()
    output = model(input)
    for i, feat in enumerate(output):
        print(feat.shape)
