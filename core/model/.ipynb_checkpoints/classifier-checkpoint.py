#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import torch
import torch.nn as nn
from torchvision import models

from IPython import embed

# source-code : https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py


class Classifier(nn.Module):

    def __init__(self, num_classes = 2, drop_ratio = 0.5):

        super(Classifier, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        x = self.resnet18.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.resnet18.fc(x)
        return x


if __name__ == "__main__":

    input  = torch.Tensor(10, 3, 224, 224)
    model  = Classifier()
    output = model(input)
    print(output.shape)
