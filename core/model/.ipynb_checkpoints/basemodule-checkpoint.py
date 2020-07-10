#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def _make_res_layer(block, inplanes, outplanes, num_blocks = 2, stride = 1, \
                    dilation = 1, norm_layer = None):

    if norm_layer is None:
        norm_layer = nn.BatchNorm2d

    downsample = None
    if stride != 1 or inplanes != outplanes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(inplanes, outplanes * block.expansion, stride),
            norm_layer(outplanes * block.expansion),
        )
    layers = []
    layers.append(block(inplanes, outplanes, stride, downsample, norm_layer=norm_layer))
    inplanes = outplanes * block.expansion
    for _ in range(1, num_blocks):
        layers.append(block(inplanes, outplanes))
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
