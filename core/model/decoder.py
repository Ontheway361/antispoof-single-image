#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F
from basemodule import BasicBlock, _make_res_layer
from IPython import embed


class DeCoder(nn.Module):

    def __init__(self):

        super(DeCoder, self).__init__()
        self.in_channels  = (64, 64, 128, 256, 512)
        self.out_channels = (512, 256, 128, 64, 64, 3)
        self.res_layers   = []
        self.conv1x1      = []
        self.conv2x2      = []
        self._make_layers()

    def _make_layers(self):

        for i in range(len(self.in_channels)-1, -1, -1):

            res_layer = _make_res_layer(
                            BasicBlock,
                            inplanes=128 if i == 1 else self.in_channels[i],
                            outplanes=self.out_channels[-(i+1)],
                            num_blocks=2,
                            norm_layer=nn.InstanceNorm2d)

            out_planes = self.in_channels[i] if i < 2 else int(self.in_channels[i] / 2)
            conv2x2 = nn.Sequential(
                          nn.Conv2d(self.in_channels[i], out_planes, kernel_size=2, bias=False),
                          nn.InstanceNorm2d(out_planes),
                          nn.ReLU(inplace=True))

            conv1x1 = nn.Sequential(
                          nn.Conv2d(in_channels  = 128 if i == 1 else self.in_channels[i],
                                    out_channels = self.out_channels[-(i+1)],
                                    kernel_size=1,
                                    bias=False),
                          nn.InstanceNorm2d(out_planes))
            self.res_layers.append(res_layer)
            self.conv2x2.append(conv2x2)
            self.conv1x1.append(conv1x1)
        self.res_layers = nn.ModuleList(self.res_layers)
        self.conv2x2    = nn.ModuleList(self.conv2x2)
        self.conv1x1    = nn.ModuleList(self.conv1x1)

    def forward(self, x):

        assert len(x) == len(self.in_channels)

        out  = x[-1]
        outs = []
        outs.append(out)

        for i in range(len(self.in_channels)):

            out = F.interpolate(out, scale_factor=2, mode='nearest')
            out = F.pad(out, [0, 1, 0, 1])
            out = self.conv2x2[i](out)
            if i < 4:
                out = torch.cat([out, x[-(i+2)]], dim=1)
            identity = self.conv1x1[i](out)
            out = self.res_layers[i](out) + identity
            outs.append(out)
        outs[-1] = torch.tanh(outs[-1])
        return outs
