#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import torch
from decoder import DeCoder
from encoder import EnCoder
from IPython import embed

if __name__ == "__main__":

    input = torch.Tensor(10, 3, 224, 224)
    encoding = EnCoder()
    decoding = DeCoder()
    print(decoding)
    # res_feat = encoding(input)
    # for feat in res_feat:
    #     print(feat.shape)
    # out_feat = decoding(res_feat)
    # for feat in out_feat:
    #     print(feat.shape)
