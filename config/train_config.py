#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import os
import argparse

root_dir  = '/home/jovyan/jupyter/benchmark_images/faceu'

def training_args():

    parser = argparse.ArgumentParser(description='PyTorch metricface')

    # -- env
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu_ids', type=list, default=[0, 1])
    parser.add_argument('--workers', type=int,  default=4)

    # -- model
    parser.add_argument('--in_size',    type=tuple,  default=(224, 224))   # FIXED
    parser.add_argument('--drop_ratio', type=float,  default=0.4)          # TODO
    parser.add_argument('--margin',     type=float,  default=0..5)
    parser.add_argument('--loss_coef',  type=dict,   default={'clf_loss':5.0, 'reg_loss':5.0, 'trip_loss':1.0})

    # -- optimizer
    parser.add_argument('--start_epoch', type=int,   default=1)        #
    parser.add_argument('--end_epoch',   type=int,   default=15)
    parser.add_argument('--batch_size',  type=int,   default=96)      # TODO | 300
    parser.add_argument('--base_lr',     type=float, default=5e-4)      # default = 0.1
    parser.add_argument('--milestones',  type=list,  default=[5, 8, 12])
    parser.add_argument('--gamma',       type=float, default=0.3)      # FIXED
    parser.add_argument('--weight_decay',type=float, default=5e-4)     # FIXED
    parser.add_argument('--resume',      type=str,    default='')      # checkpoint


    # -- dataset
    parser.add_argument('--data_path',  type=str, default=root_dir)
    parser.add_argument('--train_file', type=str, default=osp.join(casia_dir, 'anno_file/casia_landmark.txt'))
    parser.add_argument('--test_file',  type=str, default=osp.join(casia_dir, 'anno_file/casia_landmark.txt'))

    # -- save or print
    parser.add_argument('--is_debug',  type=str,   default=True)   # TODO
    parser.add_argument('--save_to',   type=str,   default=osp.join(cp_dir, 'dul'))
    parser.add_argument('--print_freq',type=int,   default=300)  # v0 : 454589, v1 : 500396, v2 : 509539, v3 : 513802, v4 : 517673
    parser.add_argument('--save_freq', type=int,   default=3)  # TODO

    args = parser.parse_args()

    return args
