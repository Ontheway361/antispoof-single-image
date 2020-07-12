#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import argparse
import os.path as osp

root_dir = '/home/jovyan/jupyter/benchmark_videos/siw'
cp_dir   = '/home/jovyan/jupyter/checkpoints_zoo/face-antisp/single-1.0' 

def training_args():

    parser = argparse.ArgumentParser(description='PyTorch metricface')

    # -- env
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu_ids', type=list, default=[0, 1, 2, 3])
    parser.add_argument('--workers', type=int,  default=4)

    # -- model
    parser.add_argument('--in_size',    type=tuple,  default=(224, 224))   # FIXED
    parser.add_argument('--drop_ratio', type=float,  default=0.4)          # TODO
    parser.add_argument('--margin',     type=float,  default=0.5)          # paper-setting
    parser.add_argument('--loss_coef',  type=dict,   default={'clf_loss':5.0, 'reg_loss':5.0, 'trip_loss':1.0}) # paper-setting

    # -- optimizer
    parser.add_argument('--start_epoch', type=int,   default=1)         #
    parser.add_argument('--n_epoches',   type=int,   default=20)        # paper-setting
    parser.add_argument('--batch_size',  type=int,   default=32)         # paper-setting
    parser.add_argument('--base_lr',     type=float, default=5e-4)      # paper-setting : 1e-3 | org-repo : 5e-4
    parser.add_argument('--milestones',  type=list,  default=[5, 8, 12])# org-repo
    parser.add_argument('--gamma',       type=float, default=0.3)       # org-repo TODO
    parser.add_argument('--resume',      type=str,   default='')        # checkpoint

    # -- dataset
    parser.add_argument('--data_path',  type=str, default=root_dir)
    parser.add_argument('--train_file', type=str, default=osp.join(root_dir, 'anno_file/train_normal_sample.csv'))  # 2316
    parser.add_argument('--test_file',  type=str, default=osp.join(root_dir, 'anno_file/test_normal_sample.csv'))   # 1967

    # -- save or print
    parser.add_argument('--is_debug',  type=str, default=False)
    parser.add_argument('--save_to',   type=str, default=osp.join(cp_dir, 'siw_baseline'))
    parser.add_argument('--print_freq',type=int, default=36)   # (2316, 32, 73)
    parser.add_argument('--save_freq', type=int, default=2)    

    args = parser.parse_args()

    return args
