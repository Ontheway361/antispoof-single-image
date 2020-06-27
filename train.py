#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import torch
import numpy as np
import pandas as pd
from sklearn import metrics
from torch.nn import functional as F
from torch.utils.data import DataLoader
from catalyst.data.sampler import BalanceClassSampler

import core as clib
import dataset as dlib


class LGSCTrainer(object):

    def __init__(self, args):

        self.args   = args
        self.model  = dict()
        self.data   = dict()
        self.device = args.use_gpu and torch.cuda.is_available()


    def _model_loader(self):

        self.model['lgsc']      = clib.LGSC(drop_ratio=args.drop_ratio)
        self.model['triplet']   = clib.TripletLoss(margin=args.margin)
        self.model['optimizer'] = torch.optim.Adam(
                                      self.model['lgsc'].parameters(), lr=self.args.base_lr)
        self.model['scheduler'] = torch.optim.lr_scheduler.MultiStepLR(
                                      self.model['optimizer'], \
                                      milestones=self.args.milestones, gamma=self.args.gamma)
        if self.device:
            self.model['lgsc'] = self.model['lgsc'].cuda()
            if len(args.gpu_ids) > 1:
                self.model['lgsc'] = torch.nn.DataParallel(self.model['lgsc'], device_ids=self.args.gpu_ids)
                print('Parallel mode is going ...')

        if len(self.resume) > 3:
            checkpoint = torch.load(self.args.resume, map_location=lambda storage, loc: storage)
            self.args.start_epoch = checkpoint['epoch']
            self.model['lgsc'].load_state_dict(checkpoint['backbone'])
            print('Resuming the train process at %3d epoches ...' % self.args.start_epoch)
        print('Model loading was finished ...')


    def _data_loader(self):

        train_trans = dlib.get_train_augmentations()
        df_train = pd.read_csv(self.args.train_file)
        dataset = dlib.DataBase(df_train, self.args.data_path, train_trans)
        labels = list(df_train.target.values)
        sampler = BalanceClassSampler(labels, mode="upsampling")
        self.data['train_loader'] = DataLoader(
                                        dataset,
                                        batch_size=self.args.batch_size,
                                        num_workers=self.args.workers,
                                        sampler=sampler)

        test_trans = dlib.get_test_augmentations()
        df_test = pd.read_csv(self.args.test_file)
        dataset = dlib.DataBase(df_test, self.args.data_path, test_trans)
        self.data['test_loader'] = DataLoader(
                                        dataset,
                                        batch_size=self.args.batch_size, \
                                        num_workers=self.args.workers,
                                        shuffle=True,
                                    )
        print('Data loading was finished ...')


    def calc_losses(self, outs, clf_out, target):
        ''' calculte the loss for LGSC '''

        clf_loss = F.cross_entropy(clf_out, target) * self.args.loss_coef['clf_loss']
        cue = target.reshape(-1, 1, 1, 1) * outs[-1]
        num_reg = (torch.sum(target) * cue.shape[1] * cue.shape[2] * cue.shape[3]).type(torch.float)
        reg_loss = (torch.sum(torch.abs(cue)) / (num_reg + 1e-9)) * self.args.loss_coef['reg_loss']

        trip_loss = 0
        batchsize = outs[-1].shape[0]
        for feat in outs[:-1]:
            feat = F.adaptive_avg_pool2d(feat, [1, 1]).view(batchsize, -1)
            trip_loss += self.triplet_loss(feat, target) * self.args.loss_coef['trip_loss']
        total_loss = clf_loss + reg_loss + trip_loss

        return total_loss


    def training_step(self, batch, batch_idx):
        input_ = batch[0]
        target = batch[1]
        outs, clf_out = self(input_)
        loss = self.calc_losses(outs, clf_out, target)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}


    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {
            "train_avg_loss": avg_loss,
        }
        return {"train_avg_loss": avg_loss, "log": tensorboard_logs}


    def validation_step(self, batch, batch_idx):
        input_ = batch[0]
        target = batch[1]
        outs, clf_out = self(input_)
        loss = self.calc_losses(outs, clf_out, target)
        val_dict = {
            "val_loss": loss,
            "score": clf_out.cpu().numpy(),
            "target": target.cpu().numpy(),
        }
        return val_dict

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        targets = np.hstack([output["target"] for output in outputs])
        scores = np.vstack([output["score"] for output in outputs])[:, 1]
        roc_auc = metrics.roc_auc_score(targets, scores)
        tensorboard_logs = {"val_loss": avg_loss, "val_roc_auc": roc_auc}
        return {"val_loss": avg_loss, "log": tensorboard_logs}
