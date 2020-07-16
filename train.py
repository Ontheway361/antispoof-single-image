#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import os
import time
import torch
import shutil
import numpy as np
import pandas as pd
from sklearn import metrics
from torch.nn import functional as F
from torch.utils.data import DataLoader
from catalyst.data.sampler import BalanceClassSampler

import core as corelib
import dataset as dlib
from config import training_args

from IPython import embed

class LGSCTrainer(object):

    def __init__(self, args):

        self.args     = args
        self.model    = dict()
        self.data     = dict()
        self.result   = dict()
        self.softmax  = torch.nn.Softmax(dim=1)
        self.use_cuda = args.use_gpu and torch.cuda.is_available()


    def _model_loader(self):

        self.model['lgsc']      = corelib.LGSC(drop_ratio=self.args.drop_ratio)
        self.model['triplet']   = corelib.TripletLoss(margin=self.args.margin, squared=False)
        self.model['optimizer'] = torch.optim.Adam(
                                      self.model['lgsc'].parameters(), lr=self.args.base_lr)
        self.model['scheduler'] = torch.optim.lr_scheduler.MultiStepLR(
                                      self.model['optimizer'], \
                                      milestones=self.args.milestones, gamma=self.args.gamma)
        if self.use_cuda:
            self.model['lgsc'] = self.model['lgsc'].cuda()
            if len(self.args.gpu_ids) > 1:
                self.model['lgsc'] = torch.nn.DataParallel(self.model['lgsc'], device_ids=self.args.gpu_ids)
                print('Parallel mode is going ...')

        if len(self.args.resume) > 3:
            checkpoint = torch.load(self.args.resume, map_location=lambda storage, loc: storage)
            self.args.start_epoch = checkpoint['epoch']
            self.model['lgsc'].load_state_dict(checkpoint['backbone'])
            print('Resuming the train process at %3d epoches ...' % self.args.start_epoch)
        print('Model loading was finished ...')


    def _data_loader(self):

        train_trans = dlib.get_train_augmentations()
        test_trans  = dlib.get_test_augmentations()
        df_train = pd.read_csv(self.args.train_file)
        df_test  = pd.read_csv(self.args.test_file)
        if self.args.is_debug:
            k_times = 2
            df_train = df_train.sample(frac=1.0).iloc[:k_times*self.args.batch_size, :] # TODO
            df_test  = df_test.sample(frac=1.0).iloc[:k_times*self.args.batch_size, :]  # TODO
            print('### Debug mode was going ###')
            
        # labels = list(df_train.target.values)
        # sampler = BalanceClassSampler(labels, mode="upsampling")
        self.data['train_loader'] = DataLoader(
                                        dlib.DataBase(df_train, self.args.data_path, train_trans),
                                        batch_size=self.args.batch_size,
                                        shuffle=True,
                                        #sampler=sampler
                                    )
        self.data['test_loader'] = DataLoader(
                                        dlib.DataBase(df_test, self.args.data_path, test_trans),
                                        batch_size=self.args.batch_size, \
                                        shuffle=True,
                                        drop_last=False\
                                   )
        print('Data loading was finished ...')


    def calc_losses(self, outs, clf_out, target):
        ''' calculte the loss for LGSC '''

        clf_loss = F.cross_entropy(clf_out, target) * self.args.loss_coef['clf_loss']
        cue_feat = target.reshape(-1, 1, 1, 1).float() * outs[-1]
        num_reg  = (torch.sum(target) * np.prod(cue_feat.shape[1:])).type(torch.float)
        reg_loss = (torch.sum(torch.abs(cue_feat)) / (num_reg + 1e-9)) * self.args.loss_coef['reg_loss']

        trip_loss = 0
        batchsize = outs[-1].shape[0]
        for feat in outs[:-1]:
            feat = F.adaptive_avg_pool2d(feat, [1, 1]).view(batchsize, -1)
            trip_loss += self.model['triplet'](feat, target) * self.args.loss_coef['trip_loss']
        total_loss = clf_loss + reg_loss + trip_loss
        return total_loss
    
    
    def train_one_epoch(self):
        
        iter_loss_list = []
        self.model['lgsc'].train()
        for idx, (imgs, gtys) in enumerate(self.data['train_loader']):
            
            imgs.requires_grad = False
            gtys.requires_grad = False
            
            if self.use_cuda:
                imgs = imgs.cuda()
                gtys = gtys.cuda()
            
            imgs_feats, clf_out = self.model['lgsc'](imgs)
            loss = self.calc_losses(imgs_feats, clf_out, gtys)
            self.model['optimizer'].zero_grad()
            loss.backward()
            self.model['optimizer'].step()
            iter_loss_list.append(loss.item())
            if (idx + 1) % self.args.print_freq == 0:
                print('epoch : %2d|%2d, iter : %4d|%4d,  loss : %.4f' % (self.result['epoch'], 
                           self.args.n_epoches, idx+1, len(self.data['train_loader']), np.mean(iter_loss_list)))
        train_loss = np.mean(iter_loss_list)
        print('train_loss : %.4f' % train_loss)
        return train_loss
    
    
    def test_one_epoch(self):
        
        self.model['lgsc'].eval()
        with torch.no_grad():
            iter_loss_list = []
            iter_gtys_list = []
            iter_pred_list = []
            for idx, (imgs, gtys) in enumerate(self.data['test_loader']):
                
                if self.use_cuda:
                    imgs = imgs.cuda()
                    gtys = gtys.cuda()
                imgs_feats, clf_out = self.model['lgsc'](imgs)
                loss  = self.calc_losses(imgs_feats, clf_out, gtys)
                score = self.softmax(clf_out).cpu().numpy()
                predy = np.argmax(score, axis=1).tolist()
                iter_loss_list.append(loss.item())
                iter_gtys_list.extend(gtys.cpu().numpy().tolist())
                iter_pred_list.extend(predy)
        eval_info = {}
        eval_info['loss'] = np.mean(iter_loss_list)
        eval_info['racc'] = self.calculate_acc(iter_gtys_list, iter_pred_list)
        print('test_loss : %.4f, racc_score : %.4f' % (eval_info['loss'], eval_info['racc']))
        return eval_info
            
            
    @staticmethod
    def calculate_acc(gt_y, pred_y):
        
        auc        = metrics.roc_auc_score(gt_y, pred_y)
        acc        = metrics.accuracy_score(gt_y, pred_y)
        recall     = metrics.recall_score(gt_y, pred_y)
        f1_score   = metrics.f1_score(gt_y, pred_y)
        precision  = metrics.precision_score(gt_y, pred_y)
        print('auc : %.4f, acc : %.4f, precision : %.4f, recall : %.4f, f1_score : %.4f' % \
                  (auc, acc, precision, recall, f1_score))
        print('%s gt vs. pred %s' % ('-' * 36, '-' * 36))
        print(metrics.classification_report(gt_y, pred_y, digits=4))
        print(metrics.confusion_matrix(gt_y, pred_y))
        print('-' * 85)
        return acc
        
        
    def save_weights(self, testinfo = {}):
        ''' save the weights during the process of training '''
        
        if not os.path.exists(self.args.save_to):
            os.mkdir(self.args.save_to)
            
        freq_flag = self.result['epoch'] % self.args.save_freq == 0
        sota_flag = self.result['min_loss'] > testinfo['loss'] or self.result['max_acc'] < testinfo['racc']
        save_name = '%s/epoch_%02d-loss_%.4f-racc_%.4f.pth' % \
                         (self.args.save_to, self.result['epoch'], testinfo['loss'], testinfo['racc'])
        if sota_flag:
            save_name = '%s/sota.pth' % self.args.save_to
            self.result['min_loss'] = testinfo['loss']
            self.result['max_acc']  = testinfo['racc']
            print('%s Yahoo, SOTA model was updated %s' % ('*'*16, '*'*16))
        
        if sota_flag or freq_flag:
            torch.save({
                'epoch'   : self.result['epoch'], 
                'backbone': self.model['lgsc'].state_dict(),
                'loss'    : testinfo['loss'],
                'racc'    : testinfo['racc']}, save_name)
            
        if sota_flag and freq_flag:
            normal_name = '%s/epoch_%02d-loss_%.4f-racc_%.4f.pth' % \
                               (self.args.save_to, self.result['epoch'], testinfo['loss'], testinfo['racc'])
            shutil.copy(save_name, normal_name)
            

    def lgsc_training(self):
        
        self.result['max_acc']  = -1.0
        self.result['min_loss'] = 100
        for epoch in range(self.args.start_epoch, self.args.n_epoches + 1):
            
            start_time = time.time()
            self.result['epoch'] = epoch
            train_loss = self.train_one_epoch()
            test_info  = self.test_one_epoch()
            self.model['scheduler'].step()
            finish_time = time.time()
            print('single epoch costs %.4f mins' % ((finish_time - start_time) / 60))
            self.save_weights(test_info)
            if self.args.is_debug:
                break
    
    def main_runner(self):
        
        self._model_loader()
        self._data_loader()
        self.lgsc_training()

            
if __name__ == "__main__":
    
    lgsc = LGSCTrainer(args=training_args())
    lgsc.main_runner()