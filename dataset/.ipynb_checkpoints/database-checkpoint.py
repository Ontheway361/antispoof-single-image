#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import os
import cv2
import torch
import numpy as np
# from PIL import Image
import albumentations as alt
from albumentations.pytorch import ToTensorV2 as ToTensor


def get_train_augmentations():
    return alt.Compose(
        [
            alt.LongestMaxSize(512),
            alt.CoarseDropout(20),
            alt.Rotate(30),
            alt.RandomCrop(224, 224, p=0.5),
            alt.LongestMaxSize(224),
            alt.PadIfNeeded(224, 224, 0),
            alt.Normalize(),
            ToTensor(),
        ]
    )


def get_test_augmentations():
    return alt.Compose([alt.LongestMaxSize(224), alt.PadIfNeeded(224, 224, 0), alt.Normalize(), ToTensor()])


class DataBase(torch.utils.data.Dataset):
    ''' Dataset-base-vision for benchmark '''
    def __init__(self, df, root, transforms):

        self.df = df
        self.root = root
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        path = os.path.join(self.root, self.df.iloc[index].path)
        file = np.random.choice(os.listdir(path))
        full_path = os.path.join(path, file)
        # image = np.array(Image.open(full_path))
        image = cv2.imread(full_path)
        image = self.transforms(image=image)['image']
        target = self.df.iloc[index].target
        return image, target
