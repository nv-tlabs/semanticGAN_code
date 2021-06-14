"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import torch
import cv2
import albumentations
import albumentations.augmentations as A

class HistogramEqualization(object):
    def __call__(self, img):
        img_eq = ImageOps.equalize(img)
        
        return img_eq

class AdjustGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma
    
    def __call__(self, img):
        img_gamma = transforms.functional.adjust_gamma(img, self.gamma)

        return img_gamma

class CelebAMaskDataset(Dataset):
    def __init__(self, args, dataroot, unlabel_transform=None, latent_dir=None, is_label=True, phase='train', 
                    limit_size=None, unlabel_limit_size=None, aug=False, resolution=256):

        self.args = args
        self.is_label = is_label


        if is_label == True:
            self.latent_dir = latent_dir
            self.data_root = os.path.join(dataroot, 'label_data')
        
            if phase == 'train':
                if limit_size is None:
                    self.idx_list = np.loadtxt(os.path.join(self.data_root, 'train_full_list.txt'), dtype=str)
                else:
                    self.idx_list = np.loadtxt(os.path.join(self.data_root, 
                                            'train_{}_list.txt'.format(limit_size)), dtype=str).reshape(-1)
            elif phase == 'val':
                if limit_size is None:
                    self.idx_list = np.loadtxt(os.path.join(self.data_root, 'val_full_list.txt'), dtype=str)
                else:
                    self.idx_list = np.loadtxt(os.path.join(self.data_root, 
                                            'val_{}_list.txt'.format(limit_size)), dtype=str).reshape(-1)
            elif phase == 'train-val':
                # concat both train and val
                if limit_size is None:
                    train_list = np.loadtxt(os.path.join(self.data_root, 'train_full_list.txt'), dtype=str)
                    val_list = np.loadtxt(os.path.join(self.data_root, 'val_full_list.txt'), dtype=str)
                    self.idx_list = list(train_list) + list(val_list)
                else:
                    train_list = np.loadtxt(os.path.join(self.data_root, 
                                            'train_{}_list.txt'.format(limit_size)), dtype=str).reshape(-1)
                    val_list = np.loadtxt(os.path.join(self.data_root, 
                                            'val_{}_list.txt'.format(limit_size)), dtype=str).reshape(-1)
                    self.idx_list = list(train_list) + list(val_list)
            else:
                self.idx_list = np.loadtxt(os.path.join(self.data_root, 'test_list.txt'), dtype=str)
        else:
            self.data_root = os.path.join(dataroot, 'unlabel_data')
            if unlabel_limit_size is None:
                self.idx_list = np.loadtxt(os.path.join(self.data_root, 'unlabel_list.txt'), dtype=str)
            else:
                self.idx_list = np.loadtxt(os.path.join(self.data_root, 'unlabel_{}_list.txt'.format(unlabel_limit_size)), dtype=str)

        self.img_dir = os.path.join(self.data_root, 'image')
        self.label_dir = os.path.join(self.data_root, 'label')

        self.phase = phase
        self.color_map = {
            0: [  0,   0,   0],
            1: [ 0,0,205],
            2: [132,112,255],
            3: [ 25,25,112],
            4: [187,255,255],
            5: [ 102,205,170],
            6: [ 227,207,87],
            7: [ 142,142,56]
        }

        self.data_size = len(self.idx_list)
        self.resolution = resolution

        self.aug = aug
        if aug == True:
            self.aug_t = albumentations.Compose([
                            A.transforms.HorizontalFlip(p=0.5),
                            A.transforms.ShiftScaleRotate(shift_limit=0.1,
                                                scale_limit=0.2,
                                                rotate_limit=15,
                                                border_mode=cv2.BORDER_CONSTANT,
                                                value=0,
                                                mask_value=0,
                                                p=0.5),
                    ])

        self.unlabel_transform = unlabel_transform
        

    def _mask_labels(self, mask_np):
        label_size = len(self.color_map.keys())
        labels = np.zeros((label_size, mask_np.shape[0], mask_np.shape[1]))
        for i in range(label_size):
            labels[i][mask_np==i] = 1.0
        
        return labels

    
    @staticmethod
    def preprocess(img):
        image_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5), inplace=True)
                    ]
                )
        img_tensor = image_transform(img)
        # normalize
        # img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
        # img_tensor = (img_tensor - 0.5) / 0.5

        return img_tensor
        

    def __len__(self):
        if hasattr(self.args, 'n_gpu') == False:
            return self.data_size
        # make sure dataloader size is larger than batchxngpu size
        return max(self.args.batch*self.args.n_gpu, self.data_size)
    
    def __getitem__(self, idx):
        if idx >= self.data_size:
            idx = idx % (self.data_size)
        img_idx = self.idx_list[idx]
        img_pil = Image.open(os.path.join(self.img_dir, img_idx)).convert('RGB').resize((self.resolution, self.resolution))
        mask_pil = Image.open(os.path.join(self.label_dir, img_idx)).convert('L').resize((self.resolution, self.resolution), resample=0)
        
        if self.is_label:
            if (self.phase == 'train' or self.phase == 'train-val') and self.aug:
                augmented = self.aug_t(image=np.array(img_pil), mask=np.array(mask_pil))
                aug_img_pil = Image.fromarray(augmented['image'])
                # apply pixel-wise transformation
                img_tensor = self.preprocess(aug_img_pil)

                mask_np = np.array(augmented['mask'])
                labels = self._mask_labels(mask_np)

                mask_tensor = torch.tensor(labels, dtype=torch.float)
                mask_tensor = (mask_tensor - 0.5) / 0.5

            else:
                img_tensor = self.preprocess(img_pil)
                mask_np = np.array(mask_pil)
                labels = self._mask_labels(mask_np)

                mask_tensor = torch.tensor(labels, dtype=torch.float)
                mask_tensor = (mask_tensor - 0.5) / 0.5
            
            return {
                'image': img_tensor,
                'mask': mask_tensor
            }
        else:
            img_tensor = self.unlabel_transform(img_pil)
            return {
                'image': img_tensor,
            }
