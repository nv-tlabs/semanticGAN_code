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

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

import argparse
from utils import inception_utils
from dataloader import (CelebAMaskDataset)
import pickle

@torch.no_grad()
def extract_features(args, loader, inception, device):
    pbar = loader

    pools, logits = [], []

    for data in pbar:
        img = data['image']
            
        # check img dim
        if img.shape[1] != 3:
            img = img.expand(-1,3,-1,-1)

        img = img.to(device)
        pool_val, logits_val = inception(img)
        
        pools.append(pool_val.cpu().numpy())
        logits.append(F.softmax(logits_val, dim=1).cpu().numpy())

    pools = np.concatenate(pools, axis=0)
    logits = np.concatenate(logits, axis=0)

    return pools, logits


def get_dataset(args):
    if args.dataset_name == 'celeba-mask':
        unlabel_dataset = CelebAMaskDataset(args, args.path, is_label=False)
        train_val_dataset = CelebAMaskDataset(args, args.path, is_label=True, phase='train-val')
        dataset = ConcatDataset([unlabel_dataset, train_val_dataset])
    else:
        raise Exception('No such a dataloader!')
    return dataset

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(
        description='Calculate Inception v3 features for datasets'
    )
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch', default=64, type=int, help='batch size')
    parser.add_argument('--n_sample', type=int, default=50000)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--image_mode', type=str, default='RGB')
    parser.add_argument('--dataset_name', type=str, help='[celeba-mask]')
    parser.add_argument('path', metavar='PATH', help='path to datset dir')

    args = parser.parse_args()

    inception = inception_utils.load_inception_net()

    dset = get_dataset(args)
    loader = DataLoader(dset, batch_size=args.batch, num_workers=4)

    pools, logits = extract_features(args, loader, inception, device)

    # pools = pools[: args.n_sample]
    # logits = logits[: args.n_sample]

    print(f'extracted {pools.shape[0]} features')

    print('Calculating inception metrics...')
    IS_mean, IS_std = inception_utils.calculate_inception_score(logits)
    print('Training data from dataloader has IS of %5.5f +/- %5.5f' % (IS_mean, IS_std))
    print('Calculating means and covariances...')

    mean = np.mean(pools, axis=0)
    cov = np.cov(pools, rowvar=False)

    with open(args.output, 'wb') as f:
        pickle.dump({'mean': mean, 'cov': cov, 'size': args.size, 'path': args.path}, f)
