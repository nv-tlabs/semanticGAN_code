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

import argparse
import math
import random
import os
import sys
sys.path.append('..')

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter

from models.stylegan2_seg import GeneratorSeg, Discriminator, MultiscaleDiscriminator, GANLoss
from dataloader.dataset import CelebAMaskDataset

from utils.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

import functools
from utils.inception_utils import sample_gema, prepare_inception_metrics
import cv2
import random

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

def validate(args, d_img, d_seg, val_loader, device, writer, step):
    with torch.no_grad():
        d_img_val_scores = []
        d_seg_val_scores = []
        for i, data in enumerate(val_loader):
            img, mask = data['image'].to(device), data['mask'].to(device)
            
            d_img_val_score = d_img(img)
            d_seg_val_score = d_seg(prep_dseg_input(args, img, mask, is_real=True))
            d_seg_val_score = torch.tensor([feat[-1].mean() for feat in d_seg_val_score])

            d_img_val_scores.append(d_img_val_score.mean().item())
            d_seg_val_scores.append(d_seg_val_score.mean().item())
        
        d_img_val_scores = np.array(d_img_val_scores).mean()
        d_seg_val_scores = np.array(d_seg_val_scores).mean()

        print("d_img val scores: {0:.4f}, d_seg val scores: {1:.4f}".format(d_img_val_scores, d_seg_val_scores))

        writer.add_scalar('scores/d_img_val', d_img_val_scores, global_step=step)
        writer.add_scalar('scores/d_seg_val', d_seg_val_scores, global_step=step)

def prep_dseg_input(args, img, mask, is_real):
    dseg_in = torch.cat([img, mask], dim=1)

    return dseg_in

def prep_dseg_output(args, pred, use_feat=False):
    if use_feat:
        return pred
    else:
        for i in range(len(pred)):
            for j in range(len(pred[i])-1):
                pred[i][j] = pred[i][j].detach()
        return pred

def create_heatmap(mask_tensor):
    mask_np = mask_tensor.detach().cpu().numpy()
    batch_size = mask_tensor.shape[0]
    heatmap_tensors = []
    for i in range(batch_size):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask_np[i][0]), cv2.COLORMAP_JET)
        # convert BGR to RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap_tensor = torch.tensor(heatmap, dtype=torch.float)
        heatmap_tensor = heatmap_tensor.permute(2,0,1)
        heatmap_tensors.append(heatmap_tensor)
    heatmap_tensors = torch.stack(heatmap_tensors, dim=0)
    return heatmap_tensors

def train(args, ckpt_dir, img_loader, seg_loader, seg_val_loader, generator, discriminator_img,
            discriminator_seg, g_optim, d_img_optim, d_seg_optim, g_ema, device, writer):

    get_inception_metrics = prepare_inception_metrics(args.inception, False)
    # sample func for calculate FID
    sample_fn = functools.partial(sample_gema, g_ema=g_ema, device=device, 
                        truncation=1.0, mean_latent=None, batch_size=args.batch)

    # d_seg gan loss
    seg_gan_loss = GANLoss(gan_mode='hinge', tensor=torch.cuda.FloatTensor)

    img_loader = sample_data(img_loader)
    seg_loader = sample_data(seg_loader)
    pbar = range(args.iter)

    mean_path_length = 0

    d_loss_val = 0
    r1_img_loss = torch.tensor(0.0, device=device)
    r1_seg_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_img_module = discriminator_img.module
        d_seg_module = discriminator_seg.module
    else:
        g_module = generator
        d_img_module = discriminator_img
        d_seg_module = discriminator_seg
        
    accum = 0.5 ** (32 / (10 * 1000))

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')

            break
        
        # train image and segmentation discriminator
        real_img = next(img_loader)['image']

        real_img = real_img.to(device)
       
        seg_data = next(seg_loader)
        seg_img, seg_mask = seg_data['image'], seg_data['mask']
        seg_img, seg_mask = seg_img.to(device), seg_mask.to(device)

        # =============================== Step1: train the d_img ===================================
        requires_grad(generator, False)
        requires_grad(discriminator_img, True)
        requires_grad(discriminator_seg, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, fake_seg = generator(noise)

        # detach fake seg
        fake_seg = fake_seg.detach()

        fake_img_pred = discriminator_img(fake_img)
        
        real_img_pred = discriminator_img(real_img)

        d_img_loss = d_logistic_loss(real_img_pred, fake_img_pred)

        loss_dict['d_img'] = d_img_loss
        loss_dict['d_img_real_score'] = real_img_pred.mean()
        loss_dict['d_img_fake_score'] = fake_img_pred.mean()

        discriminator_img.zero_grad()
        d_img_loss.backward()
        d_img_optim.step()
        
        # =============================== Step2: train the d_seg ===================================
        requires_grad(generator, False)
        requires_grad(discriminator_img, False)
        requires_grad(discriminator_seg, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, fake_seg = generator(noise)

        fake_seg_pred = discriminator_seg(prep_dseg_input(args, fake_img, fake_seg, is_real=False))
        real_seg_pred = discriminator_seg(prep_dseg_input(args, seg_img, seg_mask, is_real=True))

        # prepare output
        fake_seg_pred = prep_dseg_output(args, fake_seg_pred, use_feat=False)
        real_seg_pred = prep_dseg_output(args, real_seg_pred, use_feat=False)

        d_seg_loss = (seg_gan_loss(fake_seg_pred, False, for_discriminator=True).mean() + seg_gan_loss(real_seg_pred, True, for_discriminator=True).mean()) / 2.0
     
        loss_dict['d_seg'] = d_seg_loss
        loss_dict['d_seg_real_score'] = (real_seg_pred[0][-1].mean()+real_seg_pred[1][-1].mean()+real_seg_pred[2][-1].mean()) / 3.0
        loss_dict['d_seg_fake_score'] = (fake_seg_pred[0][-1].mean()+fake_seg_pred[1][-1].mean()+fake_seg_pred[2][-1].mean()) / 3.0

        discriminator_seg.zero_grad()
        d_seg_loss.backward()
        d_seg_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred = discriminator_img(real_img)
            r1_img_loss = d_r1_loss(real_pred, real_img)

            discriminator_img.zero_grad()
            (args.r1 / 2 * r1_img_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_img_optim.step()
            
            # seg discriminator regulate
            real_img_seg = prep_dseg_input(args, seg_img, seg_mask, is_real=True)
            real_img_seg.requires_grad = True
 
            real_pred = discriminator_seg(real_img_seg)
            real_pred = prep_dseg_output(args, real_pred, use_feat=False)

            # select three D
            real_pred = real_pred[0][-1].mean() + real_pred[1][-1].mean() + real_pred[2][-1].mean()

            r1_seg_loss = d_r1_loss(real_pred, real_img_seg)
            
            discriminator_seg.zero_grad()
            (args.r1 / 2 * r1_seg_loss * args.d_reg_every + 0 * real_pred).backward()

            d_seg_optim.step()

        loss_dict['r1_img'] = r1_img_loss
        loss_dict['r1_seg'] = r1_seg_loss

        # =============================== Step3: train the generator ===================================
        requires_grad(generator, True)
        requires_grad(discriminator_img, False)
        requires_grad(discriminator_seg, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, fake_seg = generator(noise)

        fake_img_pred = discriminator_img(fake_img)
 
        # stop gradient from d_seg to g_img
        fake_seg_pred = discriminator_seg(prep_dseg_input(args, fake_img.detach(), fake_seg, is_real=False))
        real_seg_pred = discriminator_seg(prep_dseg_input(args, seg_img, seg_mask, is_real=True))

        # prepare output
        fake_seg_pred = prep_dseg_output(args, fake_seg_pred, use_feat=True)
        real_seg_pred = prep_dseg_output(args, real_seg_pred, use_feat=False)

        g_img_loss = g_nonsaturating_loss(fake_img_pred)
        
        # g seg adv loss
        g_seg_adv_loss = seg_gan_loss(fake_seg_pred, True, for_discriminator=False).mean()

        # g seg feat loss
        g_seg_feat_loss = 0.0
        feat_weights = 1.0
        D_weights = 1.0 / 3.0

        for D_i in range(len(fake_seg_pred)):
            for D_j in range(len(fake_seg_pred[D_i])-1):
                g_seg_feat_loss += D_weights * feat_weights * \
                    F.l1_loss(fake_seg_pred[D_i][D_j], real_seg_pred[D_i][D_j].detach()) * args.lambda_dseg_feat

        g_loss = g_img_loss + g_seg_adv_loss + g_seg_feat_loss
  
        loss_dict['g_img'] = g_img_loss
        loss_dict['g_seg_adv'] = g_seg_adv_loss
        loss_dict['g_seg_feat'] = g_seg_feat_loss
        loss_dict['g'] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(
                path_batch_size, args.latent, args.mixing, device
            )
            fake_img, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()
      
            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict['path'] = path_loss
        loss_dict['path_length'] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_img_loss_val = loss_reduced['d_img'].mean().item()
        d_seg_loss_val = loss_reduced['d_seg'].mean().item()
        g_loss_val = loss_reduced['g'].mean().item()
        g_img_loss_val = loss_reduced['g_img'].mean().item()

        g_seg_adv_loss_val = loss_reduced['g_seg_adv'].mean().item()
        g_seg_feat_loss_val = loss_reduced['g_seg_feat'].mean().item()
        r1_img_val = loss_reduced['r1_img'].mean().item()
        r1_seg_val = loss_reduced['r1_seg'].mean().item()
        

        d_img_real_score_val = loss_reduced['d_img_real_score'].mean().item()
        d_img_fake_score_val = loss_reduced['d_img_fake_score'].mean().item()
        d_seg_real_score_val = loss_reduced['d_seg_real_score'].mean().item()
        d_seg_fake_score_val = loss_reduced['d_seg_fake_score'].mean().item()

        path_loss_val = loss_reduced['path'].mean().item()
        path_length_val = loss_reduced['path_length'].mean().item()

        if get_rank() == 0:
            # write to tensorboard
            writer.add_scalars('scores/d_img',{'real_score': d_img_real_score_val,
                                                'fake_score': d_img_fake_score_val
                                                    }, global_step=i)

            writer.add_scalars('scores/d_seg',{'real_score': d_seg_real_score_val,
                                                'fake_score': d_seg_fake_score_val
                                                    }, global_step=i)
            
            writer.add_scalar('r1/d_img', r1_img_val, global_step=i)
            writer.add_scalar('r1/d_seg', r1_seg_val, global_step=i)

            writer.add_scalar('path/path_loss', path_loss_val, global_step=i)
            writer.add_scalar('path/path_length', path_length_val, global_step=i)

            writer.add_scalar('g/img_loss', g_img_loss_val, global_step=i)
            writer.add_scalar('g/seg_adv_loss', g_seg_adv_loss_val, global_step=i)
            writer.add_scalar('g/seg_feat_loss', g_seg_feat_loss_val, global_step=i)


            if i % args.viz_every == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample_img, sample_seg = g_ema([sample_z])
                    sample_img = sample_img.detach().cpu()
                    sample_seg = sample_seg.detach().cpu()

                    if args.seg_name == 'celeba-mask':
                        sample_seg = torch.argmax(sample_seg, dim=1)
                        color_map = seg_val_loader.dataset.color_map
                        sample_mask = torch.zeros((sample_seg.shape[0], sample_seg.shape[1], sample_seg.shape[2], 3), dtype=torch.float)
                        for key in color_map:
                            sample_mask[sample_seg==key] = torch.tensor(color_map[key], dtype=torch.float)
                        sample_mask = sample_mask.permute(0,3,1,2)
                    
                    else:
                        raise Exception('No such a dataloader!')

                    os.makedirs(os.path.join(ckpt_dir, 'sample'), exist_ok=True)
                    utils.save_image(
                        sample_img,
                        os.path.join(ckpt_dir, f'sample/img_{str(i).zfill(6)}.png'),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
                    
                    utils.save_image(
                            sample_mask,
                            os.path.join(ckpt_dir, f'sample/mask_{str(i).zfill(6)}.png'),
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                    )

            

            if i % args.eval_every == 0:
                print("==================Start calculating validation scores==================")
                validate(args, discriminator_img, discriminator_seg, seg_val_loader, device, writer, i)
                
            if i % args.save_every == 0:
                print("==================Start calculating FID==================")
                IS_mean, IS_std, FID = get_inception_metrics(sample_fn, num_inception_images=10000, use_torch=False)
                print("iteration {0:08d}: FID: {1:.4f}, IS_mean: {2:.4f}, IS_std: {3:.4f}".format(i, FID, IS_mean, IS_std))


                writer.add_scalar('metrics/FID', FID, global_step=i)
                writer.add_scalar('metrics/IS_mean', IS_mean, global_step=i)
                writer.add_scalar('metrics/IS_std', IS_std, global_step=i)

                writer.add_text('metrics/FID', 'FID is {0:.4f}'.format(FID), global_step=i)
                
                
                os.makedirs(os.path.join(ckpt_dir, 'ckpt'), exist_ok=True)
                torch.save(
                    {
                        'g': g_module.state_dict(),
                        'd_img': d_img_module.state_dict(),
                        'd_seg': d_seg_module.state_dict(),
                        'g_ema': g_ema.state_dict(),
                        'args': args,
                    },
                    os.path.join(ckpt_dir, f'ckpt/{str(i).zfill(6)}.pt'),
                )

def get_seg_dataset(args, phase='train'):
    if args.seg_name == 'celeba-mask':
        seg_dataset = CelebAMaskDataset(args, args.seg_dataset, is_label=True, phase=phase,
                                            limit_size=args.limit_data, aug=args.seg_aug, resolution=args.size)
   
    else:
        raise Exception('No such a dataloader!')
    
    return seg_dataset

def get_transformation(args):
    if args.seg_name == 'celeba-mask':
        transform = transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5), inplace=True)
                    ]
                )
    
    else:
        raise Exception('No such a dataloader!')
    
    return transform

if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--img_dataset', type=str, required=True)
    parser.add_argument('--seg_dataset', type=str, required=True)
    parser.add_argument('--inception', type=str, help='inception pkl', required=True)

    parser.add_argument('--seg_name', type=str, help='segmentation dataloader name[celeba-mask]', default='celeba-mask')
    parser.add_argument('--iter', type=int, default=800000)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--n_sample', type=int, default=64)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--path_regularize', type=float, default=2)
    parser.add_argument('--path_batch_shrink', type=int, default=2)
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--d_use_seg_every', type=int, help='frequency mixing seg image with real image', default=-1)
    parser.add_argument('--viz_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=2000)

    parser.add_argument('--mixing', type=float, default=0.9)
    parser.add_argument('--lambda_dseg_feat', type=float, default=2.0)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    
    parser.add_argument('--limit_data', type=str, default=None, help='number of limited label data point to use')
    parser.add_argument('--unlabel_limit_data', type=str, default=None, help='number of limited unlabel data point to use')

    parser.add_argument('--image_mode', type=str, default='RGB', help='Image mode RGB|L')
    parser.add_argument('--seg_dim', type=int, default=8)
    parser.add_argument('--seg_aug', action='store_true', help='seg augmentation')

    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/')

    args = parser.parse_args()

    # build checkpoint dir
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    ckpt_dir = os.path.join(args.checkpoint_dir, 'run-'+current_time)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(ckpt_dir, 'logs'))

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.n_gpu = n_gpu

    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    generator = GeneratorSeg(
        args.size, args.latent, args.n_mlp, seg_dim=args.seg_dim,
        image_mode=args.image_mode, channel_multiplier=args.channel_multiplier
    ).to(device)

    if args.image_mode == 'RGB':
        d_input_dim = 3
    else:
        d_input_dim = 1

    d_seg_input_dim = d_input_dim + args.seg_dim

    discriminator_img = Discriminator(
        args.size, input_dim=d_input_dim, channel_multiplier=args.channel_multiplier
    ).to(device)

    discriminator_seg = MultiscaleDiscriminator(input_nc=d_seg_input_dim, getIntermFeat=True).to(device)
 
    g_ema = GeneratorSeg(
        args.size, args.latent, args.n_mlp, seg_dim=args.seg_dim,
        image_mode=args.image_mode, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_img_optim = optim.Adam(
        discriminator_img.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    d_seg_optim = optim.Adam(
        discriminator_seg.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print('load model:', args.ckpt)
        
        ckpt = torch.load(args.ckpt)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
            
        except ValueError:
            pass
            
        generator.load_state_dict(ckpt['g'])
        discriminator_img.load_state_dict(ckpt['d_img'])
        discriminator_seg.load_state_dict(ckpt['d_seg'])
        g_ema.load_state_dict(ckpt['g_ema'])

        g_optim.load_state_dict(ckpt['g_optim'])
        d_img_optim.load_state_dict(ckpt['d_img_optim'])
        d_seg_optim.load_state_dict(ckpt['d_seg_optim'])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        discriminator_img = nn.parallel.DistributedDataParallel(
            discriminator_img,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        discriminator_seg = nn.parallel.DistributedDataParallel(
            discriminator_seg,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )


    if args.seg_name == 'celeba-mask':
        transform = get_transformation(args)
        img_dataset = CelebAMaskDataset(args, args.img_dataset, unlabel_transform=transform, unlabel_limit_size=args.unlabel_limit_data,
                                                is_label=False, resolution=args.size)
    else:
        raise Exception('No such a dataloader!')

    print("Loading unlabel dataloader with size ", img_dataset.data_size)

    img_loader = data.DataLoader(
        img_dataset,
        batch_size=args.batch,
        sampler=data_sampler(img_dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    seg_dataset = get_seg_dataset(args, phase='train')

    print("Loading train dataloader with size ", seg_dataset.data_size)

    seg_loader = data.DataLoader(
        seg_dataset,
        batch_size=args.batch,
        sampler=data_sampler(seg_dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    seg_val_dataset = get_seg_dataset(args, phase='val')

    print("Loading val dataloader with size ", seg_val_dataset.data_size)

    seg_val_loader = data.DataLoader(
        seg_val_dataset,
        batch_size=args.batch,
        shuffle=False,
        drop_last=True,
    )

    torch.backends.cudnn.benchmark = True
    
    train(args, ckpt_dir, img_loader, seg_loader, seg_val_loader, generator, discriminator_img, discriminator_seg,
                    g_optim, d_img_optim, d_seg_optim, g_ema, device, writer)
