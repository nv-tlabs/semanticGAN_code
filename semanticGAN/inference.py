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
import os
from PIL import Image
import argparse
import sys
sys.path.append('..')

from models.stylegan2_seg import GeneratorSeg
from models.encoder_model import FPNEncoder

import torch
from torch import optim
import torch.nn.functional as F
import math
from torchvision import transforms
from models import lpips


def mask2rgb(args, mask):
    if args.dataset_name == 'celeba-mask':
        color_table = torch.tensor(
                        [[  0,   0,   0],
                        [ 0,0,205],
                        [132,112,255],
                        [ 25,25,112],
                        [187,255,255],
                        [ 102,205,170],
                        [ 227,207,87],
                        [ 142,142,56]], dtype=torch.float)

    else:
        raise Exception('No such a dataloader!')

    rgb_tensor = F.embedding(mask, color_table).permute(0,3,1,2)
    return rgb_tensor

def make_mask(args, tensor, threshold=0.5):
    if args.seg_dim == 1:
        seg_prob = torch.sigmoid(tensor)
        seg_mask = torch.zeros_like(tensor)
        seg_mask[seg_prob > threshold] = 1.0
        seg_mask = (seg_mask.to('cpu')
                       .mul(255)
                       .type(torch.uint8)
                       .permute(0, 2, 3, 1)
                       .numpy())
    else:
        seg_prob = torch.argmax(tensor, dim=1)
        seg_mask = mask2rgb(args, seg_prob)
        seg_mask = (seg_mask.to('cpu')
                       .type(torch.uint8)
                       .permute(0, 2, 3, 1)
                       .numpy())
    

    return seg_mask

def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to('cpu')
        .numpy()
    )


def overlay_img_and_mask(args, img_pil, mask_pil, alpha=0.3):
    img_pil = img_pil.convert('RGBA')
    mask_pil = mask_pil.convert('RGBA')

    overlay_pil = Image.blend(img_pil, mask_pil, alpha)
    
    return overlay_pil

def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

def get_transformation(args):
    if args.dataset_name == 'celeba-mask':
        transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
                ]
            )
    elif args.dataset_name == 'cxr':
        transform = transforms.Compose(
                        [
                            HistogramEqualization(),
                            AdjustGamma(0.5),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                        ]
                    )
    elif args.dataset_name == 'isic':
        transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
                ]
            )
    else:
        raise Exception('No such a dataloader!')
    
    return transform



if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    parser.add_argument('--dataset_name', type=str, help='segmentation dataloader name [celeba-mask|cxr|isic]', default='celeba-mask')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--image_mode', type=str, default='RGB')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seg_dim', type=int, default=8)
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0])

    parser.add_argument('--mean_init', action='store_true', help='initialize latent code with mean')
    parser.add_argument('--no_noises', action='store_true')
    parser.add_argument('--w_plus', action='store_true', help='optimize in w+ space, otherwise w space')

    parser.add_argument('--save_latent', action='store_true')
    parser.add_argument('--save_steps', action='store_true', help='if to save intermediate optimization results')

    parser.add_argument('--truncation', type=float, default=1, help='truncation tricky, trade-off between quality and diversity')
    parser.add_argument('--truncation_mean', type=int, default=4096)

    parser.add_argument('--lr_rampup', type=float, default=0.05)
    parser.add_argument('--lr_rampdown', type=float, default=0.25)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--noise_ramp', type=float, default=0.75)
    parser.add_argument('--step', type=int, default=100, help='optimization steps [100-500 should give good results]')
    parser.add_argument('--noise_regularize', type=float, default=1e2)
    parser.add_argument('--lambda_mse', type=float, default=0.1)
    parser.add_argument('--lambda_mean', type=float, default=0.01)
    parser.add_argument('--lambda_label', type=float, default=1.0)
    parser.add_argument('--lambda_encoder', type=float, default=1e-3)
    parser.add_argument('--lambda_encoder_init', type=float, default=0.0)

    args = parser.parse_args()
    print(args)

    args.latent = 512
    args.n_mlp = 8

    checkpoint = torch.load(args.ckpt)
 
    g_ema = GeneratorSeg(args.size, args.latent, args.n_mlp, seg_dim=args.seg_dim, image_mode=args.image_mode,
        channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.load_state_dict(checkpoint['g_ema'], strict=False)
    g_ema.eval()

    if args.truncation < 1:
        with torch.no_grad():
            tru_mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        tru_mean_latent = None

    if args.image_mode == 'RGB':
        d_input_dim = 3
    else:
        d_input_dim = 1

    if args.mean_init == True:
        encoder = None
    else:
        encoder = FPNEncoder(input_dim=d_input_dim, n_latent=g_ema.n_latent).to(device)
        encoder.load_state_dict(checkpoint['e'])
        encoder.eval()

    percept = lpips.PerceptualLoss(
        model='net-lin', net='vgg', use_gpu=device.startswith('cuda')
    ).to(device)
    

    img_list = sorted(os.listdir(args.img_dir))
    os.makedirs(os.path.join(args.outdir, 'recon'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'mask_rgb'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'target_overlay'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'recon_overlay'), exist_ok=True)
        
    if args.save_steps:
        os.makedirs(os.path.join(args.outdir, 'steps'), exist_ok=True)

    if args.save_latent:
        os.makedirs(os.path.join(args.outdir, 'latent'), exist_ok=True)

    resize = min(args.size, 256)

    transform = get_transformation(args)

    for image_name in img_list:
        img_p = os.path.join(args.img_dir, image_name)

        pbar = range(args.step)
        latent_path = []
        # load target image
        target_pil = Image.open(img_p).convert(args.image_mode).resize((args.size,args.size), resample=Image.LANCZOS)
    
        target_img_tensor = transform(target_pil).unsqueeze(0).to(device)
        noises = g_ema.make_noise()

        for noise in noises:
            noise.requires_grad = True

        # initialization
        with torch.no_grad():
            if args.mean_init:
                n_mean_latent = 10000
                with torch.no_grad():
                    noise_sample = torch.randn(n_mean_latent, 512, device=device)
                    latent_out = g_ema.style(noise_sample)

                    latent_mean = latent_out.mean(0)
                    latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

                latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(args.batch_size, 1)
                if args.w_plus:
                    latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
            
            else:
                latent_in = encoder(target_img_tensor)
                latent_enc_init = latent_in.clone().detach()
   
        latent_in.requires_grad = True

        if args.no_noises:
            optimizer = optim.Adam([latent_in], lr=args.lr)
        else:
            optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

        if args.step == 0:
            with torch.no_grad():
                # do not optimize
                if args.no_noises:
                    img_gen, img_seg = g_ema([latent_in], input_is_latent=True)
                else:
                    img_gen, img_seg = g_ema([latent_in], input_is_latent=True, noise=noises)

                latent_path.append(latent_in.detach().clone())
        else:
            for i in pbar:
                t = i / args.step
                lr = get_lr(t, args.lr)
                optimizer.param_groups[0]['lr'] = lr
                
                if args.no_noises:
                    img_gen, img_seg = g_ema([latent_in], input_is_latent=True)
                else:
                    img_gen, img_seg = g_ema([latent_in], input_is_latent=True, noise=noises)

                batch, channel, height, width = img_gen.shape
                
                if height > 256:
                    factor = height // 256

                    img_gen = img_gen.reshape(
                        batch, channel, height // factor, factor, width // factor, factor
                    )
                    img_gen = img_gen.mean([3, 5])

                p_loss = percept(img_gen, target_img_tensor).mean()
                
                mse_loss = F.mse_loss(img_gen, target_img_tensor)

                n_loss = noise_regularize(noises)

                if args.mean_init:
                    if args.w_plus == True:
                        latent_mean_loss = F.mse_loss(latent_in, latent_mean.unsqueeze(0).repeat(batch, g_ema.n_latent, 1))
                    else:
                        latent_mean_loss = F.mse_loss(latent_in, latent_mean.repeat(batch, 1))

                    # main loss function
                    loss = (p_loss + 
                            args.noise_regularize * n_loss + 
                            mse_loss * args.lambda_mse + 
                            latent_mean_loss * args.lambda_mean)

                    
                    print(f'p loss: {p_loss.item():.4f}; noise regularize:{n_loss.item():.4f}; mse loss: {mse_loss.item():.4f}; ')
                        
                else:
                    encoder_init_loss = F.mse_loss(latent_in, latent_enc_init)
                    encoder_loss = F.mse_loss(latent_in, encoder(img_gen).detach())

                    # main loss function
                    loss = p_loss + \
                           args.noise_regularize * n_loss + \
                           mse_loss * args.lambda_mse + \
                           encoder_loss * args.lambda_encoder + \
                           encoder_init_loss * args.lambda_encoder_init

                    print(
                        (
                            f'p loss: {p_loss.item():.4f}; noise regularize:{n_loss.item():.4f}; mse loss: {mse_loss.item():.4f}; '
                            f'encoder loss: {encoder_loss.item():.4f}; encoder init loss: {encoder_init_loss.item():.4f}; '
                        )
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                noise_normalize_(noises)

                latent_path.append(latent_in.detach().clone())

        
        # save results
        with torch.no_grad():
            if args.no_noises:
                img_gen, img_seg = g_ema([latent_path[-1]], input_is_latent=True)
            else:
                img_gen, img_seg = g_ema([latent_path[-1]], input_is_latent=True, noise=noises,
                                            truncation=args.truncation, truncation_latent=tru_mean_latent)

            img_gen = img_gen.cpu()
            img_seg = img_seg.cpu()

            img_gen = make_image(img_gen).squeeze()

            if args.seg_dim != 1:
                mask_gen = torch.argmax(img_seg, dim=1).squeeze().type(torch.uint8).numpy()
            else:
                mask_gen = torch.sigmoid(img_seg)
                mask_gen[mask_gen>0.5] = 1.0
                mask_gen = mask_gen.squeeze().type(torch.uint8).numpy()

            img_seg = make_mask(args, img_seg).squeeze()

            img_gen_pil = Image.fromarray(img_gen)
            img_seg_pil = Image.fromarray(img_seg)
            mask_gen_pil = Image.fromarray(mask_gen)

            pil_target_overlay = overlay_img_and_mask(args, target_pil, img_seg_pil)
            pil_gen_overlay = overlay_img_and_mask(args, img_gen_pil, img_seg_pil)

            if args.dataset_name == 'celeba-mask':
                if image_name.endswith('.jpg'):
                    image_name = image_name.replace('.jpg', '.png')

            img_gen_pil.save(os.path.join(args.outdir, 'recon/'+image_name))
            pil_gen_overlay.save(os.path.join(args.outdir, 'recon_overlay/'+image_name))
            mask_gen_pil.save(os.path.join(args.outdir, 'mask/'+image_name))
            img_seg_pil.save(os.path.join(args.outdir, 'mask_rgb/'+image_name))
            pil_target_overlay.save(os.path.join(args.outdir, 'target_overlay/'+image_name))
            
            if args.save_latent:
                latent_np = latent_path[-1].detach().cpu().numpy()
                np.save(os.path.join(args.outdir, 'latent/'+image_name.replace('.png', '.npy')), latent_np)
                
            if args.save_steps:
                total_steps = args.step
                for i in range(0, total_steps, 25):
                    img_gen, img_seg = g_ema([latent_path[i]], input_is_latent=True, noise=noises)
                    
                    img_gen = img_gen.cpu()
                    img_seg = img_seg.cpu()

                    img_gen = make_image(img_gen).squeeze()

                    img_seg = make_mask(args, img_seg).squeeze()

                    img_gen_pil = Image.fromarray(img_gen)
                    img_seg_pil = Image.fromarray(img_seg)

                    image_name = image_name.split('.')[0]
                    os.makedirs(os.path.join(args.outdir, 'steps', image_name), exist_ok=True)

                    img_gen_pil.save(os.path.join(args.outdir, 'steps/', image_name, 'img_step_{0:05d}.png'.format(i)))
                    img_seg_pil.save(os.path.join(args.outdir, 'steps/', image_name, 'label_step_{0:05d}.png'.format(i)))
            
