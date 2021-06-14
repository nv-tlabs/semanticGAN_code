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
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from torch.nn.utils import clip_grad_norm_
from torchvision import transforms, utils

from torch.utils.tensorboard import SummaryWriter

from models.stylegan2_seg import GeneratorSeg
from models.encoder_model import FPNEncoder, ResEncoder

from dataloader.dataset import CelebAMaskDataset

from utils.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
)
from models import lpips
from PIL import Image

from semanticGAN.losses import SoftmaxLoss, SoftBinaryCrossEntropyLoss, DiceLoss
from semanticGAN.ranger import Ranger


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


def batch_pix_accuracy(output, target):
    _, predict = torch.max(output, 1)

    predict = predict.int() + 1
    target = target.int() + 1

    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target) * (target > 0)).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()


def batch_intersection_union(output, target, num_class):
    _, predict = torch.max(output, 1)
    predict = predict + 1
    target = target + 1

    predict = predict * (target > 0).long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy()


def eval_metrics(output, target, num_classes, ignore_index):
    target = target.clone()
    target[target == ignore_index] = -1
    correct, labeled = batch_pix_accuracy(output.data, target)
    inter, union = batch_intersection_union(output.data, target, num_classes)
    return [np.round(correct, 5), np.round(labeled, 5), np.round(inter, 5), np.round(union, 5)]


def validate(args, encoder, generator, val_loader, device, writer, step):
    with torch.no_grad():
        encoder.eval()
        generator.eval()
        total_inter, total_union = 0.0, 0.0
        total_correct, total_label = 0.0, 0.0
        for i, data in enumerate(val_loader):
            img, mask = data['image'].to(device), data['mask'].to(device)
            # shift mask to 0 - 1
            mask = (mask + 1.0) / 2.0

            latent_w = encoder(img)
            recon_img, recon_seg = generator([latent_w], input_is_latent=True)

            if args.seg_dim == 1:
                label_pred = torch.sigmoid(recon_seg)
                bg_pred = 1.0 - label_pred
                mask_pred = torch.cat([bg_pred, label_pred], dim=1)
                true_mask = mask.squeeze(1)
                n_class = 2
            else:
                mask_pred = torch.softmax(recon_seg, dim=1)
                true_mask = torch.argmax(mask, dim=1)
                n_class = args.seg_dim

            correct, labeled, inter, union = eval_metrics(mask_pred, true_mask, n_class, -100)

            total_inter, total_union = total_inter + inter, total_union + union
            total_correct, total_label = total_correct + correct, total_label + labeled

        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        mIoU = IoU.mean()

        print("===========val miou scores: {0:.4f}, pixel acc: {1:.4f} ========================".format(mIoU, pixAcc))
        writer.add_scalar('scores/miou', mIoU, global_step=step)
        writer.add_scalar('scores/pixel_acc', pixAcc, global_step=step)
        for i in range(IoU.shape[0]):
            print("===========val {0} miou scores: {1:.4f} ========================".format(i, IoU[i]))
            writer.add_scalar('scores/{} val_miou'.format(i), IoU[i], global_step=step)


def make_image(tensor):
    return (
        tensor.detach()
            .clamp_(min=-1, max=1)
            .add(1)
            .div_(2)
            .mul(255)
            .type(torch.uint8)
            .to('cpu')
            .numpy()
    )


def mask2rgb(args, mask):
    if args.seg_name == 'celeba-mask':
        color_table = torch.tensor(
            [[0, 0, 0],
             [0, 0, 205],
             [132, 112, 255],
             [25, 25, 112],
             [187, 255, 255],
             [102, 205, 170],
             [227, 207, 87],
             [142, 142, 56]], dtype=torch.float)
    else:
        raise Exception('No such a dataloader!')

    rgb_tensor = F.embedding(mask, color_table).permute(0, 3, 1, 2)
    return rgb_tensor


def batch_overlay(args, img_tensor, mask_tensor, alpha=0.3):
    b = img_tensor.shape[0]
    overlays = []
    imgs_np = make_image(img_tensor)
    if args.seg_dim == 1:
        idx = np.nonzero(mask_tensor.detach().cpu().numpy()[:, 0, :, :])
        masks_np = np.zeros((mask_tensor.shape[0], mask_tensor.shape[2], mask_tensor.shape[3], 3), dtype=np.uint8)
        masks_np[idx] = (0, 255, 0)
    else:
        masks_np = mask_tensor.detach().cpu().permute(0, 2, 3, 1).type(torch.uint8).numpy()

    for i in range(b):
        img_pil = Image.fromarray(imgs_np[i][0]).convert('RGBA')
        mask_pil = Image.fromarray(masks_np[i]).convert('RGBA')

        overlay_pil = Image.blend(img_pil, mask_pil, alpha)
        overlay_tensor = transforms.functional.to_tensor(overlay_pil)
        overlays.append(overlay_tensor)
    overlays = torch.stack(overlays, dim=0)

    return overlays


def sample_val_viz_imgs(args, seg_val_loader, encoder, generator):
    with torch.no_grad():
        encoder.eval()
        generator.eval()
        val_count = 0
        recon_imgs = []
        recon_segs = []
        real_imgs = []
        real_segs = []
        real_overlays = []
        fake_overlays = []

        for i, data in enumerate(seg_val_loader):
            if val_count >= args.n_sample:
                break
            val_count += data['image'].shape[0]

            real_img, real_mask = data['image'].to(device), data['mask'].to(device)

            latent_w = encoder(real_img)
            recon_img, recon_seg = generator([latent_w], input_is_latent=True)

            recon_img = recon_img.detach().cpu()
            recon_seg = recon_seg.detach().cpu()

            real_mask = (real_mask + 1.0) / 2.0
            real_img = real_img.detach().cpu()
            real_mask = real_mask.detach().cpu()

            if args.seg_dim == 1:
                sample_seg = torch.sigmoid(recon_seg)
                sample_mask = torch.zeros_like(sample_seg)
                sample_mask[sample_seg > 0.5] = 1.0

            else:
                sample_seg = torch.softmax(recon_seg, dim=1)
                sample_mask = torch.argmax(sample_seg, dim=1)
                sample_mask = mask2rgb(args, sample_mask)

                real_mask = torch.argmax(real_mask, dim=1)
                real_mask = mask2rgb(args, real_mask)

            real_overlay = batch_overlay(args, real_img, real_mask)
            fake_overlay = batch_overlay(args, real_img, sample_mask)

            recon_imgs.append(recon_img)
            recon_segs.append(sample_mask)

            real_imgs.append(real_img)
            real_segs.append(real_mask)

            real_overlays.append(real_overlay)
            fake_overlays.append(fake_overlay)

        recon_imgs = torch.cat(recon_imgs, dim=0)
        recon_segs = torch.cat(recon_segs, dim=0)
        real_imgs = torch.cat(real_imgs, dim=0)
        real_segs = torch.cat(real_segs, dim=0)
        recon_imgs = torch.cat([real_imgs, recon_imgs], dim=0)
        recon_segs = torch.cat([real_segs, recon_segs], dim=0)

        real_overlays = torch.cat(real_overlays, dim=0)
        fake_overlays = torch.cat(fake_overlays, dim=0)
        overlay = torch.cat([real_overlays, fake_overlays], dim=0)

        return (recon_imgs, recon_segs, overlay)


def sample_unlabel_viz_imgs(args, unlabel_n_sample, unlabel_loader, encoder, generator):
    with torch.no_grad():
        encoder.eval()
        generator.eval()
        val_count = 0
        real_imgs = []
        recon_imgs = []
        recon_segs = []
        fake_overlays = []

        for i, data in enumerate(unlabel_loader):
            if val_count >= unlabel_n_sample:
                break
            if args.seg_name == 'CXR' or args.seg_name == 'CXR-single':
                val_count += data.shape[0]
                real_img = data.to(device)
            else:
                val_count += data['image'].shape[0]
                real_img = data['image'].to(device)

            latent_w = encoder(real_img)
            recon_img, recon_seg = generator([latent_w], input_is_latent=True)

            recon_img = recon_img.detach().cpu()
            recon_seg = recon_seg.detach().cpu()
            real_img = real_img.detach().cpu()

            if args.seg_dim == 1:
                sample_seg = torch.sigmoid(recon_seg)
                sample_mask = torch.zeros_like(sample_seg)
                sample_mask[sample_seg > 0.5] = 1.0

            else:
                sample_seg = torch.softmax(recon_seg, dim=1)
                sample_mask = torch.argmax(sample_seg, dim=1)
                sample_mask = mask2rgb(args, sample_mask)

            fake_overlay = batch_overlay(args, real_img, sample_mask)

            real_imgs.append(real_img)
            recon_imgs.append(recon_img)
            recon_segs.append(sample_mask)
            fake_overlays.append(fake_overlay)

        real_imgs = torch.cat(real_imgs, dim=0)
        recon_imgs = torch.cat(recon_imgs, dim=0)
        recon_imgs = torch.cat([real_imgs, recon_imgs], dim=0)

        recon_segs = torch.cat(recon_segs, dim=0)
        fake_overlays = torch.cat(fake_overlays, dim=0)

        return (recon_imgs, recon_segs, fake_overlays)


def update_learning_rate(args, i, optimizer):
    if i < args.lr_decay_iter_start:
        pass
    elif i < args.lr_decay_iter_end:
        lr_max = args.lr
        lr_min = args.lr_decay
        t_max = args.lr_decay_iter_end - args.lr_decay_iter_start
        t_cur = i - args.lr_decay_iter_start

        optimizer.param_groups[0]['lr'] = lr_min + 0.5 * (lr_max - lr_min) * (
                    1 + math.cos(t_cur * 1.0 / t_max * math.pi))
    else:
        pass


def train(args, ckpt_dir, img_loader, seg_loader, seg_val_loader, generator, percept,
          encoder, e_label_optim, e_unlabel_optim, device, writer):
    img_loader = sample_data(img_loader)
    seg_loader = sample_data(seg_loader)

    if args.seg_dim == 1:
        ce_loss_func = SoftBinaryCrossEntropyLoss(tau=0.3)
        dice_loss_func = DiceLoss(sigmoid_tau=0.3, include_bg=True)
    else:
        ce_loss_func = SoftmaxLoss(tau=0.1)
        dice_loss_func = DiceLoss(sigmoid_tau=0.3)

    pbar = range(args.iter)

    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        e_module = encoder.module

    else:
        g_module = generator
        e_module = encoder

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')

            break

        if args.seg_name == 'celeba-mask':
            real_img = next(img_loader)['image']
        else:
            raise Exception('No such a dataloader!')

        real_img = real_img.to(device)

        seg_data = next(seg_loader)

        seg_img, seg_mask = seg_data['image'], seg_data['mask']
        seg_img, seg_mask = seg_img.to(device), seg_mask.to(device)

        # train encoder
        requires_grad(generator, False)
        requires_grad(encoder, True)

        # =================Step 1: train with unlabel data ==============================================
        latent_w = encoder(real_img)
        fake_img, fake_seg = generator([latent_w], input_is_latent=True)

        # detach fake seg
        fake_seg = fake_seg.detach()

        e_unlabel_mse_loss = F.mse_loss(fake_img, real_img)

        e_unlabel_lpips_loss = percept(fake_img, real_img).mean()

        e_unlabel_loss = (e_unlabel_mse_loss * args.lambda_unlabel_mse +
                          e_unlabel_lpips_loss * args.lambda_unlabel_lpips)

        loss_dict['e_unlabel_mse'] = e_unlabel_mse_loss
        loss_dict['e_unlabel_lpips'] = e_unlabel_lpips_loss
        loss_dict['e_unlabel_loss'] = e_unlabel_loss

        encoder.zero_grad()
        e_unlabel_loss.backward()

        if args.profile_grad_norm == True:
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in e_module.parameters()]), 2.0)
            print("e_unlabel_average grad norm: {0:.6f}".format(total_norm))

        if args.no_grad_clip != True:
            clip_grad_norm_(e_module.parameters(), args.unlabel_grad_clip)

        e_unlabel_optim.step()
        update_learning_rate(args, i, e_unlabel_optim)

        # =================Step 2: train with label data ==============================================
        latent_w = encoder(seg_img)
        fake_img, fake_seg = generator([latent_w], input_is_latent=True)

        if args.seg_dim == 1:
            # shift to 0-1
            seg_mask = (seg_mask + 1.0) / 2.0
            e_label_ce_loss = ce_loss_func(fake_seg, seg_mask)
            e_label_dice_loss = dice_loss_func(fake_seg, seg_mask)
        else:
            # make seg mask to label
            seg_mask_ce = torch.argmax(seg_mask, dim=1)
            seg_mask_dice = (seg_mask + 1.0) / 2.0
            e_label_ce_loss = ce_loss_func(fake_seg, seg_mask_ce)
            e_label_dice_loss = dice_loss_func(fake_seg, seg_mask_dice)

        e_label_mse_loss = F.mse_loss(fake_img, seg_img)

        e_label_lpips_loss = percept(fake_img, seg_img).mean()

        e_label_loss = (e_label_ce_loss * args.lambda_label_ce +
                        e_label_dice_loss * args.lambda_label_dice +
                        e_label_mse_loss * args.lambda_label_mse +
                        e_label_lpips_loss * args.lambda_label_lpips)

        loss_dict['e_label_mse'] = e_label_mse_loss
        loss_dict['e_label_lpips'] = e_label_lpips_loss
        loss_dict['e_label_ce'] = e_label_ce_loss
        loss_dict['e_label_dice'] = e_label_dice_loss
        loss_dict['e_label_loss'] = e_label_loss

        encoder.zero_grad()
        e_label_loss.backward()

        if args.profile_grad_norm == True:
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in e_module.parameters()]), 2.0)
            print("e_label_average grad norm: {0:.6f}".format(total_norm))

        if args.no_grad_clip != True:
            # gradient clipping
            clip_grad_norm_(e_module.parameters(), args.label_grad_clip)

        e_label_optim.step()
        update_learning_rate(args, i, e_label_optim)

        loss_reduced = reduce_loss_dict(loss_dict)

        e_unlabel_mse_loss_val = loss_reduced['e_unlabel_mse'].mean().item()
        e_unlabel_lpips_loss_val = loss_reduced['e_unlabel_lpips'].mean().item()
        e_unlabel_loss_val = loss_reduced['e_unlabel_loss'].mean().item()

        e_label_mse_loss_val = loss_reduced['e_label_mse'].mean().item()
        e_label_ce_loss_val = loss_reduced['e_label_ce'].mean().item()
        e_label_dice_loss_val = loss_reduced['e_label_dice'].mean().item()
        e_label_lpips_loss_val = loss_reduced['e_label_lpips'].mean().item()
        e_label_loss_val = loss_reduced['e_label_loss'].mean().item()

        if get_rank() == 0:
            # write to tensorboard
            writer.add_scalar('e_unlabel/mse_loss', e_unlabel_mse_loss_val, global_step=i)
            writer.add_scalar('e_unlabel/lpips_loss', e_unlabel_lpips_loss_val, global_step=i)
            writer.add_scalar('e_unlabel/total_loss', e_unlabel_loss_val, global_step=i)

            writer.add_scalar('e_label/mse_loss', e_label_mse_loss_val, global_step=i)
            writer.add_scalar('e_label/lpips_loss', e_label_lpips_loss_val, global_step=i)
            writer.add_scalar('e_label/ce_loss', e_label_ce_loss_val, global_step=i)
            writer.add_scalar('e_label/dice_loss', e_label_dice_loss_val, global_step=i)
            writer.add_scalar('e_label/total_loss', e_label_loss_val, global_step=i)

            # learning rate
            writer.add_scalar('lr/e_label_lr', e_label_optim.param_groups[0]['lr'], global_step=i)
            writer.add_scalar('lr/e_unlabel_lr', e_unlabel_optim.param_groups[0]['lr'], global_step=i)

            if i % args.viz_every == 0:
                with torch.no_grad():
                    val_recon_img, val_recon_seg, _ = sample_val_viz_imgs(args, seg_val_loader, encoder,
                                                                                    generator)
                    val_n_sample = min(len(seg_val_loader), args.n_sample * 2)

                    unlabel_n_sample = 32
                    unlabel_recon_img, unlabel_recon_seg, _ = sample_unlabel_viz_imgs(args,
                                                                                    unlabel_n_sample,
                                                                                    img_loader, encoder,
                                                                                    generator)

                    os.makedirs(os.path.join(ckpt_dir, 'sample'), exist_ok=True)

                    utils.save_image(
                        val_recon_img,
                        os.path.join(ckpt_dir, f'sample/val_recon_img_{str(i).zfill(6)}.png'),
                        nrow=8,
                        normalize=True,
                        range=(-1, 1),
                    )
                    utils.save_image(
                        val_recon_seg,
                        os.path.join(ckpt_dir, f'sample/val_recon_seg_{str(i).zfill(6)}.png'),
                        nrow=8,
                        normalize=True,
                    )
                  
                    utils.save_image(
                        unlabel_recon_img,
                        os.path.join(ckpt_dir, f'sample/unlabel_recon_img_{str(i).zfill(6)}.png'),
                        nrow=8,
                        normalize=True,
                        range=(-1, 1),
                    )
                    utils.save_image(
                        unlabel_recon_seg,
                        os.path.join(ckpt_dir, f'sample/unlabel_recon_seg_{str(i).zfill(6)}.png'),
                        nrow=8,
                        normalize=True,
                    )
  

            if i % args.eval_every == 0:
                print("==================Start calculating validation scores==================")
                validate(args, encoder, g_module, seg_val_loader, device, writer, i)

            if i % args.save_every == 0:
                os.makedirs(os.path.join(ckpt_dir, 'ckpt'), exist_ok=True)
                torch.save(
                    {
                        'g_ema': g_module.state_dict(),
                        'e': e_module.state_dict(),
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
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
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

    parser.add_argument('--ckpt', type=str, help='checkpoint of pretrained gan', required=True)

    parser.add_argument('--seg_name', type=str, help='segmentation dataloader name [celeba-mask]', default='celeba-mask')
    parser.add_argument('--iter', type=int, default=800000)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--val_batch', type=int, default=16)
    parser.add_argument('--n_sample', type=int, default=32)
    parser.add_argument('--size', type=int, default=256)

    parser.add_argument('--viz_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=2000)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=0.00001)
    parser.add_argument('--lambda_lr', type=float, default=0.999)
    parser.add_argument('--lr_decay_iter_start', type=int, default=30000)
    parser.add_argument('--lr_decay_iter_end', type=int, default=100000)
    parser.add_argument('--channel_multiplier', type=int, default=2)

    parser.add_argument('--lambda_label_lpips', type=float, default=10.0)
    parser.add_argument('--lambda_label_mse', type=float, default=1.0)
    parser.add_argument('--lambda_label_ce', type=float, default=1.0)
    parser.add_argument('--lambda_label_depth', type=float, default=1.0)
    parser.add_argument('--lambda_label_dice', type=float, default=1.0)
    parser.add_argument('--lambda_label_latent', type=float, default=0.0)
    parser.add_argument('--lambda_label_adv', type=float, default=0.1)

    parser.add_argument('--lambda_unlabel_lpips', type=float, default=10.0)
    parser.add_argument('--lambda_unlabel_mse', type=float, default=1.0)
    parser.add_argument('--lambda_unlabel_adv', type=float, default=0.1)

    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--image_mode', type=str, default='RGB', help='Image mode RGB|L')

    parser.add_argument('--seg_dim', type=int, default=8)
    parser.add_argument('--seg_aug', action='store_true', help='seg augmentation')

    parser.add_argument('--no_grad_clip', action='store_true', help='if use gradient clipping')
    parser.add_argument('--profile_grad_norm', action='store_true', help='if profile average grad norm')
    parser.add_argument('--label_grad_clip', type=float, help='gradient clip norm value for labeled dataloader',
                        default=5.0)
    parser.add_argument('--unlabel_grad_clip', type=float, help='gradient clip norm value for unlabeled dataloader',
                        default=2.0)

    parser.add_argument('--enc_backbone', type=str, help='encoder backbone[res|fpn]', default='fpn')
    parser.add_argument('--optimizer', type=str, help='encoder backbone[adam|ranger]', default='ranger')

    parser.add_argument('--limit_data', type=str, default=None, help='number of limited label data point to use')
    parser.add_argument('--unlabel_limit_data', type=str, default=None,
                        help='number of limited unlabel data point to use')

    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/')

    args = parser.parse_args()
    print(args)

    # build checkpoint dir
    from datetime import datetime

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    ckpt_dir = os.path.join(args.checkpoint_dir, 'run-' + current_time)
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

    if args.image_mode == 'RGB':
        d_input_dim = 3
    else:
        d_input_dim = 1

    checkpoint = torch.load(args.ckpt)
    g_ema = GeneratorSeg(args.size, args.latent, args.n_mlp, image_mode=args.image_mode,
                         channel_multiplier=args.channel_multiplier, seg_dim=args.seg_dim
                         ).to(device)
    g_ema.load_state_dict(checkpoint['g_ema'])
    g_ema.eval()

    # percep
    percept = lpips.PerceptualLoss(
        model='net-lin', net='vgg', use_gpu=False, gpu_ids=[args.local_rank]
    ).to(device)

    # build encoder
    if args.enc_backbone == 'res':
        encoder = ResEncoder(
            args.size, input_dim=d_input_dim, n_latent=g_ema.n_latent, channel_multiplier=args.channel_multiplier).to(
            device)
    else:
        encoder = FPNEncoder(input_dim=d_input_dim, n_latent=g_ema.n_latent).to(device)

    if args.optimizer == 'adam':
        e_label_optim = optim.Adam(
            encoder.parameters(),
            lr=args.lr,
        )
        e_unlabel_optim = optim.Adam(
            encoder.parameters(),
            lr=args.lr,
        )
    else:
        e_label_optim = Ranger(
            encoder.parameters(),
            lr=args.lr,
        )
        e_unlabel_optim = Ranger(
            encoder.parameters(),
            lr=args.lr,
        )

    if args.distributed:
        g_ema = nn.parallel.DistributedDataParallel(
            g_ema,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        percept = nn.parallel.DistributedDataParallel(
            percept,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        encoder = nn.parallel.DistributedDataParallel(
            encoder,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    if args.seg_name == 'celeba-mask':
        transform = get_transformation(args)
        img_dataset = CelebAMaskDataset(args, args.img_dataset, unlabel_transform=transform,
                                        unlabel_limit_size=args.unlabel_limit_data,
                                        is_label=False, resolution=args.size)
    else:
        raise Exception('No such a dataloader!')

    img_loader = data.DataLoader(
        img_dataset,
        batch_size=args.batch,
        sampler=data_sampler(img_dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )
    print("Loading unlabel dataloader with size ", len(img_dataset))

    if args.seg_name == 'celeba-mask':
        seg_dataset = get_seg_dataset(args, phase='train')
        seg_loader = data.DataLoader(
            seg_dataset,
            batch_size=args.batch,
            sampler=data_sampler(seg_dataset, shuffle=True, distributed=args.distributed),
            drop_last=True,
            pin_memory=True,
            num_workers=4,
        )
    else:
        raise Exception('No such a dataloader!')

    print("Loading train dataloader with size ", seg_dataset.data_size)

    if args.seg_name == 'celeba-mask':
        seg_val_dataset = get_seg_dataset(args, phase='val')
    else:
        raise Exception('No such a dataloader!')

    print("Loading val dataloader with size ", seg_val_dataset.data_size)

    seg_val_loader = data.DataLoader(
        seg_val_dataset,
        batch_size=args.val_batch,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    print("local rank: {}, Start training!".format(args.local_rank))

    # setup benchmark
    torch.backends.cudnn.benchmark = True

    train(args, ckpt_dir, img_loader, seg_loader, seg_val_loader,
          g_ema, percept, encoder, e_label_optim, e_unlabel_optim, device, writer)
