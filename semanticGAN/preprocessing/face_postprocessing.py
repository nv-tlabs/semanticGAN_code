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

import os
import numpy as np
from PIL import Image
import json
import argparse

def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def main(args):
    img_list = sorted(os.listdir(args.img_dir))
    meta_list = sorted(os.listdir(args.meta_dir))
    raw_list = sorted(os.listdir(args.raw_dir))

    for img_p, meta_p, raw_p in zip(img_list, meta_list, raw_list):
        img_n = img_p.split('.')[0]

        img_p = os.path.join(args.img_dir, img_p)
        meta_p = os.path.join(args.meta_dir, meta_p)
        raw_p = os.path.join(args.raw_dir, raw_p)

        with open(meta_p, 'r') as f:
            meta_json = json.load(f)
        
        kps = meta_json['quad']
        crop_box = meta_json['bbox']
        size = meta_json['size']
        pad = meta_json['pad']
        shrink = meta_json['shrink']

        upper_left = kps[0:2]
        lower_left = kps[2:4]
        lower_right = kps[4:6]
        upper_right= kps[6:]
        all_kps = [upper_left, lower_left, lower_right, upper_right]
        pa =  all_kps 
        pb =  [[0,0 ], [0, args.size], [args.size, args.size], [args.size,0]]

        coeffs = find_coeffs(pa, pb)

        left, top, right, bottom = crop_box

        width = size[0]
        height = size[1]

        img_pil = Image.open(img_p).convert('RGB')
        
        img_pil = img_pil.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BILINEAR)

        #unpad
        img_np = np.array(img_pil)
        if (pad[0] == 0 and
            pad[1] == 0 and 
            pad[2] == 0 and 
            pad[3] == 0):
            pass
        else:
            if pad[3] != 0 and pad[2] != 0:
                img_np = img_np[pad[1]:-pad[3], pad[0]:-pad[2]]
            elif pad[3] == 0 and pad[2] != 0:
                img_np = img_np[pad[1]:, pad[0]:-pad[2]]
            elif pad[3] != 0 and pad[2] == 0:
                img_np = img_np[pad[1]:-pad[3], pad[0]:]
            else:
                img_np = img_np[pad[1]:, pad[0]:]

        crop_width = crop_box[2] - crop_box[0]
        crop_height = crop_box[3] - crop_box[1]
        #unshrink
        if shrink > 1:
            img_pil = Image.fromarray(img_np)
            rsize = (int(np.rint(float(img_pil.size[0]) * shrink)), int(np.rint(float(img_pil.size[1]) * shrink)))
            img_pil = img_pil.resize(rsize, resample=Image.LANCZOS)
            crop_width *= shrink
            crop_height *= shrink
            crop_box[3] *= shrink
            crop_box[2] *= shrink
            img_np = np.array(img_pil)

        assert crop_width == img_np.shape[1]
        assert crop_height == img_np.shape[0]

        img_ori_pil = Image.open(raw_p).convert('RGB')
        img_ori_np = np.array(img_ori_pil)
        
        img_ori_np[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]] = img_np

        img_ori_pil = Image.fromarray(img_ori_np)

        img_ori_pil.save(os.path.join(depth_out, img_n + '.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--meta_dir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--size', type=int, default=256)
    
    args = parser.parse_args() 

    os.makedirs(args.outdir, exist_ok=True)

    main(args)   