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
import dlib
import numpy as np
import json
import scipy.ndimage
import PIL.Image
import argparse


def main(args):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.detector)

    target_size = args.size
    supersampling = 4
    face_shrink = 2
    enable_padding = True


    img_out_dir = os.path.join(args.out_dir, 'image')
    meta_out_dir = os.path.join(args.out_dir, 'meta')

    img_list = sorted(os.listdir(args.img_dir))

    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(meta_out_dir, exist_ok=True)

    def rot90(v) -> np.ndarray:
        return np.array([-v[1], v[0]])

    for img_n in img_list:
        img_p = os.path.join(args.img_dir, img_n)
        detector_img = dlib.load_rgb_image(img_p)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(detector_img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        if len(dets) > 1:
            continue

        for k, d in enumerate(dets):

            # Get the landmarks/parts for the face in box d.
            shape = predictor(detector_img, d)
            all_parts = shape.parts()
            lm = np.array([ [item.x,item.y ] for item in all_parts])
            landmarks = np.float32(lm) + 0.5
            assert landmarks.shape == (68, 2)

            lm_eye_left      = landmarks[36 : 42]  # left-clockwise
            lm_eye_right     = landmarks[42 : 48]  # left-clockwise
            lm_mouth_outer   = landmarks[48 : 60]  # left-clockwise
            
            # Calculate auxiliary vectors.
            eye_left     = np.mean(lm_eye_left, axis=0)
            eye_right    = np.mean(lm_eye_right, axis=0)
            eye_avg      = (eye_left + eye_right) * 0.5
            eye_to_eye   = eye_right - eye_left
            mouth_left   = lm_mouth_outer[0]
            mouth_right  = lm_mouth_outer[6]
            mouth_avg    = (mouth_left + mouth_right) * 0.5
            eye_to_mouth = mouth_avg - eye_avg

            # Choose oriented crop rectangle.
            x = eye_to_eye - rot90(eye_to_mouth)
            x /= np.hypot(*x)
            x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
            y = rot90(x)
            c = eye_avg + eye_to_mouth * 0.1

            # Calculate auxiliary data.
            qsize = np.hypot(*x) * 2
            quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
            lo = np.min(quad, axis=0)
            hi = np.max(quad, axis=0)
            lm_rel = np.dot(landmarks - c, np.transpose([x, y])) / qsize**2 * 2 + 0.5
            rp = np.dot(np.random.RandomState(123).uniform(-1, 1, size=(1024, 2)), [x, y]) + c

            # Load.
            img_ori = PIL.Image.open(img_p).convert('RGB')
            img = PIL.Image.open(img_p).convert('RGB')

            # Shrink.
            shrink = int(np.floor(qsize / target_size * 0.5))
            if shrink > 1:
                rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
                img = img.resize(rsize, PIL.Image.ANTIALIAS)
                quad /= shrink
                qsize /= shrink
            
            # Crop.
            border = max(int(np.rint(qsize * 0.1)), 3)
            crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
            crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
            if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
                img = img.crop(crop)
                quad -= crop[0:2]

            # Pad.
            pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
            pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
            if enable_padding and max(pad) > border - 4:
                pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
                img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
                h, w, _ = img.shape
                y, x, _ = np.ogrid[:h, :w, :1]
                mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
                blur = qsize * 0.02
                img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
                img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
                img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
                quad += pad[:2]
            else:
                pad = (0,0,0,0)

            meta = {
                'bbox': list(crop), 
                'quad': list((quad.astype(float) + 0.5).flatten()),
                'size': list(img.size),
                'pad': [int(p) for p in list(pad)],
                'shrink': shrink,
                }

            # Transform.
            super_size = target_size * supersampling
            img = img.transform((super_size, super_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
            if target_size < super_size:
                img = img.resize((target_size, target_size), PIL.Image.ANTIALIAS)
        
            img_name = os.path.basename(img_p).split('.')[0]

            # save
            with open(os.path.join(meta_out_dir, img_name + '.json'), 'w') as f:
                json.dump(meta, f)
            
            img.save(os.path.join(img_out_dir, img_name + '.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--outdir', type=str, required=True)

    parser.add_argument('--detector', type=str, default='./shape_predictor_68_face_landmarks.dat')
    parser.add_argument('--size', type=int, default=256)
    
    args = parser.parse_args()    