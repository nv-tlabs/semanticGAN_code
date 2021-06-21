# SemanticGAN
This is the official code for:

#### Semantic Segmentation with Generative Models: Semi-Supervised Learning and Strong Out-of-Domain Generalization

[Daiqing Li](https://scholar.google.ca/citations?user=8q2ISMIAAAAJ&hl=en), [Junlin Yang](https://scholar.google.com/citations?user=QYkscc4AAAAJ&hl=en), [Karsten Kreis](https://scholar.google.de/citations?user=rFd-DiAAAAAJ&hl=de), [Antonio Torralba](https://groups.csail.mit.edu/vision/torralbalab/), [Sanja Fidler](http://www.cs.toronto.edu/~fidler/)

CVPR 2021
**[[Paper](https://arxiv.org/abs/2104.05833)]  [[Supp](https://nv-tlabs.github.io/semanticGAN/resources/SemanticGAN_supp.pdf)]**

<img src = "./figs/method.png" width="100%"/>


In this paper, we utilize the GAN as the inference network via test-time optimization. datasetGAN is faster at test time and can handle less training data, while semGAN shows very strong performance on out-of-distribution data.

## Requirements
- Python 3.6 or 3.7 are supported.
- Pytorch 1.4.0 + is recommended.
- This code is tested with CUDA 10.2 toolkit and CuDNN 7.5.
- Please check the python package requirement from [`requirements.txt`](requirements.txt), and install using
```
pip install -r requirements.txt
```

## Training 

To reproduce paper **Semantic Segmentation with Generative Models: Semi-Supervised Learning and Strong Out-of-Domain Generalization**: 

1. Run **Step1: Semantic GAN training**
2. Run **Step2: encoder training**
3. Run **Inference & Optimization**.  


---

#### 1. GAN Training

For training GAN with both image and its label,

```
python train_seg_gan.py \
--img_dataset [path-to-img-folder] \
--seg_dataset [path-to-seg-folder] \
--inception [path-to-inception file] \
--seg_name celeba-mask \
--checkpoint_dir [path-to-ckpt-dir] \
```

To use multi-gpus training in the cloud,

```
python -m torch.distributed.launch \
--nproc_per_node=N_GPU \
--master_port=PORTtrain_gan.py \
train_gan.py \
--img_dataset [path-to-img-folder] \
--inception [path-to-inception file] \
--dataset_name celeba-mask \
--checkpoint_dir [path-to-ckpt-dir] \
```

#### 2. Encoder Triaining

```
python train_enc.py \
--img_dataset [path-to-img-folder] \
--seg_dataset [path-to-seg-folder] \
--ckpt [path-to-pretrained GAN model] \
--seg_name celeba-mask \
--enc_backboend [fpn|res] \
--checkpoint_dir [path-to-ckpt-dir] \
```

## Inference

1. For Face Parts Segmentation Task

![img](./figs/face-parts-seg.png?lastModify=1616189357)

```
python inference.py \
--ckpt [path-to-ckpt] \
--img_dir [path-to-test-folder] \
--outdir [path-to-output-folder] \
--dataset_name celeba-mask \
--w_plus \
--image_mode RGB \
--seg_dim 8 \
--step 200 [optimization steps] \
```

Visualization of different optimization steps

![img](./figs/face-parts-opt-steps.png)

2. For Chest X-ray Segmentation Task,

![img](./figs/cxr-seg.png?lastModify=1616189357)



```
python inference.py \
--ckpt [path-to-ckpt] \
--img_dir [path-to-test-folder] \
--outdir [path-to-output-folder] \
--dataset_name cxr \
--w_plus \
--image_mode L \
--seg_dim 1 \
--step 200 [optimization steps] \
```


## Citation 

Please cite the follow paper is you used the code in this repository.

```
@inproceedings{semanticGAN, 
title={Semantic Segmentation with Generative Models: Semi-Supervised Learning and Strong Out-of-Domain Generalization}, 
booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)}, 
author={Li, Daiqing and Yang, Junlin and Karsten, Kreis and Antonio, Torralba and Fidler, Sanja}, 
year={2021}, 
}
```



## License 

The MIT License (MIT)

Copyright (c) 2021 NVIDIA Corporation. 

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
