<img src = "figs/teaser3.png" width="100%"/>


# Semantic Segmentation GAN

Official Repository for

1. Semantic Segmentation with Generative Models: Semi-Supervised Learning and Strong Out-of-Domain Generalization (**CVPR'21**)

    [paper](https://arxiv.org/abs/2104.05833)  [supplementary](https://nv-tlabs.github.io/semanticGAN/resources/SemanticGAN_supp.pdfg) 

   In this paper, we utilize the GAN as the inference network via test-time optimization. datasetGAN is faster at test time and can handle less training data, while semGAN shows very strong performance on out-of-distribution data.

2. DatasetGAN: Efficient Labeled Data Factory with Minimal Human Effort (**CVPR'21, Oral**)

   [paper](https://drive.google.com/file/d/1vSOt_x6_mIb38RvMQ_cxXKYyBs9L7Jno/view?usp=sharing)  [supplementary ](https://drive.google.com/file/d/1td1nP8FP0axHXFxl9_EXCtHQHhnaUQl8/view?usp=sharing)[code](./datasetGAN)

   In this paper, we utilize a GAN to synthesize a labeled dataset of both images and labels on which we train any downstream network.

## Requirements
- Python 3.6 or 3.7 are supported.
- Pytorch 1.4.0 + is recommended.
- This code is tested with CUDA 10.2 toolkit and CuDNN 7.5.
- Please check the python package requirement from [`requirements.txt`](semanticGAN/requirements.txt), and install using
```
pip install -r requirements.txt
```

## Citation 

Please cite the follow paper is you used the code in this repository.

```
@inproceedings{zhang21,
  title={DatasetGAN: Efficient Labeled Data Factory with Minimal Human Effort},
  author={Zhang, Yuxuan and Ling, Huan and Gao, Jun and Yin, Kangxue and Lafleche, Jean-Francois and Barriuso, Adela and Torralba, Antonio and Fidler, Sanja},
  booktitle={CVPR},
  year={2021}
}

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



 