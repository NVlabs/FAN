# Fully Attentional Networks

<p align="center">
<img src="demo/Teaser.png" width=60% height=60% 
class="center">
</p>

### [Project Page](https://github.com/NVlabs/FAN) | [Technical Report](https://arxiv.org/abs/2204.12451)

Understanding The Robustness in Vision Transformers. \
[Daquan Zhou](https://scholar.google.com/citations?user=DdCAbWwAAAAJ&hl=en), [Zhiding Yu](https://chrisding.github.io/), [Enze Xie](https://xieenze.github.io/), [Chaowei Xiao](https://xiaocw11.github.io/), [Anima Anandkumar](https://research.nvidia.com/person/anima-anandkumar), [Jiashi Feng](https://sites.google.com/site/jshfeng/home) and [Jose M. Alvarez](https://alvarezlopezjosem.github.io/). \
Technical Report, 2022.


This repository will contain the official Pytorch implementation of the training/evaluation code and the pretrained models of [Fully Attentional Network](https://arxiv.org/abs/2204.12451) (**FAN**).

**FAN** is a family of general-purpose Vision Transformer backbones highly robust to unseen natural corruptions in various image recognition tasks.

## Catalog
- [ ] Pre-trained Model Release
- [ ] ImageNet-22K Fine-tuning Code Release
- [ ] Downstream Transfer (Detection, Segmentation) Code Release
- [ ] ImageNet-1K Training & Fine-tuning Code Release
- [x] Init Repo



<!-- ✅ ⬜️  -->

## Results and Pre-trained Models
### FAN-ViT ImageNet-1K trained models

| name | resolution |ImageNet | ImageNet-C| ImageNet-A| ImageNet-R |#params | model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:|
| FAN-T-ViT | 224x224 | 79.2 | 57.5| 15.6 | 42.5 | 7.3M | [model]() |
| FAN-S-ViT | 224x224 | 82.9 | 64.5| 29.1 | 50.4 | 28.0M  | [model]() |
| FAN-B-ViT | 224x224 | 83.6 | 67.0| 35.4 | 51.8 | 54.0M  | [model]() |
| FAN-L-ViT | 224x224 | 83.9 | 67.7| 37.2 | 53.1 | 80.5M | [model]() |

### FAN-Hybrid ImageNet-1K trained models
| name | resolution |ImageNet / ImageNet-C| Cityscape / Cityscape-C| COCO / COCO-C |#params | model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| FAN-T-Hybrid | 224x224 | 80.1/57.4 | 81.2/57.1| 45.8/29.7 | 7.4M  | [model]() |
| FAN-S-Hybrid | 224x224 | 83.5/64.7 | 81.5/66.4| 49.1/35.5 | 26.3M  | [model]() |
| FAN-B-Hybrid | 224x224 | 83.9/66.4| 82.2/66.9 | 54.2/40.6 | 50.4M  | [model]() |
| FAN-L-Hybrid | 224x224 | 84.3/68.3| 82.3/68.7| 55.1/42.0 |76.8M | [model]() |

### FAN-Hybrid ImageNet-22K trained models
| name | resolution |ImageNet/ImageNet-C|#params | model |
|:---:|:---:|:---:|:---:|:---:|
| FAN-B-Hybrid | 224x224 | 85.3/70.5 | 50.4M  | [model]() |
| FAN-B-Hybrid | 384x384 | 85.6/- | 50.4M  | [model]() |
| FAN-L-Hybrid | 224x224 | 86.5/73.6 | 76.8M | [model]() |
| FAN-L-Hybrid | 384x384 | 87.1/- | 76.8M | [model]() |

## Demos
### Semantic Segmentation on Cityscapes-C
![demo image](demo/Demo_CityC.gif)

## Citation
If you find this repository helpful, please consider citing:
```
@Article{zhou2022understanding,
  author  = { Daquan Zhou, Zhiding Yu, Enze Xie, Chaowei Xiao, Anima Anandkumar, Jiashi Feng, Jose M. Alvarez},
  title   = {Understanding The Robustness in Vision Transformers},
  journal = {arXiv:2204.12451},
  year    = {2022},
}
```
