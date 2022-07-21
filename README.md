##  Deep Semantic Statistics Matching (D2SM) Denoising Network

<br><sub>Official PyTorch Implementation of the ECCV 2022 Paper</sub><br><sub>[Project](https://kfmei.page/d2sm/) | [arXiv](https://arxiv.org/abs/2207.09302)

This repo contains training and evaluation code for the following paper:

> [**Deep Semantic Statistics Matching (D2SM) Denoising Network**](hhttps://kfmei.page/d2sm/)<br>
> [Kangfu Mei](https://kfmei.page/), [Vishal M. Patel ](https://engineering.jhu.edu/vpatel36/vishal-patel/), and [Rui Huang ](https://myweb.cuhk.edu.cn/ruihuang)<br>
> *European conference on computer vision (**ECCV**) 2022*<br>


## Getting Started
Use D2SM in your project can be done in two commands as
```bash
pip install git+https://github.com/MKFMIKU/d2sm.git
```

```python
from d2sm import DeST
dest = DeST("54", args.patch_size, args.patch_size, Q=args.dest_q, K=args.dest_k, mean_shift=True)

dest_loss = dest(self.hr, self.denoised_hr)
```

## Development
`!!!` The code style is based on [Black](https://github.com/psf/black).

## Enviroments
First install pytorch

```bash
pip install -i https://opentuna.cn/pypi/web/simple torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

Than install requirements
```bash
pip install -r ./requirements.txt
```

## Commands

- For D2SM training on FFDNet with CityscapesNoise dataset

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python train.py --data_root /ram_data/Refined-Cityscapes-Denoising/ --multiprocessing-distributed --world-size 1 --rank 0 --dist-url tcp://localhost:10001 --batch-size 8 --workers 16 --model FFDNet --dataset CityscapesNoise --rgb_range 1 --lr 1e-4 --checkpoint /data/Experiments/PyAnole/DeST --epochs 400 --patch_size 512 --display_freq 5000 --print_freq 10 --test_freq 20 --name FFDNet_InnerDeST --dest
```

- For D2SM testing on FFDNet with CityscapesNoise dataset
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python test.py --data_root /ram_data/Refined-Cityscapes-Denoising/val/ --multiprocessing-distributed --world-size 1 --rank 0 --dist-url tcp://localhost:10001 --batch-size 4 --workers 16 --model FFDNet --dataset imgfolder --rgb_range 1 --lr 1e-4 --checkpoint /data/Experiments/PyAnole/DeST --epochs 800 --display_freq 1000 --print_freq 10 --test_freq 10 --name FFDNet_L1 --output /data/Experiments/PyAnole/Results/FFDNet_L1 --resume /data/Experiments/PyAnole/DeST/FFDNet_L1/checkpoint_0400.pth.tar --sigma 25
```

- Utilized measurement in the paper

```bash
python tools/quality_measure.py --dir2 /tmp_data/running/Refined-Cityscapes-Denoising/val/ --dir1 /data/Experiments/TOFNet/Results/FFDNet/inpktvgg_25/output/ --cuda -b 8
```

## Reference
- https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
- https://github.com/facebookresearch/moco
- https://github.com/open-mmlab/mmediting