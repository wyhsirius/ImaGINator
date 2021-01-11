# ImaGINator: Conditional Spatio-Temporal GAN for Video Generation
Yaohui Wang, Piotr Bilinski, Francois Bremond and Antitza Dantcheva

## Requirements
- Python 3.6
- cuda 9.2
- cudnn 7.1
- PyTorch 1.4+
- scikit-video
- tensoboard
- moviepy
- PyAV

## Dataset
You can download the original MUG datest from https://mug.ee.auth.gr/fed/ and use https://github.com/1adrianb/face-alignment to crop face regions. You can also download our preprocessed version from [here](https://drive.google.com/file/d/1zMbkzuik5O4Qjv_zVerkIPZYEZYuX_EF/view?usp=sharing) and save it under $DATA_PATH.

## Pretrained model
Download the pretrained model on MUG from [here](https://drive.google.com/file/d/1tRK6lFg0MddWmfOMQtUK51wxUQi0leFI/view?usp=sharing) and put it under ./pretrained.

## Inference
Generate videos and save them under ./demos/mug

```shell script
python demo.py --dataset mug --model_path ./pretrained/mug.pth
```

## Training
```shell script
python train.py --data_path $DATA_PATH
```

## Citation
If you find this code useful for your research, please consider citing our paper:
```bibtex
@InProceedings{WANG_2020_WACV,
author = {WANG, Yaohui and Bilinski, Piotr and Bremond, Francois and Dantcheva, Antitza},
title = {ImaGINator: Conditional Spatio-Temporal GAN for Video Generation},
booktitle = {The IEEE Winter Conference on Applications of Computer Vision (WACV)},
month = {March},
year = {2020}
}
```








