# AlexNet-HEp2-Classification
[![license](https://img.shields.io/github/license/MuGeminorum/Medical_Image_Computing.svg)](https://github.com/MuGeminorum/Medical_Image_Computing/blob/master/LICENSE)
[![Python application](https://github.com/MuGeminorum/Medical_Image_Computing/actions/workflows/python-app.yml/badge.svg?branch=hep2)](https://github.com/MuGeminorum/Medical_Image_Computing/actions/workflows/python-app.yml)
[![Github All Releases](https://img.shields.io/github/downloads-pre/MuGeminorum/Medical_Image_Computing/v1.2/total)](https://github.com/MuGeminorum/Medical_Image_Computing/releases/tag/v1.2)
[![](https://img.shields.io/badge/wiki-HEp2-3572a5.svg)](https://github.com/MuGeminorum/Medical_Image_Computing/wiki/Chapter-III-%E2%80%90-Classifying-Cell-Images-Using-Deep-Learning-Models)
[![](https://img.shields.io/badge/HF-HEp2-ffd21e.svg)](https://huggingface.co/spaces/MuGeminorum/HEp2)
[![](https://img.shields.io/badge/ModelScope-HEp2-624aff.svg)](https://www.modelscope.cn/studios/MuGeminorum/HEp2)

Classifying Cell Images Using Deep Learning Models

## Requirements
```bash
conda create -n hep2 python=3.9
conda activate hep2
echo y | conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## Usage
### Maintenance
```bash
git clone -b hep2 git@github.com:MuGeminorum/Medical_Image_Computing.git
cd Medical_Image_Computing
```

### Train
```bash
python train.py
```

### Draw training curves
```bash
python plotter.py
```
It will automatically find the latest log to plot.

## Results
| ![Figure_2](https://github.com/MuGeminorum/AlexNet-HEp2-Classification/assets/20459298/5355ea0d-58c2-46d5-9aa6-88d07b237ba9) | ![Figure_1](https://github.com/MuGeminorum/AlexNet-HEp2-Classification/assets/20459298/f8f14be5-a6db-494c-b11a-36b1a3b36a26) |
| :--------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------: |
|                                                        **Loss curve**                                                        |                                             **Training and validation accuracy**                                             |
