import os
import time
import torch
import requests
from tqdm import tqdm
from torchvision.transforms import *

compose = Compose([
    Resize(300),
    CenterCrop(300),
    RandomAffine(5),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dir(log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)


def time_stamp():
    now = int(round(time.time()*1000))
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(now/1000))


def url_download(url: str, fname: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def download_model(pre_model_url='https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'):
    model_dir = './model/'
    pre_model_path = model_dir + pre_model_url.split('/')[-1]
    create_dir(model_dir)

    if not os.path.exists(pre_model_path):
        url_download(pre_model_url, pre_model_path)

    return pre_model_path
