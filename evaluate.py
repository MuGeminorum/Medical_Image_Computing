import os
import torch
import argparse
from PIL import Image
from datasets import load_dataset
from utils import *


parser = argparse.ArgumentParser(description='predict')
parser.add_argument(
    '--target',
    type=str,
    default='./test/Golgi.png',
    help='Select pic to be predicted.'
)
args = parser.parse_args()


def embeding(img_path):
    img = Image.open(img_path).convert("RGB")
    return compose(img).to(device)


if __name__ == "__main__":
    classes = load_dataset(
        "MuGeminorum/HEp2", split="test").features['label'].names
    saved_model_path = './model/save.pt'

    if not os.path.exists(saved_model_path):
        print('No trained model found, downloading one...')
        download_model(
            'https://huggingface.co/MuGeminorum/alexnet-hep2/resolve/main/save.pt')

    model = torch.load(saved_model_path).to(device)
    torch.cuda.empty_cache()
    input = embeding(args.target)
    output = model(input.unsqueeze(0))
    predict = torch.max(output.data, 1)[1]
    print('\nPrediction result: ' + classes[predict])
