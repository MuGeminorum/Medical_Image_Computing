import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train import train
from scipy import signal as ss
from utils import *


def show_point(max_id, list):
    show_max = '['+str(max_id+1)+' '+str(list[max_id])+']'
    plt.annotate(show_max, xytext=(
        max_id+1, list[max_id]), xy=(max_id+1, list[max_id]))


def smooth(y):
    return ss.savgol_filter(y, 95, 3)


def plot_acc(tra_acc_list, val_acc_list):
    x_acc = []
    for i in range(len(tra_acc_list)):
        x_acc.append(i + 1)

    x = np.array(x_acc)
    y1 = np.array(tra_acc_list)
    y2 = np.array(val_acc_list)
    max1 = np.argmax(y1)
    max2 = np.argmax(y2)

    plt.title('Accuracy of training and validation')
    plt.xlabel('Epoch')
    plt.ylabel('Acc(%)')
    plt.plot(x, y1, label="Training")
    plt.plot(x, y2, label="Validation")
    plt.plot(1+max1, y1[max1], 'r-o')
    plt.plot(1+max2, y2[max2], 'r-o')
    show_point(max1, y1)
    show_point(max2, y2)
    plt.legend()
    plt.show()


def plot_loss(loss_list):
    x_loss = []
    for i in range(len(loss_list)):
        x_loss.append(i + 1)

    plt.title('Loss curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(x_loss, smooth(loss_list))
    plt.legend()
    plt.show()


def get_latest_csv(path):
    lists = os.listdir(path)
    lists.sort(key=lambda x: os.path.getmtime((path+"\\"+x)))
    return lists[-1][-24:]


def load_history():
    log_path = './logs'

    create_dir(log_path)

    if len(os.listdir(log_path)) == 0:
        print('No trained model found, training one...')
        train()

    latest_csv = get_latest_csv(log_path)
    latest_acc = './logs/history-acc' + latest_csv
    latest_loss = './logs/history-loss' + latest_csv
    acc_list = pd.read_csv(latest_acc)
    tra_acc_list = acc_list['tra_acc_list'].tolist()
    val_acc_list = acc_list['val_acc_list'].tolist()
    loss_list = pd.read_csv(latest_loss)['loss_list'].tolist()
    return tra_acc_list, val_acc_list, loss_list


if __name__ == "__main__":
    tra_acc_list, val_acc_list, loss_list = load_history()
    plot_acc(tra_acc_list, val_acc_list)
    plot_loss(loss_list)
