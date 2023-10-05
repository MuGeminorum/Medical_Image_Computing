import csv
import argparse
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from datasets import load_dataset
from utils import *


parser = argparse.ArgumentParser(description='train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epoch', type=int, default=40, help='epoch_num')
parser.add_argument('--iter', type=int, default=10, help='iteration')
parser.add_argument('--df', type=bool, default=False, help='deep-finetune')
args = parser.parse_args()


def eval_model_train(model, traLoader, device, tra_acc_list):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in traLoader:
            images, labels = data['image'].to(device), data['label'].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of training: %.2f %%' % (100.0 * correct / total))
    tra_acc_list.append(100.0 * correct / total)


def eval_model_validation(model, validationLoader, device, val_acc_list):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validationLoader:
            images, labels = data['image'].to(device), data['label'].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of validation: %.2f %%' % (100.0 * correct / total))
    val_acc_list.append(100.0 * correct / total)


def eval_model_test(model, tesLoader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tesLoader:
            images, labels = data['image'].to(device), data['label'].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of test: %.2f %%' % (100.0 * correct / total))


def save_history(tra_acc_list, val_acc_list, loss_list):
    acc_len = len(tra_acc_list)
    timestamp = time_stamp()
    with open("./logs/history-acc-" + timestamp + ".csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tra_acc_list", "val_acc_list"])
        for i in range(acc_len):
            writer.writerow([tra_acc_list[i], val_acc_list[i]])

    loss_len = len(loss_list)
    with open("./logs/history-loss-" + timestamp + ".csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["loss_list"])
        for i in range(loss_len):
            writer.writerow([loss_list[i]])


def transform(example_batch):
    inputs = [compose(x.convert('RGB')) for x in example_batch["image"]]
    example_batch["image"] = inputs
    return example_batch


def prepare_data():
    print('Preparing data...')
    ds = load_dataset("MuGeminorum/HEp2")
    trainset = ds['train'].with_transform(transform)
    validset = ds['validation'].with_transform(transform)
    testset = ds['test'].with_transform(transform)

    traLoader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    valLoader = DataLoader(validset, batch_size=4, shuffle=True, num_workers=2)
    tesLoader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)
    print('Data loaded.')

    return traLoader, valLoader, tesLoader


def AlexNet():
    pre_model_path = download_model()
    model = models.alexnet()
    model.load_state_dict(torch.load(pre_model_path))

    for parma in model.parameters():
        parma.requires_grad = args.df

    model.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 1000),
        nn.ReLU(inplace=True),
        nn.Linear(1000, 6)
    )

    return model


def train():
    # init args
    lr = args.lr
    epoch_num = args.epoch
    iteration = args.iter

    # load data
    tra_acc_list, val_acc_list, loss_list = [], [], []
    traLoader, valLoader, tesLoader = prepare_data()

    # init model
    model = AlexNet()

    # optimizer and loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True,
        threshold=lr, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    # gpu
    torch.cuda.empty_cache()
    model = model.to(device)
    criterion = criterion.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # train process
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        epoch_str = f' Epoch {epoch + 1}/{epoch_num} '
        print(f'{epoch_str:-^40s}')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
        running_loss = 0.0
        for i, data in enumerate(traLoader, 0):
            # get the inputs
            inputs, labels = data['image'].to(device), data['label'].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % iteration == iteration - 1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.4f' %
                      (epoch + 1, i + 1, running_loss / iteration))
                loss_list.append(running_loss / iteration)
            running_loss = 0.0

        eval_model_train(model, traLoader, device, tra_acc_list)
        eval_model_validation(model, valLoader, device, val_acc_list)
        scheduler.step(loss.item())

    print('Finished Training')
    eval_model_test(model, tesLoader, device)
    save_history(tra_acc_list, val_acc_list, loss_list)
    torch.save(model, './model/save.pt')


if __name__ == "__main__":
    train()
