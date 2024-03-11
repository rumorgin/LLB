import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
from dataloader.cifar100.cifar import CIFAR100
from dataloader.cub200.cub200 import CUB200
from dataloader.miniimagenet.miniimagenet import MiniImageNet
from models.resnet18_encoder import resnet18
from utils import *

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


if __name__ == '__main__':

    def get_command_line_parser():
        parser = argparse.ArgumentParser()

        # about dataset and network
        parser.add_argument('-dataset', type=str, default='cifar100',
                            choices=['mini_imagenet', 'cub200', 'cifar100'])
        parser.add_argument('-data_root', type=str, default='D:/LLB/data/')

        # about pre-training
        parser.add_argument('-epochs', type=int, default=1)  ## for code test set to 1 default: 100
        parser.add_argument('-lr', type=float, default=0.1)
        parser.add_argument('-beta', type=float, default=0.5)
        parser.add_argument('-decay', type=float, default=0.0005)
        parser.add_argument('-momentum', type=float, default=0.9)
        parser.add_argument('-batch_size', type=int, default=256)


        # about training
        parser.add_argument('-num_workers', type=int, default=4)
        parser.add_argument('-seed', type=int, default=1)
        parser.add_argument('-use_gpu', type=bool, default=True)

        return parser

    parser = get_command_line_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    job_id = os.environ.get('SLURM_JOB_ID')

    # Define the device to use for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the ResNet-18 model
    if args.dataset == 'miniimagenet':
        model = resnet18(num_classes=100, pretrain=True)
        num_class = 60
    elif args.dataset == 'cifar100':
        model = resnet18(num_classes=100, pretrain=True)
        num_class = 60
    elif args.dataset == 'cub200':
        model = resnet18(num_classes=200, pretrained=True, pretrain=True)
        num_class = 100
    else:
        raise RuntimeError('error dataset')

    model = model.to(device)

    # Define the transforms to apply to the images

    if args.dataset == 'miniimagenet':
        trainset = MiniImageNet(root=args.data_root, train=True, index=np.arange(num_class), base_sess=1, autoaug=0)
        testset = MiniImageNet(root=args.data_root, train=False, index=np.arange(num_class), base_sess=1, autoaug=0)
    elif args.dataset == 'cifar100':
        trainset = CIFAR100(root=args.data_root, train=True, index=np.arange(num_class), base_sess=1, autoaug=0)
        testset = CIFAR100(root=args.data_root, train=False, index=np.arange(num_class), base_sess=1, autoaug=0)
    elif args.dataset == 'cub200':
        trainset = CUB200(root=args.data_root, train=True, index=np.arange(num_class), base_sess=1, autoaug=0)
        testset = CUB200(root=args.data_root, train=False, index=np.arange(num_class), base_sess=1, autoaug=0)
    else:
        raise RuntimeError('error dataset')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Train the model
    best_acc = 0.0

    for epoch in range(args.epochs):
        tqdm_gen = tqdm(trainloader)
        embedding_list = []
        label_list = []
        for data in tqdm_gen:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.long().to(device)

            logits = model(inputs)

            train_loss = criterion(logits[:, :num_class], labels)
            train_acc = count_acc(logits[:, :num_class], labels)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            lrc = scheduler.get_last_lr()[0]

            tqdm_gen.set_description(
                'Epoch {:2d} | lrc: {:.4f} | Train Loss: {:.4f} | Train Acc: {:.4f} '.format(epoch, lrc, train_loss,
                                                                                             train_acc))

        # Evaluate the model on the test set
        with torch.no_grad():
            tqdm_gen = tqdm(testloader)
            for data in tqdm_gen:
                images, labels = data
                images, labels = images.to(device), labels.long().to(device)
                logits = model(images)

                test_acc = count_acc(logits[:, :num_class], labels)

                tqdm_gen.set_description(
                    'Epoch {:2d} |  Test Acc: {:.4f} '.format(epoch, test_acc))

        # Save the best performing model parameters
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "./pretrain/%s_id_%s_bs_%s.pth" % (args.dataset, job_id, args.batch_size))

        scheduler.step()

    print(f"Load the best performing model parameters and evaluate it on the test set")
    model.load_state_dict(torch.load("./pretrain/%s_id_%s_bs_%s.pth" % (args.dataset, job_id, args.batch_size)))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.long().to(device)
            logits = model(images)

            test_acc = count_acc(logits[:, :num_class], labels)

    print(f"Best Test Acc: {test_acc :.4f}")
