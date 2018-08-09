import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
from torchvision.transforms import transforms

from utils.utils import *
from models.lenet import LeNet
from models.resnet import *
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    # arg setting
    parser = argparse.ArgumentParser(description='Cifar toy.')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--type', default=10, type=int)
    parser.add_argument('--lr', default=0.1)
    parser.add_argument('--batch_size', default=256, type=int, metavar='N', help='number of batch')
    # parser.add_argument('--tensorboard', default=0.1)
    # parser.add_argument('--topk', default=1, type=int)
    # parser.add_argument('--adam', default=1, type=int)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Loading data.
    if args.type == 10:
        img_normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        trainset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=True, 
            download=True, 
            transform=transforms.Compose([
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                img_normalize]))
        evalset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=False, 
            download=True, 
            transform=transforms.Compose([
                transforms.RandomCrop(size=32, padding=4),
                transforms.ToTensor(),
                img_normalize]))
    else:
        img_normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        trainset = torchvision.datasets.CIFAR100(
            root='./data', 
            train=True, 
            download=True, 
            transform=transforms.Compose([
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                img_normalize]))
        evalset = torchvision.datasets.CIFAR100(
            root='./data', 
            train=False, 
            download=True, 
            transform=transforms.Compose([
                transforms.RandomCrop(size=32, padding=4),
                transforms.ToTensor(),
                img_normalize]))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    eval_loader = torch.utils.data.DataLoader(evalset, batch_size=args.batch_size, shuffle=True, num_workers=16)

    writer = SummaryWriter()
    # Import model.
    print('==> Building model..')
    # reflection mechanism
    # net = LeNet()
    net = preact_resnet152()

    net = net.to(device)
    writer.add_graph(net, torch.rand(1, 3, 32, 32).to(device))

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # # Set loss and optimization method.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    # Training loop.
    for epoch in range(args.epochs):
        adjust_lr(optimizer, net, epoch)
        # Training step.
        net.train()
        print('Epoch: %d' % epoch)
        train_loss = 0.0
        for batch_idx, [inputs, targets] in enumerate(train_loader):
            # print(targets)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar(batch_idx, len(train_loader),
                'Loss: %.6f' % (train_loss / (batch_idx + 1)))

        writer.add_scalar('loss', train_loss / (batch_idx + 1), epoch)
        for name, param in net.named_parameters():
            writer.add_histogram(name, param, epoch)

        # Evaluation step.
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, [inputs, targets] in enumerate(eval_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                _, prediction = outputs.max(1)
                correct += prediction.eq(targets).sum().item()
                total += targets.size(0)

                progress_bar(batch_idx, len(eval_loader),
                    'Acc: %.6f' % (correct/total))
    
        writer.add_scalar('accuracy', correct/total, epoch)
    writer.close()
