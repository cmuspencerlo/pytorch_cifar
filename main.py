import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
from torchvision.transforms import transforms

from utils.utils import *
from models.lenet import LeNet

if __name__ == "__main__":
    # arg setting
    parser = argparse.ArgumentParser(description='Cifar toy.')
    parser.add_argument('--lr', default=0.1)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Loading data.
    train_transform = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=16)
    evalset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=eval_transform)
    eval_loader = torch.utils.data.DataLoader(evalset, batch_size=128, shuffle=True, num_workers=16)

    # Import model.
    print('==> Building model..')
    # reflection mechanism
    net = LeNet()
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    # 
    # # Set loss and optimization method.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    # Training loop.
    for epoch in range(10):
        # Training step.
        net.train()
        print('Epoch: %d' % epoch)
        train_loss = 0.0
        for batch_idx, [inputs, targets] in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar(batch_idx, len(train_loader),
                'Loss: %.6f' % (train_loss / (batch_idx + 1)))

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
                    'Acc: %.6f' % (correct / total))

