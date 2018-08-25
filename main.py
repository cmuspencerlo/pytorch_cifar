import argparse
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from torchvision.transforms import transforms

from utils.utils import *
from models import *
from tensorboardX import SummaryWriter

import numpy as np
import copy

TEST_FREQUENCY = 10

def build_dataset(args):
    if args.type == 10:
        # Hack
        train_normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_transforms = transforms.Compose([
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            train_normalize])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transforms)
        # 9 - 1
        train_size = int(0.9 * len(trainset))
        eval_size = len(trainset) - train_size
        trainset, evalset = torch.utils.data.random_split(trainset, [train_size, eval_size])

        test_normalize = transforms.Normalize(mean=[0.4942, 0.4851, 0.4504], std=[0.2467, 0.2429, 0.2616])
        test_transforms = transforms.Compose([
            transforms.RandomCrop(size=32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            test_normalize])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transforms)
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
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    return train_loader, eval_loader, test_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cifar toy.')
    parser.add_argument('--loops', default=1, type=int, metavar='N', help='number of loops to run')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--type', default=10, type=int)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=256, type=int, metavar='N', help='number of batch')
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='use tensorboard')
    parser.add_argument('--no-tensorboard', dest='tensorboard', action='store_false', help='use tensorboard')
    parser.set_defaults(tensorboard=False)
    parser.add_argument('--init_weight', dest='init_weight', action='store_true', help='init weight')
    parser.add_argument('--no_init_weight', dest='init_weight', action='store_false', help='init weight')
    parser.set_defaults(init_weight=False)
    parser.add_argument('--adjust_lr', dest='adjust_lr', action='store_true', help='adjust_lr')
    parser.add_argument('--no_adjust_lr', dest='adjust_lr', action='store_false', help='adjust_lr')
    parser.set_defaults(adjust_lr=True)
    parser.add_argument('--topk', default=1, type=int, metavar='K', help='top-k')
    parser.add_argument("--optimizer", default='SGD', type=str, help="optimizer options")

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Import model.
    print('==> Building model..')
    # reflection mechanism
    # net = LeNet()
    net = densenet121()
    net = net.to(device)

    if args.tensorboard:
        writer = SummaryWriter()
        writer.add_graph(net, torch.rand(1, 3, 32, 32).to(device))
        net_tag = type(net).__name__
        param_dict = build_params(net, args)
        add_params(writer, net_tag, param_dict)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # Set loss and optimization method.
    criterion = nn.CrossEntropyLoss()

    # Training loop.
    loss_t = np.zeros([args.loops, args.epochs])
    train_accuracy_t = np.zeros([args.loops, args.epochs])
    eval_t = np.zeros([args.loops, args.epochs])
    test_t = np.zeros([args.loops, args.epochs//TEST_FREQUENCY+1])
    for i in range(args.loops):
        print('Loop: %d' % i)
        train_loader, eval_loader, test_loader = build_dataset(args)
        # init_params(net, args.init_weight)
        reset_params(net)
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

        train_acc_window = np.array([])
        best_eval_acc = 0.0
        for epoch in range(args.epochs):
            print('Epoch: %d' % epoch)
            if args.adjust_lr:
                optimizer = adjust_lr(net, optimizer, args.lr, epoch, train_acc_window, False)
            # Training step.
            net.train()
            train_loss = 0.0
            correct = 0
            total = 0
            train_acc = 0.0
            for batch_idx, [inputs, targets] in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                _, prediction = outputs.max(1)
                correct += prediction.eq(targets).sum().item()
                total += targets.size(0)
                optimizer.zero_grad()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_acc = correct / total
                progress_bar(batch_idx, len(train_loader),
                    '[TRAIN] Loss: %.6f  Acc: %.6f' % (train_loss/(batch_idx+1), train_acc))
            loss_t[i][epoch] = train_loss / len(train_loader)
            train_accuracy_t[i][epoch] = train_acc
            if args.tensorboard:
                for name, param in net.named_parameters():
                    writer.add_histogram(name, param, epoch)
            train_acc_window = build_window(train_acc_window, train_acc)

            # Evaluation step.
            net.eval()
            correct = 0
            total = 0
            eval_acc = 0.0
            with torch.no_grad():
                for batch_idx, [inputs, targets] in enumerate(eval_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    _, prediction = outputs.max(1)
                    correct += prediction.eq(targets).sum().item()
                    total += targets.size(0)
                    eval_acc = correct / total

                    progress_bar(batch_idx, len(eval_loader), '[EVAL] Acc: %.6f' % (eval_acc))
            eval_t[i][epoch] = eval_acc
            if eval_acc > best_eval_acc:
                if save_net(epoch, net):
                    best_eval_acc = eval_acc

            # Test step.
            if (epoch + 1) % TEST_FREQUENCY == 0:
                test_net = copy.deepcopy(net)
                load_net(test_net)
                test_net.eval()
                correct = 0
                total = 0
                test_acc = 0.0
                with torch.no_grad():
                    for batch_idx, [inputs, targets] in enumerate(test_loader):
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = test_net(inputs)
                        _, prediction = outputs.max(1)
                        correct += prediction.eq(targets).sum().item()
                        total += targets.size(0)
                        test_acc = correct / total

                        progress_bar(batch_idx, len(test_loader), '[TEST] Acc: %.6f' % (test_acc))
                test_t[i][(epoch+1)//TEST_FREQUENCY] = test_acc

    loss_t = np.mean(loss_t, axis=0)
    eval_t = np.mean(eval_t, axis=0)
    test_t = np.mean(test_t, axis=0)
    if args.tensorboard:
        for epoch in range(args.epochs):
            writer.add_scalar('TRAIN/loss', loss_t[epoch], epoch)
            writer.add_scalar('EVAL/accuracy', eval_t[epoch], epoch)
            if (epoch + 1) % TEST_FREQUENCY == 0:
                tepoch = (epoch + 1) // TEST_FREQUENCY
                writer.add_scalar('TEST/accuracy', test_t[tepoch], tepoch)
        writer.close()
