'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

def gen_mean_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=False, num_workers=2)
    train = iter(dataloader).next()[0]
    mean = np.mean(train.numpy(), axis=(0, 2, 3))
    std = np.std(train.numpy(), axis=(0, 2, 3))
    return mean, std
# cifar10 = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
# print(gen_mean_std(cifar10))
# (array([0.4914009 , 0.48215896, 0.4465308 ], dtype=float32)
# (array([0.24703279, 0.24348423, 0.26158753], dtype=float32))

# (array([0.49421427, 0.4851322 , 0.45040992], dtype=float32)
#  array([0.24665268, 0.24289216, 0.2615922 ], dtype=float32))

def init_params(net, flag=False):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            if flag:
                init.normal_(m.weight)
            else:
                init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

def reset_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

WINDOW_SIZE = 30
decay_cnt = 0
keep_lr = 0
def build_window(train_acc_window, train_acc):
    train_acc_window = np.append(train_acc_window, train_acc)
    while(len(train_acc_window) > WINDOW_SIZE):
        train_acc_window = np.delete(train_acc_window, 0)
    # print('t_mean: %.6f  t_std: %.6f' % (np.mean(train_acc), np.std(train_acc)))
    return train_acc_window

def save_net(epoch, net):
    if epoch == 0 or epoch + 1 >= WINDOW_SIZE:
        print('Saving...')
        net_tag = type(net).__name__
        state = {
            'net': net.state_dict(),
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + net_tag +'_ckpt.pkl')
        return True
    return False

def load_net(net):
    # Load checkpoint.
    print('Loading net...')
    net_tag = type(net).__name__
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/' + net_tag +'_ckpt.pkl')
    net.load_state_dict(checkpoint['net'])
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    return optimizer

def adjust_lr(net, optimizer, lr, epoch, train_acc_window, default=True):
    if default:
        if epoch < 120:
            lr = lr
        elif epoch < 160:
            lr *= 0.1
        else:
            lr *= 0.01
    else:
        global keep_lr, decay_cnt
        if epoch == 0:
            keep_lr = lr
            decay_cnt = 0
        if len(train_acc_window) == WINDOW_SIZE:
            std = np.std(train_acc_window)
            std_threshold = 1e-2 / (10**decay_cnt)
            if decay_cnt >= 2:
                std_threshold *= 2
            if std < std_threshold:
                decay_cnt += 1
                keep_lr /= 10
                optimizer = load_net(net)
                print('Trigger lr: %lf' % keep_lr)
        lr = keep_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def build_params(net, args):
    param_dict = dict()

    trivial_terms = ['shortcut', 'bn']
    num_params = 0
    cnt = 0
    for name, param in net.named_parameters():
        num_params += param.numel()
        if 'weight' in name:
            display_flag = True
            for trivial_term in trivial_terms:
                if trivial_term in name:
                    display_flag = False
                    break
            if display_flag:
                cnt += 1
    print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
    print('[Network] Total layers: %d' % cnt)

    param_dict['lr'] = args.lr
    param_dict['adjust_lr'] = args.adjust_lr
    param_dict['layer_num'] = cnt
    param_dict['param_num'] = str(round(num_params / 1e6, 3)) + 'M'
    param_dict['epochs'] = args.epochs
    param_dict['loops'] = args.loops
    param_dict['optimizer'] = args.optimizer
    param_dict['batch_size'] = args.batch_size
    param_dict['init_weight'] = args.init_weight

    return param_dict

def add_params(writer, tag, param_dict):
    writer.add_text(tag, str(param_dict))

