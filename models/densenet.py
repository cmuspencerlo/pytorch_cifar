import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # Like preact block
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], dim=1)
        return out

class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        # (B, C, W, H) -> (B, C, 1/2W, 1/2H)
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet(nn.Module):
    # Note that the definition of block is totally different in resnet and densenet
    def __init__(self, block, nblocks, growth_rate=32, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.reduction = reduction
        self.growth_rate = growth_rate
        self.in_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)

        def _make_dense_layer(block, nblock):
            layers = []
            for i in range(nblock):
                layers.append(block(self.in_planes, growth_rate))
                self.in_planes += growth_rate

            return nn.Sequential(*layers)

        def _make_transition_layer(trans_block):
            out_planes = int(math.floor(self.in_planes*self.reduction))
            trans = trans_block(self.in_planes, out_planes)
            self.in_planes = out_planes
            return trans

        trans_block = Transition
        self.dense1 = _make_dense_layer(block, nblocks[0])
        # self.in_planes == nblocks[0] * growth_rate + num_planes
        self.trans1 = _make_transition_layer(trans_block)
        self.dense2 = _make_dense_layer(block, nblocks[1])
        self.trans2 = _make_transition_layer(trans_block)
        self.dense3 = _make_dense_layer(block, nblocks[2])
        self.trans3 = _make_transition_layer(trans_block)
        self.dense4 = _make_dense_layer(block, nblocks[3])

        self.linear = nn.Linear(self.in_planes, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        # torch.Size([1, 2*growth_rate, 16, 16])
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def densenet121():
    return DenseNet(Bottleneck, [6, 12, 24, 16])

def densenet169():
    return DenseNet(Bottleneck, [6, 12, 32, 32])

def densenet201():
    return DenseNet(Bottleneck, [6, 12, 48, 32])

def densenet161():
    return DenseNet(Bottleneck, [6, 12, 36, 24], 48)

def test():
    device = 'cuda'
    net = densenet161()
    net = net.to(device)
    y = net(torch.randn(1, 3, 32, 32).to(device))
    print(y.size())

#test()
