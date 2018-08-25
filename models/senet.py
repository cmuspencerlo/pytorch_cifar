import torch
import torch.nn as nn
import torch.nn.functional as F

class PreActBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, block_planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, block_planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(block_planes)
        self.conv2 = nn.Conv2d(block_planes, block_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(block_planes)
        self.conv3 = nn.Conv2d(block_planes, self.expansion*block_planes, kernel_size=1, bias=False)

        # Note that we do not need to use BatchNorm2d here
        # for this shortcut would not connect to the activation layer
        if in_planes != self.expansion*block_planes or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*block_planes, kernel_size=1, stride=stride, bias=False))

        self.fc1 = nn.Conv2d(self.expansion*block_planes, self.expansion*block_planes//16, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(self.expansion*block_planes//16, self.expansion*block_planes, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.bn1(x)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(F.relu(out))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out

class SENet(nn.Module):
    in_planes = 64
    def __init__(self, block, num_block_list, num_classes=10):
        super(SENet, self).__init__()
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        def _make_layer(block, block_planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, block_planes, stride=stride))
                #
                self.in_planes = block.expansion * block_planes
            return nn.Sequential(*layers)

        self.layer1 = _make_layer(block, 64, num_block_list[0], stride=1)
        self.layer2 = _make_layer(block, 128, num_block_list[1], stride=2)
        self.layer3 = _make_layer(block, 256, num_block_list[2], stride=2)
        if len(num_block_list) == 4:
            self.layer4 = _make_layer(block, 512, num_block_list[3], stride=2)
        self.linear = nn.Linear(self.in_planes, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, kernel_size=2, stride=2)
        # torch.Size([1, 64, 16, 16])
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if hasattr(self, 'layer4'):
            out = self.layer4(out)
            out = F.avg_pool2d(out, 2)
        else:
            out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def senet152():
    return SENet(PreActBottleneck, [3, 8, 36, 3])

def senet164():
    return SENet(PreActBottleneck, [18, 18, 18])

def senet1001():
    return SENet(PreActBottleneck, [333, 333, 333])

def test():
    device = 'cuda'
    net = senet164()
    net = net.to(device)
    y = net(torch.randn(1, 3, 32, 32).to(device))
    print(y.size())

# test()
