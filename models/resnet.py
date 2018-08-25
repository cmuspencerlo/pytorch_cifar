import torch
import torch.nn as nn
import torch.nn.functional as F

# We can use conv setting to control W*H*D
# Trick here:
# K=3, S=1, P=1 to keep W*H
# K=3, S=2, P=1 to shrinkage 1/2W*1/2H
# K=1 to tune D
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, block_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, block_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(block_planes)
        self.conv2 = nn.Conv2d(block_planes, block_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(block_planes)

        self.shortcut = nn.Sequential()
        # Adjust the input x when
        # 1. in_planes != out_planes
        # 2. Difference in W*H
        if in_planes != self.expansion*block_planes or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*block_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*block_planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, block_planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, block_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(block_planes)
        self.conv2 = nn.Conv2d(block_planes, block_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(block_planes)
        self.conv3 = nn.Conv2d(block_planes, self.expansion*block_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*block_planes)

        self.shortcut = nn.Sequential()
        if in_planes != self.expansion*block_planes or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*block_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*block_planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class PreActBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, block_planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, block_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(block_planes)
        self.conv2 = nn.Conv2d(block_planes, block_planes, kernel_size=3, stride=1, padding=1, bias=False)

        if in_planes != self.expansion*block_planes or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*block_planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = self.bn1(x)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(F.relu(out))
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

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

    def forward(self, x):
        # Key here is to add activation layer asap
        # There are multiple ways to build shortcut here
        # Note that it is not conventional use x as the input of shortcut
        # out = self.bn1(x)
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        # out = self.conv1(F.relu(out))
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class ResNet(nn.Module):
    in_planes = 64
    def __init__(self, block, num_block_list, num_classes=10):
        super(ResNet, self).__init__()
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

        # No clue why we fail here for gpu version
        # Maybe it goes wrong for this line:
        # self.layers = []
        # for num_block in num_block_list:
        #     self.layers.append(_make_layer())
        # Try a plain way to get around
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

def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])

def resnet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def preact_resnet152():
    return ResNet(PreActBottleneck, [3, 8, 36, 3])

def preact_resnet164():
    return ResNet(PreActBottleneck, [18, 18, 18])

def preact_resnet1001():
    return ResNet(PreActBottleneck, [111, 111, 111])

def test():
    device = 'cuda'
    net = preact_resnet164()
    cnt= 0
    for name, param in net.named_parameters():
        if 'weight' in name and 'shortcut' not in name and 'bn' not in name:
            cnt += 1
            print(name)
    print(cnt)
    net = net.to(device)
    y = net(torch.randn(1, 3, 32, 32).to(device))
    print(y.size())

# test()
