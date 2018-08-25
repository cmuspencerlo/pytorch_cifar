import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    # Inverted part: in_planes < planes => in_planes
    def __init__(self, in_planes, target_planes, expansion, stride):
        super(Bottleneck, self).__init__()
        self.stride = stride
        planes = in_planes * expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # depthwise conv
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, target_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(target_planes)

        self.shortcut = nn.Sequential()
        # Only shortcut when H and W are the same
        if stride == 1 and in_planes != target_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, target_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(target_planes))

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out

class MobileNetV2(nn.Module):
    # (expansion, target_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        self.in_planes = 32
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        def _make_layers():
            layers = []
            for (expansion, target_planes, num_blocks, stride) in self.cfg:
                strides = [stride] + [1] * (num_blocks - 1)
                for stride in strides:
                    layers.append(Bottleneck(self.in_planes, target_planes, expansion, stride))
                    self.in_planes = target_planes
            return nn.Sequential(*layers)

        self.layers = _make_layers()
        self.conv2 = nn.Conv2d(self.in_planes, 1280, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.conv3 = nn.Conv2d(1280, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 2)
        # GENIUS!
        out = out.view(out.size(0), -1, 1, 1)
        # (batch, channel, 1, 1)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        return out

def test():
    device = 'cuda'
    net = MobileNetV2()
    cnt = 0
    for name, param in net.named_parameters():
        if 'weight' in name and 'shortcut' not in name and 'bn' not in name:
            cnt += 1
            # print(name)
    # print(cnt)
    net = net.to(device)
    y = net(torch.randn(1, 3, 32, 32).to(device))
    print(y.size())
    # net = torch.nn.DataParallel(net)
    # reset params
    for m in net.modules():
        if 'Conv2d' in type(m).__name__:
            print(type(m))
            m.reset_parameters()

# test()

