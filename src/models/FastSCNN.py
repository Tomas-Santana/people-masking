from typing import Optional
import torch.nn as nn
import torch
from torch.nn import functional as F

class DSConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class DWConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)
    
class ConvBN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 0, bias: bool = True):
        super(ConvBN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.conv(x)
    

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 0):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            ConvBN(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class LinearBottleneck(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, t: int=6, stride: int=2):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out
    
class PyramidPooling(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = ConvBNReLU(in_channels, inter_channels, 1)
        self.conv2 = ConvBNReLU(in_channels, inter_channels, 1)
        self.conv3 = ConvBNReLU(in_channels, inter_channels, 1)
        self.conv4 = ConvBNReLU(in_channels, inter_channels, 1)
        self.out = ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x
    
class LearningToDownsample(nn.Module):
    def __init__(self, dw_channels1: int = 32, dw_channels2=48, out_channels: int=64):
        super(LearningToDownsample, self).__init__()
        self.conv1 = ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = DSConv(dw_channels2, out_channels, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x
    

class GlobalFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int=64, block_channels:tuple[int]=(64, 96, 128), out_channels: int=128, t:int=6, num_blocks:tuple[int]=(3,3,3)):
        super(GlobalFeatureExtractor, self).__init__()
        self.btneck1 = self.stack(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.btneck2 = self.stack(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.btneck3 = self.stack(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppool = PyramidPooling(block_channels[2], out_channels)

    def stack(self, block, inplanes:int, planes:int, blocks:int, t:int=6, stride:int=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.btneck1(x)
        x = self.btneck2(x)
        x = self.btneck3(x)
        x = self.ppool(x)
        return x
    
class FeatureFusion(nn.Module):
    """Feature fusion module"""

    def __init__(self, ltd_channels: int, gfe_channels: int, out_channels: int, scale_factor=4):
        super(FeatureFusion, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = DWConv(gfe_channels, out_channels, 1)
        self.gfe_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.ltd_conv = nn.Sequential(
            nn.Conv2d(ltd_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, ltd, gfe):
        gfe = F.interpolate(gfe, scale_factor=4, mode='bilinear', align_corners=True)
        gfe = self.dwconv(gfe)
        gfe = self.gfe_conv(gfe)

        ltd = self.ltd_conv(ltd)
        out = ltd + gfe
        return self.relu(out)

class Classifier(nn.Module):
    def __init__(self, in_channels: int=128, stride=1,  num_classes: int=1):
        super(Classifier, self).__init__()
        self.dsconv1 = DSConv(in_channels, in_channels, stride)
        self.dsconv2 = DSConv(in_channels, in_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(in_channels, num_classes, 1, bias=False),
        )        

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x

class FastSCNN(nn.Module):
    def __init__(self, num_classes: int=1):
        super(FastSCNN, self).__init__()
        self.ltd = LearningToDownsample()
        self.gfe = GlobalFeatureExtractor()
        self.ffu = FeatureFusion(64, 128, 128)
        self.clf = Classifier(num_classes=num_classes)
    
    def forward(self, x):
        size = x.size()[2:]
        ltd = self.ltd(x)
        gfe = self.gfe(ltd)
        ffu = self.ffu(gfe, ltd)
        out = self.clf(ffu)
        out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        return out
    
def fastscnn(num_classes: int=1, state_path:Optional[str]=None) -> FastSCNN:
    model = FastSCNN(num_classes)

    if state_path:
        state_dict = torch.load(state_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded state dict from {state_path}")

    return model