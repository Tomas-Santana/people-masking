from torch import nn
import torch
import torch.nn.functional as F
import torchvision.transforms as TF


class UNET(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super(UNET, self).__init__()
        N = 128
        # encoder
        self.enc1 = EncoderBlock(in_channels, N)
        self.enc2 = EncoderBlock(N, N)
        self.enc3 = EncoderBlock(N, N)
        self.enc4 = EncoderBlock(N, N)
        self.enc5 = EncoderBlock(N, N)


        # bottleneck
        self.bottleneck = ConvBlock(N, N)

        # decoder
        self.dec5 = DecoderBlock(N, N)
        self.dec4 = DecoderBlock(N, N)
        self.dec3 = DecoderBlock(N, N)
        self.dec2 = DecoderBlock(N, N)
        self.dec1 = DecoderBlock(N, N)

        self.final_conv = nn.Conv2d(N, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1, p1 = self.enc1(x)
        enc2, p2 = self.enc2(p1)
        enc3, p3 = self.enc3(p2)
        enc4, p4 = self.enc4(p3)
        enc5, p5 = self.enc5(p4)

        # Bottleneck
        bottleneck = self.bottleneck(p5)

        # Decoder
        dec5 = self.dec5(bottleneck, enc5)
        dec4 = self.dec4(dec5, enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)

        return torch.sigmoid(self.final_conv(dec1))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.1),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(DecoderBlock, self).__init__()
        if bilinear:
            self.upconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upconv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        # Adjust the input channels for the ConvBlock to account for concatenation
        self.conv = ConvBlock(in_channels + out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)

        x = torch.cat((x2, x1), dim=1)
        return self.conv(x)



