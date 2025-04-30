from torch import nn
import torch

class UNET(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, channel_factor: int = 32):
        super(UNET, self).__init__()
        # encoder

        self.enc1 = EncoderBlock(in_channels, channel_factor, kernel_size=5, padding=2, double_conv=True)

        self.enc2 = EncoderBlock(channel_factor, channel_factor * 2)
        self.enc3 = EncoderBlock(channel_factor * 2, channel_factor * 4)

        # bottleneck
        self.bottleneck = ConvBlock(channel_factor * 4, channel_factor * 8, kernel_size=3, padding=1)

        # decoder
        self.dec3 = DecoderBlock(channel_factor * 8, channel_factor * 4)
        self.dec2 = DecoderBlock(channel_factor * 4, channel_factor * 2)
        self.dec1 = DecoderBlock(channel_factor * 2, channel_factor)

        self.final_conv = nn.Conv2d(channel_factor, out_channels, kernel_size=1)

        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Encoder
        enc1, p1 = self.enc1(x)
        enc2, p2 = self.enc2(p1)
        enc3, p3 = self.enc3(p2)

        # Bottleneck
        bottleneck = self.bottleneck(p3)
        # bottleneck = self.dropout(bottleneck)

        # Decoder
        dec3 = self.dec3(bottleneck, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)

        return torch.sigmoid(self.final_conv(dec1))
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, double_conv=False):
        super(ConvBlock, self).__init__()

        steps = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if double_conv:
            steps = [
                *steps,
                nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
                # nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]


        self.conv = nn.Sequential(*steps)
        

    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, double_conv=False):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels, kernel_size, padding, double_conv)
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

            
        