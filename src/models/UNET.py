from torch import nn
from torch.nn import functional as F
import torch

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, max_channels=256):
        super(UNET, self).__init__()
        # N = 256

        # going down
        self.in_to_over1 = nn.Conv2d(in_channels=in_channels, out_channels=max_channels//32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(max_channels//32)
        self.over1_to_down2 = nn.Conv2d(in_channels=max_channels//32, out_channels=max_channels//16, kernel_size=3, padding=1)  # 1/2
        self.bn2 = nn.BatchNorm2d(max_channels//16)
        self.down2_to_down3 = nn.Conv2d(in_channels=max_channels//16, out_channels=max_channels//8, kernel_size=3, padding=1)  # 1/4
        self.bn3 = nn.BatchNorm2d(max_channels//8)
        self.down3_to_down4 = nn.Conv2d(in_channels=max_channels//8, out_channels=max_channels//4, kernel_size=3, padding=1)  # 1/8
        self.bn4 = nn.BatchNorm2d(max_channels//4)
        self.down4_to_down5 = nn.Conv2d(in_channels=max_channels//4, out_channels=max_channels//2, kernel_size=3, padding=1)  # 1/16
        self.bn5 = nn.BatchNorm2d(max_channels//2)

        # horizontal
        self.down5_to_over6 = nn.Conv2d(in_channels=max_channels//2, out_channels=max_channels, kernel_size=1)

        # going up
        self.over6_to_up7 = nn.ConvTranspose2d(in_channels=max_channels, out_channels=max_channels//2, kernel_size=4, stride=2, padding=1)  # 1/8
        self.up7_to_up8 = nn.ConvTranspose2d(in_channels=max_channels//2 + max_channels//2, out_channels=max_channels//4, kernel_size=4, stride=2, padding=1)  # 1/4
        self.up8_to_up9 = nn.ConvTranspose2d(in_channels=max_channels//4 + max_channels//4, out_channels=max_channels//8, kernel_size=4, stride=2, padding=1)  # 1/2
        self.up9_to_up10 = nn.ConvTranspose2d(in_channels=max_channels//8 + max_channels//8, out_channels=max_channels//16, kernel_size=4, stride=2, padding=1)  # 1
        self.up10_to_out = nn.ConvTranspose2d(in_channels=max_channels//16, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        # 128 x 128 image
        x = F.relu(self.bn1(self.in_to_over1(x)))
        # z.shape: torch.Size([1, 16, 128, 128])
        z = F.relu(self.bn2(self.over1_to_down2(x)))
        x = F.max_pool2d(z, kernel_size=2)  # 1/2 x 1/2
        # now: 64 x 64 image
        # y.shape: torch.Size([1, 32, 64, 64])
        y = F.relu(self.bn3(self.down2_to_down3(x)))
        x = F.max_pool2d(y, kernel_size=2)  # 1/4 x 1/4
        # now: 32 x 32 image
        # z.shape: torch.Size([1, 64, 32, 32])
        z = F.relu(self.bn4(self.down3_to_down4(x)))
        x = F.max_pool2d(z, kernel_size=2)  # 1/8 x 1/8
        # now: 16 x 16 image
        # w.shape: torch.Size([1, 128, 16, 16])
        w = F.relu(self.bn5(self.down4_to_down5(x)))
        x = F.max_pool2d(w, kernel_size=2)  # 1/16 x 1/16
        # now: 8 x 8 image

        x = F.relu(self.down5_to_over6(x))

        x = F.relu(self.over6_to_up7(x))  # 1/8 x 1/8
        # now: 16 x 16 image
        x = torch.cat((x, w), dim=1)
        x = F.relu(self.up7_to_up8(x))  # 1/4 x 1/4
        # now: 32 x 32 image
        x = torch.cat((x, z), dim=1)
        x = F.relu(self.up8_to_up9(x))  # 1/2 x 1/2
        x = torch.cat((x, y), dim=1)
        x = F.relu(self.up9_to_up10(x))  # 1 x 1
        return self.up10_to_out(x)

    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()

        steps = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        ]

        # if double_conv:
        #     steps = [
        #         *steps,
        #         nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
        #         # nn.BatchNorm2d(out_channels),
        #         nn.ReLU(inplace=True),
        #     ]


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

            
        