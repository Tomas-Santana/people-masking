from torch import nn
import torch
import torch.nn.functional as F
import torchvision.transforms as TF

class UNET(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super(UNET, self).__init__()
        # encoder
        self.encoder1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.encoder2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.encoder3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

       
        self.decoder3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.decoder2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.decoder1 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        
        enc1 = F.relu(self.encoder1(x)) # 128x128x32
        enc1_pooled = F.max_pool2d(enc1, 2) # 64x64x32
        enc2 = F.relu(self.encoder2(enc1_pooled)) # 64x64x32
        enc2_pooled = F.max_pool2d(enc2, 2) # 32x32x32
        enc3 = F.relu(self.encoder3(enc2_pooled)) # 32x32x32
        

        enc3_resized = F.interpolate(enc3, size=(64, 64), mode='bilinear', align_corners=False) # 64x64x32


        dec3 = torch.cat((enc3_resized, enc2), dim=1) # 64x64x64
        dec3 = F.relu(self.decoder3(dec3))

        dec3_resized = F.interpolate(dec3, size=(128, 128), mode='bilinear', align_corners=False)

        dec2 = torch.cat((dec3_resized, enc1), dim=1) # 128x128x64
        dec2 = F.relu(self.decoder2(dec2))

        out = F.relu(self.decoder1(dec2)) # 128x128x1
        return out