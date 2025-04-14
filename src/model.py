import config
import params
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import CenterCrop
import torch

class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, channels: tuple[int] = (3, 16, 32, 64)):
        super().__init__()
        self.blocks = nn.ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        block_outputs = []

        for block in self.blocks:
            x = block(x)
            block_outputs.append(x)
            x = self.pool(x)

        return block_outputs
    
class Decoder(nn.Module):
    def __init__(self, channels: tuple[int] = (64, 32, 16)):
        super().__init__()
        self.channels = channels
        self.blocks = nn.ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        self.upconv = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=2, stride=2) for i in range(len(channels) - 1)])

    def forward(self, x: torch.Tensor, enc_features: list[torch.Tensor]) -> torch.Tensor:
            # Reverse enc_features to match the decoder's order
        enc_features = enc_features[:-1][::-1]  # Exclude the last feature (already used as input)
        
        for i in range(len(self.channels)):
            x = self.upconv[i](x) 
            enc_feature = self.crop(enc_features[i], x)  
            
            if x.shape[1] + enc_feature.shape[1] != self.channels[i]:
                raise ValueError(f"Channel mismatch: {x.shape[1]} + {enc_feature.shape[1]} != {self.channels[i]}")
            
            x = torch.cat((x, enc_feature), dim=1)  
            x = self.blocks[i](x)  

            return x
    
    def crop(self, enc_feature: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        enc_feature = CenterCrop((h, w))(enc_feature)
        return enc_feature
class UNet(nn.Module):
    def __init__(self, enc_channels: tuple[int] = (3, 64, 128, 256, 512), dec_channels: tuple[int] = (512, 256, 128, 64), num_classes: int = 1, out_size: tuple[int] = config.INPUT_IMAGE_SIZE):
        super().__init__()
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)
        self.final_conv = nn.Conv2d(dec_channels[-1], num_classes, kernel_size=1)
        self.out_size = out_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_features = self.encoder(x)
        
        dec_features = self.decoder(enc_features[-1], enc_features[::-1])

        x = self.final_conv(dec_features)
        x = F.interpolate(x, size=self.out_size)

        return x