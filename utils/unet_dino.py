import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # in_ch: channels from lower level
        # skip_ch: channels from skip connection
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Assume spatial sizes are compatible; if off by 1, you can crop/pad here
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            # simple center crop to match
            _, _, h, w = skip.shape
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
class DinoUNetDecoder(nn.Module):
    def __init__(self, in_ch=768, base_ch=128, out_ch=1):
        super().__init__()

        # "Encoder" path on top of DINO fmap (shallow to keep tiny objects & speed)
        self.enc1 = ConvBlock(in_ch, base_ch)        # S x S
        self.pool1 = nn.MaxPool2d(2)                 # S/2

        self.enc2 = ConvBlock(base_ch, base_ch * 2)  # S/2
        self.pool2 = nn.MaxPool2d(2)                 # S/4

        # bottleneck at lowest resolution (S/4)
        self.bottleneck = ConvBlock(base_ch * 2, base_ch * 4)

        # Decoder with U-Net skips
        self.up2 = UpBlock(in_ch=base_ch * 4, skip_ch=base_ch * 2, out_ch=base_ch * 2)
        self.up1 = UpBlock(in_ch=base_ch * 2, skip_ch=base_ch,     out_ch=base_ch)

        # final 1x1 conv to logits
        self.final_conv = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, f, size):
        S = f.shape[-1]
    
        x1 = self.enc1(f)      # S
        x2_in = self.pool1(x1) # S/2
    
        if S >= 16:
            # делаем полный двухуровневый UNet
            x2 = self.enc2(x2_in)      # S/2
            x3_in = self.pool2(x2)     # S/4
            x3 = self.bottleneck(x3_in)
    
            x = self.up2(x3, x2)
            x = self.up1(x, x1)
        else:
            # слишком мелко — делаем один даун/ап, попроще
            x2 = self.enc2(x2_in)
            x = self.up1(x2, x1)
    
        logits = self.final_conv(x)
        logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
        return logits