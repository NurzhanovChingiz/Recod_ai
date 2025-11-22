from torch import nn
import torch.nn.functional as F

class DinoTinyDecoder(nn.Module):
    def __init__(self, in_ch=768, out_ch=1, bottleneck_ch=96):
        super().__init__()
        # Ultra-fast head:
        # 1x1 conv to bottleneck channels
        # 3x3 depthwise conv
        # 1x1 conv to output channels
        # Shape
        # Input: (B, in_ch, S, S)
        # Output: (B, out_ch, S, S) upsampled later
        
        
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, bottleneck_ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                bottleneck_ch,
                bottleneck_ch,
                kernel_size=3,
                padding=1,
                groups=bottleneck_ch,  # depthwise conv
                bias=False,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(bottleneck_ch, out_ch, kernel_size=1, bias=True),
        )
        # self.net = nn.Sequential(
        #     nn.Conv2d(in_ch, 256, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 64, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, out_ch, 1)
        # )
        self._initialize_weights()
    def forward(self, f, size):
        # f: (B, in_ch, S, S) from DINO
        # all convs at SxS, then upsample logits once
        x = self.net(f)  # (B, out_ch, S, S)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x
    
    def _initialize_weights(self):
        """Initialize weights to prevent NaN issues"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    