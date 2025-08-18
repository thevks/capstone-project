# unet3d.py
"""
3D U-Net model (encoder-decoder). Also exposes a function to get feature embeddings
from the bottleneck (global-pooled).
PyTorch implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class UpConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
    def forward(self, x): return self.up(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, base_filters=32, embedding_pool='avg'):
        super().__init__()
        f = base_filters
        self.enc1 = ConvBlock3D(in_channels, f)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(f, f*2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(f*2, f*4)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = ConvBlock3D(f*4, f*8)
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock3D(f*8, f*16)

        self.up4 = UpConv3D(f*16, f*8)
        self.dec4 = ConvBlock3D(f*16, f*8)
        self.up3 = UpConv3D(f*8, f*4)
        self.dec3 = ConvBlock3D(f*8, f*4)
        self.up2 = UpConv3D(f*4, f*2)
        self.dec2 = ConvBlock3D(f*4, f*2)
        self.up1 = UpConv3D(f*2, f)
        self.dec1 = ConvBlock3D(f*2, f)

        self.out_conv = nn.Conv3d(f, n_classes, kernel_size=1)

        if embedding_pool == 'avg':
            self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        else:
            self.pool = nn.AdaptiveMaxPool3d((1,1,1))

    def forward(self, x):
        # x shape: (B, C, D, H, W)
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)  # bottleneck feature map
        # decoder
        u4 = self.up4(b)
        u4 = torch.cat([u4, e4], dim=1)
        d4 = self.dec4(u4)
        u3 = self.up3(d4)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)
        u2 = self.up2(d3)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)
        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        out = self.out_conv(d1)  # segmentation logits if needed

        # global embedding from bottleneck
        emb = self.pool(b).view(b.size(0), -1)  # (B, D)
        return out, emb

if __name__ == "__main__":
    # quick smoke test
    x = torch.randn(2, 1, 9, 128, 128)  # batch=2, depth=9
    m = UNet3D(in_channels=1, n_classes=2, base_filters=16)
    out, emb = m(x)
    print("out:", out.shape, "emb:", emb.shape)

'''
Notes:

embedding_pool gives you a fixed-size embedding for any input volume size.

For per-slice 2D experiments you can reuse the 2D UNet (earlier message) or convert this to 2D by replacing Conv3d/MaxPool3d with 2D counterparts.
'''
