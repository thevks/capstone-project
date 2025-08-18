import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.d1 = DoubleConv(in_ch, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)

        self.u1 = DoubleConv(512, 256)
        self.u2 = DoubleConv(256, 128)
        self.u3 = DoubleConv(128, 64)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.emb_layer = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))

        u1 = self.up(c4)
        u1 = self.u1(u1)
        u2 = self.up(u1)
        u2 = self.u2(u2)
        u3 = self.up(u2)
        u3 = self.u3(u3)

        emb = self.emb_layer(c4).view(x.size(0), -1)  # 512-d embedding
        return emb
