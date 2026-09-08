import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + x, inplace=True)

class MultiScaleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_ch, out_ch, kernel_size=5, padding=2, bias=False)
        self.conv7 = nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(out_ch * 3)
        self.fuse = nn.Conv2d(out_ch * 3, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        out = torch.cat([x3, x5, x7], dim=1)
        out = F.relu(self.bn(out), inplace=True)
        out = self.fuse(out)
        return out

class CsiNetPlus(nn.Module):
    def __init__(self, h, w, cr):
        super().__init__()
        self.h = h
        self.w = w
        self.in_dim = 2 * h * w
        self.out_dim = self.in_dim
        self.comp_dim = self.in_dim // cr

        # Encoder
        self.enc_conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            ResBlock(16),
            ResBlock(16)
        )
        self.enc_fc = nn.Linear(16 * h * w, self.comp_dim)

        # Decoder
        self.dec_fc = nn.Linear(self.comp_dim, self.out_dim)

        # Refine: multi-scale + residual
        self.refine = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            MultiScaleBlock(16, 16),
            ResBlock(16),
            nn.Conv2d(16, 2, kernel_size=3, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        # x: [B, 2, H, W]
        x = self.enc_conv(x)
        x = x.view(x.size(0), -1)
        z = self.enc_fc(x)

        x_hat = self.dec_fc(z)
        x_hat = x_hat.view(x_hat.size(0), 2, self.h, self.w)

        out = self.refine(x_hat)
        return out
