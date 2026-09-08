# models/se_csinet_res.py
import torch
import torch.nn as nn
from models.se_module import SEBlock

class ResWrap(nn.Module):
    def __init__(self, block, use_residual=True):
        super().__init__()
        self.block = block
        self.use_residual = use_residual

    def forward(self, x):
        y = self.block(x)
        if self.use_residual:
            if y.shape == x.shape:
                return y + x
        return y

class SECsiNetRes(nn.Module):
    """
    在SE-CsiNet基础上，可控地给每个block加残差:
    - res_encoder_conv
    - res_encoder_se
    - res_refine_se
    """
    def __init__(self, h, w, cr, se_reduction=8,
                 res_encoder_conv=False, res_encoder_se=False, res_refine_se=False):
        super().__init__()
        self.h = h
        self.w = w
        self.in_dim = 2 * h * w
        self.out_dim = self.in_dim
        self.comp_dim = self.in_dim // cr

        encoder_conv = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        self.encoder_conv = ResWrap(encoder_conv, use_residual=res_encoder_conv)
        self.encoder_se = ResWrap(SEBlock(2, reduction=se_reduction), use_residual=res_encoder_se)
        self.encoder_fc = nn.Linear(self.in_dim, self.comp_dim)

        self.decoder_fc = nn.Linear(self.comp_dim, self.out_dim)

        self.refine = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.refine_se = ResWrap(SEBlock(16, reduction=se_reduction), use_residual=res_refine_se)
        self.refine_out = nn.Sequential(
            nn.Conv2d(16, 2, kernel_size=3, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder_conv(x)
        x = self.encoder_se(x)
        x = x.view(x.size(0), -1)
        z = self.encoder_fc(x)

        x_hat = self.decoder_fc(z)
        x_hat = x_hat.view(x_hat.size(0), 2, self.h, self.w)

        x_hat = self.refine(x_hat)
        x_hat = self.refine_se(x_hat)
        out = self.refine_out(x_hat)
        return out
