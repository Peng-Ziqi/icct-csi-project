import torch
import torch.nn as nn
from models.se_module import SEBlock


class AWGNChannel(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def power_normalize(x, eps=1e-8):
        power = torch.mean(x ** 2, dim=1, keepdim=True)
        return x / torch.sqrt(power + eps)

    @staticmethod
    def add_awgn(x, snr_db):
        # snr_db: float or tensor [B,1]
        if isinstance(snr_db, (float, int)):
            snr_linear = 10 ** (float(snr_db) / 10.0)
            noise_std = (1.0 / snr_linear) ** 0.5
            return x + torch.randn_like(x) * noise_std
        else:
            snr_linear = torch.pow(10.0, snr_db / 10.0)
            noise_std = torch.sqrt(1.0 / snr_linear)
            return x + torch.randn_like(x) * noise_std

    def forward(self, x_code, snr_db):
        x_norm = self.power_normalize(x_code)
        return self.add_awgn(x_norm, snr_db)


class SNRHead(nn.Module):
    """从输入CSI估计SNR（回归）"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(8, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)  # [B,1]


class SECsiNetDynamic(nn.Module):
    """
    动态压缩 v2:
    - teacher gate: 用真实snr分档（训练稳定、保证多档码长生效）
    - estimate gate: 用snr_head估计分档（部署模式）
    """
    def __init__(self, h, w, cr, se_reduction=8, snr_low=8.0, snr_high=15.0):
        super().__init__()
        self.h = h
        self.w = w
        self.in_dim = 2 * h * w
        self.comp_dim_max = self.in_dim // cr
        self.out_dim = self.in_dim

        self.snr_low = snr_low
        self.snr_high = snr_high

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(2, 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        self.encoder_se = SEBlock(2, reduction=se_reduction)
        self.encoder_fc = nn.Linear(self.in_dim, self.comp_dim_max)

        self.snr_head = SNRHead()
        self.channel = AWGNChannel()

        self.decoder_fc = nn.Linear(self.comp_dim_max, self.out_dim)
        self.refine = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.refine_se = SEBlock(16, reduction=se_reduction)
        self.refine_out = nn.Sequential(
            nn.Conv2d(16, 2, 3, padding=1, bias=False),
            nn.Tanh()
        )

    def _to_snr_tensor(self, snr_db, b, device):
        if isinstance(snr_db, (float, int)):
            return torch.full((b, 1), float(snr_db), device=device)
        if snr_db.dim() == 1:
            return snr_db.view(-1, 1).to(device)
        return snr_db.to(device)

    def _ratio_from_snr(self, snr):
        # snr: [B,1]
        # <8 -> 1.0, [8,15) -> 0.5, >=15 -> 0.25
        ratio = torch.ones_like(snr)
        ratio = torch.where((snr >= self.snr_low) & (snr < self.snr_high), torch.full_like(ratio, 0.5), ratio)
        ratio = torch.where(snr >= self.snr_high, torch.full_like(ratio, 0.25), ratio)
        return ratio

    def _make_mask(self, ratio):
        b = ratio.size(0)
        d = self.comp_dim_max
        k = torch.clamp((ratio * d).long(), min=1, max=d)  # [B,1]
        idx = torch.arange(d, device=ratio.device).view(1, d).repeat(b, 1)
        mask = (idx < k).float()
        return mask, k.squeeze(1)

    def forward(self, x, snr_db, gate_mode="teacher"):
        """
        gate_mode:
          - teacher: 用真实snr_db分档（推荐训练）
          - estimate: 用snr_head分档（部署测试）
        """
        b = x.size(0)
        device = x.device
        snr_true = self._to_snr_tensor(snr_db, b, device)
        snr_est = self.snr_head(x)

        # encode
        feat = self.encoder_conv(x)
        feat = self.encoder_se(feat)
        feat = feat.view(b, -1)
        z = self.encoder_fc(feat)

        # gate
        if gate_mode == "teacher":
            ratio = self._ratio_from_snr(snr_true)
        elif gate_mode == "estimate":
            ratio = self._ratio_from_snr(snr_est.detach())
        else:
            raise ValueError(f"Unknown gate_mode: {gate_mode}")

        mask, k = self._make_mask(ratio)
        z_masked = z * mask

        # channel
        z_noisy = self.channel(z_masked, snr_true)

        # decode
        x_hat = self.decoder_fc(z_noisy)
        x_hat = x_hat.view(b, 2, self.h, self.w)
        x_hat = self.refine(x_hat)
        x_hat = self.refine_se(x_hat)
        out = self.refine_out(x_hat)

        return out, snr_est, ratio, k
