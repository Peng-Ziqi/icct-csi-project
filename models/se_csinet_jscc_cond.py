import torch
import torch.nn as nn
from models.se_module import SEBlock


class AWGNChannel(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def power_normalize(x, eps=1e-8):
        # x: [B, D]
        power = torch.mean(x ** 2, dim=1, keepdim=True)
        x_norm = x / torch.sqrt(power + eps)
        return x_norm

    @staticmethod
    def add_awgn(x, snr_db):
        # snr_db: float 或 shape=[B,1]
        if isinstance(snr_db, (float, int)):
            snr_linear = 10 ** (float(snr_db) / 10.0)
            noise_var = 1.0 / snr_linear
            noise_std = noise_var ** 0.5
            noise = torch.randn_like(x) * noise_std
            return x + noise
        else:
            # tensor [B,1]
            snr_linear = torch.pow(10.0, snr_db / 10.0)
            noise_var = 1.0 / snr_linear
            noise_std = torch.sqrt(noise_var)
            noise = torch.randn_like(x) * noise_std
            return x + noise

    def forward(self, x_code, snr_db):
        x_norm = self.power_normalize(x_code)
        y = self.add_awgn(x_norm, snr_db)
        return y


class SECsiNetJSCCCond(nn.Module):
    """
    SNR-conditioned JSCC:
    Input -> Encoder -> codeword -> power norm + AWGN -> [codeword || snr_embed] -> Decoder -> Output
    """
    def __init__(self, h, w, cr, se_reduction=8, snr_min=0.0, snr_max=20.0):
        super().__init__()
        self.h = h
        self.w = w
        self.in_dim = 2 * h * w
        self.comp_dim = self.in_dim // cr
        self.out_dim = self.in_dim

        self.snr_min = snr_min
        self.snr_max = snr_max

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        self.encoder_se = SEBlock(2, reduction=se_reduction)
        self.encoder_fc = nn.Linear(self.in_dim, self.comp_dim)

        # Channel
        self.channel = AWGNChannel()

        # SNR embedding: 1 -> 16
        self.snr_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True)
        )

        # Decoder: input变成 comp_dim + 16
        self.decoder_fc = nn.Linear(self.comp_dim + 16, self.out_dim)

        self.refine = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.refine_se = SEBlock(16, reduction=se_reduction)
        self.refine_out = nn.Sequential(
            nn.Conv2d(16, 2, kernel_size=3, padding=1, bias=False),
            nn.Tanh()
        )

    def _snr_to_tensor(self, snr_db, batch_size, device):
        if isinstance(snr_db, (float, int)):
            snr = torch.full((batch_size, 1), float(snr_db), device=device)
        else:
            # tensor shape [B] or [B,1]
            if snr_db.dim() == 1:
                snr = snr_db.view(-1, 1).to(device)
            else:
                snr = snr_db.to(device)
        return snr

    def _normalize_snr(self, snr):
        # 归一化到[0,1]
        return (snr - self.snr_min) / (self.snr_max - self.snr_min + 1e-8)

    def encode(self, x):
        x = self.encoder_conv(x)
        x = self.encoder_se(x)
        x = x.view(x.size(0), -1)
        z = self.encoder_fc(x)
        return z

    def decode(self, z_noisy, snr_db):
        b = z_noisy.size(0)
        device = z_noisy.device
        snr = self._snr_to_tensor(snr_db, b, device)
        snr_norm = self._normalize_snr(snr)
        snr_feat = self.snr_embed(snr_norm)

        z_cat = torch.cat([z_noisy, snr_feat], dim=1)
        x_hat = self.decoder_fc(z_cat)
        x_hat = x_hat.view(x_hat.size(0), 2, self.h, self.w)

        x_hat = self.refine(x_hat)
        x_hat = self.refine_se(x_hat)
        out = self.refine_out(x_hat)
        return out

    def forward(self, x, snr_db):
        z = self.encode(x)
        z_noisy = self.channel(z, snr_db)
        out = self.decode(z_noisy, snr_db)
        return out
