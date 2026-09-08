import torch
import torch.nn as nn
from models.se_module import SEBlock


class AWGNChannel(nn.Module):
    @staticmethod
    def power_norm(x, eps=1e-8):
        p = torch.mean(x ** 2, dim=1, keepdim=True)
        return x / torch.sqrt(p + eps)

    @staticmethod
    def add_awgn(x, snr_db):
        # snr_db: float or [B,1]
        if isinstance(snr_db, (float, int)):
            snr_lin = 10 ** (float(snr_db) / 10.0)
            std = (1.0 / snr_lin) ** 0.5
            n = torch.randn_like(x) * std
        else:
            snr_lin = torch.pow(10.0, snr_db / 10.0)
            std = torch.sqrt(1.0 / snr_lin)
            n = torch.randn_like(x) * std
        return x + n

    def forward(self, z, snr_db):
        z = self.power_norm(z)
        return self.add_awgn(z, snr_db)


class SNRHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(8, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)  # [B,1]


class SEJSCCSNRPrior(nn.Module):
    """
    统一主线：
    Encoder -> codeword(max) -> SNR先验门控(mask) -> PowerNorm+AWGN -> Decoder(SNR条件) -> Output
    """
    def __init__(self, h=32, w=256, cr=4, se_reduction=8, snr_low=8.0, snr_high=15.0):
        super().__init__()
        self.h, self.w = h, w
        self.in_dim = 2 * h * w
        self.comp_dim = self.in_dim // cr
        self.out_dim = self.in_dim
        self.snr_low = snr_low
        self.snr_high = snr_high

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(2, 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(2), nn.ReLU(inplace=True)
        )
        self.encoder_se = SEBlock(2, reduction=se_reduction)
        self.encoder_fc = nn.Linear(self.in_dim, self.comp_dim)

        self.snr_head = SNRHead()
        self.channel = AWGNChannel()

        # decoder侧加入SNR条件embedding
        self.snr_embed = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 16), nn.ReLU(inplace=True)
        )
        self.decoder_fc = nn.Linear(self.comp_dim + 16, self.out_dim)

        self.refine = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )
        self.refine_se = SEBlock(16, reduction=se_reduction)
        self.refine_out = nn.Sequential(
            nn.Conv2d(16, 2, 3, padding=1, bias=False),
            nn.Tanh()
        )

    def _snr_to_tensor(self, snr_db, b, device):
        if isinstance(snr_db, (float, int)):
            return torch.full((b, 1), float(snr_db), device=device)
        return snr_db.view(-1, 1).to(device) if snr_db.dim() == 1 else snr_db.to(device)

    def _ratio_from_snr(self, snr):
        r = torch.ones_like(snr)  # low SNR: full length
        r = torch.where((snr >= self.snr_low) & (snr < self.snr_high), torch.full_like(r, 0.5), r)
        r = torch.where(snr >= self.snr_high, torch.full_like(r, 0.25), r)
        return r

    def _mask(self, ratio):
        b = ratio.size(0)
        d = self.comp_dim
        k = torch.clamp((ratio * d).long(), min=1, max=d)  # [B,1]
        idx = torch.arange(d, device=ratio.device).view(1, d).repeat(b, 1)
        m = (idx < k).float()
        return m, k.squeeze(1)

    def forward(self, x, snr_db, gate_mode="teacher"):
        b = x.size(0)
        dev = x.device
        snr_true = self._snr_to_tensor(snr_db, b, dev)
        snr_est = self.snr_head(x)

        # encode
        f = self.encoder_conv(x)
        f = self.encoder_se(f)
        f = f.view(b, -1)
        z = self.encoder_fc(f)

        # gate
        if gate_mode == "teacher":
            ratio = self._ratio_from_snr(snr_true)
        else:
            ratio = self._ratio_from_snr(snr_est.detach())

        m, k = self._mask(ratio)
        z = z * m

        # channel
        zc = self.channel(z, snr_true)

        # snr condition for decoder
        snr_norm = torch.clamp(snr_true / 20.0, 0.0, 1.0)
        se = self.snr_embed(snr_norm)
        z_in = torch.cat([zc, se], dim=1)

        # decode
        y = self.decoder_fc(z_in).view(b, 2, self.h, self.w)
        y = self.refine(y)
        y = self.refine_se(y)
        y = self.refine_out(y)
        return y, snr_est, ratio, k
