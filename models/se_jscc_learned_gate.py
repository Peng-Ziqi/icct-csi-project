import torch
import torch.nn as nn
import torch.nn.functional as F
from models.se_module import SEBlock


class AWGNChannel(nn.Module):
    @staticmethod
    def power_norm(x, eps=1e-8):
        p = torch.mean(x ** 2, dim=1, keepdim=True)
        return x / torch.sqrt(p + eps)

    @staticmethod
    def add_awgn(x, snr_db):
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


class GateNet(nn.Module):
    """输出3档logits: [full, half, quarter]"""
    def __init__(self, in_ch=2):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.mlp = nn.Sequential(
            nn.Linear(8 + 1, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 3)
        )

    def forward(self, x, snr_db_tensor):
        f = self.feat(x)
        snr_norm = torch.clamp(snr_db_tensor / 20.0, 0.0, 1.0)
        inp = torch.cat([f, snr_norm], dim=1)
        return self.mlp(inp)  # [B,3]


class SEJSCCLearnedGate(nn.Module):
    def __init__(self, h=32, w=256, cr=4, se_reduction=8):
        super().__init__()
        self.h, self.w = h, w
        self.in_dim = 2 * h * w
        self.comp_dim = self.in_dim // cr
        self.out_dim = self.in_dim

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(2, 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(2), nn.ReLU(inplace=True)
        )
        self.encoder_se = SEBlock(2, reduction=se_reduction)
        self.encoder_fc = nn.Linear(self.in_dim, self.comp_dim)

        self.gate_net = GateNet(in_ch=2)
        self.channel = AWGNChannel()

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

        # 三档比例：full/half/quarter
        self.register_buffer("ratio_table", torch.tensor([1.0, 0.5, 0.25]).view(1, 3))

    def _snr_to_tensor(self, snr_db, b, device):
        if isinstance(snr_db, (float, int)):
            return torch.full((b, 1), float(snr_db), device=device)
        return snr_db.view(-1, 1).to(device) if snr_db.dim() == 1 else snr_db.to(device)

    def _build_mask(self, ratio):
        b = ratio.size(0)
        d = self.comp_dim
        k = torch.clamp((ratio * d).long(), min=1, max=d)
        idx = torch.arange(d, device=ratio.device).view(1, d).repeat(b, 1)
        m = (idx < k).float()
        return m, k

    def forward(self, x, snr_db, tau=1.0, hard=True):
        b = x.size(0)
        dev = x.device
        snr_t = self._snr_to_tensor(snr_db, b, dev)

        # encode
        f = self.encoder_conv(x)
        f = self.encoder_se(f)
        z = self.encoder_fc(f.view(b, -1))

        # learned gate
        logits = self.gate_net(x, snr_t)  # [B,3]
        gate_prob = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=1)  # one-hot近似

        # ratio = prob * [1,0.5,0.25]
        ratio = torch.sum(gate_prob * self.ratio_table.to(dev), dim=1, keepdim=True)  # [B,1]
        mask, k = self._build_mask(ratio)
        z = z * mask

        # channel
        zc = self.channel(z, snr_t)

        # decoder with snr condition
        snr_norm = torch.clamp(snr_t / 20.0, 0.0, 1.0)
        se = self.snr_embed(snr_norm)
        z_in = torch.cat([zc, se], dim=1)

        y = self.decoder_fc(z_in).view(b, 2, self.h, self.w)
        y = self.refine(y)
        y = self.refine_se(y)
        y = self.refine_out(y)

        # 返回开销相关中间量用于loss
        return y, logits, gate_prob, ratio, k
