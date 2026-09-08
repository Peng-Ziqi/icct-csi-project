import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] long/int
        if t.dim() == 2:
            t = t.squeeze(-1)
        half = self.dim // 2
        device = t.device
        scale = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device) * -scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class SNRConditionEmbedding(nn.Module):
    """
    snr_db: [B] or [B,1] -> [B, cond_dim]
    """
    def __init__(self, cond_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, cond_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, snr_db: torch.Tensor) -> torch.Tensor:
        if snr_db.dim() == 1:
            snr_db = snr_db.view(-1, 1)
        # 对齐常用SNR范围 0~20dB
        snr_norm = torch.clamp(snr_db, 0.0, 20.0) / 20.0
        return self.net(snr_norm.float())


class DWFiLMBlock(nn.Module):
    def __init__(self, ch: int, cond_dim: int):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.SiLU(inplace=True)
        self.to_gamma_beta = nn.Linear(cond_dim, ch * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.dw(x)
        h = self.pw(h)
        h = self.bn(h)
        gb = self.to_gamma_beta(cond)  # [B,2C]
        gamma, beta = torch.chunk(gb, 2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        h = h * (1.0 + gamma) + beta
        h = self.act(h)
        return x + h


class DiffusionRefiner(nn.Module):
    """
    输入:
      x_t:    [B,2,32,256]
      coarse: [B,2,32,256]
      t:      [B]
      snr_db: [B] or [B,1] (当use_snr_cond=True时)
    输出:
      eps_hat: [B,2,32,256]
    """
    def __init__(self, in_ch=2, base_ch=32, cond_dim=64, use_snr_cond=True):
        super().__init__()
        self.use_snr_cond = use_snr_cond
        self.t_embed = SinusoidalTimeEmbedding(cond_dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
        )
        if use_snr_cond:
            self.snr_embed = SNRConditionEmbedding(cond_dim=cond_dim)
        else:
            self.snr_embed = None

        # 拼接 x_t 和 coarse
        self.in_proj = nn.Conv2d(in_ch * 2, base_ch, 3, padding=1, bias=False)
        self.b1 = DWFiLMBlock(base_ch, cond_dim)
        self.b2 = DWFiLMBlock(base_ch, cond_dim)
        self.b3 = DWFiLMBlock(base_ch, cond_dim)
        self.out_proj = nn.Conv2d(base_ch, in_ch, 3, padding=1, bias=False)

    def forward(self, x_t, t, coarse, snr_db=None):
        cond = self.t_mlp(self.t_embed(t))
        if self.use_snr_cond:
            if snr_db is None:
                raise ValueError("use_snr_cond=True 但未传 snr_db")
            cond = cond + self.snr_embed(snr_db)

        x = torch.cat([x_t, coarse], dim=1)  # [B,4,H,W]
        x = self.in_proj(x)
        x = self.b1(x, cond)
        x = self.b2(x, cond)
        x = self.b3(x, cond)
        eps_hat = self.out_proj(x)
        return eps_hat
