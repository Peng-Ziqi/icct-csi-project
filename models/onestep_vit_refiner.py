import torch
import torch.nn as nn


class SinusoidalTimeSNRMLP(nn.Module):
    def __init__(self, dim=48, use_snr_cond=True):
        super().__init__()
        self.use_snr_cond = use_snr_cond

        in_dim = 1 + (1 if use_snr_cond else 0)

        self.net = nn.Sequential(
            nn.Linear(in_dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t, snr_db=None):
        if t.ndim == 1:
            t = t.view(-1, 1)

        inputs = [t.float()]

        if self.use_snr_cond:
            if snr_db is None:
                raise ValueError("use_snr_cond=True but snr_db is None")
            if snr_db.ndim == 1:
                snr_db = snr_db.view(-1, 1)
            # 简单归一化到大致 0~1 范围
            snr_norm = snr_db.float() / 20.0
            inputs.append(snr_norm)

        x = torch.cat(inputs, dim=1)
        return self.net(x)


class OneStepViTRefiner(nn.Module):
    """
    Vanilla global Transformer refiner baseline.

    Same interface as OneStepVideoSwinRefiner:
        input:
            z_t:    [B,T,2,32,256]
            coarse: [B,T,2,32,256]
            t:      [B,1]
            snr_db: [B,1] or None
        output:
            dict with res_full and res_repa
    """

    def __init__(
        self,
        in_ch=2,
        cond_dim=128,
        dim=48,
        depth=6,
        num_heads=4,
        patch_size=(1, 4, 16),
        seq_len=8,
        height=32,
        width=256,
        use_snr_cond=True,
        dropout=0.0,
    ):
        super().__init__()

        self.in_ch = in_ch
        self.dim = dim
        self.patch_size = patch_size
        self.seq_len = seq_len
        self.height = height
        self.width = width
        self.use_snr_cond = use_snr_cond

        pt, ph, pw = patch_size
        assert seq_len % pt == 0
        assert height % ph == 0
        assert width % pw == 0

        self.gt = seq_len // pt
        self.gh = height // ph
        self.gw = width // pw
        self.num_tokens = self.gt * self.gh * self.gw

        # z_t 和 coarse 拼接，所以输入通道为 4
        self.patch_embed = nn.Conv3d(
            in_channels=in_ch * 2,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.cond_mlp = SinusoidalTimeSNRMLP(dim=dim, use_snr_cond=use_snr_cond)

        self.norm = nn.LayerNorm(dim)

        self.full_head = nn.ConvTranspose3d(
            in_channels=dim,
            out_channels=in_ch,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.repa_head = nn.ConvTranspose3d(
            in_channels=dim,
            out_channels=in_ch,
            kernel_size=patch_size,
            stride=patch_size,
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, z_t, coarse, t, snr_db=None):
        """
        z_t/coarse: [B,T,2,32,256]
        """
        if z_t.ndim != 5:
            raise ValueError(f"Expected z_t [B,T,C,H,W], got {tuple(z_t.shape)}")

        x = torch.cat([z_t, coarse], dim=2)  # [B,T,4,H,W]

        # Conv3d expects [B,C,T,H,W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        feat = self.patch_embed(x)  # [B,dim,T',H',W']
        b, d, gt, gh, gw = feat.shape

        tokens = feat.flatten(2).transpose(1, 2).contiguous()  # [B,N,D]

        cond = self.cond_mlp(t, snr_db=snr_db).unsqueeze(1)  # [B,1,D]

        tokens = tokens + self.pos_embed[:, :tokens.size(1), :] + cond

        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        feat = tokens.transpose(1, 2).contiguous().view(b, d, gt, gh, gw)

        res_full = self.full_head(feat)  # [B,2,T,H,W]
        res_repa = self.repa_head(feat)

        # back to [B,T,2,H,W]
        res_full = res_full.permute(0, 2, 1, 3, 4).contiguous()
        res_repa = res_repa.permute(0, 2, 1, 3, 4).contiguous()

        return {
            "res_full": res_full,
            "res_repa": res_repa,
        }
