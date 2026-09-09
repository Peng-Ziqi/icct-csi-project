import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition_3d(x, window_size):
    """
    Args:
        x: [B, D, H, W, C]
        window_size: (Wd, Wh, Ww)

    Returns:
        windows: [B*num_windows, Wd*Wh*Ww, C]
    """
    B, D, H, W, C = x.shape
    Wd, Wh, Ww = window_size

    x = x.view(
        B,
        D // Wd, Wd,
        H // Wh, Wh,
        W // Ww, Ww,
        C,
    )
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, Wd * Wh * Ww, C)
    return windows


def window_reverse_3d(windows, window_size, B, D, H, W):
    """
    Args:
        windows: [B*num_windows, Wd*Wh*Ww, C]

    Returns:
        x: [B, D, H, W, C]
    """
    Wd, Wh, Ww = window_size
    C = windows.shape[-1]

    x = windows.view(
        B,
        D // Wd,
        H // Wh,
        W // Ww,
        Wd,
        Wh,
        Ww,
        C,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, D, H, W, C)
    return x


def compute_attn_mask_3d(D, H, W, window_size, shift_size, device):
    """
    Shifted-window attention mask.
    """
    Wd, Wh, Ww = window_size
    Sd, Sh, Sw = shift_size

    if Sd == 0 and Sh == 0 and Sw == 0:
        return None

    img_mask = torch.zeros((1, D, H, W, 1), device=device)

    d_slices = (
        slice(0, -Wd),
        slice(-Wd, -Sd),
        slice(-Sd, None),
    )
    h_slices = (
        slice(0, -Wh),
        slice(-Wh, -Sh),
        slice(-Sh, None),
    )
    w_slices = (
        slice(0, -Ww),
        slice(-Ww, -Sw),
        slice(-Sw, None),
    )

    cnt = 0
    for d in d_slices:
        for h in h_slices:
            for w in w_slices:
                img_mask[:, d, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition_3d(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)

    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
    attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
    return attn_mask


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if x.ndim == 2:
            x = x.squeeze(-1)

        half = self.dim // 2
        device = x.device
        scale = math.log(10000.0) / max(half - 1, 1)

        freqs = torch.exp(torch.arange(half, device=device) * -scale)
        emb = x.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class SNRConditionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, snr_db):
        if snr_db.ndim == 1:
            snr_db = snr_db.view(-1, 1)

        snr_norm = torch.clamp(snr_db.float(), 0.0, 20.0) / 20.0
        return self.net(snr_norm)


class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size=(2, 4, 8), num_heads=4, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        num_tokens = window_size[0] * window_size[1] * window_size[2]
        self.relative_position_bias = nn.Parameter(torch.zeros(num_heads, num_tokens, num_tokens))
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B*nW, N, C]
            mask: [nW, N, N] or None
        """
        B_, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn + self.relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VideoSwinBlock3D(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim,
        input_resolution,
        num_heads=4,
        window_size=(2, 4, 8),
        shift_size=(0, 0, 0),
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
    ):
        super().__init__()

        self.dim = dim
        self.cond_dim = cond_dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        D, H, W = input_resolution
        Wd, Wh, Ww = window_size

        if D % Wd != 0 or H % Wh != 0 or W % Ww != 0:
            raise ValueError(
                f"input_resolution={input_resolution} must be divisible by window_size={window_size}"
            )

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio, drop=drop)

        self.cond1 = nn.Linear(cond_dim, dim * 2)
        self.cond2 = nn.Linear(cond_dim, dim * 2)

        nn.init.zeros_(self.cond1.weight)
        nn.init.zeros_(self.cond1.bias)
        nn.init.zeros_(self.cond2.weight)
        nn.init.zeros_(self.cond2.bias)

    def _modulate(self, x, cond, layer):
        gamma_beta = layer(cond)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)

        while gamma.ndim < x.ndim:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        return x * (1.0 + gamma) + beta

    def forward(self, x, cond):
        """
        Args:
            x: [B,D,H,W,C]
            cond: [B,cond_dim]
        """
        B, D, H, W, C = x.shape
        Sd, Sh, Sw = self.shift_size

        shortcut = x

        h = self.norm1(x)
        h = self._modulate(h, cond, self.cond1)

        if Sd > 0 or Sh > 0 or Sw > 0:
            shifted_h = torch.roll(h, shifts=(-Sd, -Sh, -Sw), dims=(1, 2, 3))
            attn_mask = compute_attn_mask_3d(
                D=D,
                H=H,
                W=W,
                window_size=self.window_size,
                shift_size=self.shift_size,
                device=x.device,
            )
        else:
            shifted_h = h
            attn_mask = None

        windows = window_partition_3d(shifted_h, self.window_size)
        attn_windows = self.attn(windows, mask=attn_mask)
        shifted_h = window_reverse_3d(attn_windows, self.window_size, B, D, H, W)

        if Sd > 0 or Sh > 0 or Sw > 0:
            h = torch.roll(shifted_h, shifts=(Sd, Sh, Sw), dims=(1, 2, 3))
        else:
            h = shifted_h

        x = shortcut + h

        shortcut = x
        h = self.norm2(x)
        h = self._modulate(h, cond, self.cond2)
        h = self.mlp(h)
        x = shortcut + h

        return x


class ResidualDecodeHead(nn.Module):
    def __init__(self, dim, out_ch=2):
        super().__init__()
        hidden = max(dim // 2, 16)

        self.net = nn.Sequential(
            nn.ConvTranspose3d(
                dim,
                hidden,
                kernel_size=(1, 2, 4),
                stride=(1, 2, 4),
                padding=0,
                bias=False,
            ),
            nn.GroupNorm(num_groups=8 if hidden % 8 == 0 else 4, num_channels=hidden),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8 if hidden % 8 == 0 else 4, num_channels=hidden),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, out_ch, kernel_size=3, padding=1, bias=True),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        """
        Args:
            x: [B,C,T,H',W']

        Returns:
            residual: [B,T,2,32,256]
        """
        x = self.net(x)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x


class OneStepVideoSwinRefiner(nn.Module):
    """
    One-step SNR-conditioned Video-Swin residual refiner.

    Inputs:
        z_t:    [B,8,2,32,256] noisy residual / initial residual
        coarse: [B,8,2,32,256] JSCC coarse sequence
        t:      [B] or [B,1] continuous denoising level in [0,1]
        snr_db: [B] or [B,1], optional

    Outputs:
        {
            "res_full": [B,8,2,32,256],
            "res_repa": [B,8,2,32,256]
        }
    """

    def __init__(
        self,
        in_ch=2,
        cond_dim=128,
        dim=48,
        depth=6,
        num_heads=4,
        window_size=(2, 4, 8),
        use_snr_cond=True,
    ):
        super().__init__()

        self.in_ch = in_ch
        self.use_snr_cond = use_snr_cond
        self.depth = depth

        self.t_embed = SinusoidalEmbedding(cond_dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
        )

        if use_snr_cond:
            self.snr_embed = SNRConditionEmbedding(cond_dim)
        else:
            self.snr_embed = None

        self.stem = nn.Sequential(
            nn.Conv3d(
                in_ch * 2,
                dim,
                kernel_size=(3, 3, 7),
                stride=(1, 2, 4),
                padding=(1, 1, 3),
                bias=False,
            ),
            nn.GroupNorm(num_groups=8 if dim % 8 == 0 else 4, num_channels=dim),
            nn.SiLU(inplace=True),
        )

        self.latent_resolution = (8, 16, 64)
        shift_size = tuple(w // 2 for w in window_size)

        blocks = []
        for i in range(depth):
            blocks.append(
                VideoSwinBlock3D(
                    dim=dim,
                    cond_dim=cond_dim,
                    input_resolution=self.latent_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0, 0) if i % 2 == 0 else shift_size,
                    mlp_ratio=4.0,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.full_head = ResidualDecodeHead(dim=dim, out_ch=in_ch)
        self.repa_head = ResidualDecodeHead(dim=dim, out_ch=in_ch)

    def _make_cond(self, t, snr_db):
        cond = self.t_mlp(self.t_embed(t))

        if self.use_snr_cond:
            if snr_db is None:
                raise ValueError("use_snr_cond=True but snr_db is None")
            cond = cond + self.snr_embed(snr_db)

        return cond

    def forward(self, z_t, coarse, t, snr_db=None):
        if z_t.ndim != 5:
            raise ValueError(f"Expected z_t [B,T,2,32,256], got {tuple(z_t.shape)}")
        if coarse.shape != z_t.shape:
            raise ValueError(f"coarse shape {tuple(coarse.shape)} must match z_t {tuple(z_t.shape)}")
        if z_t.shape[1:] != (8, 2, 32, 256):
            raise ValueError(f"Expected [B,8,2,32,256], got {tuple(z_t.shape)}")

        cond = self._make_cond(t, snr_db)

        x = torch.cat([z_t, coarse], dim=2)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        x = self.stem(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        repa_feat = None
        repa_index = max(self.depth // 2 - 1, 0)

        for i, block in enumerate(self.blocks):
            x = block(x, cond)
            if i == repa_index:
                repa_feat = x

        full_feat = x

        full_feat = full_feat.permute(0, 4, 1, 2, 3).contiguous()
        repa_feat = repa_feat.permute(0, 4, 1, 2, 3).contiguous()

        res_full = self.full_head(full_feat)
        res_repa = self.repa_head(repa_feat)

        return {
            "res_full": res_full,
            "res_repa": res_repa,
        }
