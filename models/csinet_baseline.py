import torch
import torch.nn as nn


class CsiNetRefineBlock(nn.Module):
    def __init__(self, channels=2, hidden1=8, hidden2=16):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden1),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(hidden1, hidden2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden2),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(hidden2, channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x):
        return x + self.net(x)


class CsiNetBaseline(nn.Module):
    """
    CsiNet-style frame-wise CSI autoencoder adapted to [B,2,32,256].

    Input:
        x: [B,2,32,256]

    Output:
        x_hat: [B,2,32,256]

    Notes:
        - CR=4 means latent_dim = input_dim / 4.
        - If use_awgn=True, the latent code is power-normalized and passed
          through an AWGN feedback channel at the given SNR.
        - This is a fair CsiNet-style noisy-feedback baseline, not the original
          noiseless COST2100-only CsiNet setting.
    """

    def __init__(self, cr=4, use_awgn=True, input_shape=(2, 32, 256)):
        super().__init__()

        self.cr = int(cr)
        self.use_awgn = bool(use_awgn)
        self.input_shape = input_shape

        c, h, w = input_shape
        self.input_dim = c * h * w
        if self.input_dim % self.cr != 0:
            raise ValueError(f"input_dim={self.input_dim} must be divisible by cr={self.cr}")

        self.latent_dim = self.input_dim // self.cr

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(c, 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.3, inplace=True),
        )

        self.encoder_fc = nn.Linear(self.input_dim, self.latent_dim)

        self.decoder_fc = nn.Linear(self.latent_dim, self.input_dim)

        self.decoder_refine = nn.Sequential(
            CsiNetRefineBlock(channels=2),
            CsiNetRefineBlock(channels=2),
            nn.Conv2d(2, 2, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
        )

    def power_normalize(self, z, eps=1e-8):
        power = torch.mean(z ** 2, dim=1, keepdim=True)
        return z / torch.sqrt(power + eps)

    def awgn_channel(self, z, snr_db):
        if isinstance(snr_db, (float, int)):
            snr_db = torch.full((z.size(0), 1), float(snr_db), device=z.device)

        if not torch.is_tensor(snr_db):
            raise TypeError(f"Unsupported snr_db type: {type(snr_db)}")

        snr_db = snr_db.to(z.device).float()
        if snr_db.ndim == 1:
            snr_db = snr_db.view(-1, 1)

        snr_linear = torch.pow(10.0, snr_db / 10.0)
        noise_std = torch.sqrt(1.0 / snr_linear)
        noise = torch.randn_like(z) * noise_std

        return z + noise

    def forward(self, x, snr_db=None):
        if x.ndim != 4 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(f"Expected x [B,2,32,256], got {tuple(x.shape)}")

        h = self.encoder_conv(x)
        h = h.reshape(h.size(0), -1)

        z = self.encoder_fc(h)

        if self.use_awgn:
            if snr_db is None:
                raise ValueError("use_awgn=True but snr_db is None")
            z = self.power_normalize(z)
            z = self.awgn_channel(z, snr_db=snr_db)

        y = self.decoder_fc(z)
        y = y.view(x.size(0), *self.input_shape)
        y = self.decoder_refine(y)

        return y
