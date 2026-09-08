import torch


class DiffusionSchedule:
    """
    Generic DDPM schedule.

    Supports:
        x: [B,C,H,W]
        x: [B,T,C,H,W]
    """

    def __init__(self, T=8, beta_start=1e-4, beta_end=2e-2, device="cpu"):
        self.T = int(T)
        betas = torch.linspace(beta_start, beta_end, self.T, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def _expand(self, values, x):
        """
        values: [B]
        x: [B,...]
        return: [B,1,1,...]
        """
        return values.view(x.shape[0], *([1] * (x.ndim - 1)))

    def q_sample(self, x0, t, noise):
        """
        Forward diffusion:
            x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * noise
        """
        ab = self._expand(self.alpha_bars[t], x0)
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise

    def predict_x0_from_eps(self, x_t, t, eps_hat):
        """
        Predict x0 from noisy x_t and predicted noise eps_hat.
        """
        ab = self._expand(self.alpha_bars[t], x_t)
        return (x_t - torch.sqrt(1.0 - ab) * eps_hat) / torch.sqrt(ab)

    @torch.no_grad()
    def p_sample_step(self, denoiser, x_t, t, coarse, snr_db=None, add_noise=False):
        """
        Reverse diffusion step.

        add_noise=False is recommended for CSI refinement because deterministic
        denoising is more stable than stochastic sampling.
        """
        eps_hat = denoiser(x_t=x_t, t=t, coarse=coarse, snr_db=snr_db)

        a = self._expand(self.alphas[t], x_t)
        ab = self._expand(self.alpha_bars[t], x_t)
        b = self._expand(self.betas[t], x_t)

        mean = (1.0 / torch.sqrt(a)) * (x_t - (b / torch.sqrt(1.0 - ab)) * eps_hat)

        if not add_noise:
            return mean

        noise = torch.randn_like(x_t)
        nz = self._expand((t > 0).float(), x_t)
        return mean + nz * torch.sqrt(b) * noise
