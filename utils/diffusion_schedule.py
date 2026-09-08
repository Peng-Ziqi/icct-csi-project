import torch


class DiffusionSchedule:
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

    def q_sample(self, x0, t, noise):
        """
        前向扩散:
          x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise
        """
        ab = self.alpha_bars[t].view(-1, 1, 1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise

    def predict_x0_from_eps(self, x_t, t, eps_hat):
        """
        由噪声预测反推 x0
        """
        ab = self.alpha_bars[t].view(-1, 1, 1, 1)
        return (x_t - torch.sqrt(1.0 - ab) * eps_hat) / torch.sqrt(ab)

    @torch.no_grad()
    def p_sample_step(self, denoiser, x_t, t, coarse, snr_db=None, add_noise=True):
        """
        单步反推:
        - add_noise=True  : 标准随机采样式 DDPM
        - add_noise=False : 确定性反推（更适合 refinement 任务）
        """
        eps_hat = denoiser(x_t=x_t, t=t, coarse=coarse, snr_db=snr_db)

        a = self.alphas[t].view(-1, 1, 1, 1)
        ab = self.alpha_bars[t].view(-1, 1, 1, 1)
        b = self.betas[t].view(-1, 1, 1, 1)

        # DDPM mean
        mean = (1.0 / torch.sqrt(a)) * (x_t - (b / torch.sqrt(1.0 - ab)) * eps_hat)

        if not add_noise:
            return mean

        noise = torch.randn_like(x_t)
        nz = (t > 0).float().view(-1, 1, 1, 1)
        return mean + nz * torch.sqrt(b) * noise
