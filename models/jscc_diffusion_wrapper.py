import torch
import torch.nn as nn
from utils.diffusion_schedule import DiffusionSchedule


class JSCCDiffusionWrapper(nn.Module):
    """
    兼容两类 base model:
    1) 返回 tensor: y = model(x, snr_db)
    2) 返回 tuple/list: y, ... = model(x, snr_db=..., gate_mode=...)
    """

    def __init__(self, base_model: nn.Module, refiner: nn.Module,
                 T=8, beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        self.base_model = base_model
        self.refiner = refiner
        self.schedule = DiffusionSchedule(T=T, beta_start=beta_start, beta_end=beta_end)

    def _get_coarse(self, x, snr_db, gate_mode="estimate"):
        """
        获取 coarse 重建 CSI，不改动 base model 内部结构
        """
        try:
            # 兼容 SEJSCCSNRPrior 风格
            out = self.base_model(x, snr_db=snr_db, gate_mode=gate_mode)
        except TypeError:
            # 兼容 SECsiNetJSCC / SECsiNetJSCCCond 风格
            out = self.base_model(x, snr_db)

        if torch.is_tensor(out):
            coarse = out
            aux = None
        elif isinstance(out, (tuple, list)):
            coarse = out[0]
            aux = out[1:]
        elif isinstance(out, dict):
            if "y" in out:
                coarse = out["y"]
            elif "x_hat" in out:
                coarse = out["x_hat"]
            else:
                raise KeyError("dict 输出未找到 y/x_hat 键")
            aux = out
        else:
            raise TypeError(f"不支持的 base_model 输出类型: {type(out)}")

        return coarse, aux

    @torch.no_grad()
    def refine(self, coarse, snr_db=None, steps=None, residual_scale=1.0, deterministic=True):
        """
        Residual Diffusion:
          - 采样 residual r_hat
          - refined = coarse + r_hat

        Args:
            coarse: [B,2,32,256]
            snr_db: [B,1] 或 None
            steps:  反推步数
            residual_scale: 初始化 residual 噪声尺度
            deterministic: True=确定性反推(不加噪), False=随机采样反推
        Returns:
            refined: [B,2,32,256]
            r_hat:   [B,2,32,256]
        """
        dev = coarse.device
        self.schedule.to(dev)
        T = self.schedule.T if steps is None else int(steps)

        # residual 初始化
        r_t = torch.randn_like(coarse) * float(residual_scale)

        for ti in reversed(range(T)):
            t = torch.full((coarse.size(0),), ti, device=dev, dtype=torch.long)
            r_t = self.schedule.p_sample_step(
                denoiser=self.refiner,
                x_t=r_t,
                t=t,
                coarse=coarse,
                snr_db=snr_db,
                add_noise=(not deterministic)
            )

        r_hat = r_t
        refined = coarse + r_hat
        return refined, r_hat

    def forward(self, x, snr_db, gate_mode="estimate", refine=True, steps=None,
                residual_scale=1.0, deterministic=True):
        coarse, aux = self._get_coarse(x, snr_db, gate_mode=gate_mode)
        if not refine:
            return {
                "coarse": coarse,
                "refined": coarse,
                "residual": torch.zeros_like(coarse),
                "aux": aux
            }

        refined, r_hat = self.refine(
            coarse=coarse,
            snr_db=snr_db,
            steps=steps,
            residual_scale=residual_scale,
            deterministic=deterministic
        )
        return {
            "coarse": coarse,
            "refined": refined,
            "residual": r_hat,
            "aux": aux
        }
