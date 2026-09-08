import torch

def nmse_per_sample(pred, target, eps=1e-12):
    """
    返回每个样本的 NMSE（线性域，不是dB）
    pred/target: [B, C, H, W]
    return: [B]
    """
    num = torch.sum((pred - target) ** 2, dim=(1, 2, 3))
    den = torch.sum(target ** 2, dim=(1, 2, 3)) + eps
    return num / den

def nmse(pred, target, eps=1e-12):
    """
    保持与你现有训练代码兼容：
    返回 batch 平均 NMSE（标量 tensor）
    """
    return nmse_per_sample(pred, target, eps=eps).mean()

def nmse_db_from_linear(nmse_linear, eps=1e-12):
    """
    线性NMSE -> dB
    """
    if isinstance(nmse_linear, torch.Tensor):
        return 10 * torch.log10(nmse_linear + eps)
    return 10 * torch.log10(torch.tensor(nmse_linear + eps))

