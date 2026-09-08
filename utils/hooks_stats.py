# utils/hooks_stats.py
import os
import csv
import torch

class ActivationStatsHook:
    """
    收集模块输出分布统计:
    - mean, std, var, min, max
    - p01/p05/p50/p95/p99
    - zero_ratio (接近0比例)
    """
    def __init__(self, save_dir="stats", eps=1e-8, near_zero_th=1e-4):
        self.save_dir = save_dir
        self.eps = eps
        self.near_zero_th = near_zero_th
        self.handles = []
        self.records = []
        os.makedirs(self.save_dir, exist_ok=True)

    def _tensor_stats(self, x: torch.Tensor):
        x = x.detach().float().reshape(-1)
        if x.numel() == 0:
            return None
        mean = torch.mean(x).item()
        var = torch.var(x, unbiased=False).item()
        std = (var + self.eps) ** 0.5
        minv = torch.min(x).item()
        maxv = torch.max(x).item()

        q = torch.quantile(x, torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=x.device))
        p01, p05, p50, p95, p99 = [v.item() for v in q]
        zero_ratio = (torch.abs(x) < self.near_zero_th).float().mean().item()

        return {
            "mean": mean, "var": var, "std": std, "min": minv, "max": maxv,
            "p01": p01, "p05": p05, "p50": p50, "p95": p95, "p99": p99,
            "zero_ratio": zero_ratio
        }

    def _make_hook(self, name):
        def hook(module, inp, out):
            if not torch.is_tensor(out):
                return
            st = self._tensor_stats(out)
            if st is None:
                return
            st["module"] = name
            self.records.append(st)
        return hook

    def register(self, model, module_name_filter=None):
        """
        module_name_filter: 可选函数 f(name, module)->bool
        """
        for name, m in model.named_modules():
            if len(list(m.children())) > 0:
                continue  # 只挂在叶子层，避免重复
            if module_name_filter is not None and (not module_name_filter(name, m)):
                continue
            h = m.register_forward_hook(self._make_hook(name))
            self.handles.append(h)

    def clear_records(self):
        self.records = []

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def save_csv(self, file_name="activation_stats.csv"):
        path = os.path.join(self.save_dir, file_name)
        if len(self.records) == 0:
            return path
        keys = ["module", "mean", "var", "std", "min", "max", "p01", "p05", "p50", "p95", "p99", "zero_ratio"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in self.records:
                writer.writerow({k: r.get(k, None) for k in keys})
        return path
