import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.metrics import nmse
from utils.logger import setup_log_redirect
from utils.quantize import quantize_model_copy, model_size_bits
from models.se_module import SEBlock


# ===== 与 se_jscc_cond_cr4.pt 对齐的结构 =====
class SECsiNetJSCCCondCompat(nn.Module):
    def __init__(self, h=32, w=256, cr=4, se_reduction=8):
        super().__init__()
        self.h, self.w = h, w
        self.in_dim = 2 * h * w
        self.comp_dim = self.in_dim // cr
        self.out_dim = self.in_dim

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(2, 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.encoder_se = SEBlock(2, reduction=se_reduction)
        self.encoder_fc = nn.Linear(self.in_dim, self.comp_dim)

        # conditioned关键：snr_embed + decoder_fc输入扩展
        self.snr_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True),
        )
        self.decoder_fc = nn.Linear(self.comp_dim + 16, self.out_dim)

        self.refine = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.refine_se = SEBlock(16, reduction=se_reduction)
        self.refine_out = nn.Sequential(
            nn.Conv2d(16, 2, 3, padding=1, bias=False),
            nn.Tanh()
        )

    @staticmethod
    def power_norm(x, eps=1e-8):
        p = torch.mean(x ** 2, dim=1, keepdim=True)
        return x / torch.sqrt(p + eps)

    @staticmethod
    def awgn(x, snr_db):
        if isinstance(snr_db, (float, int)):
            snr_lin = 10 ** (float(snr_db) / 10.0)
            std = (1.0 / snr_lin) ** 0.5
            n = torch.randn_like(x) * std
        else:
            snr_lin = torch.pow(10.0, snr_db / 10.0)
            std = torch.sqrt(1.0 / snr_lin)
            n = torch.randn_like(x) * std
        return x + n

    def _snr_to_tensor(self, snr_db, b, device):
        if isinstance(snr_db, (float, int)):
            return torch.full((b, 1), float(snr_db), device=device)
        if snr_db.dim() == 1:
            return snr_db.view(-1, 1).to(device)
        return snr_db.to(device)

    def forward(self, x, snr_db):
        b = x.size(0)
        dev = x.device
        snr_t = self._snr_to_tensor(snr_db, b, dev)

        f = self.encoder_conv(x)
        f = self.encoder_se(f)
        z = self.encoder_fc(f.view(b, -1))

        z = self.power_norm(z)
        z = self.awgn(z, snr_t)

        snr_norm = torch.clamp(snr_t / 20.0, 0.0, 1.0)
        se = self.snr_embed(snr_norm)
        z_in = torch.cat([z, se], dim=1)

        y = self.decoder_fc(z_in).view(b, 2, self.h, self.w)
        y = self.refine(y)
        y = self.refine_se(y)
        y = self.refine_out(y)
        return y


def load_pt_x(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        x = obj
    elif isinstance(obj, dict):
        x = obj.get("x", obj.get("data", obj.get("H", None)))
        if x is None:
            raise KeyError(f"{path} dict无x/data/H键")
    elif isinstance(obj, (list, tuple)):
        x = obj[0]
    else:
        raise TypeError(f"{path}类型不支持")
    if x.ndim == 5 and x.shape[2] == 1:
        x = x.squeeze(2)
    return x.float()


def parse_snr_list(s):
    return [float(v) for v in s.split(",")]


@torch.no_grad()
def eval_avg_nmse(model, loader, device, snr_list):
    model.eval()
    vals = []
    for snr in snr_list:
        s, n = 0.0, 0
        for x, _ in loader:
            x = x.to(device)
            y = model(x, snr_db=snr)
            s += nmse(y, x).item() * x.size(0)
            n += x.size(0)
        vals.append(s / n)
    return sum(vals) / len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--cr", type=int, default=4)
    ap.add_argument("--step", type=int, default=1, choices=[1, 2])
    ap.add_argument("--bits-list", type=str, default="8,4")
    ap.add_argument("--snr-list", type=str, default="0,5,10,15,20")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log-file", type=str, default="")
    args = ap.parse_args()

    if args.log_file == "":
        args.log_file = f"logs/test_snr_aware_jscc_baseline_quant_cr{args.cr}_step{args.step}.log"
    _ = setup_log_redirect(args.log_file)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snr_list = parse_snr_list(args.snr_list)
    bits_list = [int(v) for v in args.bits_list.split(",")]

    xte = load_pt_x(os.path.join(args.data_root, "processed", "test.pt"))
    loader = DataLoader(TensorDataset(xte, torch.zeros(len(xte))),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SECsiNetJSCCCondCompat(h=32, w=256, cr=args.cr, se_reduction=8).to(device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    print("[Info] loaded conditioned baseline checkpoint successfully.")

    base_nmse = eval_avg_nmse(model, loader, device, snr_list)
    base_size = model_size_bits(model, default_bits=32, rules={}, keep_bias_fp32=True)
    print(f"[Base] NMSE={base_nmse:.6f}, Size={base_size/8/1024/1024:.3f}MB")

    os.makedirs("logs", exist_ok=True)
    csvf = f"logs/test_snr_aware_jscc_baseline_quant_cr{args.cr}_step{args.step}.csv"
    with open(csvf, "w", encoding="utf-8") as f:
        f.write("bits,avg_nmse,size_mb,size_reduction\n")

    for b in bits_list:
        if args.step == 1:
            rules = {"encoder_fc.weight": b}
        else:
            rules = {
                "encoder_fc.weight": b,
                "encoder_se.fc": b,
                "refine_se.fc": b
            }

        qmodel = quantize_model_copy(model, rules=rules, keep_bias_fp32=True).to(device)
        q_nmse = eval_avg_nmse(qmodel, loader, device, snr_list)
        q_size = model_size_bits(qmodel, default_bits=32, rules=rules, keep_bias_fp32=True)

        print(f"[Q{b}-S{args.step}] NMSE={q_nmse:.6f}, Size={q_size/8/1024/1024:.3f}MB, Reduce={1-q_size/base_size:.4f}")
        with open(csvf, "a", encoding="utf-8") as f:
            f.write(f"{b},{q_nmse:.8f},{q_size/8/1024/1024:.6f},{1-q_size/base_size:.6f}\n")

    print(f"[Done] csv={csvf}")


if __name__ == "__main__":
    main()
