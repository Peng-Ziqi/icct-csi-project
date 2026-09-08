# test_all_scenarios_snr.py
import os
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.csinet import CsiNet
from models.csinet_plus import CsiNetPlus
from models.se_csinet import SECsiNet


def parse_int_list(s):
    s = s.replace(",", " ")
    vals = [v.strip() for v in s.split() if v.strip()]
    return [int(v) for v in vals]


def _extract_x_from_obj(obj):
    if isinstance(obj, torch.Tensor):
        x = obj
    elif isinstance(obj, dict):
        if "x" in obj:
            x = obj["x"]
        elif "data" in obj:
            x = obj["data"]
        elif "H" in obj:
            x = obj["H"]
        else:
            raise KeyError(f"pt是dict，但找不到x/data/H键，现有键: {list(obj.keys())}")
    elif isinstance(obj, (list, tuple)):
        x = obj[0]
    else:
        raise TypeError(f"不支持的pt类型: {type(obj)}")
    return x


def _to_2ch_real_imag(x: torch.Tensor) -> torch.Tensor:
    """
    把输入统一为 [N,2,32,256]
    兼容:
      - complex: [N,1,32,256] / [N,32,256] / [N,32,256,1]
      - real:    [N,2,32,256] / [N,2,1,32,256]
    """
    # complex -> real/imag two-channel
    if torch.is_complex(x):
        # 常见: [N,1,32,256]
        if x.ndim == 4 and x.shape[1] == 1:
            xr = x.real.squeeze(1)  # [N,32,256]
            xi = x.imag.squeeze(1)
            x = torch.stack([xr, xi], dim=1)  # [N,2,32,256]
            return x.float()

        # [N,32,256]
        if x.ndim == 3:
            xr = x.real
            xi = x.imag
            x = torch.stack([xr, xi], dim=1)
            return x.float()

        # [N,32,256,1] -> squeeze last
        if x.ndim == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)
            xr = x.real
            xi = x.imag
            x = torch.stack([xr, xi], dim=1)
            return x.float()

        raise ValueError(f"复杂张量形状暂不支持: {tuple(x.shape)}")

    # real tensor
    x = x.float()

    # [N,2,1,32,256] -> [N,2,32,256]
    if x.ndim == 5 and x.shape[2] == 1:
        x = x.squeeze(2)

    # 如果是 [N,1,32,256]（实数单通道），补一个零虚部
    if x.ndim == 4 and x.shape[1] == 1:
        zero = torch.zeros_like(x)
        x = torch.cat([x, zero], dim=1)  # [N,2,32,256]

    # [N,32,256] -> [N,2,32,256]（虚部置0）
    if x.ndim == 3:
        x = x.unsqueeze(1)  # [N,1,32,256]
        zero = torch.zeros_like(x)
        x = torch.cat([x, zero], dim=1)

    if x.ndim != 4 or x.shape[1] != 2:
        raise ValueError(f"归一化后维度应为[N,2,H,W]，当前: {tuple(x.shape)}")

    return x


def load_test_data(data_root, scene):
    raw_scene_path = os.path.join(data_root, "raw", f"{scene}.pt")
    if os.path.exists(raw_scene_path):
        obj = torch.load(raw_scene_path, map_location="cpu", weights_only=False)
        x = _extract_x_from_obj(obj)
        x = _to_2ch_real_imag(x)
        y_dummy = torch.zeros(len(x))
        return TensorDataset(x, y_dummy)

    test_path = os.path.join(data_root, "processed", "test.pt")
    if os.path.exists(test_path):
        obj = torch.load(test_path, map_location="cpu", weights_only=False)
        x = _extract_x_from_obj(obj)
        x = _to_2ch_real_imag(x)
        y_dummy = torch.zeros(len(x))
        return TensorDataset(x, y_dummy)

    raise FileNotFoundError(f"未找到场景文件: {raw_scene_path}，且无fallback: {test_path}")


def build_model(model_name, cr):
    h, w = 32, 256
    name = model_name.lower()
    if name == "csinet":
        return CsiNet(h=h, w=w, cr=cr)
    elif name == "csinetplus":
        return CsiNetPlus(h=h, w=w, cr=cr)
    elif name == "se-csinet":
        return SECsiNet(h=h, w=w, cr=cr)
    else:
        raise ValueError(f"未知模型: {model_name}")


def load_ckpt(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    return model


def add_awgn_by_snr(x, snr_db):
    sig_pow = torch.mean(x * x, dim=(1, 2, 3), keepdim=True)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_pow = sig_pow / (snr_linear + 1e-12)
    noise_std = torch.sqrt(noise_pow + 1e-12)
    noise = torch.randn_like(x) * noise_std
    return x + noise


def achievable_rate(pred, target, snr_db):
    pr = pred[:, 0].reshape(pred.size(0), -1)
    pi = pred[:, 1].reshape(pred.size(0), -1)
    tr = target[:, 0].reshape(target.size(0), -1)
    ti = target[:, 1].reshape(target.size(0), -1)

    inner_re = torch.sum(tr * pr + ti * pi, dim=1)
    inner_im = torch.sum(tr * pi - ti * pr, dim=1)
    inner_abs2 = inner_re * inner_re + inner_im * inner_im

    t_pow = torch.sum(tr * tr + ti * ti, dim=1) + 1e-12
    p_pow = torch.sum(pr * pr + pi * pi, dim=1) + 1e-12
    eta = inner_abs2 / (t_pow * p_pow + 1e-12)
    eta = torch.clamp(eta, 0.0, 1.0)

    snr_linear = 10 ** (snr_db / 10.0)
    rate = torch.log2(1.0 + snr_linear * eta)
    return rate.mean().item()


@torch.no_grad()
def evaluate_model(model, loader, device, snr_db, noise_mode="input"):
    model.eval()
    total_num, total_den = 0.0, 0.0
    total_rate, count = 0.0, 0

    for x, _ in loader:
        x = x.to(device)
        x_in = add_awgn_by_snr(x, snr_db) if noise_mode == "input" else x
        pred = model(x_in)

        diff = pred - x
        total_num += torch.sum(diff * diff).item()
        total_den += torch.sum(x * x).item()

        b_rate = achievable_rate(pred, x, snr_db)
        total_rate += b_rate * x.size(0)
        count += x.size(0)

    nmse_linear = total_num / (total_den + 1e-12)
    nmse_db = 10 * np.log10(nmse_linear + 1e-12)
    avg_rate = total_rate / max(count, 1)
    return nmse_linear, nmse_db, avg_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--ckpt-root", type=str, default="checkpoints")
    parser.add_argument("--out-dir", type=str, default="results_full_test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--scenes", type=str, default="UMa_LOS,UMa_NLOS,UMi_LOS,UMi_NLOS")
    parser.add_argument("--cr-list", type=str, default="4,8,16")
    parser.add_argument("--snr-list", type=str, default="-10,-5,0,5,10,15,20")
    parser.add_argument("--models", type=str, default="csinet,csinetplus,se-csinet")
    parser.add_argument("--noise-mode", type=str, default="input", choices=["none", "input"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    cr_list = parse_int_list(args.cr_list)
    snr_list = parse_int_list(args.snr_list)
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]

    out_csv = os.path.join(args.out_dir, "raw_results_all.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "scene", "cr", "snr_db", "nmse_linear", "nmse_db", "rate", "noise_mode"])

        for scene in scenes:
            dataset = load_test_data(args.data_root, scene)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

            for cr in cr_list:
                for model_name in model_list:
                    model = build_model(model_name, cr).to(device)

                    name = model_name.lower()
                    if name == "csinet":
                        ckpt_path = os.path.join(args.ckpt_root, f"csinet_cr{cr}.pt")
                    elif name == "csinetplus":
                        ckpt_path = os.path.join(args.ckpt_root, f"csinet_plus_cr{cr}.pt")
                    elif name == "se-csinet":
                        ckpt_path = os.path.join(args.ckpt_root, f"se_csinet_cr{cr}.pt")
                    else:
                        continue

                    if not os.path.exists(ckpt_path):
                        print(f"[跳过] 未找到权重: {ckpt_path}")
                        continue

                    model = load_ckpt(model, ckpt_path, device)

                    for snr_db in snr_list:
                        nmse_linear, nmse_db, rate = evaluate_model(
                            model=model, loader=loader, device=device, snr_db=snr_db, noise_mode=args.noise_mode
                        )
                        writer.writerow([model_name, scene, cr, snr_db, nmse_linear, nmse_db, rate, args.noise_mode])
                        print(f"[OK] model={model_name:10s} scene={scene:8s} cr={cr:2d} snr={snr_db:>3d} | "
                              f"NMSE={nmse_db:8.3f} dB | Rate={rate:7.4f}")

    print(f"\n结果已保存: {out_csv}")


if __name__ == "__main__":
    main()
