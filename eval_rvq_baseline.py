# eval_rvq_baseline.py
# 传统RVQ码本反馈基线评估：输出 NMSE / Achievable Rate / Overhead
#
# 用法示例：
# python -u eval_rvq_baseline.py --data-root data --snr-list 0,5,10,15,20 --bits-list 8,10,12 --num-codebooks 3
#
# 输出：
#   logs/eval_rvq_baseline.csv
#   logs/eval_rvq_baseline.log

import os
import argparse
import numpy as np
import torch
from utils.logger import setup_log_redirect


def load_pt_x(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        x = obj
    elif isinstance(obj, dict):
        x = obj.get("x", obj.get("data", obj.get("H", None)))
        if x is None:
            raise KeyError(f"{path} dict中未找到 x/data/H")
    elif isinstance(obj, (list, tuple)):
        x = obj[0]
    else:
        raise TypeError(f"不支持的数据类型: {type(obj)}")

    # [N,2,H,W] 或 [N,2,D]
    if x.ndim == 5 and x.shape[2] == 1:
        x = x.squeeze(2)
    return x.float()


def to_complex_vec(x):
    """
    x: [N,2,H,W] or [N,2,D]
    return hc: [N,D] complex64
    """
    if x.ndim == 4:
        n = x.shape[0]
        r = x[:, 0].reshape(n, -1).cpu().numpy()
        i = x[:, 1].reshape(n, -1).cpu().numpy()
    elif x.ndim == 3:
        n = x.shape[0]
        r = x[:, 0, :].cpu().numpy()
        i = x[:, 1, :].cpu().numpy()
    else:
        raise ValueError(f"输入维度不支持: {x.shape}")

    return r.astype(np.float32) + 1j * i.astype(np.float32)


def nmse_complex(h_hat, h, eps=1e-12):
    num = np.sum(np.abs(h_hat - h) ** 2, axis=1)
    den = np.sum(np.abs(h) ** 2, axis=1) + eps
    return np.mean(num / den)


def unit_norm_rows(z, eps=1e-12):
    nrm = np.linalg.norm(z, axis=1, keepdims=True) + eps
    return z / nrm


def gen_rvq_codebook(dim, bits, rng):
    m = 2 ** bits
    # 复高斯随机码字
    c = (rng.standard_normal((m, dim)) + 1j * rng.standard_normal((m, dim))).astype(np.complex64)
    c = unit_norm_rows(c)
    return c  # [M,D]


def rvq_quantize(h, codebook):
    """
    h: [N,D] complex
    codebook: [M,D] complex, each row unit norm
    return:
      h_hat_dir: [N,D] 方向重构（单位范数）
      idx: [N]
    """
    h_u = unit_norm_rows(h)
    # 相似度 |h^H c|
    # h_u [N,D], c [M,D]
    sim = np.abs(h_u @ np.conjugate(codebook).T)  # [N,M]
    idx = np.argmax(sim, axis=1)
    h_hat_dir = codebook[idx]  # [N,D]
    return h_hat_dir, idx


def achievable_rate_mrt(h, h_hat_dir, snr_db):
    """
    单用户MRT近似:
      w = h_hat_dir (单位范数)
      gamma = snr * |h^H w|^2 / ||h||^0  (不再额外归一，保持相对比较口径)
      R = log2(1 + gamma)
    """
    snr_lin = 10 ** (snr_db / 10.0)
    proj = np.abs(np.sum(np.conjugate(h) * h_hat_dir, axis=1)) ** 2  # [N]
    gamma = snr_lin * proj
    r = np.log2(1.0 + gamma)
    return float(np.mean(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--snr-list", type=str, default="0,5,10,15,20")
    ap.add_argument("--bits-list", type=str, default="8,10,12")
    ap.add_argument("--num-codebooks", type=int, default=3, help="多随机码本重复平均")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--log-file", type=str, default="logs/eval_rvq_baseline.log")
    args = ap.parse_args()

    _ = setup_log_redirect(args.log_file)
    os.makedirs("logs", exist_ok=True)

    snr_list = [float(v) for v in args.snr_list.split(",")]
    bits_list = [int(v) for v in args.bits_list.split(",")]

    pt = os.path.join(args.data_root, "processed", f"{args.split}.pt")
    x = load_pt_x(pt)
    h = to_complex_vec(x)  # [N,D]
    n, d = h.shape

    print(f"[Info] split={args.split}, samples={n}, dim={d}, snrs={snr_list}, bits={bits_list}")

    rows = []
    master_rng = np.random.default_rng(args.seed)

    for b in bits_list:
        nmse_runs = []
        rate_runs = {snr: [] for snr in snr_list}

        for k in range(args.num_codebooks):
            rng = np.random.default_rng(master_rng.integers(1, 10**9))
            cb = gen_rvq_codebook(d, b, rng)  # [2^b, D]

            h_hat_dir, _ = rvq_quantize(h, cb)

            # NMSE：方向重构后按原范数缩放
            h_norm = np.linalg.norm(h, axis=1, keepdims=True) + 1e-12
            h_hat = h_hat_dir * h_norm
            nm = nmse_complex(h_hat, h)
            nmse_runs.append(nm)

            for snr in snr_list:
                r = achievable_rate_mrt(h, h_hat_dir, snr)
                rate_runs[snr].append(r)

        nmse_mean = float(np.mean(nmse_runs))
        nmse_std = float(np.std(nmse_runs))

        # 反馈开销：bits / 维度（每复维）
        overhead = b / d

        row = {
            "method": "RVQ",
            "bits": b,
            "nmse_mean": nmse_mean,
            "nmse_std": nmse_std,
            "feedback_bits": b,
            "dim": d,
            "overhead_bits_per_dim": overhead
        }
        for snr in snr_list:
            row[f"rate_snr{int(snr)}"] = float(np.mean(rate_runs[snr]))
            row[f"rate_snr{int(snr)}_std"] = float(np.std(rate_runs[snr]))

        rows.append(row)

        print(f"[RVQ-{b}bit] NMSE={nmse_mean:.6f}±{nmse_std:.6f}, Overhead={overhead:.6f}")
        for snr in snr_list:
            print(f"  SNR={snr:.1f} dB | Rate={row[f'rate_snr{int(snr)}']:.6f}")

    # 保存CSV
    import pandas as pd
    df = pd.DataFrame(rows)
    out_csv = "logs/eval_rvq_baseline.csv"
    df.to_csv(out_csv, index=False)

    print(f"[Done] csv={out_csv}, log={args.log_file}")


if __name__ == "__main__":
    main()
