# test_innov4_dynamic.py
import os
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.se_csinet_dynamic import SECsiNetDynamic
from utils.metrics import nmse
from utils.logger import setup_log_redirect


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--cr", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--snr-list", type=str, default="0,5,10,15,20")
    parser.add_argument("--gate", type=str, default="teacher", choices=["teacher", "estimate"])
    parser.add_argument("--log-file", type=str, default="")
    args = parser.parse_args()

    if args.log_file == "":
        args.log_file = f"logs/test_innov4_dynamic_cr{args.cr}_{args.gate}.log"
    _tee = setup_log_redirect(args.log_file)

    print("[Start] test_innov4_dynamic.py loaded.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snr_list = parse_snr_list(args.snr_list)

    print(f"[Info] device={device}, cr={args.cr}, gate={args.gate}, snr_list={snr_list}")
    print(f"[Info] ckpt={args.ckpt}")

    test_x = load_pt_x(os.path.join(args.data_root, "processed", "test.pt"))
    print(f"[Data] test={tuple(test_x.shape)}")

    test_loader = DataLoader(
        TensorDataset(test_x, torch.zeros(len(test_x))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = SECsiNetDynamic(h=32, w=256, cr=args.cr, se_reduction=8).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    dmax = model.comp_dim_max
    os.makedirs("logs", exist_ok=True)
    csvf = f"logs/test_innov4_dynamic_cr{args.cr}_{args.gate}.csv"
    with open(csvf, "w", encoding="utf-8") as f:
        f.write("snr_db,nmse,avg_k,overhead_ratio\n")

    all_nmse, all_ratio = [], []
    with torch.no_grad():
        for snr in snr_list:
            s_nmse, s_k, n = 0.0, 0.0, 0
            for x, _ in test_loader:
                x = x.to(device)
                y, snr_est, ratio, k = model(x, snr_db=snr, gate_mode=args.gate)
                s_nmse += nmse(y, x).item() * x.size(0)
                s_k += k.float().sum().item()
                n += x.size(0)

            nmse_v = s_nmse / n
            avg_k = s_k / n
            overhead = avg_k / dmax

            all_nmse.append(nmse_v)
            all_ratio.append(overhead)

            print(f"[{args.gate}] SNR={snr:.1f} dB | NMSE={nmse_v:.6f} | AvgK={avg_k:.2f}/{dmax} | Overhead={overhead:.4f}")
            with open(csvf, "a", encoding="utf-8") as f:
                f.write(f"{snr},{nmse_v:.8f},{avg_k:.4f},{overhead:.6f}\n")

    print(f"[Summary-{args.gate}] Avg NMSE={sum(all_nmse)/len(all_nmse):.6f}, Avg Overhead={sum(all_ratio)/len(all_ratio):.6f}")
    print(f"[Done] csv={csvf}, log={args.log_file}")


if __name__ == "__main__":
    main()
