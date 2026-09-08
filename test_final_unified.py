# test_final_unified.py
import os
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.se_jscc_snrprior import SEJSCCSNRPrior
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--cr", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--snr-list", type=str, default="0,5,10,15,20")
    ap.add_argument("--gate", type=str, default="teacher", choices=["teacher", "estimate"])
    ap.add_argument("--log-file", type=str, default="")
    args = ap.parse_args()

    if args.log_file == "":
        args.log_file = f"logs/test_final_unified_cr{args.cr}_{args.gate}.log"
    _ = setup_log_redirect(args.log_file)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snrs = parse_snr_list(args.snr_list)

    xte = load_pt_x(os.path.join(args.data_root, "processed", "test.pt"))
    loader = DataLoader(TensorDataset(xte, torch.zeros(len(xte))),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SEJSCCSNRPrior(cr=args.cr).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False))
    model.eval()

    dmax = model.comp_dim
    os.makedirs("logs", exist_ok=True)
    csvf = f"logs/test_final_unified_cr{args.cr}_{args.gate}.csv"
    with open(csvf, "w", encoding="utf-8") as f:
        f.write("snr_db,nmse,avg_k,overhead\n")

    all_nmse, all_over = [], []
    with torch.no_grad():
        for snr in snrs:
            s_nmse, s_k, n = 0.0, 0.0, 0
            for x, _ in loader:
                x = x.to(dev)
                y, snr_est, ratio, k = model(x, snr_db=snr, gate_mode=args.gate)
                s_nmse += nmse(y, x).item() * x.size(0)
                s_k += k.float().sum().item()
                n += x.size(0)
            nmsev = s_nmse / n
            avgk = s_k / n
            over = avgk / dmax
            all_nmse.append(nmsev)
            all_over.append(over)

            print(f"[{args.gate}] SNR={snr:.1f} dB | NMSE={nmsev:.6f} | AvgK={avgk:.1f}/{dmax} | Overhead={over:.4f}")
            with open(csvf, "a", encoding="utf-8") as f:
                f.write(f"{snr},{nmsev:.8f},{avgk:.4f},{over:.6f}\n")

    print(f"[Summary-{args.gate}] Avg NMSE={sum(all_nmse)/len(all_nmse):.6f}, Avg Overhead={sum(all_over)/len(all_over):.6f}")
    print(f"[Done] csv={csvf}, log={args.log_file}")


if __name__ == "__main__":
    main()
