import os
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.se_csinet_jscc_cond import SECsiNetJSCCCond
from utils.metrics import nmse
from utils.logger import setup_log_redirect


def load_pt_x(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        x = obj
    elif isinstance(obj, dict):
        if "x" in obj: x = obj["x"]
        elif "data" in obj: x = obj["data"]
        elif "H" in obj: x = obj["H"]
        else: raise KeyError(f"{path} dict无x/data/H键")
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
    parser.add_argument("--log-file", type=str, default="")
    args = parser.parse_args()

    if args.log_file == "":
        args.log_file = f"logs/test_innov3_jscc_cond_cr{args.cr}.log"
    _tee = setup_log_redirect(args.log_file)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snr_list = parse_snr_list(args.snr_list)

    test_x = load_pt_x(os.path.join(args.data_root, "processed", "test.pt"))
    test_loader = DataLoader(TensorDataset(test_x, torch.zeros(len(test_x))),
                             batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SECsiNetJSCCCond(h=32, w=256, cr=args.cr, se_reduction=8,
                             snr_min=min(snr_list), snr_max=max(snr_list)).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    os.makedirs("logs", exist_ok=True)
    csv_path = f"logs/test_innov3_jscc_cond_cr{args.cr}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("snr_db,nmse\n")

    with torch.no_grad():
        for snr in snr_list:
            s, c = 0.0, 0
            for x, _ in test_loader:
                x = x.to(device)
                y = model(x, snr_db=snr)
                s += nmse(y, x).item() * x.size(0)
                c += x.size(0)
            v = s / c
            print(f"SNR={snr:.1f} dB | NMSE={v:.6f}")
            with open(csv_path, "a", encoding="utf-8") as f:
                f.write(f"{snr},{v:.8f}\n")

    print(f"[Done] csv={csv_path}, log={args.log_file}")


if __name__ == "__main__":
    main()
