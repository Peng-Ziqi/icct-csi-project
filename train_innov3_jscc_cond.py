import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.se_csinet_jscc_cond import SECsiNetJSCCCond
from utils.metrics import nmse
from utils.logger import setup_log_redirect


def seed_all(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def evaluate(model, loader, device, snr_list):
    model.eval()
    res = {}
    with torch.no_grad():
        for snr in snr_list:
            s, c = 0.0, 0
            for x, _ in loader:
                x = x.to(device)
                y = model(x, snr_db=snr)
                s += nmse(y, x).item() * x.size(0)
                c += x.size(0)
            res[snr] = s / c
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--cr", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--snr-list", type=str, default="0,5,10,15,20")
    parser.add_argument("--log-file", type=str, default="")
    args = parser.parse_args()

    if args.log_file == "":
        args.log_file = f"logs/train_innov3_jscc_cond_cr{args.cr}.log"
    _tee = setup_log_redirect(args.log_file)

    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snr_list = parse_snr_list(args.snr_list)

    print(f"[Info] device={device}, cr={args.cr}, snr_list={snr_list}, epochs={args.epochs}")

    train_x = load_pt_x(os.path.join(args.data_root, "processed", "train.pt"))
    val_x = load_pt_x(os.path.join(args.data_root, "processed", "val.pt"))

    train_loader = DataLoader(TensorDataset(train_x, torch.zeros(len(train_x))),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(val_x, torch.zeros(len(val_x))),
                            batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SECsiNetJSCCCond(h=32, w=256, cr=args.cr, se_reduction=8,
                             snr_min=min(snr_list), snr_max=max(snr_list)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    ckpt = f"checkpoints/se_jscc_cond_cr{args.cr}.pt"
    csv_path = f"logs/train_innov3_jscc_cond_cr{args.cr}.csv"

    with open(csv_path, "w", encoding="utf-8") as f:
        head = "epoch,train_loss," + ",".join([f"val_nmse_snr{int(s)}" for s in snr_list]) + ",val_nmse_avg\n"
        f.write(head)

    best_avg = 1e9

    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0

        for x, _ in train_loader:
            x = x.to(device)
            snr = random.choice(snr_list)

            optimizer.zero_grad()
            y = model(x, snr_db=snr)
            loss = criterion(y, x)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)

        train_loss = loss_sum / len(train_loader.dataset)
        val_nmse = evaluate(model, val_loader, device, snr_list)
        val_avg = sum(val_nmse.values()) / len(val_nmse)

        line = f"{ep},{train_loss:.8f}," + ",".join([f"{val_nmse[s]:.8f}" for s in snr_list]) + f",{val_avg:.8f}"
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        print(f"Epoch {ep:03d} | Loss {train_loss:.6f} | ValAvgNMSE {val_avg:.6f} | " +
              " ".join([f"SNR{int(s)}:{val_nmse[s]:.4f}" for s in snr_list]))

        if val_avg < best_avg:
            best_avg = val_avg
            torch.save(model.state_dict(), ckpt)

    print(f"[Done] best_avg_nmse={best_avg:.6f}, ckpt={ckpt}, csv={csv_path}, log={args.log_file}")


if __name__ == "__main__":
    main()
