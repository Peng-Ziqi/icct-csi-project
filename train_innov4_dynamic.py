# train_innov4_dynamic.py
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.se_csinet_dynamic import SECsiNetDynamic
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


def evaluate(model, loader, device, snr_list, gate_mode="teacher"):
    model.eval()
    out = {}
    with torch.no_grad():
        for snr in snr_list:
            s_nmse, s_k, n = 0.0, 0.0, 0
            for x, _ in loader:
                x = x.to(device)
                y, snr_est, ratio, k = model(x, snr_db=snr, gate_mode=gate_mode)
                s_nmse += nmse(y, x).item() * x.size(0)
                s_k += k.float().sum().item()
                n += x.size(0)
            out[snr] = {"nmse": s_nmse / n, "avg_k": s_k / n}
    return out


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
    parser.add_argument("--lambda-snr", type=float, default=0.01)
    parser.add_argument("--log-file", type=str, default="")
    args = parser.parse_args()

    if args.log_file == "":
        args.log_file = f"logs/train_innov4_dynamic_cr{args.cr}.log"
    _tee = setup_log_redirect(args.log_file)

    print("[Start] train_innov4_dynamic.py loaded.")
    seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snr_list = parse_snr_list(args.snr_list)
    print(f"[Info] device={device}, cr={args.cr}, epochs={args.epochs}, snr_list={snr_list}, seed={args.seed}")

    train_x = load_pt_x(os.path.join(args.data_root, "processed", "train.pt"))
    val_x = load_pt_x(os.path.join(args.data_root, "processed", "val.pt"))
    print(f"[Data] train={tuple(train_x.shape)}, val={tuple(val_x.shape)}")

    train_loader = DataLoader(
        TensorDataset(train_x, torch.zeros(len(train_x))),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(val_x, torch.zeros(len(val_x))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = SECsiNetDynamic(h=32, w=256, cr=args.cr, se_reduction=8).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    ckpt = f"checkpoints/se_dynamic_cr{args.cr}.pt"
    csvf = f"logs/train_innov4_dynamic_cr{args.cr}.csv"
    with open(csvf, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_nmse_avg_teacher,val_k_avg_teacher,val_nmse_avg_est,val_k_avg_est\n")

    best = 1e9
    dmax = model.comp_dim_max

    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0

        for x, _ in train_loader:
            x = x.to(device)
            snr = random.choice(snr_list)
            snr_t = torch.full((x.size(0), 1), float(snr), device=device)

            opt.zero_grad()
            # 训练时用 teacher gate，保证三档码长机制生效
            y, snr_est, ratio, k = model(x, snr_db=snr, gate_mode="teacher")
            loss_rec = mse(y, x)
            loss_snr = mse(snr_est, snr_t)
            loss = loss_rec + args.lambda_snr * loss_snr
            loss.backward()
            opt.step()

            loss_sum += loss.item() * x.size(0)
            n += x.size(0)

        train_loss = loss_sum / n

        val_teacher = evaluate(model, val_loader, device, snr_list, gate_mode="teacher")
        val_est = evaluate(model, val_loader, device, snr_list, gate_mode="estimate")

        nmse_teacher = sum([val_teacher[s]["nmse"] for s in snr_list]) / len(snr_list)
        k_teacher = sum([val_teacher[s]["avg_k"] for s in snr_list]) / len(snr_list)

        nmse_est = sum([val_est[s]["nmse"] for s in snr_list]) / len(snr_list)
        k_est = sum([val_est[s]["avg_k"] for s in snr_list]) / len(snr_list)

        with open(csvf, "a", encoding="utf-8") as f:
            f.write(f"{ep},{train_loss:.8f},{nmse_teacher:.8f},{k_teacher:.4f},{nmse_est:.8f},{k_est:.4f}\n")

        print(f"Epoch {ep:03d} | Loss {train_loss:.6f} | "
              f"Teacher NMSE {nmse_teacher:.6f} K {k_teacher:.1f}/{dmax} | "
              f"Estimate NMSE {nmse_est:.6f} K {k_est:.1f}/{dmax}")

        if nmse_teacher < best:
            best = nmse_teacher
            torch.save(model.state_dict(), ckpt)

    print(f"[Done] best_teacher_avg_nmse={best:.6f}, ckpt={ckpt}, csv={csvf}, log={args.log_file}")


if __name__ == "__main__":
    main()
