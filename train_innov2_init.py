# train_innov2_init.py
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.se_csinet import SECsiNet
from utils.metrics import nmse
from utils.init_utils import init_model
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
        if "x" in obj:
            x = obj["x"]
        elif "data" in obj:
            x = obj["data"]
        elif "H" in obj:
            x = obj["H"]
        else:
            raise KeyError(f"{path} dict无x/data/H键")
    elif isinstance(obj, (list, tuple)):
        x = obj[0]
    else:
        raise TypeError(f"{path}类型不支持")
    if x.ndim == 5 and x.shape[2] == 1:
        x = x.squeeze(2)
    return x.float()


def print_weight_stats(model, title=""):
    print(f"\n[WeightStats] {title}")
    for n, p in model.named_parameters():
        if p.requires_grad and p.dim() >= 2:
            print(f"{n:40s} mean={p.data.mean().item():+.6f} std={p.data.std().item():.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--cr", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--init",
        type=str,
        default="default",
        choices=["default", "he_all", "he_conv_only", "he_conv_mainfc"]
    )
    parser.add_argument("--log-file", type=str, default="")
    args = parser.parse_args()

    # 日志
    if args.log_file == "":
        args.log_file = f"logs/train_innov2_init_{args.init}_cr{args.cr}_seed{args.seed}.log"
    _tee = setup_log_redirect(args.log_file)

    # 固定随机种子
    seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Info] device={device}, init={args.init}, cr={args.cr}, epochs={args.epochs}, seed={args.seed}")

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

    model = SECsiNet(h=32, w=256, cr=args.cr, se_reduction=8).to(device)
    model = init_model(model, scheme=args.init)
    print("[Model] initialized.")
    print_weight_stats(model, title=f"init={args.init}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    ckpt = f"checkpoints/se_init_{args.init}_cr{args.cr}_seed{args.seed}.pt"
    csv_log = f"logs/train_innov2_init_{args.init}_cr{args.cr}_seed{args.seed}.csv"

    with open(csv_log, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_nmse,best_nmse\n")

    best = 1e9

    for ep in range(1, args.epochs + 1):
        # train
        model.train()
        loss_sum = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            y = model(x)
            loss = criterion(y, x)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
        train_loss = loss_sum / len(train_loader.dataset)

        # val
        model.eval()
        s, c = 0.0, 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                y = model(x)
                s += nmse(y, x).item() * x.size(0)
                c += x.size(0)
        val_nmse = s / c

        if val_nmse < best:
            best = val_nmse
            torch.save(model.state_dict(), ckpt)

        with open(csv_log, "a", encoding="utf-8") as f:
            f.write(f"{ep},{train_loss:.8f},{val_nmse:.8f},{best:.8f}\n")

        print(f"[{args.init}] Epoch {ep:03d} | Loss {train_loss:.6f} | Val NMSE {val_nmse:.6f} | Best {best:.6f}")

    print(f"[Done] init={args.init}, best_nmse={best:.6f}, ckpt={ckpt}, csv={csv_log}, log={args.log_file}")


if __name__ == "__main__":
    main()
