# train_se.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.se_csinet import SECsiNet
from utils.metrics import nmse

def load_dataset(data_root):
    # 使用 processed 数据集（与你 train.py 一致）
    train_path = os.path.join(data_root, "processed", "train.pt")
    val_path = os.path.join(data_root, "processed", "val.pt")

    def load_pt(path):
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

        # squeeze 5D -> 4D
        if x.ndim == 5 and x.shape[2] == 1:
            x = x.squeeze(2)
        return x.float()

    x_train = load_pt(train_path)
    x_val = load_pt(val_path)

    train_ds = TensorDataset(x_train, torch.zeros(len(x_train)))
    val_ds = TensorDataset(x_val, torch.zeros(len(x_val)))
    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--cr", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Start training SE-CsiNet | CR={args.cr} | epochs={args.epochs} | batch={args.batch_size} | lr={args.lr}")

    train_ds, val_ds = load_dataset(args.data_root)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SECsiNet(h=32, w=256, cr=args.cr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_nmse = 1e9
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/se_csinet_cr{args.cr}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # val
        model.eval()
        with torch.no_grad():
            nmse_sum = 0.0
            count = 0
            for x, _ in val_loader:
                x = x.to(device)
                pred = model(x)
                nmse_sum += nmse(pred, x).item() * x.size(0)
                count += x.size(0)
            val_nmse = nmse_sum / count

        print(f"Epoch {epoch:03d} | Train Loss {avg_loss:.6f} | Val NMSE {val_nmse:.6f}")

        if val_nmse < best_nmse:
            best_nmse = val_nmse
            torch.save(model.state_dict(), ckpt_path)

    print(f"训练完成，最佳模型保存到 {ckpt_path}")


if __name__ == "__main__":
    main()