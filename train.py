import os
import json
import argparse
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from models.csinet import CsiNet
from utils.metrics import nmse

def load_dataset(path):
    data = torch.load(path, weights_only=True)
    return data["x"].float(), data["y"].long()

def preprocess_input(x):
    # x: [B, 2, N_RX, N_TX, N_SC] -> [B, 2, N_TX, N_SC] (N_RX=1)
    if x.dim() == 5:
        x = x.squeeze(2)
    return x

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--cr", type=int, default=4, choices=[4, 8, 16])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--append-log", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"train_cr{args.cr}.log")
    log_mode = "a" if args.append_log else "w"
    log_file = open(log_path, log_mode, buffering=1)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    h = meta["n_rx"] * meta["n_tx"]
    w = meta["n_subcarriers"]

    train_x, _ = load_dataset(os.path.join(args.data_dir, "train.pt"))
    val_x, _ = load_dataset(os.path.join(args.data_dir, "val.pt"))

    train_loader = DataLoader(
        TensorDataset(train_x, torch.zeros(len(train_x))),
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_x, torch.zeros(len(val_x))),
        batch_size=args.batch_size,
        shuffle=False
    )

    model = CsiNet(h, w, args.cr).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_nmse = 1e9
    history = {"train_loss": [], "val_nmse": []}

    print(f"Start training | CR={args.cr} | epochs={args.epochs} | batch={args.batch_size} | lr={args.lr}")
    print(f"Log file: {log_path}")

    for epoch in range(1, args.epochs + 1):
        start_t = time.perf_counter()

        model.train()
        epoch_loss = 0.0
        for x, _ in train_loader:
            x = preprocess_input(x).to(args.device)
            pred = model(x)
            loss = criterion(pred, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)

        epoch_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_nmse = 0.0
            for x, _ in val_loader:
                x = preprocess_input(x).to(args.device)
                pred = model(x)
                val_nmse += nmse(pred, x).item() * x.size(0)
            val_nmse /= len(val_loader.dataset)

        history["train_loss"].append(epoch_loss)
        history["val_nmse"].append(val_nmse)

        if val_nmse < best_nmse:
            best_nmse = val_nmse
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/csinet_cr{args.cr}.pt")

        epoch_time = time.perf_counter() - start_t
        print(f"Epoch {epoch:03d} | Train Loss {epoch_loss:.6f} | Val NMSE {val_nmse:.6f} | Time {epoch_time:.2f}s")

    os.makedirs("plots", exist_ok=True)
    plt.figure()
    plt.plot(history["val_nmse"])
    plt.title(f"Val NMSE (CR={args.cr})")
    plt.xlabel("Epoch")
    plt.ylabel("NMSE")
    plt.grid(True)
    plt.savefig(f"plots/val_nmse_cr{args.cr}.png", dpi=200)

    plt.figure()
    plt.plot(history["train_loss"])
    plt.title(f"Train Loss (CR={args.cr})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig(f"plots/train_loss_cr{args.cr}.png", dpi=200)

if __name__ == "__main__":
    main()
