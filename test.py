import os
import json
import argparse
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.csinet import CsiNet
from utils.metrics import nmse

def load_dataset(path):
    data = torch.load(path, weights_only=True)
    return data["x"].float(), data["y"].long()

def preprocess_input(x):
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--append-log", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"test_cr{args.cr}.log")
    log_mode = "a" if args.append_log else "w"
    log_file = open(log_path, log_mode, buffering=1)
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    meta = json.load(open(os.path.join(args.data_dir, "meta.json")))
    h = meta["n_rx"] * meta["n_tx"]
    w = meta["n_subcarriers"]

    test_x, _ = load_dataset(os.path.join(args.data_dir, "test.pt"))
    test_loader = DataLoader(
        TensorDataset(test_x, torch.zeros(len(test_x))),
        batch_size=args.batch_size,
        shuffle=False
    )

    model = CsiNet(h, w, args.cr).to(args.device)
    ckpt = f"checkpoints/csinet_cr{args.cr}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=args.device, weights_only=True))
    model.eval()

    print(f"Start testing | CR={args.cr} | batch={args.batch_size}")
    print(f"Log file: {log_path}")

    with torch.no_grad():
        test_nmse = 0.0
        for x, _ in test_loader:
            x = preprocess_input(x).to(args.device)
            pred = model(x)
            test_nmse += nmse(pred, x).item() * x.size(0)
        test_nmse /= len(test_loader.dataset)

    print(f"Test NMSE (CR={args.cr}): {test_nmse:.6f}")

if __name__ == "__main__":
    main()
