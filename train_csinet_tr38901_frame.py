import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.csinet_baseline import CsiNetBaseline
from models.csinet_plus_baseline import CsiNetPlusBaseline



def seed_all(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_log(log_file):
    import sys

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    log_f = open(log_file, "w", encoding="utf-8")

    class Tee:
        def write(self, msg):
            sys.__stdout__.write(msg)
            log_f.write(msg)
            log_f.flush()

        def flush(self):
            sys.__stdout__.flush()
            log_f.flush()

    sys.stdout = Tee()
    sys.stderr = sys.stdout


def load_frame_x(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "x" not in obj:
        raise ValueError(f"{path} must be dict with key 'x'")

    x = obj["x"].float()
    if x.ndim != 4 or x.shape[1:] != (2, 32, 256):
        raise ValueError(f"Expected [N,2,32,256], got {tuple(x.shape)}")

    return x


def nmse_torch(pred, target, eps=1e-12):
    numerator = torch.sum((pred - target) ** 2, dim=(1, 2, 3))
    denominator = torch.sum(target ** 2, dim=(1, 2, 3)) + eps
    return torch.mean(numerator / denominator)


def evaluate(model, loader, device, snrs, use_awgn):
    model.eval()

    results = {}

    with torch.no_grad():
        for snr in snrs:
            total_nmse = 0.0
            total_mse = 0.0
            total_n = 0

            for (x,) in loader:
                x = x.to(device, non_blocking=True)

                if use_awgn:
                    y = model(x, snr_db=float(snr))
                else:
                    y = model(x, snr_db=None)

                mse = F.mse_loss(y, x).item()
                nmse = nmse_torch(y, x).item()

                total_mse += mse * x.size(0)
                total_nmse += nmse * x.size(0)
                total_n += x.size(0)

            results[float(snr)] = {
                "mse": total_mse / total_n,
                "nmse": total_nmse / total_n,
            }

    avg_nmse = sum(v["nmse"] for v in results.values()) / len(results)
    avg_mse = sum(v["mse"] for v in results.values()) / len(results)

    return results, avg_nmse, avg_mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--model", type=str, default="csinet", choices=["csinet", "csinet_plus"])
    parser.add_argument("--cr", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--snr-list", type=str, default="0,5,10,15,20")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-awgn", action="store_true", help="Train noiseless original CsiNet-style AE.")
    parser.add_argument("--tag-suffix", type=str, default="")
    args = parser.parse_args()

    seed_all(args.seed)

    use_awgn = not args.no_awgn
    base_name = "csinet_plus" if args.model == "csinet_plus" else "csinet"
    tag = f"{base_name}_awgn" if use_awgn else f"{base_name}_noiseless"
    if args.tag_suffix:
        tag = f"{tag}_{args.tag_suffix}"

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    log_file = f"logs/train_{tag}_tr38901_cr{args.cr}.log"
    setup_log(log_file)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snrs = [float(v) for v in args.snr_list.split(",")]

    train_path = os.path.join(args.data_root, "processed", "train_tr38901_frame.pt")
    val_path = os.path.join(args.data_root, "processed", "val_tr38901_frame.pt")

    train_x = load_frame_x(train_path)
    val_x = load_frame_x(val_path)

    train_loader = DataLoader(
        TensorDataset(train_x),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        TensorDataset(val_x),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    if args.model == "csinet":
        model = CsiNetBaseline(cr=args.cr, use_awgn=use_awgn).to(device)
    elif args.model == "csinet_plus":
        model = CsiNetPlusBaseline(cr=args.cr, use_awgn=use_awgn).to(device)
    else:
        raise ValueError(args.model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    ckpt_path = f"checkpoints/{tag}_tr38901_cr{args.cr}.pt"
    last_path = f"checkpoints/{tag}_tr38901_cr{args.cr}_last.pt"

    best_nmse = float("inf")

    print("=" * 80)
    print("[Train CsiNet-style Baseline on TR 38.901 Frame Dataset]")
    print("model tag:", tag)
    print("use_awgn:", use_awgn)
    print("train:", tuple(train_x.shape))
    print("val:", tuple(val_x.shape))
    print("cr:", args.cr)
    print("latent_dim:", model.latent_dim)
    print("epochs:", args.epochs)
    print("batch_size:", args.batch_size)
    print("lr:", args.lr)
    print("snrs:", snrs)
    print("device:", device)
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_nmse = 0.0
        total_n = 0

        for (x,) in train_loader:
            x = x.to(device, non_blocking=True)

            if use_awgn:
                snr = random.choice(snrs)
                y = model(x, snr_db=float(snr))
            else:
                y = model(x, snr_db=None)

            loss = F.mse_loss(y, x)
            batch_nmse = nmse_torch(y, x)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_nmse += batch_nmse.item() * x.size(0)
            total_n += x.size(0)

        train_mse = total_loss / total_n
        train_nmse = total_nmse / total_n

        val_results, val_avg_nmse, val_avg_mse = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            snrs=snrs,
            use_awgn=use_awgn,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_mse={train_mse:.6f} | "
            f"train_nmse={train_nmse:.6f} | "
            f"val_avg_nmse={val_avg_nmse:.6f} | "
            f"val_avg_mse={val_avg_mse:.6f}"
        )

        details = " | ".join([f"{int(s)}dB:{val_results[s]['nmse']:.6f}" for s in snrs])
        print("  Val NMSE:", details)

        torch.save(model.state_dict(), last_path)

        if val_avg_nmse < best_nmse:
            best_nmse = val_avg_nmse
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [Best] save -> {ckpt_path}")

    print("=" * 80)
    print("[Done]")
    print("best_val_avg_nmse:", best_nmse)
    print("ckpt:", ckpt_path)
    print("last:", last_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
