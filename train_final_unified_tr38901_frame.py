import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.se_jscc_snrprior import SEJSCCSNRPrior


def seed_all(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TeeLogger:
    def __init__(self, log_file):
        self.terminal = None
        self.log = open(log_file, "w", encoding="utf-8")

    def write(self, message):
        import sys
        sys.__stdout__.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        import sys
        sys.__stdout__.flush()
        self.log.flush()


def setup_log(log_file):
    import sys
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    sys.stdout = TeeLogger(log_file)
    sys.stderr = sys.stdout


def load_frame_x(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        x = obj
    elif isinstance(obj, dict):
        if "x" not in obj:
            raise KeyError(f"{path} is dict but missing key 'x'")
        x = obj["x"]
    else:
        raise TypeError(f"Unsupported data type in {path}: {type(obj)}")

    if x.ndim != 4:
        raise ValueError(f"Expected frame data [N,2,32,256], got {tuple(x.shape)}")

    if x.shape[1:] != (2, 32, 256):
        raise ValueError(f"Unexpected CSI shape {tuple(x.shape)}, expected [N,2,32,256]")

    return x.float()


def nmse_torch(pred, target, eps=1e-12):
    numerator = torch.sum((pred - target) ** 2, dim=(1, 2, 3))
    denominator = torch.sum(target ** 2, dim=(1, 2, 3)) + eps
    return torch.mean(numerator / denominator)


def get_model_output(model, x, snr_db, gate_mode="teacher"):
    """
    Compatible with SEJSCCSNRPrior:
        y, snr_est, ratio, k = model(x, snr_db=..., gate_mode=...)
    Also supports tensor/dict output defensively.
    """
    try:
        out = model(x, snr_db=snr_db, gate_mode=gate_mode)
    except TypeError:
        out = model(x, snr_db)

    if torch.is_tensor(out):
        return out

    if isinstance(out, (tuple, list)):
        return out[0]

    if isinstance(out, dict):
        if "y" in out:
            return out["y"]
        if "x_hat" in out:
            return out["x_hat"]
        if "recon" in out:
            return out["recon"]
        raise KeyError("Model dict output missing y/x_hat/recon")

    raise TypeError(f"Unsupported model output type: {type(out)}")


def evaluate(model, loader, device, snrs, gate_mode="estimate"):
    model.eval()

    results = {}
    with torch.no_grad():
        for snr in snrs:
            total_nmse = 0.0
            total_mse = 0.0
            total_n = 0

            for (x,) in loader:
                x = x.to(device, non_blocking=True)
                y = get_model_output(model, x, snr_db=float(snr), gate_mode=gate_mode)

                batch_nmse = nmse_torch(y, x).item()
                batch_mse = F.mse_loss(y, x).item()

                total_nmse += batch_nmse * x.size(0)
                total_mse += batch_mse * x.size(0)
                total_n += x.size(0)

            results[float(snr)] = {
                "nmse": total_nmse / total_n,
                "mse": total_mse / total_n,
            }

    avg_nmse = sum(v["nmse"] for v in results.values()) / len(results)
    avg_mse = sum(v["mse"] for v in results.values()) / len(results)
    return results, avg_nmse, avg_mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--cr", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--snr-list", type=str, default="0,5,10,15,20")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-gate-mode", type=str, default="estimate", choices=["teacher", "estimate"])
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--log-file", type=str, default="")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    if args.log_file == "":
        args.log_file = f"logs/train_final_unified_tr38901_frame_cr{args.cr}.log"

    setup_log(args.log_file)
    seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snrs = [float(v) for v in args.snr_list.split(",")]

    print("=" * 80)
    print("[Train Spatial JSCC Backbone on Standard TR 38.901 Frame Dataset]")
    print("CR:", args.cr)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("LR:", args.lr)
    print("Device:", device)
    print("SNR list:", snrs)
    print("Eval gate mode:", args.eval_gate_mode)
    print("=" * 80)

    train_path = os.path.join(args.data_root, "processed", "train_tr38901_frame.pt")
    val_path = os.path.join(args.data_root, "processed", "val_tr38901_frame.pt")

    train_x = load_frame_x(train_path)
    val_x = load_frame_x(val_path)

    print("[Data]")
    print("train:", tuple(train_x.shape))
    print("val:  ", tuple(val_x.shape))

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

    model = SEJSCCSNRPrior(cr=args.cr).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_nmse = float("inf")
    best_ckpt = os.path.join(args.save_dir, f"final_unified_tr38901_cr{args.cr}.pt")
    last_ckpt = os.path.join(args.save_dir, f"final_unified_tr38901_cr{args.cr}_last.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_nmse = 0.0
        total_n = 0

        for (x,) in train_loader:
            x = x.to(device, non_blocking=True)

            snr = random.choice(snrs)
            y = get_model_output(model, x, snr_db=float(snr), gate_mode="teacher")

            loss = F.mse_loss(y, x)
            batch_nmse = nmse_torch(y, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_nmse += batch_nmse.item() * x.size(0)
            total_n += x.size(0)

        train_loss = total_loss / total_n
        train_nmse = total_nmse / total_n

        val_results, val_avg_nmse, val_avg_mse = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            snrs=snrs,
            gate_mode=args.eval_gate_mode,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_mse={train_loss:.6f} | "
            f"train_nmse={train_nmse:.6f} | "
            f"val_avg_nmse={val_avg_nmse:.6f} | "
            f"val_avg_mse={val_avg_mse:.6f}"
        )

        detail = " | ".join([f"{int(s)}dB:{val_results[s]['nmse']:.6f}" for s in snrs])
        print("  Val NMSE:", detail)

        torch.save(model.state_dict(), last_ckpt)

        if val_avg_nmse < best_nmse:
            best_nmse = val_avg_nmse
            torch.save(model.state_dict(), best_ckpt)
            print(f"  [Best] save -> {best_ckpt}")

    print("=" * 80)
    print("[Done]")
    print("Best val avg NMSE:", best_nmse)
    print("Best checkpoint:", best_ckpt)
    print("Last checkpoint:", last_ckpt)
    print("=" * 80)


if __name__ == "__main__":
    main()
