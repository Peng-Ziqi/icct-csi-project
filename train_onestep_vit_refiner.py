import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.se_jscc_snrprior import SEJSCCSNRPrior
from models.sequence_jscc_wrapper import SequenceJSCCWrapper
from models.onestep_vit_refiner import OneStepViTRefiner


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


def load_seq_x(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "x" not in obj:
        raise ValueError(f"{path} must be dict with key 'x'")

    x = obj["x"].float()

    if x.ndim != 5:
        raise ValueError(f"Expected [N,8,2,32,256], got {tuple(x.shape)}")
    if x.shape[1:] != (8, 2, 32, 256):
        raise ValueError(f"Expected [N,8,2,32,256], got {tuple(x.shape)}")

    return x


def sample_logit_normal(batch_size, device, mean=0.0, std=1.0, eps=1e-4):
    z = torch.randn(batch_size, 1, device=device) * std + mean
    t = torch.sigmoid(z)
    t = torch.clamp(t, eps, 1.0 - eps)
    return t


def expand_t(t, x):
    return t.view(x.shape[0], *([1] * (x.ndim - 1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-suffix", type=str, default="")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--base-ckpt", type=str, required=True)
    parser.add_argument("--cr", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--snr-list", type=str, default="0,5,10,15,20")
    parser.add_argument("--use-snr-cond", action="store_true")
    parser.add_argument("--lambda-rec", type=float, default=0.2)
    parser.add_argument("--lambda-repa", type=float, default=0.5)
    parser.add_argument("--lambda-v", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-gate-mode", type=str, default="teacher", choices=["teacher", "estimate"])
    args = parser.parse_args()

    seed_all(args.seed)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    tag = "vit_snrcond" if args.use_snr_cond else "vit_nocond"
    if args.tag_suffix:
        tag = f"{tag}_{args.tag_suffix}"
    log_file = f"logs/train_onestep_video_swin_tr38901_cr{args.cr}_{tag}.log"
    setup_log(log_file)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snrs = [float(v) for v in args.snr_list.split(",")]

    train_x = load_seq_x(os.path.join(args.data_root, "processed", "train_tr38901_seq.pt"))
    val_x = load_seq_x(os.path.join(args.data_root, "processed", "val_tr38901_seq.pt"))

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

    base = SEJSCCSNRPrior(cr=args.cr).to(device)
    base.load_state_dict(torch.load(args.base_ckpt, map_location=device, weights_only=False))
    base.eval()

    for p in base.parameters():
        p.requires_grad = False

    seq_jscc = SequenceJSCCWrapper(base).to(device)
    seq_jscc.eval()

    refiner = OneStepViTRefiner(
        in_ch=2,
        cond_dim=128,
        dim=48,
        depth=6,
        num_heads=4,
        patch_size=(1, 4, 16),
        use_snr_cond=args.use_snr_cond,
    ).to(device)

    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=0.01)

    ckpt_path = f"checkpoints/onestep_video_swin_tr38901_cr{args.cr}_{tag}.pt"
    last_path = f"checkpoints/onestep_video_swin_tr38901_cr{args.cr}_{tag}_last.pt"

    best_val = float("inf")

    print("=" * 80)
    print("[Train One-Step Video-Swin Representation Diffusion Refiner]")
    print("train:", tuple(train_x.shape))
    print("val:", tuple(val_x.shape))
    print("base_ckpt:", args.base_ckpt)
    print("cr:", args.cr)
    print("epochs:", args.epochs)
    print("batch_size:", args.batch_size)
    print("lr:", args.lr)
    print("snrs:", snrs)
    print("use_snr_cond:", args.use_snr_cond)
    print("lambda_rec:", args.lambda_rec)
    print("lambda_repa:", args.lambda_repa)
    print("lambda_v:", args.lambda_v)
    print("device:", device)
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        refiner.train()

        total_loss = 0.0
        total_full = 0.0
        total_repa = 0.0
        total_rec = 0.0
        total_v = 0.0
        total_n = 0

        for (x,) in train_loader:
            x = x.to(device, non_blocking=True)

            snr = random.choice(snrs)
            snr_tensor = torch.full((x.size(0), 1), float(snr), device=device)

            with torch.no_grad():
                coarse = seq_jscc(x, snr_db=float(snr), gate_mode=args.train_gate_mode)

            res_gt = x - coarse
            noise = torch.randn_like(res_gt)

            t = sample_logit_normal(x.size(0), device=device)
            t_view = expand_t(t, res_gt)

            z_t = t_view * res_gt + (1.0 - t_view) * noise

            out = refiner(
                z_t=z_t,
                coarse=coarse,
                t=t,
                snr_db=snr_tensor if args.use_snr_cond else None,
            )

            res_full = out["res_full"]
            res_repa = out["res_repa"]

            loss_full = F.mse_loss(res_full, res_gt)
            loss_repa = F.mse_loss(res_repa, res_gt)
            loss_rec = F.mse_loss(coarse + res_full, x)

            if args.lambda_v > 0:
                denom = torch.clamp(1.0 - t_view, min=1e-4)
                v_gt = (res_gt - z_t) / denom
                v_pred = (res_full - z_t) / denom
                loss_v = F.mse_loss(v_pred, v_gt)
            else:
                loss_v = torch.zeros((), device=device)

            loss = (
                loss_full
                + args.lambda_repa * loss_repa
                + args.lambda_rec * loss_rec
                + args.lambda_v * loss_v
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=1.0)
            optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_full += loss_full.item() * bs
            total_repa += loss_repa.item() * bs
            total_rec += loss_rec.item() * bs
            total_v += loss_v.item() * bs
            total_n += bs

        train_loss = total_loss / total_n
        train_full = total_full / total_n
        train_repa = total_repa / total_n
        train_rec = total_rec / total_n
        train_v = total_v / total_n

        refiner.eval()
        val_sum = 0.0
        val_n = 0

        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(device, non_blocking=True)

                snr = random.choice(snrs)
                snr_tensor = torch.full((x.size(0), 1), float(snr), device=device)

                coarse = seq_jscc(x, snr_db=float(snr), gate_mode=args.train_gate_mode)
                res_gt = x - coarse
                noise = torch.randn_like(res_gt)

                t = sample_logit_normal(x.size(0), device=device)
                t_view = expand_t(t, res_gt)
                z_t = t_view * res_gt + (1.0 - t_view) * noise

                out = refiner(
                    z_t=z_t,
                    coarse=coarse,
                    t=t,
                    snr_db=snr_tensor if args.use_snr_cond else None,
                )

                res_full = out["res_full"]
                res_repa = out["res_repa"]

                loss_full = F.mse_loss(res_full, res_gt)
                loss_repa = F.mse_loss(res_repa, res_gt)
                loss_rec = F.mse_loss(coarse + res_full, x)

                val_loss = loss_full + args.lambda_repa * loss_repa + args.lambda_rec * loss_rec

                val_sum += val_loss.item() * x.size(0)
                val_n += x.size(0)

        val_loss = val_sum / val_n

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_loss:.6f} | "
            f"full={train_full:.6f} | "
            f"repa={train_repa:.6f} | "
            f"rec={train_rec:.6f} | "
            f"v={train_v:.6f} | "
            f"val={val_loss:.6f}"
        )

        torch.save(refiner.state_dict(), last_path)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(refiner.state_dict(), ckpt_path)
            print(f"  [Best] save -> {ckpt_path}")

    print("=" * 80)
    print("[Done]")
    print("best_val:", best_val)
    print("ckpt:", ckpt_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
