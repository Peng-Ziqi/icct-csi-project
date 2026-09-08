import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.se_jscc_snrprior import SEJSCCSNRPrior
from models.diffusion_refiner import DiffusionRefiner
from models.jscc_diffusion_wrapper import JSCCDiffusionWrapper
from utils.diffusion_schedule import DiffusionSchedule
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
        x = obj.get("x", obj.get("data", obj.get("H")))
    else:
        x = obj[0]
    if x.ndim == 5 and x.shape[2] == 1:
        x = x.squeeze(2)
    return x.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--base-ckpt", type=str, required=True, help="checkpoints/final_unified_cr*.pt")
    ap.add_argument("--cr", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--snr-list", type=str, default="0,5,10,15,20")
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--use-snr-cond", action="store_true")
    ap.add_argument("--lambda-rec", type=float, default=0.2, help="reconstruction loss weight")
    ap.add_argument("--log-file", type=str, default="")
    args = ap.parse_args()

    if args.log_file == "":
        tag = "snrcond" if args.use_snr_cond else "nocond"
        args.log_file = f"logs/train_diffusion_refiner_cr{args.cr}_{tag}.log"
    _ = setup_log_redirect(args.log_file)

    seed_all(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snrs = [float(v) for v in args.snr_list.split(",")]

    # Data
    tr = load_pt_x(os.path.join(args.data_root, "processed", "train.pt"))
    va = load_pt_x(os.path.join(args.data_root, "processed", "val.pt"))
    trl = DataLoader(TensorDataset(tr, torch.zeros(len(tr))), batch_size=args.batch_size, shuffle=True)
    val = DataLoader(TensorDataset(va, torch.zeros(len(va))), batch_size=args.batch_size, shuffle=False)

    # Base JSCC (frozen)
    base = SEJSCCSNRPrior(cr=args.cr).to(dev)
    base.load_state_dict(torch.load(args.base_ckpt, map_location=dev, weights_only=False))
    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    # Refiner
    refiner = DiffusionRefiner(in_ch=2, base_ch=32, cond_dim=64, use_snr_cond=args.use_snr_cond).to(dev)

    # Wrapper (仅用于拿 coarse)
    wrapper = JSCCDiffusionWrapper(base, refiner, T=args.T).to(dev)
    schedule = DiffusionSchedule(T=args.T, device=dev).to(dev)

    opt = torch.optim.Adam(refiner.parameters(), lr=args.lr)

    os.makedirs("checkpoints", exist_ok=True)
    tag = "snrcond" if args.use_snr_cond else "nocond"
    ckpt = f"checkpoints/diffusion_refiner_cr{args.cr}_{tag}.pt"

    best = 1e9

    for ep in range(1, args.epochs + 1):
        refiner.train()
        sum_loss, n = 0.0, 0

        for x, _ in trl:
            x = x.to(dev)  # gt CSI [B,2,32,256]
            snr = random.choice(snrs)
            snr_t = torch.full((x.size(0), 1), float(snr), device=dev)

            # coarse from base model
            with torch.no_grad():
                coarse, _ = wrapper._get_coarse(x, snr_db=snr, gate_mode="teacher")

            # residual target
            res_gt = x - coarse

            # diffusion forward on residual
            t = torch.randint(0, args.T, (x.size(0),), device=dev, dtype=torch.long)
            noise = torch.randn_like(res_gt)
            r_t = schedule.q_sample(res_gt, t, noise)

            if args.use_snr_cond:
                eps_hat = refiner(x_t=r_t, t=t, coarse=coarse, snr_db=snr_t)
            else:
                eps_hat = refiner(x_t=r_t, t=t, coarse=coarse, snr_db=None)

            # loss
            loss_diff = F.mse_loss(eps_hat, noise)
            res_hat = schedule.predict_x0_from_eps(r_t, t, eps_hat)
            x_hat = coarse + res_hat
            loss_rec = F.mse_loss(x_hat, x)
            loss = loss_diff + args.lambda_rec * loss_rec

            opt.zero_grad()
            loss.backward()
            opt.step()

            sum_loss += loss.item() * x.size(0)
            n += x.size(0)

        train_loss = sum_loss / n

        # validation (noise-pred loss only, same as diffusion objective)
        refiner.eval()
        v_sum, v_n = 0.0, 0
        with torch.no_grad():
            for x, _ in val:
                x = x.to(dev)
                snr = random.choice(snrs)
                snr_t = torch.full((x.size(0), 1), float(snr), device=dev)

                coarse, _ = wrapper._get_coarse(x, snr_db=snr, gate_mode="teacher")
                res_gt = x - coarse

                t = torch.randint(0, args.T, (x.size(0),), device=dev, dtype=torch.long)
                noise = torch.randn_like(res_gt)
                r_t = schedule.q_sample(res_gt, t, noise)

                if args.use_snr_cond:
                    eps_hat = refiner(x_t=r_t, t=t, coarse=coarse, snr_db=snr_t)
                else:
                    eps_hat = refiner(x_t=r_t, t=t, coarse=coarse, snr_db=None)

                v_loss = F.mse_loss(eps_hat, noise)
                v_sum += v_loss.item() * x.size(0)
                v_n += x.size(0)

        val_loss = v_sum / v_n
        print(f"Epoch {ep:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best:
            best = val_loss
            torch.save(refiner.state_dict(), ckpt)
            print(f"[Best] save -> {ckpt}")

    print(f"[Done] best_val={best:.6f}")


if __name__ == "__main__":
    main()
