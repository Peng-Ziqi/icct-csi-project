import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.se_jscc_snrprior import SEJSCCSNRPrior
from models.diffusion_refiner import DiffusionRefiner
from models.jscc_diffusion_wrapper import JSCCDiffusionWrapper
from utils.metrics import nmse
from utils.logger import setup_log_redirect


def seed_all(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 为了可复现
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


def snr_to_steps(snr):
    # 你当前默认的动态步数策略
    if snr <= 5:
        return 8
    elif snr <= 10:
        return 6
    else:
        return 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--base-ckpt", type=str, required=True)
    ap.add_argument("--refiner-ckpt", type=str, required=True)
    ap.add_argument("--cr", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--snr-list", type=str, default="0,5,10,15,20")
    ap.add_argument("--use-snr-cond", action="store_true")
    ap.add_argument("--fixed-steps", type=int, default=8, help="fixed模式下的步数")
    ap.add_argument("--steps-mode", type=str, default="fixed", choices=["fixed", "dynamic"],
                    help="fixed: 所有SNR用同一步数; dynamic: 按snr_to_steps")
    ap.add_argument("--gate-mode", type=str, default="estimate", choices=["teacher", "estimate"])
    ap.add_argument("--residual-scale", type=float, default=0.02, help="residual sampling init scale")
    ap.add_argument("--alpha", type=float, default=0.2, help="final blend: coarse + alpha*(refined_raw-coarse)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--log-file", type=str, default="")
    args = ap.parse_args()

    # 日志文件名
    if args.log_file == "":
        tag = "snrcond" if args.use_snr_cond else "nocond"
        args.log_file = (
            f"logs/test_diffusion_jscc_cr{args.cr}_{tag}_"
            f"{args.steps_mode}_seed{args.seed}.log"
        )
    _ = setup_log_redirect(args.log_file)

    seed_all(args.seed)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snrs = [float(v) for v in args.snr_list.split(",")]

    # Data
    x = load_pt_x(os.path.join(args.data_root, "processed", "test.pt"))
    loader = DataLoader(TensorDataset(x, torch.zeros(len(x))), batch_size=args.batch_size, shuffle=False)

    # Base model
    base = SEJSCCSNRPrior(cr=args.cr).to(dev)
    base.load_state_dict(torch.load(args.base_ckpt, map_location=dev, weights_only=False))
    base.eval()

    # Refiner
    refiner = DiffusionRefiner(use_snr_cond=args.use_snr_cond).to(dev)
    refiner.load_state_dict(torch.load(args.refiner_ckpt, map_location=dev, weights_only=False))
    refiner.eval()

    # Wrapper (内部默认 deterministic=True)
    wrapper = JSCCDiffusionWrapper(base, refiner, T=max(args.fixed_steps, 8)).to(dev)
    wrapper.eval()

    tag = "snrcond" if args.use_snr_cond else "nocond"
    csvf = (
        f"logs/test_diffusion_jscc_cr{args.cr}_{tag}_"
        f"{args.steps_mode}_seed{args.seed}.csv"
    )
    with open(csvf, "w", encoding="utf-8") as f:
        f.write("snr_db,nmse_coarse,nmse_refined\n")

    with torch.no_grad():
        for snr in snrs:
            s_c, s_r, n = 0.0, 0.0, 0

            # 统一步数选择逻辑（与是否use_snr_cond解耦）
            if args.steps_mode == "fixed":
                steps = args.fixed_steps
            else:
                steps = snr_to_steps(snr)

            for xb, _ in loader:
                xb = xb.to(dev)

                out = wrapper(xb, snr_db=snr, gate_mode=args.gate_mode, refine=False)
                coarse = out["coarse"]

                if args.use_snr_cond:
                    snr_t = torch.full((xb.size(0), 1), float(snr), device=dev)
                else:
                    snr_t = None

                refined_raw, r_hat = wrapper.refine(
                    coarse=coarse,
                    snr_db=snr_t,
                    steps=steps,
                    residual_scale=args.residual_scale,
                    deterministic=True
                )

                # 保守融合
                refined = coarse + args.alpha * (refined_raw - coarse)

                s_c += nmse(coarse, xb).item() * xb.size(0)
                s_r += nmse(refined, xb).item() * xb.size(0)
                n += xb.size(0)

            nmse_c = s_c / n
            nmse_r = s_r / n
            print(f"SNR={snr:.1f} | steps={steps} | coarse={nmse_c:.6f} | refined={nmse_r:.6f}")
            with open(csvf, "a", encoding="utf-8") as f:
                f.write(f"{snr},{nmse_c:.8f},{nmse_r:.8f}\n")

    print(f"[Done] csv={csvf}")


if __name__ == "__main__":
    main()
