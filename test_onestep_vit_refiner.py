import os
import csv
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.se_jscc_snrprior import SEJSCCSNRPrior
from models.sequence_jscc_wrapper import SequenceJSCCWrapper
from models.onestep_vit_refiner import OneStepViTRefiner


def seed_all(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_seq_obj(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "x" not in obj:
        raise ValueError(f"{path} must be dict with key 'x'")

    x = obj["x"].float()
    speed = obj["speed_kmh"].float()
    scenario = obj["scenario_id"].long()

    if x.shape[1:] != (8, 2, 32, 256):
        raise ValueError(f"Expected [N,8,2,32,256], got {tuple(x.shape)}")

    return x, speed, scenario


def nmse_per_sample(pred, target, eps=1e-12):
    numerator = torch.sum((pred - target) ** 2, dim=(1, 2, 3, 4))
    denominator = torch.sum(target ** 2, dim=(1, 2, 3, 4)) + eps
    return numerator / denominator


def temporal_consistency_error(pred, target, eps=1e-12):
    pred_diff = pred[:, 1:] - pred[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]

    numerator = torch.sum((pred_diff - target_diff) ** 2, dim=(1, 2, 3, 4))
    denominator = torch.sum(target_diff ** 2, dim=(1, 2, 3, 4)) + eps
    return numerator / denominator


def evaluate_group(
    seq_jscc,
    refiner,
    loader,
    device,
    snr,
    use_snr_cond,
    gate_mode,
    alpha,
    guidance_scale,
    t_infer,
    speed_filter=None,
    scenario_filter=None,
):
    total_coarse_nmse = 0.0
    total_refined_nmse = 0.0
    total_coarse_tce = 0.0
    total_refined_tce = 0.0
    total_n = 0

    with torch.no_grad():
        for x, speed, scenario in loader:
            mask = torch.ones(x.size(0), dtype=torch.bool)

            if speed_filter is not None:
                mask = mask & (speed == float(speed_filter))

            if scenario_filter is not None:
                mask = mask & (scenario == int(scenario_filter))

            if mask.sum().item() == 0:
                continue

            x = x[mask].to(device, non_blocking=True)

            coarse = seq_jscc(x, snr_db=float(snr), gate_mode=gate_mode)

            z_init = torch.zeros_like(coarse)
            t = torch.full((x.size(0), 1), float(t_infer), device=device)
            snr_tensor = torch.full((x.size(0), 1), float(snr), device=device)

            out = refiner(
                z_t=z_init,
                coarse=coarse,
                t=t,
                snr_db=snr_tensor if use_snr_cond else None,
            )

            res_full = out["res_full"]
            res_repa = out["res_repa"]

            res_hat = res_full + guidance_scale * (res_full - res_repa)
            refined = coarse + alpha * res_hat

            coarse_nmse = nmse_per_sample(coarse, x)
            refined_nmse = nmse_per_sample(refined, x)

            coarse_tce = temporal_consistency_error(coarse, x)
            refined_tce = temporal_consistency_error(refined, x)

            bs = x.size(0)

            total_coarse_nmse += coarse_nmse.sum().item()
            total_refined_nmse += refined_nmse.sum().item()
            total_coarse_tce += coarse_tce.sum().item()
            total_refined_tce += refined_tce.sum().item()
            total_n += bs

    if total_n == 0:
        return None

    return {
        "coarse_nmse": total_coarse_nmse / total_n,
        "refined_nmse": total_refined_nmse / total_n,
        "coarse_tce": total_coarse_tce / total_n,
        "refined_tce": total_refined_tce / total_n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--base-ckpt", type=str, required=True)
    parser.add_argument("--refiner-ckpt", type=str, required=True)
    parser.add_argument("--cr", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--snr-list", type=str, default="0,5,10,15,20")
    parser.add_argument("--use-snr-cond", action="store_true")
    parser.add_argument("--gate-mode", type=str, default="estimate", choices=["teacher", "estimate"])
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--t-infer", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-csv", type=str, default="")
    args = parser.parse_args()

    seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snrs = [float(v) for v in args.snr_list.split(",")]

    seq_path = os.path.join(args.data_root, "processed", f"{args.split}_tr38901_seq.pt")
    x, speed, scenario = load_seq_obj(seq_path)


    loader = DataLoader(
        TensorDataset(x, speed, scenario),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
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


    refiner.load_state_dict(torch.load(args.refiner_ckpt, map_location=device, weights_only=False))
    refiner.eval()

    tag = "snrcond" if args.use_snr_cond else "nocond"

    if args.out_csv == "":
        args.out_csv = f"logs/test_onestep_video_swin_tr38901_cr{args.cr}_{tag}.csv"

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    speeds = [float(v.item()) for v in torch.unique(speed)]
    scenarios = [int(v.item()) for v in torch.unique(scenario)]

    print("=" * 80)
    print("[Test One-Step Video-Swin Refiner]")
    print("split:", args.split)
    print("data:", tuple(x.shape))
    print("base_ckpt:", args.base_ckpt)
    print("refiner_ckpt:", args.refiner_ckpt)
    print("cr:", args.cr)
    print("use_snr_cond:", args.use_snr_cond)
    print("alpha:", args.alpha)
    print("guidance_scale:", args.guidance_scale)
    print("t_infer:", args.t_infer)
    print("device:", device)
    print("=" * 80)

    rows = []

    for snr in snrs:
        overall = evaluate_group(
            seq_jscc=seq_jscc,
            refiner=refiner,
            loader=loader,
            device=device,
            snr=snr,
            use_snr_cond=args.use_snr_cond,
            gate_mode=args.gate_mode,
            alpha=args.alpha,
            guidance_scale=args.guidance_scale,
            t_infer=args.t_infer,
        )

        print(
            f"SNR={snr:.1f} | "
            f"coarse={overall['coarse_nmse']:.6f} | "
            f"refined={overall['refined_nmse']:.6f} | "
            f"coarse_tce={overall['coarse_tce']:.6f} | "
            f"refined_tce={overall['refined_tce']:.6f}"
        )

        rows.append({
            "snr_db": snr,
            "group": "overall",
            "group_value": "all",
            **overall,
        })

        for sp in speeds:
            result = evaluate_group(
                seq_jscc=seq_jscc,
                refiner=refiner,
                loader=loader,
                device=device,
                snr=snr,
                use_snr_cond=args.use_snr_cond,
                gate_mode=args.gate_mode,
                alpha=args.alpha,
                guidance_scale=args.guidance_scale,
                t_infer=args.t_infer,
                speed_filter=sp,
            )
            print(
                f"  speed={sp:g} | "
                f"coarse={result['coarse_nmse']:.6f} | "
                f"refined={result['refined_nmse']:.6f} | "
                f"coarse_tce={result['coarse_tce']:.6f} | "
                f"refined_tce={result['refined_tce']:.6f}"
            )

            rows.append({
                "snr_db": snr,
                "group": "speed_kmh",
                "group_value": sp,
                **result,
            })

        for sc in scenarios:
            result = evaluate_group(
                seq_jscc=seq_jscc,
                refiner=refiner,
                loader=loader,
                device=device,
                snr=snr,
                use_snr_cond=args.use_snr_cond,
                gate_mode=args.gate_mode,
                alpha=args.alpha,
                guidance_scale=args.guidance_scale,
                t_infer=args.t_infer,
                scenario_filter=sc,
            )
            print(
                f"  scenario={sc} | "
                f"coarse={result['coarse_nmse']:.6f} | "
                f"refined={result['refined_nmse']:.6f} | "
                f"coarse_tce={result['coarse_tce']:.6f} | "
                f"refined_tce={result['refined_tce']:.6f}"
            )

            rows.append({
                "snr_db": snr,
                "group": "scenario_id",
                "group_value": sc,
                **result,
            })

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "snr_db",
                "group",
                "group_value",
                "coarse_nmse",
                "refined_nmse",
                "coarse_tce",
                "refined_tce",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("=" * 80)
    print("[Done]")
    print("csv:", args.out_csv)
    print("=" * 80)


if __name__ == "__main__":
    main()
