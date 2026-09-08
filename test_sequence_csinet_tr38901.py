import os
import csv
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.csinet_baseline import CsiNetBaseline
from models.csinet_plus_baseline import CsiNetPlusBaseline



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

    if x.ndim != 5 or x.shape[1:] != (8, 2, 32, 256):
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


@torch.no_grad()
def forward_sequence(model, x_seq, snr, use_awgn):
    """
    x_seq: [B,8,2,32,256]
    """
    b, t, c, h, w = x_seq.shape
    frames = x_seq.reshape(b * t, c, h, w)

    if use_awgn:
        out = model(frames, snr_db=float(snr))
    else:
        out = model(frames, snr_db=None)

    return out.reshape(b, t, c, h, w)


def evaluate_group(
    model,
    loader,
    device,
    snr,
    use_awgn,
    speed_filter=None,
    scenario_filter=None,
):
    model.eval()

    total_nmse = 0.0
    total_tce = 0.0
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

            y = forward_sequence(model, x, snr=snr, use_awgn=use_awgn)

            nmse = nmse_per_sample(y, x)
            tce = temporal_consistency_error(y, x)

            total_nmse += nmse.sum().item()
            total_tce += tce.sum().item()
            total_n += x.size(0)

    if total_n == 0:
        return None

    return {
        "nmse": total_nmse / total_n,
        "tce": total_tce / total_n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--model", type=str, default="csinet", choices=["csinet", "csinet_plus"])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--cr", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--snr-list", type=str, default="0,5,10,15,20")
    parser.add_argument("--no-awgn", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-csv", type=str, default="")
    args = parser.parse_args()

    seed_all(args.seed)

    use_awgn = not args.no_awgn
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snrs = [float(v) for v in args.snr_list.split(",")]

    test_path = os.path.join(args.data_root, "processed", "test_tr38901_seq.pt")
    x, speed, scenario = load_seq_obj(test_path)

    loader = DataLoader(
        TensorDataset(x, speed, scenario),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if args.model == "csinet":
        model = CsiNetBaseline(cr=args.cr, use_awgn=use_awgn).to(device)
    elif args.model == "csinet_plus":
        model = CsiNetPlusBaseline(cr=args.cr, use_awgn=use_awgn).to(device)
    else:
        raise ValueError(args.model)

    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False))
    model.eval()

    tag = "csinet_awgn" if use_awgn else "csinet_noiseless"

    if args.out_csv == "":
        args.out_csv = f"logs/test_sequence_{tag}_tr38901_cr{args.cr}.csv"

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    speeds = [float(v.item()) for v in torch.unique(speed)]
    scenarios = [int(v.item()) for v in torch.unique(scenario)]

    print("=" * 80)
    print("[Test CsiNet-style Sequence Baseline on TR 38.901]")
    print("ckpt:", args.ckpt)
    print("use_awgn:", use_awgn)
    print("cr:", args.cr)
    print("test:", tuple(x.shape))
    print("snrs:", snrs)
    print("device:", device)
    print("=" * 80)

    rows = []

    for snr in snrs:
        overall = evaluate_group(
            model=model,
            loader=loader,
            device=device,
            snr=snr,
            use_awgn=use_awgn,
        )

        print(f"SNR={snr:.1f} | nmse={overall['nmse']:.6f} | tce={overall['tce']:.6f}")

        rows.append({
            "snr_db": snr,
            "group": "overall",
            "group_value": "all",
            "nmse": overall["nmse"],
            "tce": overall["tce"],
        })

        for sp in speeds:
            result = evaluate_group(
                model=model,
                loader=loader,
                device=device,
                snr=snr,
                use_awgn=use_awgn,
                speed_filter=sp,
            )
            print(f"  speed={sp:g} | nmse={result['nmse']:.6f} | tce={result['tce']:.6f}")

            rows.append({
                "snr_db": snr,
                "group": "speed_kmh",
                "group_value": sp,
                "nmse": result["nmse"],
                "tce": result["tce"],
            })

        for sc in scenarios:
            result = evaluate_group(
                model=model,
                loader=loader,
                device=device,
                snr=snr,
                use_awgn=use_awgn,
                scenario_filter=sc,
            )
            print(f"  scenario={sc} | nmse={result['nmse']:.6f} | tce={result['tce']:.6f}")

            rows.append({
                "snr_db": snr,
                "group": "scenario_id",
                "group_value": sc,
                "nmse": result["nmse"],
                "tce": result["tce"],
            })

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["snr_db", "group", "group_value", "nmse", "tce"],
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
