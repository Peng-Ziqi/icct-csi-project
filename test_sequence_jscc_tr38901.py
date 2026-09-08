import os
import argparse
import csv
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.se_jscc_snrprior import SEJSCCSNRPrior
from models.sequence_jscc_wrapper import SequenceJSCCWrapper


def seed_all(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_seq_obj(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, got {type(obj)}")
    if "x" not in obj:
        raise KeyError(f"Missing key 'x' in {path}")

    x = obj["x"].float()
    if x.ndim != 5:
        raise ValueError(f"Expected x [N,T,2,32,256], got {tuple(x.shape)}")

    if x.shape[2:] != (2, 32, 256):
        raise ValueError(f"Unexpected CSI shape {tuple(x.shape)}, expected [N,T,2,32,256]")

    speed = obj.get("speed_kmh", torch.zeros(x.shape[0])).float()
    scenario = obj.get("scenario_id", torch.zeros(x.shape[0], dtype=torch.long)).long()

    return x, speed, scenario


def nmse_per_sample(pred, target, eps=1e-12):
    """
    pred/target: [B,T,2,32,256]
    return: [B]
    """
    numerator = torch.sum((pred - target) ** 2, dim=(1, 2, 3, 4))
    denominator = torch.sum(target ** 2, dim=(1, 2, 3, 4)) + eps
    return numerator / denominator


def temporal_consistency_error(pred, target, eps=1e-12):
    """
    Optional temporal metric.

    Measures whether inter-frame differences are reconstructed consistently.

    pred/target: [B,T,2,32,256]
    return scalar
    """
    pred_diff = pred[:, 1:] - pred[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]

    numerator = torch.sum((pred_diff - target_diff) ** 2, dim=(1, 2, 3, 4))
    denominator = torch.sum(target_diff ** 2, dim=(1, 2, 3, 4)) + eps

    return torch.mean(numerator / denominator)


def evaluate_group(
    model,
    loader,
    device,
    snr,
    gate_mode,
    speed_filter=None,
    scenario_filter=None,
):
    model.eval()

    total_nmse = 0.0
    total_tce = 0.0
    total_n = 0

    with torch.no_grad():
        for x, speed, scenario in loader:
            mask = torch.ones(x.shape[0], dtype=torch.bool)

            if speed_filter is not None:
                mask = mask & (speed == float(speed_filter))

            if scenario_filter is not None:
                mask = mask & (scenario == int(scenario_filter))

            if mask.sum().item() == 0:
                continue

            x = x[mask].to(device, non_blocking=True)
            y = model(x, snr_db=float(snr), gate_mode=gate_mode)

            nmse_vals = nmse_per_sample(y, x)
            tce_val = temporal_consistency_error(y, x).item()

            total_nmse += nmse_vals.sum().item()
            total_tce += tce_val * x.shape[0]
            total_n += x.shape[0]

    if total_n == 0:
        return None, None

    return total_nmse / total_n, total_tce / total_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--cr", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--snr-list", type=str, default="0,5,10,15,20")
    parser.add_argument("--gate-mode", type=str, default="estimate", choices=["teacher", "estimate"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-csv", type=str, default="")
    args = parser.parse_args()

    seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snrs = [float(v) for v in args.snr_list.split(",")]

    test_path = os.path.join(args.data_root, "processed", "test_tr38901_seq.pt")
    x, speed, scenario = load_seq_obj(test_path)

    print("=" * 80)
    print("[Evaluate Sequence JSCC Baseline on Standard TR 38.901 Dataset]")
    print("ckpt:", args.ckpt)
    print("cr:", args.cr)
    print("device:", device)
    print("test x:", tuple(x.shape))
    print("speeds:", torch.unique(speed))
    print("scenarios:", torch.unique(scenario))
    print("snrs:", snrs)
    print("gate_mode:", args.gate_mode)
    print("=" * 80)

    loader = DataLoader(
        TensorDataset(x, speed, scenario),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    base = SEJSCCSNRPrior(cr=args.cr).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    base.load_state_dict(state)
    base.eval()

    model = SequenceJSCCWrapper(base).to(device)
    model.eval()

    if args.out_csv == "":
        args.out_csv = f"logs/test_sequence_jscc_tr38901_cr{args.cr}.csv"

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    rows = []
    speeds = [float(v.item()) for v in torch.unique(speed)]
    scenarios = [int(v.item()) for v in torch.unique(scenario)]

    for snr in snrs:
        overall_nmse, overall_tce = evaluate_group(
            model=model,
            loader=loader,
            device=device,
            snr=snr,
            gate_mode=args.gate_mode,
        )

        print(f"SNR={snr:.1f} | overall_nmse={overall_nmse:.6f} | overall_tce={overall_tce:.6f}")

        rows.append({
            "snr_db": snr,
            "group": "overall",
            "group_value": "all",
            "nmse": overall_nmse,
            "temporal_consistency_error": overall_tce,
        })

        for sp in speeds:
            val_nmse, val_tce = evaluate_group(
                model=model,
                loader=loader,
                device=device,
                snr=snr,
                gate_mode=args.gate_mode,
                speed_filter=sp,
            )
            print(f"  speed={sp:g} km/h | nmse={val_nmse:.6f} | tce={val_tce:.6f}")

            rows.append({
                "snr_db": snr,
                "group": "speed_kmh",
                "group_value": sp,
                "nmse": val_nmse,
                "temporal_consistency_error": val_tce,
            })

        for sc in scenarios:
            val_nmse, val_tce = evaluate_group(
                model=model,
                loader=loader,
                device=device,
                snr=snr,
                gate_mode=args.gate_mode,
                scenario_filter=sc,
            )
            print(f"  scenario_id={sc} | nmse={val_nmse:.6f} | tce={val_tce:.6f}")

            rows.append({
                "snr_db": snr,
                "group": "scenario_id",
                "group_value": sc,
                "nmse": val_nmse,
                "temporal_consistency_error": val_tce,
            })

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "snr_db",
                "group",
                "group_value",
                "nmse",
                "temporal_consistency_error",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("=" * 80)
    print("[Done]")
    print("CSV:", args.out_csv)
    print("=" * 80)


if __name__ == "__main__":
    main()
