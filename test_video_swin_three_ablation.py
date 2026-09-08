#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strict evaluation for three Video-Swin component ablations.

Variants:
- full:
  SNR conditioning enabled, lambda_repa=0.5
- no_snr:
  SNR conditioning disabled, lambda_repa=0.5
- no_aux:
  SNR conditioning enabled, lambda_repa=0.0

Shared final inference protocol:
- alpha=0.9
- guidance=0.0
- only res_full is used
- t_infer=0.5
- z_t=zeros
- JSCC gate_mode=estimate

For each batch and SNR, the coarse JSCC reconstruction is computed only once
and reused by all three variants. Therefore, all variants use exactly the same
test samples and exactly the same channel-noise realizations.
"""

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.se_jscc_snrprior import SEJSCCSNRPrior
from models.sequence_jscc_wrapper import SequenceJSCCWrapper
from models.onestep_video_swin_refiner import OneStepVideoSwinRefiner


EXPECTED_SHAPE = (8, 2, 32, 256)

SNR_LIST = (
    0.0,
    5.0,
    10.0,
    15.0,
    20.0,
)

ALPHA = 0.9
GUIDANCE_SCALE = 0.0
T_INFER = 0.5
GATE_MODE = "estimate"

SCENARIO_NAMES = {
    0: "UMa",
    1: "UMi",
}


@dataclass(frozen=True)
class Variant:
    name: str
    display_name: str
    use_snr_cond: bool
    lambda_repa: float
    checkpoint: Path


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_channel_seed(
    seed: int,
    device: torch.device,
) -> None:
    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_seq_obj(
    path: Path,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = (
        "x",
        "speed_kmh",
        "scenario_id",
    )

    if not isinstance(obj, dict):
        raise ValueError(
            f"{path} must contain a dictionary."
        )

    for key in required_keys:
        if key not in obj:
            raise ValueError(
                f"{path} is missing key '{key}'."
            )

    x = obj["x"].float()
    speed = obj["speed_kmh"].float()
    scenario = obj["scenario_id"].long()

    if (
        x.ndim != 5
        or tuple(x.shape[1:]) != EXPECTED_SHAPE
    ):
        raise ValueError(
            f"Expected [N,8,2,32,256], "
            f"got {tuple(x.shape)}."
        )

    if speed.ndim != 1:
        raise ValueError(
            "speed_kmh must be one-dimensional."
        )

    if scenario.ndim != 1:
        raise ValueError(
            "scenario_id must be one-dimensional."
        )

    if len(speed) != len(x):
        raise ValueError(
            "speed_kmh length does not match x."
        )

    if len(scenario) != len(x):
        raise ValueError(
            "scenario_id length does not match x."
        )

    if not torch.isfinite(x).all():
        raise ValueError(
            f"Non-finite CSI values found in {path}."
        )

    if not torch.isfinite(speed).all():
        raise ValueError(
            f"Non-finite speed values found in {path}."
        )

    return x, speed, scenario


def build_loader(
    x: torch.Tensor,
    speed: torch.Tensor,
    scenario: torch.Tensor,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        TensorDataset(
            x,
            speed,
            scenario,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=(
            seed_worker
            if num_workers > 0
            else None
        ),
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def build_refiner(
    use_snr_cond: bool,
    device: torch.device,
) -> OneStepVideoSwinRefiner:
    return OneStepVideoSwinRefiner(
        in_ch=2,
        cond_dim=128,
        dim=48,
        depth=6,
        num_heads=4,
        window_size=(2, 4, 8),
        use_snr_cond=use_snr_cond,
    ).to(device)


def nmse_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    numerator = torch.sum(
        (pred - target) ** 2,
        dim=(1, 2, 3, 4),
    )

    denominator = (
        torch.sum(
            target ** 2,
            dim=(1, 2, 3, 4),
        )
        + eps
    )

    return numerator / denominator


def tce_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    pred_diff = (
        pred[:, 1:]
        - pred[:, :-1]
    )

    target_diff = (
        target[:, 1:]
        - target[:, :-1]
    )

    numerator = torch.sum(
        (pred_diff - target_diff) ** 2,
        dim=(1, 2, 3, 4),
    )

    denominator = (
        torch.sum(
            target_diff ** 2,
            dim=(1, 2, 3, 4),
        )
        + eps
    )

    return numerator / denominator


def new_stat() -> Dict[str, float]:
    return {
        "coarse_nmse_sum": 0.0,
        "refined_nmse_sum": 0.0,
        "coarse_tce_sum": 0.0,
        "refined_tce_sum": 0.0,
        "count": 0.0,
    }


def create_stats(
    variants: List[Variant],
    speeds: List[float],
    scenarios: List[int],
) -> Dict[
    str,
    Dict[
        Tuple[str, str],
        Dict[str, float],
    ],
]:
    stats = {}

    for variant in variants:
        groups = {
            ("overall", "all"): new_stat()
        }

        for speed in speeds:
            groups[
                ("speed_kmh", f"{speed:g}")
            ] = new_stat()

        for scenario in scenarios:
            groups[
                ("scenario_id", str(scenario))
            ] = new_stat()

        stats[variant.name] = groups

    return stats


def add_stat(
    stat: Dict[str, float],
    mask: torch.Tensor,
    coarse_nmse: torch.Tensor,
    refined_nmse: torch.Tensor,
    coarse_tce: torch.Tensor,
    refined_tce: torch.Tensor,
) -> None:
    mask = mask.bool()

    count = int(
        mask.sum().item()
    )

    if count == 0:
        return

    stat["coarse_nmse_sum"] += float(
        coarse_nmse[mask].sum().item()
    )

    stat["refined_nmse_sum"] += float(
        refined_nmse[mask].sum().item()
    )

    stat["coarse_tce_sum"] += float(
        coarse_tce[mask].sum().item()
    )

    stat["refined_tce_sum"] += float(
        refined_tce[mask].sum().item()
    )

    stat["count"] += count


def add_all_groups(
    groups: Dict[
        Tuple[str, str],
        Dict[str, float],
    ],
    speed: torch.Tensor,
    scenario: torch.Tensor,
    speeds: List[float],
    scenarios: List[int],
    coarse_nmse: torch.Tensor,
    refined_nmse: torch.Tensor,
    coarse_tce: torch.Tensor,
    refined_tce: torch.Tensor,
) -> None:
    all_mask = torch.ones(
        len(speed),
        dtype=torch.bool,
    )

    add_stat(
        stat=groups[("overall", "all")],
        mask=all_mask,
        coarse_nmse=coarse_nmse,
        refined_nmse=refined_nmse,
        coarse_tce=coarse_tce,
        refined_tce=refined_tce,
    )

    for speed_value in speeds:
        add_stat(
            stat=groups[
                (
                    "speed_kmh",
                    f"{speed_value:g}",
                )
            ],
            mask=speed == speed_value,
            coarse_nmse=coarse_nmse,
            refined_nmse=refined_nmse,
            coarse_tce=coarse_tce,
            refined_tce=refined_tce,
        )

    for scenario_value in scenarios:
        add_stat(
            stat=groups[
                (
                    "scenario_id",
                    str(scenario_value),
                )
            ],
            mask=scenario == scenario_value,
            coarse_nmse=coarse_nmse,
            refined_nmse=refined_nmse,
            coarse_tce=coarse_tce,
            refined_tce=refined_tce,
        )


def finalize_stat(
    stat: Dict[str, float],
) -> Dict[str, float]:
    count = int(
        stat["count"]
    )

    if count <= 0:
        raise RuntimeError(
            "A metric group contains no samples."
        )

    return {
        "count": count,
        "coarse_nmse": (
            stat["coarse_nmse_sum"]
            / count
        ),
        "refined_nmse": (
            stat["refined_nmse_sum"]
            / count
        ),
        "coarse_tce": (
            stat["coarse_tce_sum"]
            / count
        ),
        "refined_tce": (
            stat["refined_tce_sum"]
            / count
        ),
    }


@torch.no_grad()
def evaluate_one_snr(
    seq_jscc: SequenceJSCCWrapper,
    refiners: Dict[
        str,
        OneStepVideoSwinRefiner,
    ],
    variants: List[Variant],
    loader: DataLoader,
    device: torch.device,
    snr: float,
    snr_index: int,
    speeds: List[float],
    scenarios: List[int],
    seed: int,
) -> Dict[
    str,
    Dict[
        Tuple[str, str],
        Dict[str, float],
    ],
]:
    """
    Evaluate every variant in one shared pass.

    The coarse reconstruction is produced once for each batch and then reused
    by all three refiners.
    """
    stats = create_stats(
        variants=variants,
        speeds=speeds,
        scenarios=scenarios,
    )

    for refiner in refiners.values():
        refiner.eval()

    for batch_index, (
        x,
        speed,
        scenario,
    ) in enumerate(loader):
        x = x.to(
            device,
            non_blocking=True,
        )

        # Preserve the controlled channel-seed formula used by the existing
        # ablation test protocol.
        channel_seed = (
            seed
            + 1_000_000
            + snr_index * 100_000
            + batch_index
        )

        set_channel_seed(
            channel_seed,
            device,
        )

        # Compute coarse reconstruction once.
        coarse = seq_jscc(
            x,
            snr_db=float(snr),
            gate_mode=GATE_MODE,
        )

        z_t = torch.zeros_like(
            coarse
        )

        t = torch.full(
            (x.size(0), 1),
            T_INFER,
            device=device,
            dtype=x.dtype,
        )

        coarse_nmse = nmse_per_sample(
            coarse,
            x,
        ).cpu()

        coarse_tce = tce_per_sample(
            coarse,
            x,
        ).cpu()

        speed_cpu = speed.cpu()
        scenario_cpu = scenario.cpu()

        for variant in variants:
            snr_tensor: Optional[torch.Tensor]

            if variant.use_snr_cond:
                snr_tensor = torch.full(
                    (x.size(0), 1),
                    float(snr),
                    device=device,
                    dtype=x.dtype,
                )
            else:
                snr_tensor = None

            output = refiners[
                variant.name
            ](
                z_t=z_t,
                coarse=coarse,
                t=t,
                snr_db=snr_tensor,
            )

            if not isinstance(output, dict):
                raise RuntimeError(
                    f"{variant.name} must return a dict."
                )

            if "res_full" not in output:
                raise RuntimeError(
                    f"{variant.name} output is missing "
                    "'res_full'."
                )

            # guidance_scale=0:
            # the auxiliary head does not participate in reconstruction.
            refined = (
                coarse
                + ALPHA * output["res_full"]
            )

            refined_nmse = nmse_per_sample(
                refined,
                x,
            ).cpu()

            refined_tce = tce_per_sample(
                refined,
                x,
            ).cpu()

            add_all_groups(
                groups=stats[variant.name],
                speed=speed_cpu,
                scenario=scenario_cpu,
                speeds=speeds,
                scenarios=scenarios,
                coarse_nmse=coarse_nmse,
                refined_nmse=refined_nmse,
                coarse_tce=coarse_tce,
                refined_tce=refined_tce,
            )

    results = {}

    for variant in variants:
        results[variant.name] = {
            group_key: finalize_stat(stat)
            for group_key, stat in (
                stats[variant.name].items()
            )
        }

    return results


def group_display_name(
    group: str,
    value: str,
) -> str:
    if group == "scenario_id":
        scenario_id = int(value)

        return SCENARIO_NAMES.get(
            scenario_id,
            f"scenario_{scenario_id}",
        )

    if group == "speed_kmh":
        return f"{value} km/h"

    return "Overall"


def write_csv(
    path: Path,
    rows: List[Dict[str, object]],
) -> None:
    if not rows:
        raise ValueError(
            f"No rows available for {path}."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test the strict three-way Video-Swin "
            "component ablation."
        )
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=[
            "val",
            "test",
        ],
    )

    parser.add_argument(
        "--base-ckpt",
        type=str,
        default=(
            "checkpoints/"
            "final_unified_tr38901_cr4.pt"
        ),
    )

    parser.add_argument(
        "--full-ckpt",
        type=str,
        default=(
            "checkpoints/"
            "onestep_video_swin_tr38901_cr4"
            "_ablation_full.pt"
        ),
    )

    parser.add_argument(
        "--no-snr-ckpt",
        type=str,
        default=(
            "checkpoints/"
            "onestep_video_swin_tr38901_cr4"
            "_ablation_no_snr.pt"
        ),
    )

    parser.add_argument(
        "--no-aux-ckpt",
        type=str,
        default=(
            "checkpoints/"
            "onestep_video_swin_tr38901_cr4"
            "_ablation_no_aux.pt"
        ),
    )

    parser.add_argument(
        "--cr",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--out-csv",
        type=str,
        default=(
            "logs/"
            "test_video_swin_three_ablation"
            "_alpha09_g00.csv"
        ),
    )

    parser.add_argument(
        "--summary-csv",
        type=str,
        default=(
            "logs/"
            "test_video_swin_three_ablation"
            "_summary.csv"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.cr != 4:
        raise ValueError(
            "This ablation protocol is fixed to CR=4."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size must be positive."
        )

    seed_all(args.seed)

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        print(
            "[Warning] CUDA is unavailable; using CPU."
        )
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    sequence_path = (
        Path(args.data_root)
        / "processed"
        / f"{args.split}_tr38901_seq.pt"
    )

    x, speed, scenario = load_seq_obj(
        sequence_path
    )

    loader = build_loader(
        x=x,
        speed=speed,
        scenario=scenario,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed + 70,
        pin_memory=device.type == "cuda",
    )

    base_checkpoint = Path(
        args.base_ckpt
    )

    if not base_checkpoint.is_file():
        raise FileNotFoundError(
            f"Base checkpoint not found: "
            f"{base_checkpoint}"
        )

    base = SEJSCCSNRPrior(
        cr=args.cr
    ).to(device)

    base.load_state_dict(
        torch.load(
            base_checkpoint,
            map_location=device,
            weights_only=False,
        ),
        strict=True,
    )

    base.eval()

    for parameter in base.parameters():
        parameter.requires_grad_(False)

    seq_jscc = SequenceJSCCWrapper(
        base
    ).to(device)

    seq_jscc.eval()

    for parameter in seq_jscc.parameters():
        parameter.requires_grad_(False)

    variants = [
        Variant(
            name="full",
            display_name="Full model",
            use_snr_cond=True,
            lambda_repa=0.5,
            checkpoint=Path(args.full_ckpt),
        ),
        Variant(
            name="no_snr",
            display_name="w/o SNR conditioning",
            use_snr_cond=False,
            lambda_repa=0.5,
            checkpoint=Path(args.no_snr_ckpt),
        ),
        Variant(
            name="no_aux",
            display_name="w/o auxiliary supervision",
            use_snr_cond=True,
            lambda_repa=0.0,
            checkpoint=Path(args.no_aux_ckpt),
        ),
    ]

    refiners: Dict[
        str,
        OneStepVideoSwinRefiner,
    ] = {}

    for variant in variants:
        if not variant.checkpoint.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{variant.checkpoint}"
            )

        model = build_refiner(
            use_snr_cond=variant.use_snr_cond,
            device=device,
        )

        model.load_state_dict(
            torch.load(
                variant.checkpoint,
                map_location=device,
                weights_only=False,
            ),
            strict=True,
        )

        model.eval()

        refiners[variant.name] = model

    speeds = sorted(
        float(value.item())
        for value in torch.unique(speed)
    )

    scenarios = sorted(
        int(value.item())
        for value in torch.unique(scenario)
    )

    print("=" * 100)
    print(
        "[Three-Way Video-Swin Ablation Test]"
    )
    print(f"split: {args.split}")
    print(f"data: {tuple(x.shape)}")
    print(f"base_ckpt: {base_checkpoint}")
    print(f"alpha: {ALPHA}")
    print(
        f"guidance_scale: {GUIDANCE_SCALE}"
    )
    print(f"t_infer: {T_INFER}")
    print("z_t: zeros")
    print(f"gate_mode: {GATE_MODE}")
    print(f"snrs: {list(SNR_LIST)}")
    print(f"speed_groups: {speeds}")
    print(f"scenario_groups: {scenarios}")
    print(f"device: {device}")
    print(
        "Each batch uses one shared coarse "
        "reconstruction for all variants."
    )
    print("=" * 100)

    detailed_rows: List[
        Dict[str, object]
    ] = []

    overall_results: Dict[
        str,
        List[Dict[str, float]],
    ] = {
        variant.name: []
        for variant in variants
    }

    for snr_index, snr in enumerate(
        SNR_LIST
    ):
        results = evaluate_one_snr(
            seq_jscc=seq_jscc,
            refiners=refiners,
            variants=variants,
            loader=loader,
            device=device,
            snr=snr,
            snr_index=snr_index,
            speeds=speeds,
            scenarios=scenarios,
            seed=args.seed,
        )

        print(
            f"\nSNR={snr:g} dB"
        )

        for variant in variants:
            overall = results[
                variant.name
            ][
                ("overall", "all")
            ]

            overall_results[
                variant.name
            ].append(overall)

            print(
                f"{variant.display_name:<30} | "
                f"coarse_nmse="
                f"{overall['coarse_nmse']:.6f} | "
                f"refined_nmse="
                f"{overall['refined_nmse']:.6f} | "
                f"coarse_tce="
                f"{overall['coarse_tce']:.6f} | "
                f"refined_tce="
                f"{overall['refined_tce']:.6f}"
            )

            for (
                group,
                group_value,
            ), metric in results[
                variant.name
            ].items():
                detailed_rows.append(
                    {
                        "variant": variant.name,
                        "display_name": (
                            variant.display_name
                        ),
                        "use_snr_cond": (
                            variant.use_snr_cond
                        ),
                        "auxiliary_supervision": (
                            variant.name != "no_aux"
                        ),
                        "lambda_repa": (
                            variant.lambda_repa
                        ),
                        "snr_db": snr,
                        "group": group,
                        "group_value": group_value,
                        "group_display_name": (
                            group_display_name(
                                group,
                                group_value,
                            )
                        ),
                        "count": int(
                            metric["count"]
                        ),
                        "alpha": ALPHA,
                        "guidance_scale": (
                            GUIDANCE_SCALE
                        ),
                        "t_infer": T_INFER,
                        "z_t": "zeros",
                        "gate_mode": GATE_MODE,
                        "coarse_nmse": (
                            metric["coarse_nmse"]
                        ),
                        "refined_nmse": (
                            metric["refined_nmse"]
                        ),
                        "coarse_tce": (
                            metric["coarse_tce"]
                        ),
                        "refined_tce": (
                            metric["refined_tce"]
                        ),
                    }
                )

    summary_rows: List[
        Dict[str, object]
    ] = []

    print("\n" + "=" * 100)
    print("[Five-SNR Average]")

    for variant in variants:
        values = overall_results[
            variant.name
        ]

        mean_coarse_nmse = (
            sum(
                value["coarse_nmse"]
                for value in values
            )
            / len(values)
        )

        mean_refined_nmse = (
            sum(
                value["refined_nmse"]
                for value in values
            )
            / len(values)
        )

        mean_coarse_tce = (
            sum(
                value["coarse_tce"]
                for value in values
            )
            / len(values)
        )

        mean_refined_tce = (
            sum(
                value["refined_tce"]
                for value in values
            )
            / len(values)
        )

        summary_rows.append(
            {
                "variant": variant.name,
                "display_name": (
                    variant.display_name
                ),
                "use_snr_cond": (
                    variant.use_snr_cond
                ),
                "auxiliary_supervision": (
                    variant.name != "no_aux"
                ),
                "lambda_repa": (
                    variant.lambda_repa
                ),
                "alpha": ALPHA,
                "guidance_scale": (
                    GUIDANCE_SCALE
                ),
                "t_infer": T_INFER,
                "gate_mode": GATE_MODE,
                "mean_coarse_nmse_over_5_snrs": (
                    mean_coarse_nmse
                ),
                "mean_refined_nmse_over_5_snrs": (
                    mean_refined_nmse
                ),
                "mean_coarse_tce_over_5_snrs": (
                    mean_coarse_tce
                ),
                "mean_refined_tce_over_5_snrs": (
                    mean_refined_tce
                ),
            }
        )

        print(
            f"{variant.display_name:<30} | "
            f"mean_refined_nmse="
            f"{mean_refined_nmse:.6f} | "
            f"mean_refined_tce="
            f"{mean_refined_tce:.6f}"
        )

    write_csv(
        Path(args.out_csv),
        detailed_rows,
    )

    write_csv(
        Path(args.summary_csv),
        summary_rows,
    )

    print("=" * 100)
    print("[Done]")
    print(
        f"detailed_csv: {args.out_csv}"
    )
    print(
        f"summary_csv: {args.summary_csv}"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
