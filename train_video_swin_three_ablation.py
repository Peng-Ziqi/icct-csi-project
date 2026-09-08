#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strict three-way component ablation for the one-step Video-Swin CSI refiner.

Variants:
1) full:
   use_snr_cond=True,  lambda_repa=0.5
2) no_snr:
   use_snr_cond=False, lambda_repa=0.5
3) no_aux:
   use_snr_cond=True,  lambda_repa=0.0

All other settings exactly follow train_onestep_video_swin_refiner.py:
- frozen spatial JSCC backbone, CR=4
- epochs=300, batch_size=8, lr=1e-4
- AdamW, weight_decay=0.01
- gradient clipping max_norm=1.0
- SNRs=[0,5,10,15,20] dB
- lambda_rec=0.2, lambda_v=0.0
- logit-normal t sampling
- z_t=t*residual+(1-t)*epsilon
- training and validation use the same JSCC gate mode

Fairness controls:
- same data order for all variants
- same SNR sequence for all variants
- same coarse-JSCC channel-noise realization for all variants
- same t and epsilon sequence for all variants
- no_aux keeps the auxiliary head; only lambda_repa is set to zero
"""

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.se_jscc_snrprior import SEJSCCSNRPrior
from models.sequence_jscc_wrapper import SequenceJSCCWrapper
from models.onestep_video_swin_refiner import OneStepVideoSwinRefiner


EXPECTED_SHAPE = (8, 2, 32, 256)
SNR_LIST = (0.0, 5.0, 10.0, 15.0, 20.0)


@dataclass(frozen=True)
class VariantConfig:
    name: str
    display_name: str
    use_snr_cond: bool
    lambda_repa: float


VARIANTS = {
    "full": VariantConfig(
        name="full",
        display_name="Full model",
        use_snr_cond=True,
        lambda_repa=0.5,
    ),
    "no_snr": VariantConfig(
        name="no_snr",
        display_name="w/o SNR conditioning",
        use_snr_cond=False,
        lambda_repa=0.5,
    ),
    "no_aux": VariantConfig(
        name="no_aux",
        display_name="w/o auxiliary supervision",
        use_snr_cond=True,
        lambda_repa=0.0,
    ),
}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_channel_seed(seed: int, device: torch.device) -> None:
    """
    Fix the random realization used by the frozen spatial JSCC backbone.

    The JSCC channel simulation uses PyTorch's global random generator.
    Resetting it for each batch guarantees that all ablation variants receive
    exactly the same coarse reconstruction.
    """
    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_seq_x(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    obj = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(obj, dict) or "x" not in obj:
        raise ValueError(
            f"{path} must be a dict containing key 'x'."
        )

    x = obj["x"].float()

    if x.ndim != 5 or tuple(x.shape[1:]) != EXPECTED_SHAPE:
        raise ValueError(
            f"Expected [N,8,2,32,256], "
            f"got {tuple(x.shape)} from {path}."
        )

    if not torch.isfinite(x).all():
        raise ValueError(
            f"Non-finite values found in {path}."
        )

    return x


def build_loader(
    x: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    """
    Recreating this loader with the same seed produces the same shuffled
    sample order for every ablation variant.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        TensorDataset(x),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker if num_workers > 0 else None,
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


def sample_logit_normal(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    mean: float = 0.0,
    std: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Exact t distribution used by the original main-model training script.
    """
    z = torch.randn(
        (batch_size, 1),
        device=device,
        dtype=dtype,
        generator=generator,
    )

    z = z * std + mean
    t = torch.sigmoid(z)
    t = torch.clamp(t, eps, 1.0 - eps)

    return t


def expand_t(
    t: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    return t.view(
        x.shape[0],
        *([1] * (x.ndim - 1)),
    )


def compute_batch_losses(
    x: torch.Tensor,
    seq_jscc: SequenceJSCCWrapper,
    refiner: OneStepVideoSwinRefiner,
    snr: float,
    use_snr_cond: bool,
    lambda_repa: float,
    lambda_rec: float,
    lambda_v: float,
    gate_mode: str,
    channel_seed: int,
    residual_generator: torch.Generator,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Execute the original one-step residual training objective for one batch.
    """

    # Guarantee identical coarse JSCC reconstruction across variants.
    set_channel_seed(channel_seed, device)

    with torch.no_grad():
        coarse = seq_jscc(
            x,
            snr_db=float(snr),
            gate_mode=gate_mode,
        )

    residual_gt = x - coarse

    # Original logit-normal interpolation-state sampling.
    t = sample_logit_normal(
        batch_size=x.size(0),
        device=device,
        dtype=x.dtype,
        generator=residual_generator,
    )

    t_view = expand_t(t, residual_gt)

    epsilon = torch.randn(
        residual_gt.shape,
        device=device,
        dtype=residual_gt.dtype,
        generator=residual_generator,
    )

    z_t = (
        t_view * residual_gt
        + (1.0 - t_view) * epsilon
    )

    snr_tensor: Optional[torch.Tensor]

    if use_snr_cond:
        snr_tensor = torch.full(
            (x.size(0), 1),
            float(snr),
            device=device,
            dtype=x.dtype,
        )
    else:
        snr_tensor = None

    output = refiner(
        z_t=z_t,
        coarse=coarse,
        t=t,
        snr_db=snr_tensor,
    )

    if not isinstance(output, dict):
        raise RuntimeError(
            "OneStepVideoSwinRefiner must return a dict."
        )

    if "res_full" not in output:
        raise RuntimeError(
            "Refiner output is missing 'res_full'."
        )

    if "res_repa" not in output:
        raise RuntimeError(
            "Refiner output is missing 'res_repa'."
        )

    res_full = output["res_full"]
    res_repa = output["res_repa"]

    if res_full.shape != residual_gt.shape:
        raise RuntimeError(
            f"res_full shape {tuple(res_full.shape)} does not match "
            f"target shape {tuple(residual_gt.shape)}."
        )

    if res_repa.shape != residual_gt.shape:
        raise RuntimeError(
            f"res_repa shape {tuple(res_repa.shape)} does not match "
            f"target shape {tuple(residual_gt.shape)}."
        )

    loss_full = F.mse_loss(
        res_full,
        residual_gt,
    )

    loss_repa = F.mse_loss(
        res_repa,
        residual_gt,
    )

    loss_rec = F.mse_loss(
        coarse + res_full,
        x,
    )

    if lambda_v > 0.0:
        denom = torch.clamp(
            1.0 - t_view,
            min=1e-4,
        )

        v_gt = (
            residual_gt - z_t
        ) / denom

        v_pred = (
            res_full - z_t
        ) / denom

        loss_v = F.mse_loss(
            v_pred,
            v_gt,
        )
    else:
        loss_v = torch.zeros(
            (),
            device=device,
            dtype=x.dtype,
        )

    loss_total = (
        loss_full
        + lambda_repa * loss_repa
        + lambda_rec * loss_rec
        + lambda_v * loss_v
    )

    return {
        "total": loss_total,
        "full": loss_full,
        "repa": loss_repa,
        "rec": loss_rec,
        "v": loss_v,
    }


def empty_totals() -> Dict[str, float]:
    return {
        "total": 0.0,
        "full": 0.0,
        "repa": 0.0,
        "rec": 0.0,
        "v": 0.0,
    }


def update_totals(
    totals: Dict[str, float],
    losses: Dict[str, torch.Tensor],
    batch_size: int,
) -> None:
    for key in totals:
        totals[key] += (
            float(losses[key].detach().item())
            * batch_size
        )


def average_totals(
    totals: Dict[str, float],
    count: int,
) -> Dict[str, float]:
    if count <= 0:
        raise RuntimeError(
            "No samples were processed."
        )

    return {
        key: value / count
        for key, value in totals.items()
    }


@torch.no_grad()
def validate_one_epoch(
    refiner: OneStepVideoSwinRefiner,
    seq_jscc: SequenceJSCCWrapper,
    loader: DataLoader,
    config: VariantConfig,
    epoch: int,
    args: argparse.Namespace,
    device: torch.device,
    snr_rng: random.Random,
    residual_generator: torch.Generator,
) -> Dict[str, float]:
    """
    Validation follows the same stochastic residual objective and the same
    JSCC gate mode as the original main-model training script.
    """
    refiner.eval()

    totals = empty_totals()
    total_n = 0

    for batch_index, (x,) in enumerate(loader):
        x = x.to(
            device,
            non_blocking=True,
        )

        snr = snr_rng.choice(SNR_LIST)

        channel_seed = (
            args.seed
            + 2_000_000
            + epoch * 100_000
            + batch_index
        )

        losses = compute_batch_losses(
            x=x,
            seq_jscc=seq_jscc,
            refiner=refiner,
            snr=snr,
            use_snr_cond=config.use_snr_cond,
            lambda_repa=config.lambda_repa,
            lambda_rec=args.lambda_rec,
            lambda_v=args.lambda_v,
            gate_mode=args.train_gate_mode,
            channel_seed=channel_seed,
            residual_generator=residual_generator,
            device=device,
        )

        batch_size = x.size(0)

        update_totals(
            totals,
            losses,
            batch_size,
        )

        total_n += batch_size

    return average_totals(
        totals,
        total_n,
    )


def write_history(
    path: Path,
    rows: List[Dict[str, float]],
) -> None:
    if not rows:
        return

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


def log_line(
    log_file,
    text: str = "",
) -> None:
    print(
        text,
        flush=True,
    )

    log_file.write(
        text + "\n"
    )

    log_file.flush()


def train_variant(
    config: VariantConfig,
    train_x: torch.Tensor,
    val_x: torch.Tensor,
    seq_jscc: SequenceJSCCWrapper,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """
    Train one controlled ablation variant from scratch.
    """

    # Restart all global RNGs for each variant.
    seed_all(args.seed)

    train_loader = build_loader(
        x=train_x,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed + 10,
        pin_memory=device.type == "cuda",
    )

    val_loader = build_loader(
        x=val_x,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed + 20,
        pin_memory=device.type == "cuda",
    )

    refiner = build_refiner(
        use_snr_cond=config.use_snr_cond,
        device=device,
    )

    # Restore the optimizer used by the original main-model script.
    optimizer = torch.optim.AdamW(
        refiner.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Dedicated random streams ensure identical SNR sequences.
    train_snr_rng = random.Random(
        args.seed + 30
    )

    val_snr_rng = random.Random(
        args.seed + 31
    )

    # Dedicated generators ensure identical t and epsilon sequences.
    train_residual_generator = torch.Generator(
        device=device
    )

    train_residual_generator.manual_seed(
        args.seed + 40
    )

    val_residual_generator = torch.Generator(
        device=device
    )

    val_residual_generator.manual_seed(
        args.seed + 41
    )

    stem = (
        f"onestep_video_swin_tr38901_cr{args.cr}"
        f"_ablation_{config.name}"
    )

    checkpoint_dir = Path(
        args.checkpoint_dir
    )

    log_dir = Path(
        args.log_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        checkpoint_dir
        / f"{stem}.pt"
    )

    last_path = (
        checkpoint_dir
        / f"{stem}_last.pt"
    )

    log_path = (
        log_dir
        / f"train_{stem}.log"
    )

    history_path = (
        log_dir
        / f"train_{stem}.csv"
    )

    best_val = float("inf")
    best_epoch = 0

    history: List[Dict[str, float]] = []

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_file:
        log_line(log_file, "=" * 92)
        log_line(
            log_file,
            "[Train One-Step Video-Swin Three-Way Ablation]",
        )
        log_line(
            log_file,
            f"variant: {config.display_name}",
        )
        log_line(
            log_file,
            f"variant_name: {config.name}",
        )
        log_line(
            log_file,
            f"train: {tuple(train_x.shape)}",
        )
        log_line(
            log_file,
            f"val: {tuple(val_x.shape)}",
        )
        log_line(
            log_file,
            f"base_ckpt: {args.base_ckpt}",
        )
        log_line(
            log_file,
            f"cr: {args.cr}",
        )
        log_line(
            log_file,
            f"epochs: {args.epochs}",
        )
        log_line(
            log_file,
            f"batch_size: {args.batch_size}",
        )
        log_line(
            log_file,
            f"lr: {args.lr}",
        )
        log_line(
            log_file,
            "optimizer: AdamW",
        )
        log_line(
            log_file,
            f"weight_decay: {args.weight_decay}",
        )
        log_line(
            log_file,
            f"grad_clip: {args.grad_clip}",
        )
        log_line(
            log_file,
            f"snrs: {list(SNR_LIST)}",
        )
        log_line(
            log_file,
            f"use_snr_cond: {config.use_snr_cond}",
        )
        log_line(
            log_file,
            f"lambda_rec: {args.lambda_rec}",
        )
        log_line(
            log_file,
            f"lambda_repa: {config.lambda_repa}",
        )
        log_line(
            log_file,
            f"lambda_v: {args.lambda_v}",
        )
        log_line(
            log_file,
            (
                "gate_mode_train_and_val: "
                f"{args.train_gate_mode}"
            ),
        )
        log_line(
            log_file,
            "t_sampling: logit-normal",
        )
        log_line(
            log_file,
            "z_t = t * residual + (1 - t) * epsilon",
        )
        log_line(
            log_file,
            f"seed: {args.seed}",
        )
        log_line(
            log_file,
            f"device: {device}",
        )
        log_line(log_file, "=" * 92)

        for epoch in range(
            1,
            args.epochs + 1,
        ):
            refiner.train()

            totals = empty_totals()
            total_n = 0

            for batch_index, (x,) in enumerate(
                train_loader
            ):
                x = x.to(
                    device,
                    non_blocking=True,
                )

                snr = train_snr_rng.choice(
                    SNR_LIST
                )

                channel_seed = (
                    args.seed
                    + 1_000_000
                    + epoch * 100_000
                    + batch_index
                )

                losses = compute_batch_losses(
                    x=x,
                    seq_jscc=seq_jscc,
                    refiner=refiner,
                    snr=snr,
                    use_snr_cond=config.use_snr_cond,
                    lambda_repa=config.lambda_repa,
                    lambda_rec=args.lambda_rec,
                    lambda_v=args.lambda_v,
                    gate_mode=args.train_gate_mode,
                    channel_seed=channel_seed,
                    residual_generator=(
                        train_residual_generator
                    ),
                    device=device,
                )

                if not torch.isfinite(
                    losses["total"]
                ):
                    raise FloatingPointError(
                        "Non-finite loss at "
                        f"epoch={epoch}, "
                        f"batch={batch_index}."
                    )

                optimizer.zero_grad(
                    set_to_none=True
                )

                losses["total"].backward()

                # Restore the original gradient clipping.
                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        refiner.parameters(),
                        max_norm=args.grad_clip,
                    )
                )

                if not torch.isfinite(
                    grad_norm
                ):
                    raise FloatingPointError(
                        "Non-finite gradient norm at "
                        f"epoch={epoch}, "
                        f"batch={batch_index}."
                    )

                optimizer.step()

                batch_size = x.size(0)

                update_totals(
                    totals,
                    losses,
                    batch_size,
                )

                total_n += batch_size

            train_avg = average_totals(
                totals,
                total_n,
            )

            val_avg = validate_one_epoch(
                refiner=refiner,
                seq_jscc=seq_jscc,
                loader=val_loader,
                config=config,
                epoch=epoch,
                args=args,
                device=device,
                snr_rng=val_snr_rng,
                residual_generator=(
                    val_residual_generator
                ),
            )

            row = {
                "epoch": epoch,
                "train_total": train_avg["total"],
                "train_full": train_avg["full"],
                "train_repa": train_avg["repa"],
                "train_rec": train_avg["rec"],
                "train_v": train_avg["v"],
                "val_total": val_avg["total"],
                "val_full": val_avg["full"],
                "val_repa": val_avg["repa"],
                "val_rec": val_avg["rec"],
                "val_v": val_avg["v"],
            }

            history.append(row)

            write_history(
                history_path,
                history,
            )

            log_line(
                log_file,
                (
                    f"Epoch {epoch:03d} | "
                    f"train={train_avg['total']:.6f} | "
                    f"full={train_avg['full']:.6f} | "
                    f"repa={train_avg['repa']:.6f} | "
                    f"rec={train_avg['rec']:.6f} | "
                    f"v={train_avg['v']:.6f} | "
                    f"val={val_avg['total']:.6f} | "
                    f"val_full={val_avg['full']:.6f} | "
                    f"val_repa={val_avg['repa']:.6f} | "
                    f"val_rec={val_avg['rec']:.6f}"
                ),
            )

            # Keep raw state_dict format for compatibility with the
            # original project testing scripts.
            torch.save(
                refiner.state_dict(),
                last_path,
            )

            if val_avg["total"] < best_val:
                best_val = val_avg["total"]
                best_epoch = epoch

                torch.save(
                    refiner.state_dict(),
                    best_path,
                )

                log_line(
                    log_file,
                    (
                        f"  [Best] epoch={best_epoch}, "
                        f"val={best_val:.8f} "
                        f"-> {best_path}"
                    ),
                )

        log_line(log_file, "=" * 92)
        log_line(
            log_file,
            f"[Done] {config.display_name}",
        )
        log_line(
            log_file,
            f"best_epoch: {best_epoch}",
        )
        log_line(
            log_file,
            f"best_val: {best_val:.8f}",
        )
        log_line(
            log_file,
            f"best_checkpoint: {best_path}",
        )
        log_line(
            log_file,
            f"last_checkpoint: {last_path}",
        )
        log_line(
            log_file,
            f"history_csv: {history_path}",
        )
        log_line(log_file, "=" * 92)

    del optimizer
    del refiner

    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train exactly three controlled Video-Swin "
            "component-ablation variants."
        )
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
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
        "--variant",
        type=str,
        default="all",
        choices=[
            "all",
            "full",
            "no_snr",
            "no_aux",
        ],
    )

    parser.add_argument(
        "--cr",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--lambda-rec",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--lambda-v",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--train-gate-mode",
        type=str,
        default="teacher",
        choices=[
            "teacher",
            "estimate",
        ],
        help=(
            "The same gate mode is used for both "
            "training and validation, matching the "
            "original main-model script."
        ),
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
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
    )

    return parser.parse_args()


def enforce_protocol(
    args: argparse.Namespace,
) -> None:
    """
    Prevent accidental changes to the paper ablation protocol.
    """

    if args.cr != 4:
        raise ValueError(
            "The paper ablation is fixed to CR=4."
        )

    if args.epochs != 300:
        raise ValueError(
            "The paper ablation is fixed to 300 epochs."
        )

    if args.batch_size != 8:
        raise ValueError(
            "The paper ablation is fixed to batch size 8."
        )

    if abs(args.lr - 1e-4) > 1e-15:
        raise ValueError(
            "The paper ablation is fixed to lr=1e-4."
        )

    if abs(args.weight_decay - 0.01) > 1e-15:
        raise ValueError(
            "The paper ablation is fixed to "
            "weight_decay=0.01."
        )

    if abs(args.grad_clip - 1.0) > 1e-15:
        raise ValueError(
            "The paper ablation is fixed to "
            "gradient clipping max_norm=1.0."
        )

    if abs(args.lambda_rec - 0.2) > 1e-15:
        raise ValueError(
            "The paper ablation is fixed to "
            "lambda_rec=0.2."
        )

    if abs(args.lambda_v) > 1e-15:
        raise ValueError(
            "The paper ablation is fixed to "
            "lambda_v=0.0."
        )


def main() -> None:
    args = parse_args()
    enforce_protocol(args)
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

    data_root = Path(args.data_root)

    train_path = (
        data_root
        / "processed"
        / "train_tr38901_seq.pt"
    )

    val_path = (
        data_root
        / "processed"
        / "val_tr38901_seq.pt"
    )

    train_x = load_seq_x(train_path)
    val_x = load_seq_x(val_path)

    base_checkpoint = Path(args.base_ckpt)

    if not base_checkpoint.is_file():
        raise FileNotFoundError(
            f"Base checkpoint not found: {base_checkpoint}"
        )

    base = SEJSCCSNRPrior(
        cr=args.cr
    ).to(device)

    base_state = torch.load(
        base_checkpoint,
        map_location=device,
        weights_only=False,
    )

    base.load_state_dict(
        base_state,
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

    if args.variant == "all":
        selected = [
            VARIANTS["full"],
            VARIANTS["no_snr"],
            VARIANTS["no_aux"],
        ]
    else:
        selected = [
            VARIANTS[args.variant]
        ]

    print("=" * 92)
    print("Strict three-way Video-Swin ablation")
    print(f"train: {tuple(train_x.shape)}")
    print(f"val:   {tuple(val_x.shape)}")
    print(f"base checkpoint: {base_checkpoint}")
    print(f"device: {device}")
    print("Variants:")

    for config in selected:
        print(
            f"  - {config.display_name}: "
            f"use_snr_cond={config.use_snr_cond}, "
            f"lambda_repa={config.lambda_repa}"
        )

    print(
        "Shared final test protocol: "
        "alpha=0.9, guidance=0, t_infer=0.5"
    )
    print("=" * 92)

    for config in selected:
        train_variant(
            config=config,
            train_x=train_x,
            val_x=val_x,
            seq_jscc=seq_jscc,
            args=args,
            device=device,
        )


if __name__ == "__main__":
    main()
