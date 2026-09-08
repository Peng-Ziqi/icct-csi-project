# run_all_experiments.py
import os
import subprocess
import argparse
from datetime import datetime


def run(cmd, dry=False):
    print("\n" + "=" * 120)
    print("[CMD]", cmd)
    print("=" * 120)
    if not dry:
        subprocess.run(cmd, shell=True, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--snr-list", type=str, default="0,5,10,15,20")
    parser.add_argument("--cr-list", type=str, default="4,8,16")
    parser.add_argument("--base-ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--enable-innov4", action="store_true")
    parser.add_argument("--enable-innov5", action="store_true")
    args = parser.parse_args()

    cr_list = [int(v) for v in args.cr_list.split(",")]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)

    # 你已有的baseline ckpt命名可按需改这里
    # 例如 se_init_default_cr4_seed2026.pt
    def baseline_ckpt(cr):
        return os.path.join(args.base_ckpt_dir, f"se_init_default_cr{cr}_seed2026.pt")

    # ---------- 创新点4：动态压缩 ----------
    if args.enable_innov4:
        for cr in cr_list:
            run(
                f"python -u train_innov4_dynamic.py "
                f"--cr {cr} --epochs {args.epochs} --snr-list {args.snr_list} "
                f"--device {args.device} "
                f"--log-file logs/{timestamp}_train_innov4_dynamic_cr{cr}.log",
                dry=args.dry_run
            )
            run(
                f"python -u test_innov4_dynamic.py "
                f"--cr {cr} --ckpt checkpoints/se_dynamic_cr{cr}.pt --snr-list {args.snr_list} "
                f"--gate teacher --device {args.device} "
                f"--log-file logs/{timestamp}_test_innov4_dynamic_cr{cr}_teacher.log",
                dry=args.dry_run
            )
            run(
                f"python -u test_innov4_dynamic.py "
                f"--cr {cr} --ckpt checkpoints/se_dynamic_cr{cr}.pt --snr-list {args.snr_list} "
                f"--gate estimate --device {args.device} "
                f"--log-file logs/{timestamp}_test_innov4_dynamic_cr{cr}_estimate.log",
                dry=args.dry_run
            )

    # ---------- 创新点5：量化 ----------
    if args.enable_innov5:
        for cr in cr_list:
            ckpt = baseline_ckpt(cr)
            run(
                f"python -u test_innov5_quant_step1.py "
                f"--cr {cr} --ckpt {ckpt} --bits-list 8,4,2 --device {args.device} "
                f"--log-file logs/{timestamp}_test_innov5_step1_cr{cr}.log",
                dry=args.dry_run
            )
            run(
                f"python -u test_innov5_quant_step2.py "
                f"--cr {cr} --ckpt {ckpt} --bits-list 8,4,2 --device {args.device} "
                f"--log-file logs/{timestamp}_test_innov5_step2_cr{cr}.log",
                dry=args.dry_run
            )

    print("\n[Done] run_all_experiments finished.")


if __name__ == "__main__":
    main()
