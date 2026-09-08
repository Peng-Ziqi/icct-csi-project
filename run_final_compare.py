# run_final_compare.py
import os
import argparse
import subprocess


def run(cmd):
    print("\n" + "=" * 120)
    print("[CMD]", cmd)
    print("=" * 120)
    subprocess.run(cmd, shell=True, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--snr-list", type=str, default="0,5,10,15,20")
    ap.add_argument("--cr-list", type=str, default="4,8,16")
    args = ap.parse_args()

    crs = [int(v) for v in args.cr_list.split(",")]
    os.makedirs("logs", exist_ok=True)

    for cr in crs:
        # 1) 训练统一主线
        run(
            f"python -u train_final_unified.py "
            f"--cr {cr} --epochs {args.epochs} --snr-list {args.snr-list if False else args.snr_list} "
            f"--device {args.device} --log-file logs/train_final_unified_cr{cr}.log"
        )

        # 2) 测试 teacher / estimate
        run(
            f"python -u test_final_unified.py "
            f"--cr {cr} --ckpt checkpoints/final_unified_cr{cr}.pt --snr-list {args.snr_list} "
            f"--gate teacher --device {args.device} --log-file logs/test_final_unified_cr{cr}_teacher.log"
        )
        run(
            f"python -u test_final_unified.py "
            f"--cr {cr} --ckpt checkpoints/final_unified_cr{cr}.pt --snr-list {args.snr_list} "
            f"--gate estimate --device {args.device} --log-file logs/test_final_unified_cr{cr}_estimate.log"
        )

        # 3) 量化（以estimate门控为部署模式）
        run(
            f"python -u test_final_unified_quant.py "
            f"--cr {cr} --ckpt checkpoints/final_unified_cr{cr}.pt --step 1 --bits-list 8,4 "
            f"--gate estimate --snr-list {args.snr_list} --device {args.device} "
            f"--log-file logs/test_final_unified_quant_cr{cr}_step1.log"
        )
        run(
            f"python -u test_final_unified_quant.py "
            f"--cr {cr} --ckpt checkpoints/final_unified_cr{cr}.pt --step 2 --bits-list 8,4 "
            f"--gate estimate --snr-list {args.snr_list} --device {args.device} "
            f"--log-file logs/test_final_unified_quant_cr{cr}_step2.log"
        )

    print("\n[Done] run_final_compare finished.")


if __name__ == "__main__":
    main()
