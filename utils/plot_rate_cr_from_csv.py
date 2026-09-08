# utils/plot_rate_cr_from_csv.py
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def cr_to_ratio_label(cr: int) -> str:
    # 你当前定义：cr=4/8/16 -> 压缩比 1/4,1/8,1/16
    return f"1/{cr}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results_full_test/raw_results_all.csv")
    parser.add_argument("--out-dir", type=str, default="results_full_test/figures")
    parser.add_argument("--scene", type=str, default="indoor")
    parser.add_argument("--snr-list", type=str, default="0,10,20")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)

    # 模型显示名
    name_map = {
        "csinet": "CsiNet",
        "csinetplus": "CsiNet+",
        "se-csinet": "SE-CsiNet"
    }
    df["model_show"] = df["model"].map(lambda x: name_map.get(str(x).lower(), x))

    scene = args.scene
    snr_list = [int(x.strip()) for x in args.snr_list.split(",") if x.strip()]

    d0 = df[df["scene"] == scene].copy()
    if len(d0) == 0:
        raise ValueError(f"csv里没有 scene={scene} 数据")

    # CR排序
    cr_order = sorted(d0["cr"].unique().tolist())
    ratio_labels = [cr_to_ratio_label(c) for c in cr_order]

    for snr in snr_list:
        d = d0[d0["snr_db"] == snr].copy()
        if len(d) == 0:
            print(f"[跳过] scene={scene}, snr={snr} 无数据")
            continue

        # 给每个cr一个固定x位置
        x_map = {cr: i for i, cr in enumerate(cr_order)}

        plt.figure(figsize=(6, 4))
        for model in sorted(d["model_show"].unique()):
            dm = d[d["model_show"] == model].copy()
            dm = dm.sort_values("cr")
            xs = [x_map[c] for c in dm["cr"].tolist()]
            ys = dm["rate"].tolist()
            plt.plot(xs, ys, marker="o", linewidth=1.8, label=model)

        plt.xticks(range(len(cr_order)), ratio_labels)
        plt.xlabel("Compression ratio")
        plt.ylabel("Achievable Rate (bps/Hz)")
        plt.title(f"Rate vs Compression ratio (1/4,1/8,1/16) | Scene={scene}, SNR={snr} dB")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        out_png = os.path.join(args.out_dir, f"rate_ratio_scene-{scene}_snr-{snr}.png")
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"[保存] {out_png}")


if __name__ == "__main__":
    main()
