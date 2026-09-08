# utils/plot_results_by_scene.py
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results_full_test/raw_results_all.csv")
    parser.add_argument("--out-dir", type=str, default="results_full_test/figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)

    # 标准化模型名
    name_map = {
        "csinet": "CsiNet",
        "csinetplus": "CsiNet+",
        "se-csinet": "SE-CsiNet"
    }
    df["model_show"] = df["model"].map(lambda x: name_map.get(str(x).lower(), x))

    required_cols = ["model", "scene", "cr", "snr_db", "nmse_db", "rate"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column in CSV: {c}")

    scenes = sorted(df["scene"].unique())
    cr_list = sorted(df["cr"].unique())

    # NMSE-SNR 曲线（按场景画）
    for scene in scenes:
        sub = df[df["scene"] == scene]
        plt.figure(figsize=(6, 4))
        for cr in cr_list:
            sub_cr = sub[sub["cr"] == cr]
            for model in sorted(sub_cr["model_show"].unique()):
                d = sub_cr[sub_cr["model_show"] == model].sort_values("snr_db")
                plt.plot(
                    d["snr_db"],
                    d["nmse_db"],
                    marker="o",
                    linewidth=1.6,
                    label=f"{model} (1/{cr})"
                )
        plt.xlabel("SNR (dB)")
        plt.ylabel("NMSE (dB)")
        plt.title(f"NMSE vs SNR | {scene}")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()

        out_png = os.path.join(args.out_dir, f"NMSE_SNR_{scene}.png")
        out_pdf = os.path.join(args.out_dir, f"NMSE_SNR_{scene}.pdf")
        plt.savefig(out_png, dpi=300)
        plt.savefig(out_pdf)
        plt.close()
        print(f"[SAVED] {out_png}")
        print(f"[SAVED] {out_pdf}")

    # Rate-SNR 曲线（按场景画）
    for scene in scenes:
        sub = df[df["scene"] == scene]
        plt.figure(figsize=(6, 4))
        for cr in cr_list:
            sub_cr = sub[sub["cr"] == cr]
            for model in sorted(sub_cr["model_show"].unique()):
                d = sub_cr[sub_cr["model_show"] == model].sort_values("snr_db")
                plt.plot(
                    d["snr_db"],
                    d["rate"],
                    marker="o",
                    linewidth=1.6,
                    label=f"{model} (1/{cr})"
                )
        plt.xlabel("SNR (dB)")
        plt.ylabel("Achievable Rate (bps/Hz)")
        plt.title(f"Rate vs SNR | {scene}")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()

        out_png = os.path.join(args.out_dir, f"Rate_SNR_{scene}.png")
        out_pdf = os.path.join(args.out_dir, f"Rate_SNR_{scene}.pdf")
        plt.savefig(out_png, dpi=300)
        plt.savefig(out_pdf)
        plt.close()
        print(f"[SAVED] {out_png}")
        print(f"[SAVED] {out_pdf}")

    print("\n[DONE] All figures generated.")


if __name__ == "__main__":
    main()
