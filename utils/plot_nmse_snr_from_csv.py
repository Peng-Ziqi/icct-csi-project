# plot_nmse_snr_from_csv.py
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results_full_test/raw_results_all.csv")
    parser.add_argument("--out-dir", type=str, default="results_full_test/figures")
    parser.add_argument("--scene", type=str, default="indoor")
    parser.add_argument("--cr-list", type=str, default="4,8,16")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    name_map = {
        "csinet": "CsiNet",
        "csinetplus": "CsiNet+",
        "se-csinet": "SE-CsiNet"
    }
    df["model_show"] = df["model"].map(lambda x: name_map.get(str(x).lower(), x))

    scene = args.scene
    cr_list = [int(x.strip()) for x in args.cr_list.split(",") if x.strip()]

    d0 = df[df["scene"] == scene].copy()
    if len(d0) == 0:
        raise ValueError(f"csv里没有scene={scene}的数据")

    for cr in cr_list:
        d = d0[d0["cr"] == cr].copy()
        if len(d) == 0:
            print(f"[跳过] scene={scene}, cr={cr} 无数据")
            continue

        plt.figure(figsize=(6, 4))
        for model in sorted(d["model_show"].unique()):
            dm = d[d["model_show"] == model].sort_values("snr_db")
            plt.plot(dm["snr_db"], dm["nmse_db"], marker="o", linewidth=1.8, label=model)

        plt.xlabel("SNR (dB)")
        plt.ylabel("NMSE (dB)")
        plt.title(f"NMSE vs SNR | Scene={scene}, CR={cr}")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        out_png = os.path.join(args.out_dir, f"nmse_snr_scene-{scene}_cr-{cr}.png")
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"[保存] {out_png}")


if __name__ == "__main__":
    main()
