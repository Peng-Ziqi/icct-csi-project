# plot_thesis_figures.py
import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 140


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def plot_dynamic_nmse_overhead():
    """
    用 results/summary_innov4_dynamic.csv
    画两张图：
    1) 不同CR下 teacher/estimate 的 Avg NMSE
    2) 不同CR下 teacher/estimate 的 Avg Overhead
    """
    f = "results/summary_innov4_dynamic.csv"
    if not os.path.exists(f):
        print(f"[Skip] {f} not found.")
        return

    df = pd.read_csv(f)
    df = df.sort_values(["cr", "gate"])

    crs = sorted(df["cr"].unique())
    gates = ["teacher", "estimate"]

    # NMSE 柱状图
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    x = range(len(crs))
    for i, g in enumerate(gates):
        vals = [df[(df["cr"] == c) & (df["gate"] == g)]["avg_nmse"].values[0] for c in crs]
        pos = [p + (i - 0.5) * width for p in x]
        ax.bar(pos, vals, width=width, label=g)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"CR=1/{c}" for c in crs])
    ax.set_ylabel("Average NMSE")
    ax.set_title("Innovation4 Dynamic Compression: Avg NMSE")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig("figures/innov4_avg_nmse_bar.png")
    plt.close()

    # Overhead 柱状图
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, g in enumerate(gates):
        vals = [df[(df["cr"] == c) & (df["gate"] == g)]["avg_overhead"].values[0] for c in crs]
        pos = [p + (i - 0.5) * width for p in x]
        ax.bar(pos, vals, width=width, label=g)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"CR=1/{c}" for c in crs])
    ax.set_ylabel("Average Feedback Overhead Ratio")
    ax.set_title("Innovation4 Dynamic Compression: Avg Overhead")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig("figures/innov4_avg_overhead_bar.png")
    plt.close()

    print("[Saved] figures/innov4_avg_nmse_bar.png")
    print("[Saved] figures/innov4_avg_overhead_bar.png")


def plot_quant_nmse_size():
    """
    用 results/summary_innov5_quant.csv
    对每个CR画一张双子图：
      左：bits vs NMSE（step1/step2）
      右：bits vs model size(MB)（step1/step2）
    """
    f = "results/summary_innov5_quant.csv"
    if not os.path.exists(f):
        print(f"[Skip] {f} not found.")
        return

    df = pd.read_csv(f)
    crs = sorted(df["cr"].unique())

    for cr in crs:
        d = df[df["cr"] == cr].copy()
        d1 = d[d["type"] == "innov5_step1"].sort_values("bits")
        d2 = d[d["type"] == "innov5_step2"].sort_values("bits")

        bits = d1["bits"].tolist()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # NMSE
        axes[0].plot(bits, d1["nmse"].tolist(), marker="o", label="Step1")
        axes[0].plot(bits, d2["nmse"].tolist(), marker="s", label="Step2")
        axes[0].set_xticks(bits)
        axes[0].set_xlabel("Quantization Bits")
        axes[0].set_ylabel("NMSE")
        axes[0].set_title(f"CR=1/{cr}: NMSE vs Bits")
        axes[0].grid(alpha=0.2)
        axes[0].legend()

        # Size
        axes[1].plot(bits, d1["size_mb"].tolist(), marker="o", label="Step1")
        axes[1].plot(bits, d2["size_mb"].tolist(), marker="s", label="Step2")
        axes[1].set_xticks(bits)
        axes[1].set_xlabel("Quantization Bits")
        axes[1].set_ylabel("Model Size (MB)")
        axes[1].set_title(f"CR=1/{cr}: Size vs Bits")
        axes[1].grid(alpha=0.2)
        axes[1].legend()

        plt.tight_layout()
        out = f"figures/innov5_cr{cr}_nmse_size.png"
        plt.savefig(out)
        plt.close()
        print(f"[Saved] {out}")


def plot_dynamic_per_snr():
    """
    从 logs/test_innov4_dynamic_cr{cr}_{gate}.csv
    画每个CR下 NMSE-SNR 曲线（teacher vs estimate）
    """
    for cr in [4, 8, 16]:
        f_t = f"logs/test_innov4_dynamic_cr{cr}_teacher.csv"
        f_e = f"logs/test_innov4_dynamic_cr{cr}_estimate.csv"
        if not (os.path.exists(f_t) and os.path.exists(f_e)):
            continue

        dt = pd.read_csv(f_t).sort_values("snr_db")
        de = pd.read_csv(f_e).sort_values("snr_db")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(dt["snr_db"], dt["nmse"], marker="o", label="teacher-gate")
        ax.plot(de["snr_db"], de["nmse"], marker="s", label="estimate-gate")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("NMSE")
        ax.set_title(f"Innovation4 Dynamic NMSE-SNR (CR=1/{cr})")
        ax.grid(alpha=0.2)
        ax.legend()
        plt.tight_layout()
        out = f"figures/innov4_cr{cr}_nmse_snr.png"
        plt.savefig(out)
        plt.close()
        print(f"[Saved] {out}")


def main():
    ensure_dir("figures")
    plot_dynamic_nmse_overhead()
    plot_dynamic_per_snr()
    plot_quant_nmse_size()
    print("[Done] all figures generated.")


if __name__ == "__main__":
    main()
