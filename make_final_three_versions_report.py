import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 140


def read_quant(path, scheme, step):
    df = pd.read_csv(path)
    need = ["bits", "avg_nmse", "avg_overhead", "size_mb", "size_reduction"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"{path} 缺少列 {c}")
    df = df[need].copy()
    df["scheme"] = scheme
    df["step"] = step
    return df


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    files = [
        ("logs/test_final_unified_quant_cr4_step1_teacher.csv", "rule_teacher_gate", "Step1"),
        ("logs/test_final_unified_quant_cr4_step2_teacher.csv", "rule_teacher_gate", "Step2"),
        ("logs/test_learned_gate_jscc_quant_cr4_step1_v1.csv", "learned_gate_v1", "Step1"),
        ("logs/test_learned_gate_jscc_quant_cr4_step2_v1.csv", "learned_gate_v1", "Step2"),
        ("logs/test_learned_gate_jscc_quant_cr4_step1_v2.csv", "learned_gate_v2", "Step1"),
        ("logs/test_learned_gate_jscc_quant_cr4_step2_v2.csv", "learned_gate_v2", "Step2"),
    ]

    rows = []
    for f, s, st in files:
        if not os.path.exists(f):
            print(f"[Warn] missing: {f}")
            continue
        rows.append(read_quant(f, s, st))

    if len(rows) == 0:
        print("[Error] no files found.")
        return

    df = pd.concat(rows, ignore_index=True)
    qdf = df[df["bits"].isin([8, 4])].copy()
    qdf = qdf.sort_values(["scheme", "step", "bits"], ascending=[True, True, False]).reset_index(drop=True)
    qdf.to_csv("results/final_three_versions_step_table_cr4.csv", index=False)

    algo = qdf.groupby(["scheme", "bits"], as_index=False).agg(
        avg_nmse=("avg_nmse", "mean"),
        avg_overhead=("avg_overhead", "mean"),
        size_mb=("size_mb", "mean"),
        size_reduction=("size_reduction", "mean"),
    ).sort_values(["bits", "avg_nmse"])
    algo.to_csv("results/final_three_versions_algo_table_cr4.csv", index=False)

    # markdown
    md = ["| Scheme | Bits | Avg NMSE | Avg Overhead | Size(MB) | Size Reduction |",
          "|---|---:|---:|---:|---:|---:|"]
    for _, r in algo.iterrows():
        md.append(f"| {r['scheme']} | {int(r['bits'])} | {r['avg_nmse']:.6f} | {r['avg_overhead']:.4f} | {r['size_mb']:.3f} | {r['size_reduction']:.4f} |")
    with open("results/final_three_versions_algo_table_cr4.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # 图1 NMSE
    fig, ax = plt.subplots(figsize=(7,4))
    schemes = ["rule_teacher_gate", "learned_gate_v1", "learned_gate_v2"]
    bits = [8,4]
    w = 0.22
    x = range(len(bits))
    for i, s in enumerate(schemes):
        vals = []
        for b in bits:
            d = algo[(algo["scheme"]==s)&(algo["bits"]==b)]
            vals.append(float(d["avg_nmse"].values[0]) if len(d) else float("nan"))
        pos = [p+(i-1)*w for p in x]
        ax.bar(pos, vals, width=w, label=s)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["8-bit","4-bit"])
    ax.set_ylabel("Avg NMSE")
    ax.set_title("CR=1/4 NMSE Comparison (Three Gate Versions)")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig("figures/final_three_versions_nmse_cr4.png")
    plt.close()

    # 图2 Overhead
    fig, ax = plt.subplots(figsize=(7,4))
    over = algo[algo["bits"]==8][["scheme","avg_overhead"]].set_index("scheme").reindex(schemes)
    ax.bar(over.index, over["avg_overhead"])
    ax.set_ylabel("Avg Overhead")
    ax.set_title("CR=1/4 Overhead Comparison (8-bit row)")
    ax.grid(alpha=0.2, axis="y")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("figures/final_three_versions_overhead_cr4.png")
    plt.close()

    print("[Saved] results/final_three_versions_step_table_cr4.csv")
    print("[Saved] results/final_three_versions_algo_table_cr4.csv")
    print("[Saved] results/final_three_versions_algo_table_cr4.md")
    print("[Saved] figures/final_three_versions_nmse_cr4.png")
    print("[Saved] figures/final_three_versions_overhead_cr4.png")
    print("\n".join(md))


if __name__ == "__main__":
    main()
