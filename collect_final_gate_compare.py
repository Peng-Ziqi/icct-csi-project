import os
import pandas as pd


def read_quant_csv(path, scheme, step_name):
    df = pd.read_csv(path)
    # 统一列名
    # bits,avg_nmse,avg_overhead,size_mb,size_reduction
    need_cols = ["bits", "avg_nmse", "avg_overhead", "size_mb", "size_reduction"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"{path} 缺少列: {c}")
    df = df[need_cols].copy()
    df["scheme"] = scheme
    df["step"] = step_name
    return df


def main():
    os.makedirs("results", exist_ok=True)

    files = [
        ("logs/test_final_unified_quant_cr4_step1_teacher.csv", "rule_teacher_gate", "Step1"),
        ("logs/test_final_unified_quant_cr4_step2_teacher.csv", "rule_teacher_gate", "Step2"),
        ("logs/test_learned_gate_jscc_quant_cr4_step1.csv", "learned_gate_jscc", "Step1"),
        ("logs/test_learned_gate_jscc_quant_cr4_step2.csv", "learned_gate_jscc", "Step2"),
    ]

    all_df = []
    for f, scheme, step in files:
        if not os.path.exists(f):
            print(f"[Warn] missing: {f}")
            continue
        all_df.append(read_quant_csv(f, scheme, step))

    if len(all_df) == 0:
        print("[Error] no valid csv found.")
        return

    df = pd.concat(all_df, ignore_index=True)

    # 仅保留 bits=8/4 量化行
    qdf = df[df["bits"].isin([8, 4])].copy()

    # 生成终表
    qdf = qdf[["scheme", "step", "bits", "avg_nmse", "avg_overhead", "size_mb", "size_reduction"]]
    qdf = qdf.sort_values(["scheme", "step", "bits"], ascending=[True, True, False]).reset_index(drop=True)

    out_csv = "results/final_gate_quant_compare_cr4.csv"
    qdf.to_csv(out_csv, index=False)

    # markdown表
    md_lines = []
    md_lines.append("| Scheme | Step | Bits | Avg NMSE | Avg Overhead | Size (MB) | Size Reduction |")
    md_lines.append("|---|---|---:|---:|---:|---:|---:|")
    for _, r in qdf.iterrows():
        md_lines.append(
            f"| {r['scheme']} | {r['step']} | {int(r['bits'])} | "
            f"{r['avg_nmse']:.6f} | {r['avg_overhead']:.4f} | "
            f"{r['size_mb']:.3f} | {r['size_reduction']:.4f} |"
        )
    out_md = "results/final_gate_quant_compare_cr4.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"[Saved] {out_csv}")
    print(f"[Saved] {out_md}")
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
