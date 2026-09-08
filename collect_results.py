# collect_results.py
import os
import glob
import re
import pandas as pd


def collect_dynamic():
    rows = []
    files = glob.glob("logs/test_innov4_dynamic_cr*_*.csv")
    for f in files:
        name = os.path.basename(f).replace(".csv", "")
        # 匹配: test_innov4_dynamic_cr4_teacher / ..._estimate
        m = re.search(r"test_innov4_dynamic_cr(\d+)_(teacher|estimate)", name)
        if m is None:
            continue
        cr = int(m.group(1))
        gate = m.group(2)

        df = pd.read_csv(f)
        rows.append({
            "file": name,
            "type": "innov4_dynamic",
            "cr": cr,
            "gate": gate,
            "avg_nmse": float(df["nmse"].mean()),
            "avg_overhead": float(df["overhead_ratio"].mean())
        })
    return pd.DataFrame(rows)


def collect_quant():
    rows = []
    # Step1
    for f in glob.glob("logs/test_innov5_quant_step1_cr*.csv"):
        name = os.path.basename(f).replace(".csv", "")
        m = re.search(r"test_innov5_quant_step1_cr(\d+)", name)
        if m is None:
            continue
        cr = int(m.group(1))
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            rows.append({
                "file": name,
                "type": "innov5_step1",
                "cr": cr,
                "bits": int(r["bits"]),
                "nmse": float(r["nmse"]),
                "size_mb": float(r["size_mb"]),
                "size_reduction_ratio": float(r["size_reduction_ratio"])
            })

    # Step2
    for f in glob.glob("logs/test_innov5_quant_step2_cr*.csv"):
        name = os.path.basename(f).replace(".csv", "")
        m = re.search(r"test_innov5_quant_step2_cr(\d+)", name)
        if m is None:
            continue
        cr = int(m.group(1))
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            rows.append({
                "file": name,
                "type": "innov5_step2",
                "cr": cr,
                "bits": int(r["bits"]),
                "nmse": float(r["nmse"]),
                "size_mb": float(r["size_mb"]),
                "size_reduction_ratio": float(r["size_reduction_ratio"])
            })

    return pd.DataFrame(rows)


def main():
    os.makedirs("results", exist_ok=True)

    d1 = collect_dynamic()
    d2 = collect_quant()

    if len(d1) > 0:
        d1 = d1.sort_values(["cr", "gate"]).reset_index(drop=True)
        d1.to_csv("results/summary_innov4_dynamic.csv", index=False)
        print("[Saved] results/summary_innov4_dynamic.csv")
        print(d1)
    else:
        print("[Warn] no innov4 csv found.")

    if len(d2) > 0:
        d2 = d2.sort_values(["type", "cr", "bits"]).reset_index(drop=True)
        d2.to_csv("results/summary_innov5_quant.csv", index=False)
        print("[Saved] results/summary_innov5_quant.csv")
        print(d2.head(12))
    else:
        print("[Warn] no innov5 csv found.")


if __name__ == "__main__":
    main()
