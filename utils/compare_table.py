import os
import json
import re
import csv
import sys

# 确保从项目根目录导入
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.csinet import CsiNet
from models.se_csinet import SECsiNet
from utils.profile import count_params, count_flops

LOG_DIR = "logs"
OUT_DIR = "plots"
CRS = [4, 8, 16]

def parse_test_nmse(log_path):
    if not os.path.exists(log_path):
        return None
    pattern = re.compile(r"Test NMSE.*:\s*([0-9.]+)")
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return float(m.group(1))
    return None

def main():
    meta = json.load(open(os.path.join("data/processed", "meta.json")))
    h = meta["n_rx"] * meta["n_tx"]
    w = meta["n_subcarriers"]

    rows = []
    for cr in CRS:
        # CsiNet
        csinet = CsiNet(h, w, cr)
        c_params = count_params(csinet)
        c_macs, c_flops = count_flops(csinet, (1, 2, h, w), device="cpu")
        c_nmse = parse_test_nmse(os.path.join(LOG_DIR, f"test_cr{cr}.log"))

        # SE-CsiNet
        se = SECsiNet(h, w, cr, se_reduction=8)
        s_params = count_params(se)
        s_macs, s_flops = count_flops(se, (1, 2, h, w), device="cpu")
        s_nmse = parse_test_nmse(os.path.join(LOG_DIR, f"test_se_cr{cr}.log"))

        rows.append([
            cr,
            c_params, c_macs, c_flops, c_nmse,
            s_params, s_macs, s_flops, s_nmse
        ])

    os.makedirs(OUT_DIR, exist_ok=True)

    # CSV
    csv_path = os.path.join(OUT_DIR, "compare_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "CR",
            "CsiNet Params", "CsiNet MACs", "CsiNet FLOPs", "CsiNet Test NMSE",
            "SE-CsiNet Params", "SE-CsiNet MACs", "SE-CsiNet FLOPs", "SE-CsiNet Test NMSE"
        ])
        for r in rows:
            writer.writerow(r)

    # Markdown
    md_path = os.path.join(OUT_DIR, "compare_table.md")
    with open(md_path, "w") as f:
        f.write("| CR | CsiNet Params | CsiNet MACs | CsiNet FLOPs | CsiNet Test NMSE | "
                "SE-CsiNet Params | SE-CsiNet MACs | SE-CsiNet FLOPs | SE-CsiNet Test NMSE |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r[0]} | {r[1]:,} | {r[2]:,} | {r[3]:,} | {r[4]} | "
                    f"{r[5]:,} | {r[6]:,} | {r[7]:,} | {r[8]} |\n")

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")

if __name__ == "__main__":
    main()
