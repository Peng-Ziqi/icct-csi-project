import os
import re
import matplotlib.pyplot as plt

LOG_DIR = "logs"
PLOT_DIR = "plots"

CRS = [4, 8, 16]

def parse_val_nmse(log_path):
    vals = []
    if not os.path.exists(log_path):
        return vals
    pattern = re.compile(r"Val NMSE\s+([0-9.]+)")
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                vals.append(float(m.group(1)))
    return vals

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

def plot_nmse_epoch(model_tag, prefix):
    plt.figure()
    for cr in CRS:
        log = os.path.join(LOG_DIR, f"{prefix}_cr{cr}.log")
        vals = parse_val_nmse(log)
        if vals:
            plt.plot(vals, label=f"CR={cr}")
    plt.title(f"{model_tag} Val NMSE / Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("NMSE")
    plt.grid(True)
    plt.legend()
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOT_DIR, f"nmse_epoch_{model_tag.lower()}.png"), dpi=200)

def plot_cr_nmse_compare():
    csinet = []
    se = []
    for cr in CRS:
        csinet.append(parse_test_nmse(os.path.join(LOG_DIR, f"test_cr{cr}.log")))
        se.append(parse_test_nmse(os.path.join(LOG_DIR, f"test_se_cr{cr}.log")))

    plt.figure()
    plt.plot(CRS, csinet, marker="o", label="CsiNet")
    plt.plot(CRS, se, marker="o", label="SE-CsiNet")
    plt.title("Test NMSE vs CR")
    plt.xlabel("CR")
    plt.ylabel("Test NMSE")
    plt.grid(True)
    plt.legend()
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOT_DIR, "cr_nmse_compare.png"), dpi=200)

def main():
    # 三条曲线：各 CR 的 NMSE/epoch
    plot_nmse_epoch("CsiNet", "train")
    plot_nmse_epoch("SE-CsiNet", "train_se")

    # CR-NMSE 对比（CsiNet vs SE-CsiNet）
    plot_cr_nmse_compare()

if __name__ == "__main__":
    main()
