import os
import re
import argparse
import matplotlib.pyplot as plt

def parse_val_nmse(log_path):
    vals = []
    pattern = re.compile(r"Val NMSE ([0-9\.]+)")
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                vals.append(float(m.group(1)))
    return vals

def parse_test_nmse(log_path):
    pattern = re.compile(r"Test NMSE \(CR=\d+\): ([0-9\.]+)")
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return float(m.group(1))
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--out", type=str, default="plots/nmse_val_test.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cr_list = [4, 8, 16]
    plt.figure(figsize=(7, 5))

    for cr in cr_list:
        train_log = os.path.join(args.log_dir, f"train_cr{cr}.log")
        test_log = os.path.join(args.log_dir, f"test_cr{cr}.log")

        if not os.path.exists(train_log):
            print(f"Missing: {train_log}")
            continue

        vals = parse_val_nmse(train_log)
        if not vals:
            print(f"No Val NMSE found in {train_log}")
            continue

        label = f"CR=1/{cr} Val"
        plt.plot(vals, label=label)

        test_nmse = None
        if os.path.exists(test_log):
            test_nmse = parse_test_nmse(test_log)

        if test_nmse is not None:
            x = len(vals) - 1
            y = test_nmse
            plt.scatter([x], [y], s=40, marker="x")
            plt.text(x, y, f" Test {test_nmse:.4f}", fontsize=8)

    plt.title("CsiNet NMSE (Val Curves + Test Points)")
    plt.xlabel("Epoch")
    plt.ylabel("NMSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved plot to {args.out}")

if __name__ == "__main__":
    main()
