# utils/export_complexity_table.py
import os
import csv
import argparse
import torch

from models.csinet import CsiNet
from models.csinet_plus import CsiNetPlus
from models.se_csinet import SECsiNet


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def try_profile_macs_flops(model, input_shape=(1, 2, 32, 256), device="cpu"):
    """
    优先使用 thop 统计 MACs/FLOPs。
    FLOPs 近似按 2*MACs 计算（常见口径）。
    若 thop 不可用，返回 None。
    """
    try:
        from thop import profile
    except Exception:
        return None, None

    model = model.to(device).eval()
    x = torch.randn(*input_shape).to(device)
    with torch.no_grad():
        macs, _ = profile(model, inputs=(x,), verbose=False)
    flops = 2.0 * macs
    return macs, flops


def human_readable(n):
    if n is None:
        return "N/A"
    n = float(n)
    if n >= 1e9:
        return f"{n/1e9:.3f} G"
    if n >= 1e6:
        return f"{n/1e6:.3f} M"
    if n >= 1e3:
        return f"{n/1e3:.3f} K"
    return f"{n:.0f}"


def build_model(model_name, cr, h=32, w=256):
    name = model_name.lower()
    if name == "csinet":
        return CsiNet(h=h, w=w, cr=cr)
    elif name == "csinetplus":
        return CsiNetPlus(h=h, w=w, cr=cr)
    elif name == "se-csinet":
        return SECsiNet(h=h, w=w, cr=cr)
    else:
        raise ValueError(f"未知模型: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="results_full_test")
    parser.add_argument("--cr-list", type=str, default="4,8,16")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--h", type=int, default=32)
    parser.add_argument("--w", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cr_list = [int(x.strip()) for x in args.cr_list.split(",") if x.strip()]
    model_names = ["csinet", "se-csinet", "csinetplus"]

    rows = []
    for cr in cr_list:
        for m in model_names:
            model = build_model(m, cr=cr, h=args.h, w=args.w)

            params = count_params(model)
            trainable = count_trainable_params(model)
            macs, flops = try_profile_macs_flops(
                model, input_shape=(1, 2, args.h, args.w), device=args.device
            )

            rows.append({
                "model": m,
                "cr": cr,
                "compression_ratio": f"1/{cr}",
                "params": int(params),
                "trainable_params": int(trainable),
                "macs": None if macs is None else float(macs),
                "flops": None if flops is None else float(flops),
                "params_hr": human_readable(params),
                "macs_hr": human_readable(macs),
                "flops_hr": human_readable(flops),
            })

    # 保存CSV
    out_csv = os.path.join(args.out_dir, "complexity_table.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "cr", "compression_ratio",
            "params", "trainable_params", "macs", "flops",
            "params_hr", "macs_hr", "flops_hr"
        ])
        for r in rows:
            writer.writerow([
                r["model"], r["cr"], r["compression_ratio"],
                r["params"], r["trainable_params"], r["macs"], r["flops"],
                r["params_hr"], r["macs_hr"], r["flops_hr"]
            ])

    # 保存LaTeX表格
    out_tex = os.path.join(args.out_dir, "complexity_table.tex")
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{模型复杂度对比（参数量与计算量）}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Model & Compression Ratio & Params & MACs & FLOPs \\\\\n")
        f.write("\\hline\n")
        for r in rows:
            model_show = {
                "csinet": "CsiNet",
                "se-csinet": "SE-CsiNet",
                "csinetplus": "CsiNet+"
            }.get(r["model"], r["model"])
            f.write(
                f"{model_show} & {r['compression_ratio']} & {r['params_hr']} & {r['macs_hr']} & {r['flops_hr']} \\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:complexity_compare}\n")
        f.write("\\end{table}\n")

    print(f"[完成] CSV: {out_csv}")
    print(f"[完成] TEX: {out_tex}")
    print("提示：若 MACs/FLOPs 显示 N/A，请先安装 thop: pip install thop")


if __name__ == "__main__":
    main()
