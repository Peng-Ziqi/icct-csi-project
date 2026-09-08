import os
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.se_csinet import SECsiNet
from models.se_csinet_jscc import SECsiNetJSCC        # 你工程已有
from models.se_jscc_snrprior import SEJSCCSNRPrior
from utils.metrics import nmse
from utils.logger import setup_log_redirect
from utils.quantize import quantize_model_copy, model_size_bits


def load_pt_x(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        x = obj
    elif isinstance(obj, dict):
        x = obj.get("x", obj.get("data", obj.get("H", None)))
        if x is None:
            raise KeyError(f"{path} dict无x/data/H键")
    elif isinstance(obj, (list, tuple)):
        x = obj[0]
    else:
        raise TypeError(f"{path}类型不支持")
    if x.ndim == 5 and x.shape[2] == 1:
        x = x.squeeze(2)
    return x.float()


def parse_snr_list(s):
    return [float(v) for v in s.split(",")]


@torch.no_grad()
def eval_avg_nmse(model_name, model, loader, device, snr_list, gate="estimate"):
    """
    返回平均NMSE（对snr_list求平均）
    """
    model.eval()
    all_nmse = []
    for snr in snr_list:
        s, n = 0.0, 0
        for x, _ in loader:
            x = x.to(device)
            if model_name == "se_csinet":
                y = model(x)
            elif model_name == "se_jscc":
                y = model(x, snr_db=snr)
            elif model_name == "final_unified":
                y, _, _, _ = model(x, snr_db=snr, gate_mode=gate)
            else:
                raise ValueError(model_name)

            s += nmse(y, x).item() * x.size(0)
            n += x.size(0)
        all_nmse.append(s / n)
    return sum(all_nmse) / len(all_nmse)


def get_quant_rules(model_name, step, bits):
    """
    step1: 只量化 encoder_fc.weight
    step2: 在step1基础上再量化SE模块 fc
    """
    if step == 1:
        return {"encoder_fc.weight": bits}
    elif step == 2:
        return {
            "encoder_fc.weight": bits,
            "encoder_se.fc": bits,
            "refine_se.fc": bits
        }
    else:
        raise ValueError("step must be 1 or 2")


def build_model(model_name, cr):
    if model_name == "se_csinet":
        return SECsiNet(h=32, w=256, cr=cr, se_reduction=8)
    elif model_name == "se_jscc":
        return SECsiNetJSCC(h=32, w=256, cr=cr, se_reduction=8)
    elif model_name == "final_unified":
        return SEJSCCSNRPrior(h=32, w=256, cr=cr, se_reduction=8)
    else:
        raise ValueError(model_name)


def run_one_model(model_name, ckpt, cr, bits_list, step, loader, device, snr_list, gate):
    model = build_model(model_name, cr).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))

    base_nmse = eval_avg_nmse(model_name, model, loader, device, snr_list, gate)
    base_size = model_size_bits(model, default_bits=32, rules={}, keep_bias_fp32=True)

    rows = []
    rows.append({
        "model": model_name, "step": step, "bits": 32,
        "avg_nmse": base_nmse, "delta_nmse": 0.0,
        "size_mb": base_size / 8 / 1024 / 1024,
        "size_reduction": 0.0
    })

    for b in bits_list:
        rules = get_quant_rules(model_name, step, b)
        qmodel = quantize_model_copy(model, rules=rules, keep_bias_fp32=True).to(device)
        q_nmse = eval_avg_nmse(model_name, qmodel, loader, device, snr_list, gate)
        q_size = model_size_bits(qmodel, default_bits=32, rules=rules, keep_bias_fp32=True)

        rows.append({
            "model": model_name, "step": step, "bits": b,
            "avg_nmse": q_nmse, "delta_nmse": q_nmse - base_nmse,
            "size_mb": q_size / 8 / 1024 / 1024,
            "size_reduction": 1 - q_size / base_size
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--cr", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--snr-list", type=str, default="0,5,10,15,20")
    ap.add_argument("--bits-list", type=str, default="8,4")
    ap.add_argument("--gate", type=str, default="estimate", choices=["teacher", "estimate"])

    ap.add_argument("--ckpt-se-csinet", type=str, required=True)
    ap.add_argument("--ckpt-se-jscc", type=str, required=True)
    ap.add_argument("--ckpt-final", type=str, required=True)

    ap.add_argument("--log-file", type=str, default="")
    args = ap.parse_args()

    if args.log_file == "":
        args.log_file = f"logs/quant_compare_three_models_cr{args.cr}.log"
    _ = setup_log_redirect(args.log_file)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snr_list = parse_snr_list(args.snr_list)
    bits_list = [int(v) for v in args.bits_list.split(",")]

    xte = load_pt_x(os.path.join(args.data_root, "processed", "test.pt"))
    loader = DataLoader(TensorDataset(xte, torch.zeros(len(xte))),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_rows = []

    # step1 对比
    all_rows += run_one_model("se_csinet", args.ckpt_se_csinet, args.cr, bits_list, 1, loader, device, snr_list, args.gate)
    all_rows += run_one_model("se_jscc", args.ckpt_se_jscc, args.cr, bits_list, 1, loader, device, snr_list, args.gate)
    all_rows += run_one_model("final_unified", args.ckpt_final, args.cr, bits_list, 1, loader, device, snr_list, args.gate)

    # step2 对比
    all_rows += run_one_model("se_csinet", args.ckpt_se_csinet, args.cr, bits_list, 2, loader, device, snr_list, args.gate)
    all_rows += run_one_model("se_jscc", args.ckpt_se_jscc, args.cr, bits_list, 2, loader, device, snr_list, args.gate)
    all_rows += run_one_model("final_unified", args.ckpt_final, args.cr, bits_list, 2, loader, device, snr_list, args.gate)

    import pandas as pd
    df = pd.DataFrame(all_rows)
    df = df.sort_values(["step", "model", "bits"]).reset_index(drop=True)

    os.makedirs("logs", exist_ok=True)
    out_csv = f"logs/quant_compare_three_models_cr{args.cr}.csv"
    df.to_csv(out_csv, index=False)

    print(df)
    print(f"[Done] saved: {out_csv}")


if __name__ == "__main__":
    main()
