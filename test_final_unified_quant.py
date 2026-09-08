# test_final_unified_quant.py
import os
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

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


@torch.no_grad()
def eval_nmse_overhead(model, loader, device, snr_list, gate_mode):
    model.eval()
    dmax = model.comp_dim
    all_nmse, all_over = [], []
    for snr in snr_list:
        s_nmse, s_k, n = 0.0, 0.0, 0
        for x, _ in loader:
            x = x.to(device)
            y, snr_est, ratio, k = model(x, snr_db=snr, gate_mode=gate_mode)
            s_nmse += nmse(y, x).item() * x.size(0)
            s_k += k.float().sum().item()
            n += x.size(0)
        all_nmse.append(s_nmse / n)
        all_over.append((s_k / n) / dmax)
    return sum(all_nmse) / len(all_nmse), sum(all_over) / len(all_over)


def parse_snr_list(s):
    return [float(v) for v in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--cr", type=int, default=4)
    ap.add_argument("--bits-list", type=str, default="8,4")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--snr-list", type=str, default="0,5,10,15,20")
    ap.add_argument("--gate", type=str, default="estimate", choices=["teacher", "estimate"])
    ap.add_argument("--step", type=int, default=1, choices=[1, 2])
    ap.add_argument("--log-file", type=str, default="")
    args = ap.parse_args()

    if args.log_file == "":
        args.log_file = f"logs/test_final_unified_quant_cr{args.cr}_step{args.step}_{args.gate}.log"
    _ = setup_log_redirect(args.log_file)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bits_list = [int(v) for v in args.bits_list.split(",")]
    snrs = parse_snr_list(args.snr_list)

    xte = load_pt_x(os.path.join(args.data_root, "processed", "test.pt"))
    loader = DataLoader(TensorDataset(xte, torch.zeros(len(xte))),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SEJSCCSNRPrior(cr=args.cr).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False))

    base_nmse, base_over = eval_nmse_overhead(model, loader, dev, snrs, args.gate)
    base_size = model_size_bits(model, default_bits=32, rules={}, keep_bias_fp32=True)

    print(f"[Base] gate={args.gate}, AvgNMSE={base_nmse:.6f}, AvgOver={base_over:.4f}, Size={base_size/8/1024/1024:.3f}MB")

    os.makedirs("logs", exist_ok=True)
    csvf = f"logs/test_final_unified_quant_cr{args.cr}_step{args.step}_{args.gate}.csv"
    with open(csvf, "w", encoding="utf-8") as f:
        f.write("bits,avg_nmse,avg_overhead,size_mb,size_reduction\n")

    for b in bits_list:
        if args.step == 1:
            rules = {"encoder_fc.weight": b}
        else:
            rules = {
                "encoder_fc.weight": b,
                "encoder_se.fc": b,
                "refine_se.fc": b
            }

        qmodel = quantize_model_copy(model, rules=rules, keep_bias_fp32=True).to(dev)
        q_nmse, q_over = eval_nmse_overhead(qmodel, loader, dev, snrs, args.gate)
        q_size = model_size_bits(qmodel, default_bits=32, rules=rules, keep_bias_fp32=True)

        print(f"[Q{b}-S{args.step}] AvgNMSE={q_nmse:.6f}, AvgOver={q_over:.4f}, Size={q_size/8/1024/1024:.3f}MB, Reduce={(1-q_size/base_size):.4f}")
        with open(csvf, "a", encoding="utf-8") as f:
            f.write(f"{b},{q_nmse:.8f},{q_over:.6f},{q_size/8/1024/1024:.6f},{1-q_size/base_size:.6f}\n")

    print(f"[Done] csv={csvf}, log={args.log_file}")


if __name__ == "__main__":
    main()
