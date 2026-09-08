# test_innov5_quant_step1.py
import os
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.se_csinet import SECsiNet
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
def eval_nmse(model, loader, device):
    model.eval()
    s, c = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        y = model(x)
        s += nmse(y, x).item() * x.size(0)
        c += x.size(0)
    return s / c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--cr", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bits-list", type=str, default="8,4,2")
    parser.add_argument("--log-file", type=str, default="")
    args = parser.parse_args()

    if args.log_file == "":
        args.log_file = f"logs/test_innov5_quant_step1_cr{args.cr}.log"
    _ = setup_log_redirect(args.log_file)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bits_list = [int(v) for v in args.bits_list.split(",")]

    test_x = load_pt_x(os.path.join(args.data_root, "processed", "test.pt"))
    test_loader = DataLoader(TensorDataset(test_x, torch.zeros(len(test_x))),
                             batch_size=args.batch_size, shuffle=False)

    model = SECsiNet(h=32, w=256, cr=args.cr, se_reduction=8).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False))

    base_nmse = eval_nmse(model, test_loader, device)
    base_size = model_size_bits(model, default_bits=32, rules={}, keep_bias_fp32=True)
    print(f"[Base] NMSE={base_nmse:.6f}, Size={base_size/8/1024/1024:.3f} MB")

    csvf = f"logs/test_innov5_quant_step1_cr{args.cr}.csv"
    with open(csvf, "w", encoding="utf-8") as f:
        f.write("bits,nmse,size_mb,size_reduction_ratio\n")

    for b in bits_list:
        # 仅量化encoder_fc.weight，bias保持FP32
        rules = {"encoder_fc.weight": b}
        qmodel = quantize_model_copy(model, rules=rules, keep_bias_fp32=True).to(device)
        q_nmse = eval_nmse(qmodel, test_loader, device)
        q_size = model_size_bits(qmodel, default_bits=32, rules=rules, keep_bias_fp32=True)

        print(f"[Q{b}] NMSE={q_nmse:.6f}, Size={q_size/8/1024/1024:.3f} MB, Reduce={(1-q_size/base_size):.4f}")
        with open(csvf, "a", encoding="utf-8") as f:
            f.write(f"{b},{q_nmse:.8f},{q_size/8/1024/1024:.6f},{1-q_size/base_size:.6f}\n")

    print(f"[Done] csv={csvf}, log={args.log_file}")


if __name__ == "__main__":
    main()
