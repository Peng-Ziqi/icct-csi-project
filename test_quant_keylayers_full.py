import os
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.metrics import nmse
from utils.logger import setup_log_redirect
from utils.quantize import quantize_model_copy, model_size_bits

# 你的模型
from models.se_jscc_snrprior import SEJSCCSNRPrior
from models.se_jscc_learned_gate import SEJSCCLearnedGate
from test_snr_aware_jscc_baseline_quant import SECsiNetJSCCCondCompat  # 直接复用你已有兼容类


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
def eval_model(model_name, model, loader, device, snr_list):
    model.eval()
    vals, overs = [], []
    for snr in snr_list:
        s_nmse, s_over, n = 0.0, 0.0, 0
        for x, _ in loader:
            x = x.to(device)
            if model_name == "rule_teacher":
                y, _, _, k = model(x, snr_db=snr, gate_mode="teacher")
                over = (k.float().mean() / model.comp_dim).item()
            elif model_name in ["learned_v1", "learned_v2"]:
                y, _, _, _, k = model(x, snr_db=snr, tau=0.2, hard=True)
                over = (k.float().mean() / model.comp_dim).item()
            elif model_name == "paper_baseline":
                y = model(x, snr_db=snr)
                over = 1.0  # 固定满码长近似
            else:
                raise ValueError(model_name)

            s_nmse += nmse(y, x).item() * x.size(0)
            s_over += over * x.size(0)
            n += x.size(0)

        vals.append(s_nmse / n)
        overs.append(s_over / n)

    return sum(vals) / len(vals), sum(overs) / len(overs)


def build_model(model_name, cr, ckpt, device):
    if model_name == "rule_teacher":
        m = SEJSCCSNRPrior(cr=cr).to(device)
    elif model_name in ["learned_v1", "learned_v2"]:
        m = SEJSCCLearnedGate(cr=cr).to(device)
    elif model_name == "paper_baseline":
        m = SECsiNetJSCCCondCompat(cr=cr).to(device)
    else:
        raise ValueError(model_name)

    sd = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    return m


def keylayer_rules(bits):
    # 关键层尽可能全覆盖（按名字前缀规则）
    return {
        "encoder_fc.weight": bits,
        "decoder_fc.weight": bits,

        "encoder_conv.0.weight": bits,
        "refine.0.weight": bits,
        "refine.3.weight": bits,
        "refine_out.0.weight": bits,

        "encoder_se.fc": bits,
        "refine_se.fc": bits,

        "snr_embed.0.weight": bits,
        "snr_embed.2.weight": bits,

        "gate_net.feat.0.weight": bits,
        "gate_net.mlp.0.weight": bits,
        "gate_net.mlp.2.weight": bits,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", type=str, required=True,
                    choices=["rule_teacher", "learned_v1", "learned_v2", "paper_baseline"])
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--cr", type=int, default=4)
    ap.add_argument("--bits-list", type=str, default="8,4")
    ap.add_argument("--snr-list", type=str, default="0,5,10,15,20")
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log-file", type=str, default="")
    args = ap.parse_args()

    if args.log_file == "":
        args.log_file = f"logs/test_quant_keylayers_full_{args.model_name}_cr{args.cr}.log"
    _ = setup_log_redirect(args.log_file)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snr_list = [float(v) for v in args.snr_list.split(",")]
    bits_list = [int(v) for v in args.bits_list.split(",")]

    xte = load_pt_x(os.path.join(args.data_root, "processed", "test.pt"))
    loader = DataLoader(TensorDataset(xte, torch.zeros(len(xte))),
                        batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.model_name, args.cr, args.ckpt, device)
    base_nmse, base_over = eval_model(args.model_name, model, loader, device, snr_list)
    base_size = model_size_bits(model, default_bits=32, rules={}, keep_bias_fp32=True)

    print(f"[Base] NMSE={base_nmse:.6f}, Over={base_over:.4f}, Size={base_size/8/1024/1024:.3f}MB")

    csvf = f"logs/test_quant_keylayers_full_{args.model_name}_cr{args.cr}.csv"
    with open(csvf, "w") as f:
        f.write("bits,avg_nmse,avg_overhead,size_mb,size_reduction\n")

    for b in bits_list:
        rules = keylayer_rules(b)
        qmodel = quantize_model_copy(model, rules=rules, keep_bias_fp32=True).to(device)
        q_nmse, q_over = eval_model(args.model_name, qmodel, loader, device, snr_list)
        q_size = model_size_bits(qmodel, default_bits=32, rules=rules, keep_bias_fp32=True)
        print(f"[Q{b}] NMSE={q_nmse:.6f}, Over={q_over:.4f}, Size={q_size/8/1024/1024:.3f}MB, Reduce={1-q_size/base_size:.4f}")
        with open(csvf, "a") as f:
            f.write(f"{b},{q_nmse:.8f},{q_over:.6f},{q_size/8/1024/1024:.6f},{1-q_size/base_size:.6f}\n")

    print(f"[Done] csv={csvf}")


if __name__ == "__main__":
    main()
