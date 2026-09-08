import os, argparse, torch
from torch.utils.data import DataLoader, TensorDataset
from models.se_jscc_learned_gate import SEJSCCLearnedGate
from utils.metrics import nmse
from utils.logger import setup_log_redirect
from utils.quantize import quantize_model_copy, model_size_bits

def load_pt_x(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor): x = obj
    elif isinstance(obj, dict): x = obj.get("x", obj.get("data", obj.get("H")))
    else: x = obj[0]
    if x.ndim == 5 and x.shape[2] == 1: x = x.squeeze(2)
    return x.float()

@torch.no_grad()
def eval_avg(model, loader, device, snr_list, tau=0.3):
    model.eval()
    all_nmse, all_over = [], []
    for snr in snr_list:
        s_nmse, s_k, n = 0.0, 0.0, 0
        for x, _ in loader:
            x = x.to(device)
            y, logits, gp, ratio, k = model(x, snr_db=snr, tau=tau, hard=True)
            s_nmse += nmse(y, x).item() * x.size(0)
            s_k += k.float().sum().item()
            n += x.size(0)
        all_nmse.append(s_nmse / n)
        all_over.append((s_k / n) / model.comp_dim)
    return sum(all_nmse)/len(all_nmse), sum(all_over)/len(all_over)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--cr", type=int, default=4)
    ap.add_argument("--step", type=int, default=1, choices=[1,2])
    ap.add_argument("--bits-list", type=str, default="8,4")
    ap.add_argument("--snr-list", type=str, default="0,5,10,15,20")
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--log-file", type=str, default="")
    args = ap.parse_args()

    if args.log_file == "":
        args.log_file = f"logs/test_learned_gate_jscc_quant_cr{args.cr}_step{args.step}.log"
    _ = setup_log_redirect(args.log_file)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snr_list = [float(v) for v in args.snr_list.split(",")]
    bits_list = [int(v) for v in args.bits_list.split(",")]

    x = load_pt_x(os.path.join(args.data_root, "processed", "test.pt"))
    loader = DataLoader(TensorDataset(x, torch.zeros(len(x))), batch_size=args.batch_size, shuffle=False)

    model = SEJSCCLearnedGate(cr=args.cr).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False))

    base_nmse, base_over = eval_avg(model, loader, device, snr_list, tau=args.tau)
    base_size = model_size_bits(model, default_bits=32, rules={}, keep_bias_fp32=True)
    print(f"[Base] NMSE={base_nmse:.6f}, Over={base_over:.4f}, Size={base_size/8/1024/1024:.3f}MB")

    csvf = f"logs/test_learned_gate_jscc_quant_cr{args.cr}_step{args.step}.csv"
    with open(csvf, "w") as f:
        f.write("bits,avg_nmse,avg_overhead,size_mb,size_reduction\n")

    for b in bits_list:
        if args.step == 1:
            rules = {"encoder_fc.weight": b}
        else:
            rules = {
                "encoder_fc.weight": b,
                "encoder_se.fc": b,
                "refine_se.fc": b,
                "gate_net.mlp.0.weight": b,
                "gate_net.mlp.2.weight": b
            }

        qmodel = quantize_model_copy(model, rules=rules, keep_bias_fp32=True).to(device)
        q_nmse, q_over = eval_avg(qmodel, loader, device, snr_list, tau=args.tau)
        q_size = model_size_bits(qmodel, default_bits=32, rules=rules, keep_bias_fp32=True)

        print(f"[Q{b}-S{args.step}] NMSE={q_nmse:.6f}, Over={q_over:.4f}, Size={q_size/8/1024/1024:.3f}MB, Reduce={1-q_size/base_size:.4f}")
        with open(csvf, "a") as f:
            f.write(f"{b},{q_nmse:.8f},{q_over:.6f},{q_size/8/1024/1024:.6f},{1-q_size/base_size:.6f}\n")

    print(f"[Done] csv={csvf}, log={args.log_file}")

if __name__ == "__main__":
    main()
