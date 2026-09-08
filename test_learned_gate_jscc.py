import os, argparse, torch
from torch.utils.data import DataLoader, TensorDataset
from models.se_jscc_learned_gate import SEJSCCLearnedGate
from utils.metrics import nmse
from utils.logger import setup_log_redirect

def load_pt_x(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor): x = obj
    elif isinstance(obj, dict): x = obj.get("x", obj.get("data", obj.get("H")))
    else: x = obj[0]
    if x.ndim == 5 and x.shape[2] == 1: x = x.squeeze(2)
    return x.float()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--cr", type=int, default=4)
    ap.add_argument("--snr-list", type=str, default="0,5,10,15,20")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--log-file", type=str, default="")
    args = ap.parse_args()

    if args.log_file == "":
        args.log_file = f"logs/test_learned_gate_jscc_cr{args.cr}.log"
    _ = setup_log_redirect(args.log_file)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snrs = [float(v) for v in args.snr_list.split(",")]

    x = load_pt_x(os.path.join(args.data-root if False else args.data_root, "processed", "test.pt"))
    loader = DataLoader(TensorDataset(x, torch.zeros(len(x))), batch_size=args.batch_size, shuffle=False)

    model = SEJSCCLearnedGate(cr=args.cr).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev, weights_only=False))
    model.eval()

    csvf = f"logs/test_learned_gate_jscc_cr{args.cr}.csv"
    with open(csvf, "w") as f:
        f.write("snr_db,nmse,avg_k,overhead\n")

    all_nmse, all_over = [], []
    with torch.no_grad():
        for snr in snrs:
            s_nmse, s_k, n = 0.0, 0.0, 0
            for xb, _ in loader:
                xb = xb.to(dev)
                y, logits, gp, ratio, k = model(xb, snr_db=snr, tau=args.tau, hard=True)
                s_nmse += nmse(y, xb).item() * xb.size(0)
                s_k += k.float().sum().item()
                n += xb.size(0)
            nmsev = s_nmse / n
            avgk = s_k / n
            over = avgk / model.comp_dim
            print(f"SNR={snr:.1f} | NMSE={nmsev:.6f} | AvgK={avgk:.1f}/{model.comp_dim} | Overhead={over:.4f}")
            with open(csvf, "a") as f:
                f.write(f"{snr},{nmsev:.8f},{avgk:.4f},{over:.6f}\n")
            all_nmse.append(nmsev); all_over.append(over)

    print(f"[Summary] Avg NMSE={sum(all_nmse)/len(all_nmse):.6f}, Avg Overhead={sum(all_over)/len(all_over):.6f}")
    print(f"[Done] csv={csvf}")

if __name__ == "__main__":
    main()
