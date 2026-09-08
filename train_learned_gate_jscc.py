import os, argparse, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.se_jscc_learned_gate import SEJSCCLearnedGate
from utils.metrics import nmse
from utils.logger import setup_log_redirect

def seed_all(seed=2026):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_pt_x(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor): x = obj
    elif isinstance(obj, dict): x = obj.get("x", obj.get("data", obj.get("H")))
    else: x = obj[0]
    if x.ndim == 5 and x.shape[2] == 1: x = x.squeeze(2)
    return x.float()

@torch.no_grad()
def evaluate(model, loader, device, snr_list, tau=0.5):
    model.eval()
    out = []
    for snr in snr_list:
        s_nmse, s_k, n = 0.0, 0.0, 0
        for x, _ in loader:
            x = x.to(device)
            y, logits, gp, ratio, k = model(x, snr_db=snr, tau=tau, hard=True)
            s_nmse += nmse(y, x).item() * x.size(0)
            s_k += k.float().sum().item()
            n += x.size(0)
        out.append((snr, s_nmse / n, s_k / n))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data")
    ap.add_argument("--cr", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--snr-list", type=str, default="0,5,10,15,20")
    ap.add_argument("--target-overhead", type=float, default=0.6)  # 期望平均比例
    ap.add_argument("--lambda-over", type=float, default=0.2)
    ap.add_argument("--tau-start", type=float, default=1.0)
    ap.add_argument("--tau-end", type=float, default=0.3)
    ap.add_argument("--log-file", type=str, default="")
    args = ap.parse_args()

    if args.log_file == "":
        args.log_file = f"logs/train_learned_gate_jscc_cr{args.cr}.log"
    _ = setup_log_redirect(args.log_file)

    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snr_list = [float(v) for v in args.snr_list.split(",")]

    tr = load_pt_x(os.path.join(args.data_root, "processed", "train.pt"))
    va = load_pt_x(os.path.join(args.data_root, "processed", "val.pt"))
    trl = DataLoader(TensorDataset(tr, torch.zeros(len(tr))), batch_size=args.batch_size, shuffle=True)
    val = DataLoader(TensorDataset(va, torch.zeros(len(va))), batch_size=args.batch_size, shuffle=False)

    model = SEJSCCLearnedGate(cr=args.cr).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = f"checkpoints/learned_gate_jscc_cr{args.cr}.pt"
    best = 1e9

    for ep in range(1, args.epochs + 1):
        model.train()
        tau = args.tau_start + (args.tau_end - args.tau_start) * (ep - 1) / max(1, args.epochs - 1)
        ls, n = 0.0, 0
        for x, _ in trl:
            x = x.to(device)
            snr = random.choice(snr_list)

            y, logits, gp, ratio, k = model(x, snr_db=snr, tau=tau, hard=False)
            recon = mse(y, x)
            over_loss = (ratio.mean() - args.target_overhead) ** 2
            loss = recon + args.lambda_over * over_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            ls += loss.item() * x.size(0); n += x.size(0)

        val_res = evaluate(model, val, device, snr_list, tau=max(0.3, tau))
        avg_nmse = sum([v[1] for v in val_res]) / len(val_res)
        avg_k = sum([v[2] for v in val_res]) / len(val_res)
        print(f"Epoch {ep:03d} | Loss {ls/n:.6f} | Val NMSE {avg_nmse:.6f} | AvgK {avg_k:.1f}/{model.comp_dim} | tau={tau:.3f}")

        if avg_nmse < best:
            best = avg_nmse
            torch.save(model.state_dict(), ckpt)

    print(f"[Done] best={best:.6f}, ckpt={ckpt}")

if __name__ == "__main__":
    main()
