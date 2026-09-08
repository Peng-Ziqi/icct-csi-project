import os, argparse, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.se_jscc_snrprior import SEJSCCSNRPrior
from utils.metrics import nmse
from utils.logger import setup_log_redirect

def seed_all(seed=2026):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_pt_x(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor): x = obj
    elif isinstance(obj, dict): x = obj.get("x", obj.get("data", obj.get("H")))
    else: x = obj[0]
    if x.ndim == 5 and x.shape[2] == 1: x = x.squeeze(2)
    return x.float()

def parse_snr_list(s): return [float(v) for v in s.split(",")]

@torch.no_grad()
def eval_model(model, loader, device, snr_list, gate_mode="teacher"):
    model.eval()
    out = {}
    for snr in snr_list:
        s_nmse, s_k, n = 0.0, 0.0, 0
        for x, _ in loader:
            x = x.to(device)
            y, snr_est, ratio, k = model(x, snr_db=snr, gate_mode=gate_mode)
            s_nmse += nmse(y, x).item() * x.size(0)
            s_k += k.float().sum().item()
            n += x.size(0)
        out[snr] = {"nmse": s_nmse / n, "avg_k": s_k / n}
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
    ap.add_argument("--lambda-snr", type=float, default=0.01)
    ap.add_argument("--log-file", type=str, default="")
    args = ap.parse_args()

    if args.log_file == "":
        args.log_file = f"logs/train_final_unified_cr{args.cr}.log"
    _ = setup_log_redirect(args.log_file)

    seed_all(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    snrs = parse_snr_list(args.snr_list)
    print(f"[Info] device={dev}, cr={args.cr}, snrs={snrs}, epochs={args.epochs}")

    tr = load_pt_x(os.path.join(args.data_root, "processed", "train.pt"))
    va = load_pt_x(os.path.join(args.data_root, "processed", "val.pt"))
    trl = DataLoader(TensorDataset(tr, torch.zeros(len(tr))), batch_size=args.batch_size, shuffle=True)
    val = DataLoader(TensorDataset(va, torch.zeros(len(va))), batch_size=args.batch_size, shuffle=False)

    model = SEJSCCSNRPrior(cr=args.cr).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    os.makedirs("checkpoints", exist_ok=True); os.makedirs("logs", exist_ok=True)
    ckpt = f"checkpoints/final_unified_cr{args.cr}.pt"
    csvf = f"logs/train_final_unified_cr{args.cr}.csv"
    with open(csvf, "w") as f:
        f.write("epoch,train_loss,val_nmse_teacher,val_k_teacher,val_nmse_est,val_k_est\n")

    best = 1e9
    dmax = model.comp_dim

    for ep in range(1, args.epochs + 1):
        model.train()
        ls, n = 0.0, 0
        for x, _ in trl:
            x = x.to(dev)
            snr = random.choice(snrs)
            snr_t = torch.full((x.size(0), 1), float(snr), device=dev)

            opt.zero_grad()
            y, snr_est, ratio, k = model(x, snr_db=snr, gate_mode="teacher")
            loss = mse(y, x) + args.lambda_snr * mse(snr_est, snr_t)
            loss.backward()
            opt.step()
            ls += loss.item() * x.size(0); n += x.size(0)

        tr_loss = ls / n
        vt = eval_model(model, val, dev, snrs, "teacher")
        ve = eval_model(model, val, dev, snrs, "estimate")
        nmse_t = sum([vt[s]["nmse"] for s in snrs]) / len(snrs)
        k_t = sum([vt[s]["avg_k"] for s in snrs]) / len(snrs)
        nmse_e = sum([ve[s]["nmse"] for s in snrs]) / len(snrs)
        k_e = sum([ve[s]["avg_k"] for s in snrs]) / len(snrs)

        with open(csvf, "a") as f:
            f.write(f"{ep},{tr_loss:.8f},{nmse_t:.8f},{k_t:.4f},{nmse_e:.8f},{k_e:.4f}\n")

        print(f"Epoch {ep:03d} | Loss {tr_loss:.6f} | T {nmse_t:.6f} K {k_t:.1f}/{dmax} | E {nmse_e:.6f} K {k_e:.1f}/{dmax}")
        if nmse_t < best:
            best = nmse_t
            torch.save(model.state_dict(), ckpt)

    print(f"[Done] best_teacher_nmse={best:.6f}, ckpt={ckpt}, csv={csvf}")

if __name__ == "__main__":
    main()
